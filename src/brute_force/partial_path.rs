//! Representation of a partially explored path in brute force search

use super::FeedPair;
use crate::{
    cache::{self, CacheModel, CacheSimulation},
    FeedIdx, MAX_FEEDS, MAX_PAIRS,
};
use slotmap::{DefaultKey, SlotMap};

/// Machine word size in bits
const WORD_SIZE: u32 = (std::mem::size_of::<usize>() * 8) as u32;

/// Integer division with upper rounding
const fn div_round_up(num: usize, denom: usize) -> usize {
    (num / denom) + (num % denom != 0) as usize
}

/// Maximum number of machine words needed to store one bit per feed pair
const MAX_PAIR_WORDS: usize = div_round_up(MAX_PAIRS, WORD_SIZE as usize);

/// Path which we are in the process of exploring
pub struct PartialPath {
    /// Path steps and cumulative cache costs
    ///
    /// This is stored as a linked list with node deduplication across paths of
    /// identical origin in order to save memory capacity and bandwidth that
    /// would otherwise be spent copying previous path steps for every child of
    /// a single parent path.
    ///
    /// The price to pay, however, is that this linked tree format is
    /// extraordinarily expensive to access. Therefore, a subset of this list's
    /// information is duplicated inline below for fast access, so that the list
    /// only needs to be traversed in the rare event where a path turns out to
    /// be a new cache cost / extra distance record.
    ///
    path: PathLink,

    /// Length of the path in steps
    path_len: usize,

    /// Last path step
    curr_step: FeedPair,

    /// Total cache cost, accumulated over previous path steps
    curr_cost: cache::Cost,

    /// Deviation of path steps from basic (1, 0) steps
    extra_distance: StepDistance,

    /// Bitmap of feed pairs that we've been through
    visited_pairs: [usize; MAX_PAIR_WORDS],

    /// Current state of the cache simulation
    cache_sim: CacheSimulation,
}
//
/// Total distance that was "walked" across a path step
pub type StepDistance = f32;
//
impl PartialPath {
    /// Index of a certain coordinate in the visited_pairs bitvec
    //
    // NOTE: This operation is super hot and must be very fast
    //
    const fn coord_to_bit_index(&[x, y]: &FeedPair) -> (usize, u32) {
        let linear_index = y as usize * MAX_FEEDS as usize + x as usize;
        let word_index = linear_index / WORD_SIZE as usize;
        let bit_index = (linear_index % WORD_SIZE as usize) as u32;
        (word_index, bit_index)
    }

    // Inverse of to_bit_index
    //
    // NOTE: This operation is very rare and can be slow
    //
    const fn bit_index_to_coord(word: usize, bit: u32) -> FeedPair {
        let linear_index = word * WORD_SIZE as usize + bit as usize;
        let y = (linear_index / (MAX_FEEDS as usize)) as FeedIdx;
        let x = (linear_index % (MAX_FEEDS as usize)) as FeedIdx;
        [x, y]
    }

    /// Start a path
    //
    // NOTE: This operation is very rare and can be slow
    //
    pub fn new(
        path_elem_storage: &mut PathElemStorage,
        cache_model: &CacheModel,
        start_step: FeedPair,
    ) -> Self {
        let mut cache_sim = cache_model.start_simulation();
        let mut curr_cost = 0.0;
        for &feed in start_step.iter() {
            curr_cost += cache_sim.simulate_access(&cache_model, feed);
        }

        let path = PathLink::new(path_elem_storage, start_step, curr_cost, None);

        let mut visited_pairs = [0; MAX_PAIR_WORDS];
        for word in 0..MAX_PAIR_WORDS {
            let mut current_word = 0;
            for bit in (0..WORD_SIZE).rev() {
                let [x, y] = Self::bit_index_to_coord(word, bit);
                let visited = [x, y] == start_step;
                current_word = (current_word << 1) | (visited as usize);
            }
            visited_pairs[word] = current_word;
        }

        Self {
            path,
            path_len: 1,
            curr_step: start_step,
            curr_cost,
            visited_pairs,
            cache_sim,
            extra_distance: 0.0,
        }
    }

    /// Tell how long the path is
    //
    // NOTE: This operation is hot and must be fast
    //
    pub fn len(&self) -> usize {
        self.path_len
    }

    /// Tell how much excess distance was covered through path stepping
    //
    // NOTE: This operation is hot and must be fast
    //
    pub fn extra_distance(&self) -> StepDistance {
        self.extra_distance
    }

    /// Get the last path entry
    //
    // NOTE: This operation is hot and must be fast
    //
    pub fn last_step(&self) -> &FeedPair {
        &self.curr_step
    }

    /// Iterate over the path steps and cumulative costs in reverse step order
    //
    // NOTE: This operation can be slow, it is only called when a better path
    //       has been found (very rare) or when displaying debug output.
    //
    pub fn iter_rev<'a>(
        &'a self,
        path_elem_storage: &'a PathElemStorage,
    ) -> impl Iterator<Item = (FeedPair, cache::Cost)> + 'a {
        let mut next_node = Some(&self.path);
        std::iter::from_fn(move || {
            let node = next_node?.get(path_elem_storage);
            next_node = node.prev_steps.as_ref();
            Some((node.curr_step, node.curr_cost))
        })
    }

    /// Tell whether a path contains a certain feed pair
    //
    // NOTE: This operation is super hot and must be very fast
    //
    pub fn contains(&self, pair: &FeedPair) -> bool {
        let (word, bit) = Self::coord_to_bit_index(pair);
        debug_assert!(word < self.visited_pairs.len());
        (unsafe { self.visited_pairs.get_unchecked(word) } & (1 << bit)) != 0
    }

    /// Get the accumulated cache cost of following this path so far
    //
    // NOTE: This operation is hot and must be fast
    //
    pub fn cost_so_far(&self) -> cache::Cost {
        self.curr_cost
    }

    /// Given an extra feed pair, tell what the accumulated cache cost would
    /// become if the path was completed by this pair, and what the cache
    /// entries would then be.
    //
    // NOTE: This operation is super hot and must be very fast
    //
    pub fn evaluate_next_step(
        &self,
        cache_model: &CacheModel,
        &next_step: &FeedPair,
    ) -> NextStepEvaluation {
        let mut next_cache = self.cache_sim.clone();
        let next_cost = self.cost_so_far()
            + next_step
                .iter()
                .map(|&feed| next_cache.simulate_access(&cache_model, feed))
                .sum::<f32>();
        NextStepEvaluation {
            next_step,
            next_cost,
            next_cache,
        }
    }

    /// Create a new partial path which follows all the steps from this one,
    /// plus an extra step for which the new cache cost and updated cache model
    /// are provided.
    //
    // NOTE: This operation is relatively hot and must be quite fast
    //
    pub fn commit_next_step(
        &self,
        path_elem_storage: &mut PathElemStorage,
        next_step_eval: NextStepEvaluation,
    ) -> Self {
        let NextStepEvaluation {
            next_step,
            next_cost,
            next_cache,
        } = next_step_eval;

        let prev_steps = Some(self.path.clone(path_elem_storage));
        let next_path = PathLink::new(path_elem_storage, next_step, next_cost, prev_steps);

        let mut next_visited_pairs = self.visited_pairs.clone();
        let (word, bit) = Self::coord_to_bit_index(&next_step);
        // TODO: Make sure the bound check is elided or has negligible cost, if
        //       it is too expensive use get_unchecked.
        next_visited_pairs[word] |= 1 << bit;

        let step_length = self
            .last_step()
            .iter()
            .zip(next_step.iter())
            .map(|(&curr_coord, &next_coord)| {
                ((next_coord as StepDistance) - (curr_coord as StepDistance)).powi(2)
            })
            .sum::<StepDistance>()
            .sqrt();

        Self {
            path: next_path,
            path_len: self.path_len + 1,
            curr_step: next_step,
            curr_cost: next_cost,
            visited_pairs: next_visited_pairs,
            cache_sim: next_cache,
            extra_distance: self.extra_distance + (step_length - 1.0),
        }
    }

    /// Cleanly dispose of the path's elements
    //
    // FIXME: Abstract this away by providing an abstraction that combines a
    //        PartialPath and an &mut PathElemStorage, so that we can properly
    //        implement Drop.
    //
    pub(crate) fn drop_elems(&mut self, path_elem_storage: &mut PathElemStorage) {
        self.path.dispose(path_elem_storage);
    }
}

/// Storage for PartialPath path elements
///
/// PartialPath steps are stored as a linked list, with node deduplication so
/// that when a path forks into multiple sub-paths, we don't need to make
/// multiple copies of the parent path (which costs RAM capacity and bandwidth).
///
/// Originally, this list was stored using Rc<PathElem>, but the overhead
/// associated with allocating and liberating all those PathElems turned out to
/// be too great. So we're now reusing allocations instead.
///
pub struct PathElemStorage(SlotMap<DefaultKey, PathElem>);
//
impl PathElemStorage {
    /// Set up storage for path elements
    pub fn new() -> Self {
        Self(SlotMap::new())
    }
}

/// Reference-counted PartialPath path element
struct PathElem {
    /// Number of references to that path element in existence
    ///
    /// This is 1 when a path is created, increases to N when a path is forked
    /// into sub-paths, and once it drops to 0, all sub-paths have been fully
    /// explored, and this path an all of its parent paths can be disposed of.
    ///
    reference_count: u8,

    /// Last step that was taken on that path
    curr_step: FeedPair,

    /// Total cache cost after taking this step
    curr_cost: cache::Cost,

    /// Previous steps that were taken on this path
    prev_steps: Option<PathLink>,
}
//
#[cfg(debug_assertions)]
impl Drop for PathElem {
    fn drop(&mut self) {
        assert_eq!(
            self.reference_count, 0,
            "PathElem dropped while references still existed (according to refcount)"
        );
    }
}

/// Link to a PathElem in PathElemStorage
struct PathLink {
    /// Key of the target PathElem in the underlying PathElemStorage
    key: DefaultKey,

    /// In debug mode, we make sure that PathElems are correctly disposed of
    #[cfg(debug_assertions)]
    disposed: bool,
}
//
impl PathLink {
    /// Record a new path element
    fn new(
        storage: &mut PathElemStorage,
        curr_step: FeedPair,
        curr_cost: cache::Cost,
        prev_steps: Option<PathLink>,
    ) -> Self {
        let key = storage.0.insert(PathElem {
            reference_count: 1,
            curr_step,
            curr_cost,
            prev_steps,
        });
        Self {
            key,
            #[cfg(debug_assertions)]
            disposed: false,
        }
    }

    /// Read-only access to a path element from storage
    fn get<'storage>(&self, storage: &'storage PathElemStorage) -> &'storage PathElem {
        #[cfg(debug_assertions)]
        {
            debug_assert!(!self.disposed);
        }
        &storage.0[self.key]
    }

    /// Mutable access to a path element from storage
    fn get_mut<'storage>(&self, storage: &'storage mut PathElemStorage) -> &'storage mut PathElem {
        #[cfg(debug_assertions)]
        {
            debug_assert!(!self.disposed);
        }
        &mut storage.0[self.key]
    }

    /// Make a new PathLink pointing to the same PathElem
    fn clone(&self, storage: &mut PathElemStorage) -> Self {
        self.get_mut(storage).reference_count += 1;
        Self {
            key: self.key,
            #[cfg(debug_assertions)]
            disposed: false,
        }
    }

    /// Invalidate a PathLink, possibly disposing of the underlying storage
    fn dispose(&mut self, storage: &mut PathElemStorage) {
        #[cfg(debug_assertions)]
        {
            debug_assert!(!self.disposed);
        }
        let path_elem = self.get_mut(storage);
        path_elem.reference_count -= 1;
        if path_elem.reference_count == 0 {
            let prev_steps = path_elem.prev_steps.take();
            storage.0.remove(self.key);
            for mut prev_steps in prev_steps {
                prev_steps.dispose(storage);
            }
        }
        #[cfg(debug_assertions)]
        {
            self.disposed = true;
        }
    }
}
//
#[cfg(debug_assertions)]
impl Drop for PathLink {
    fn drop(&mut self) {
        assert!(
            self.disposed,
            "PathLink dropped without cleaning up the underlying PathElem"
        );
    }
}

/// Result of `PartialPath::evaluate_next_step()`
pub struct NextStepEvaluation {
    /// Next step to be taken (repeated to simplify commit_next_step signature)
    next_step: FeedPair,

    /// Total cache cost after taking this step
    pub next_cost: cache::Cost,

    /// Cache state after taking this step
    next_cache: CacheSimulation,
}
