//! Representation of a partially explored path in brute force search

mod history;

pub use self::history::PathElemStorage;

use self::history::PathLink;
use super::FeedPair;
use crate::{
    cache::{self, CacheModel, CacheSimulation},
    FeedIdx, MAX_FEEDS, MAX_PAIRS, _MAX_UNORDERED_PAIRS,
};
use num_traits::identities::Zero;
use static_assertions::const_assert;
use std::{
    cell::{Ref, RefCell},
    ops::Deref,
};

/// Machine word size in bits
const WORD_SIZE: u32 = (std::mem::size_of::<usize>() * 8) as u32;

/// Integer division with upper rounding
const fn div_round_up(num: usize, denom: usize) -> usize {
    (num / denom) + (num % denom != 0) as usize
}

/// Maximum number of machine words needed to store one bit per feed pair
const MAX_PAIR_WORDS: usize = div_round_up(MAX_PAIRS, WORD_SIZE as usize);

/// Path which we are in the process of exploring
pub struct PartialPath<'storage> {
    /// Inner data
    data: PartialPathData,

    /// Underlying cache model
    cache_model: &'storage CacheModel,

    /// Path element storage
    ///
    /// This must be RefCell'd because we want to mutably access the storage on
    /// Drop (to auto-delete path elements), and still have mutable access to
    /// the storage for other purposes like creating new sub-paths.
    ///
    path_elem_storage: &'storage RefCell<PathElemStorage>,
}
//
impl<'storage> PartialPath<'storage> {
    /// Start a path
    pub fn new(
        path_elem_storage: &'storage RefCell<PathElemStorage>,
        cache_model: &'storage CacheModel,
        start_step: FeedPair,
    ) -> Self {
        Self {
            data: PartialPathData::new(
                &mut *path_elem_storage.borrow_mut(),
                cache_model,
                start_step,
            ),
            cache_model,
            path_elem_storage,
        }
    }

    /// Iterate over the path steps and cumulative costs in reverse step order
    pub fn iter_rev(&'storage self) -> impl Iterator<Item = (FeedPair, cache::Cost)> + 'storage {
        self.data.iter_rev(self.path_elem_storage.borrow())
    }

    /// Given an extra feed pair, tell what the accumulated cache cost would
    /// become if the path was completed by this pair, and what the cache
    /// entries would then be.
    //
    // NOTE: This operation is super hot and must be very fast
    //
    pub fn evaluate_next_step(&self, next_step: &FeedPair) -> NextStepEvaluation {
        self.data.evaluate_next_step(self.cache_model, next_step)
    }

    /// Create a new partial path which follows all the steps from this one,
    /// plus an extra step for which the new cache cost and updated cache model
    /// are provided.
    pub fn commit_next_step(&self, next_step_eval: NextStepEvaluation) -> Self {
        Self {
            data: self
                .data
                .commit_next_step(&mut *self.path_elem_storage.borrow_mut(), next_step_eval),
            cache_model: self.cache_model,
            path_elem_storage: self.path_elem_storage,
        }
    }

    /// Compose a partial path's data and its storage into a full PartialPath
    pub(super) fn wrap(
        data: PartialPathData,
        cache_model: &'storage CacheModel,
        path_elem_storage: &'storage RefCell<PathElemStorage>,
    ) -> Self {
        // Prefetch the first path element
        //
        // We will need it to decrement the reference count when this
        // PartialPath is dropped, and may need it before that for other
        // operations like forking the path into sub-paths, so this won't be
        // wasted cache work.
        //
        data.path.prefetch(&*path_elem_storage.borrow());

        // Return the wrapped PartialPath
        Self {
            data,
            cache_model,
            path_elem_storage,
        }
    }

    /// Get back the inner PartialPathData, typically for container insertion
    pub(super) fn unwrap(self) -> PartialPathData {
        // So, I'm sure there's a cleaner way to do this, but I haven't found
        // it yet. Basically, the issue here is that...
        //
        // - I cannot move out of the wrapper because it implements Drop.
        // - I cannot remove the Drop impl because it is very much critical to
        //   the wrapper's functionality.
        // - I very much want a way to move data out of the wrapper when I'm
        //   going to store said data into a container.
        // - I cannot use Option because that would have runtime overhead which
        //   I very much do not want in this hot part of the code.
        // - I'm convinced that it is actually safe to move out of this type as
        //   long as I inhibit its destructor with `mem::forget()`
        //
        // Therefore, I'm resorting to forcing a move behind the borrow
        // checker's back, under the assumption that this is safe because...
        //
        // 1. PartialPathData is a trivial type, almost just a bunch of numbers,
        //    it does not contain data which is unsafe to copy like &mut refs.
        // 2. Double dropping will not occur because I'm forgetting `self`
        //    immediately after performing the read, with no possibility of
        //    panicking inbetween these two events.
        //
        let data_ptr = &self.data as *const PartialPathData;
        let data = unsafe { data_ptr.read() };
        std::mem::forget(self);
        data
    }
}
//
impl Deref for PartialPath<'_> {
    type Target = PartialPathData;
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}
//
impl Drop for PartialPath<'_> {
    fn drop(&mut self) {
        self.data
            .drop_elems(&mut *self.path_elem_storage.borrow_mut())
    }
}

/// Data about a path which we are in the process of exploring
///
/// This struct tries to be as compact as possible (since it pretty much
/// single-handedly dominates our memory capacity and bandwidth usage), but as a
/// result it is incomplete without secondary information that is common to all
/// partial paths, namely the location of the PathElemStorage container where
/// all path elements are stored, and the underlying CPU cache model.
///
/// This notably means that it cannot follow RAII idioms and must follow a
/// custom destruction protocol, which is error-prone. Therefore, manipulating
/// this struct directly is not recommended, unless you are building a path
/// container. You should instead use the `PartialPath` convenience wrapper.
///
// SAFETY: PartialPathData must not be made to contain types which are unsafe
//         to copy such as &mut references.
//
pub struct PartialPathData {
    /// Path steps and cumulative cache costs
    ///
    /// This is stored as a linked list with node deduplication across paths of
    /// identical origin, in order to save memory capacity and bandwidth that
    /// would otherwise be spent copying previous path steps for every child of
    /// a single parent path.
    ///
    /// The price to pay, however, is that this linked tree format is
    /// extraordinarily expensive to access. Therefore, a subset of this list's
    /// information is duplicated inline below for fast access, so that the list
    /// only needs to be traversed for debug output and in the rare event where
    /// a path turns out to beat a new cache cost / extra distance record.
    ///
    path: PathLink,

    /// Length of the path in steps
    path_len: PathLen,

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
/// Type appropriate for representing path lengths
pub type PathLen = u8;
const_assert!(PathLen::MAX as usize >= _MAX_UNORDERED_PAIRS);
//
/// Total distance that was "walked" across a path step
pub type StepDistance = f32;
//
impl PartialPathData {
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
    fn new(
        path_elem_storage: &mut PathElemStorage,
        cache_model: &CacheModel,
        start_step: FeedPair,
    ) -> Self {
        let mut cache_sim = cache_model.start_simulation();
        let mut curr_cost = cache::Cost::zero();
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
        self.path_len as usize
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
    fn iter_rev<'self_, 'storage: 'self_>(
        &'self_ self,
        path_elem_storage: Ref<'storage, PathElemStorage>,
    ) -> impl Iterator<Item = (FeedPair, cache::Cost)> + 'self_ {
        // This weak clone is safe because per the above function signature, the
        // output iterator borrows the underlying PartialPathData, which
        // prevents the underlying PathLink from being disposed of.
        let mut next_node = Some(unsafe { self.path.weak_clone() });
        std::iter::from_fn(move || {
            let node = next_node.take()?.get(&*path_elem_storage);
            // This weak clone is safe because the PathLink that we are cloning
            // is protected by the PathLink that we locked above.
            next_node = node
                .prev_steps
                .as_ref()
                .map(|link| unsafe { link.weak_clone() });
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
    fn evaluate_next_step(
        &self,
        cache_model: &CacheModel,
        &next_step: &FeedPair,
    ) -> NextStepEvaluation {
        let mut next_cache = self.cache_sim.clone();
        let next_cost = self.cost_so_far()
            + next_step
                .iter()
                .map(|&feed| next_cache.simulate_access(&cache_model, feed))
                .sum::<cache::Cost>();
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
    fn commit_next_step(
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
    ///
    /// This should be called before the PartialPathData is dropped. PartialPath
    /// will take care of this for you automatically.
    ///
    fn drop_elems(&mut self, path_elem_storage: &mut PathElemStorage) {
        self.path.dispose(path_elem_storage);
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
