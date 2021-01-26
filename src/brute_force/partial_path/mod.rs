//! Representation of a partially explored path in brute force search

mod history;

pub use history::PathElemStorage;

use super::FeedPair;
use crate::{
    cache::{self, CacheModel, CacheSimulation},
    FeedIdx, MAX_FEEDS, MAX_PAIRS,
};
use history::PathLink;

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

/// Result of `PartialPath::evaluate_next_step()`
pub struct NextStepEvaluation {
    /// Next step to be taken (repeated to simplify commit_next_step signature)
    next_step: FeedPair,

    /// Total cache cost after taking this step
    pub next_cost: cache::Cost,

    /// Cache state after taking this step
    next_cache: CacheSimulation,
}
