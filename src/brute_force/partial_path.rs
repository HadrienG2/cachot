//! Representation of a partially explored path in brute force search

use super::FeedPair;
use crate::{
    cache::{self, CacheModel, CacheSimulation},
    FeedIdx, MAX_FEEDS, MAX_PAIRS,
};
use std::rc::Rc;

/// Machine word size in bits
const WORD_SIZE: u32 = (std::mem::size_of::<usize>() * 8) as u32;

/// Integer division with upper rounding
const fn div_round_up(num: usize, denom: usize) -> usize {
    (num / denom) + (num % denom != 0) as usize
}

/// Maximum number of machine words needed to store one bit per feed pair
const MAX_PAIR_WORDS: usize = div_round_up(MAX_PAIRS, WORD_SIZE as usize);

/// Path which we are in the process of exploring
#[derive(Clone)]
pub struct PartialPath {
    // TODO: Consider replacing Rc with something that allows allocation reuse
    path: Rc<PathElems>,
    path_len: usize,
    visited_pairs: [usize; MAX_PAIR_WORDS],
    cache_sim: CacheSimulation,
    extra_distance: StepDistance,
}
//
/// Path elements are stored as a linked list to enable sharing of common nodes.
///
/// This should enable tremendous memory savings and faster path forking, both
/// of which are very important, at the cost of...
///
/// - Slowing down path iteration, but outside of debug logging scenarios we
///   only need that at the end of a path, and we don't reach the end of a path
///   very often as most paths are discarded due to excessive cost.
/// - Slowing down path length counting, which is more important. We address
///   this by double-counting the length so that we don't need to traverse the
///   list in order to know the length.
///
struct PathElems {
    curr_step: FeedPair,
    curr_cost: cache::Cost,
    prev_steps: Option<Rc<PathElems>>,
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
    pub fn new(cache_model: &CacheModel, start: FeedPair) -> Self {
        let mut cache_sim = cache_model.start_simulation();
        let mut curr_cost = 0.0;
        for &feed in start.iter() {
            curr_cost += cache_sim.simulate_access(&cache_model, feed);
        }

        let path = Rc::new(PathElems {
            curr_step: start,
            curr_cost,
            prev_steps: None,
        });

        let mut visited_pairs = [0; MAX_PAIR_WORDS];
        for word in 0..MAX_PAIR_WORDS {
            let mut current_word = 0;
            for bit in (0..WORD_SIZE).rev() {
                let [x, y] = Self::bit_index_to_coord(word, bit);
                let visited = (y < x) || ([x, y] == start);
                current_word = (current_word << 1) | (visited as usize);
            }
            visited_pairs[word] = current_word;
        }

        Self {
            path,
            path_len: 1,
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
        &self.path.curr_step
    }

    /// Iterate over the path steps and cumulative costs in reverse step order
    //
    // NOTE: This operation can be slow, it is only called when a better path
    //       has been found (very rare) or when displaying debug output.
    //
    pub fn iter_rev(&self) -> impl Iterator<Item = (FeedPair, cache::Cost)> + '_ {
        let mut next_node = Some(&self.path);
        std::iter::from_fn(move || {
            let node = next_node?;
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
        self.path.curr_cost
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
    pub fn commit_next_step(&self, next_step_eval: NextStepEvaluation) -> Self {
        let NextStepEvaluation {
            next_step,
            next_cost,
            next_cache,
        } = next_step_eval;

        let next_path = Rc::new(PathElems {
            curr_step: next_step,
            curr_cost: next_cost,
            prev_steps: Some(self.path.clone()),
        });

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
            visited_pairs: next_visited_pairs,
            cache_sim: next_cache,
            extra_distance: self.extra_distance() + (step_length - 1.0),
        }
    }
}

/// Result of `PartialPath::evaluate_next_step()`
pub struct NextStepEvaluation {
    pub next_step: FeedPair,
    pub next_cost: cache::Cost,
    next_cache: CacheSimulation,
}
