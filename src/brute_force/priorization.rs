//! Priorized container of PartialPaths
//!
//! The amount of possible paths is ridiculously high (of the order of the
//! factorial of path_length, which itself grows like the square of num_feeds),
//! so it's extremely important to...
//!
//! - Finish exploring paths reasonably quickly, to free up RAM and update
//!   the "best cost" figure of merit, which in turn allow us to...
//! - Prune paths as soon as it becomes clear that they won't beat the
//!   current best cost.
//! - Explore promising paths first, and make sure we explore large areas of the
//!   path space quickly instead of perpetually staying in the same region of
//!   the space of possible paths like depth-first search would have us do.
//!
//! To help us with these goals, we store information about the paths which
//! we are in the process of exploring in a data structure which allows
//! priorizing the most promising tracks over others.

use super::{PartialPath, StepDistance};
use crate::cache;
use rand::prelude::*;
use std::{cmp::Ordering, collections::BinaryHeap};

/// PartialPath container that enables priorization and randomization
pub struct PriorizedPartialPaths {
    /// Priorized partial paths, grouped by length
    ///
    /// - The slot at index i contains paths of length self.min_path_len + i
    /// - The first and last slot are guaranteed to contain some paths.
    //
    paths_by_len: Vec<BinaryHeap<PriorizedPath>>,

    /// Minimal path length that hasn't been fully explored
    min_path_len: usize,

    /// Maximal number of steps that must be taken to finish one minimal path
    max_path_steps: usize,

    /// Fast access to the total number of stored paths
    num_paths: usize,

    /// Mechanism to periodically leave path selection to chance
    iters_since_last_rng: usize,
}
//
impl PriorizedPartialPaths {
    /// Create the collection
    pub fn new(full_path_len: usize) -> Self {
        let max_path_len = full_path_len - 1;
        Self {
            paths_by_len: Vec::with_capacity(max_path_len),
            min_path_len: 1,
            max_path_steps: max_path_len - 1,
            num_paths: 0,
            iters_since_last_rng: 0,
        }
    }

    /// Record a new partial path
    #[inline(always)]
    pub fn push(&mut self, path: PartialPath) {
        // Create storage for paths of that length if we don't already have it
        let relative_len = path.len() - self.min_path_len + 1;
        if self.paths_by_len.len() < relative_len {
            self.paths_by_len.resize(relative_len, BinaryHeap::new());
        }

        // Push that path into the heap for paths of its length and keep track
        // of the total number of paths
        self.paths_by_len[relative_len - 1].push(PriorizedPath(path));
        self.num_paths += 1;
    }

    /// Extract one of the highest-priority paths
    #[inline(always)]
    pub fn pop(&mut self, mut rng: impl Rng) -> Option<PartialPath> {
        // Exit early if we have no path in store
        if self.num_paths == 0 {
            return None;
        }

        // Pick a minimal path length according to memory pressure
        //
        // We want to explore shortest paths first, as that gives us more
        // complete information about longer paths. But if we explored all paths
        // of length 0, then all paths of length 1, ..., we would 1/run out of
        // RAM and 2/take a huge amount of time to finish the first path, which
        // is bad because we feed back new information into the path search
        // algorithm by finishing paths.
        //
        // Therefore, we compromize by forcing exploration of longer paths when
        // we start to have too many paths in flight.
        //
        // Our general strategy for this is that we have a memory budget,
        // expressed as a number of paths that we're willing to store, and when
        // we run out of that budget, we rush to finish longer paths.
        //
        // If we did just that, however, we would accumulate a huge number of
        // short paths and then be forced to explore those short paths one by
        // one, without any priorization on longer paths. So instead, what we
        // conceptually do is that we spread the path budget between path
        // lengths: if our path budget is 10k paths and our path can take at
        // most 10 steps, then we allow ourselves to store 1k variants of the
        // first step, 1k variants of the second step, etc.
        //
        const MEMORY_PRESSURE: f32 = 1e-5;
        let max_length_idx = self.paths_by_len.len() - 1;
        debug_assert!(max_length_idx <= self.max_path_steps);
        let min_length_idx_float = ((MEMORY_PRESSURE * self.num_paths as f32).min(1.0)
            * self.max_path_steps as f32)
            .min(max_length_idx as f32);

        // Use dithering to avoid going through a steep cutoff effect when
        // memory pressure reaches a new threshold.
        let min_length_idx = min_length_idx_float as usize
            + (rng.gen::<f32>() < min_length_idx_float.fract()) as usize;

        // Find the first non-empty path length class in that range
        let (length_idx, shortest_paths) = self
            .paths_by_len
            .iter_mut()
            .enumerate()
            .skip(min_length_idx)
            .find(|(_idx, paths)| !paths.is_empty())
            .expect("Should work if paths_by_len is not empty and shrunk to fit non-empty heaps");

        // Pick the highest-priority path for that length
        //
        // TODO: Bring back the randomness, but this time use weighted index
        //       sampling (and adjust random roll occurence frequency according
        //       to the much greater cost of this method, which is expensive in
        //       and of itself and requires us to collect our BinaryHeap into a
        //       vec and then turn it back into a BinaryHeap).
        //
        //       Handle composite priority in weighted index sampling by rolling
        //       the cache cost, then the distance at that cache cost.
        //
        let path = shortest_paths
            .pop()
            .expect("Should not be empty according to above selection")
            .0;
        debug_assert_eq!(path.len(), self.min_path_len + length_idx);
        self.num_paths -= 1;

        // Keep self.paths_by_len as small as possible
        if shortest_paths.len() == 0 {
            if length_idx == 0 {
                self.paths_by_len.remove(0);
                self.min_path_len += 1;
                if self.max_path_steps > 0 {
                    self.max_path_steps -= 1;
                } else {
                    debug_assert_eq!(self.num_paths, 0);
                }
            } else if length_idx == max_length_idx {
                let new_length = self
                    .paths_by_len
                    .iter()
                    .rposition(|paths| !paths.is_empty())
                    .map(|pos| pos + 1)
                    .unwrap_or(0);
                self.paths_by_len.truncate(new_length);
            }
        }

        // Return the chosen path
        Some(path)
    }

    /// Count the total number of paths of each length that are currently stored
    pub fn paths_by_len(&self) -> Vec<usize> {
        std::iter::repeat(0)
            .take(self.min_path_len - 1)
            .chain(self.paths_by_len.iter().map(|paths| paths.len()))
            .collect()
    }
}
//
#[derive(Clone)]
struct PriorizedPath(PartialPath);
//
/// Priorize cache cost, then given equal cache cost priorize simplest path
type Priority = (cache::Cost, StepDistance);
//
impl PriorizedPath {
    /// Prioritize a certain path with respect to other paths of equal length.
    /// A higher priority means that a path will be processed first.
    fn priority(&self) -> Priority {
        (-self.0.cost_so_far(), -self.0.extra_distance())
    }
}
//
impl PartialEq for PriorizedPath {
    fn eq(&self, other: &Self) -> bool {
        self.priority().eq(&other.priority())
    }
}
//
impl PartialOrd for PriorizedPath {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.priority().partial_cmp(&other.priority())
    }
}
//
impl Eq for PriorizedPath {}
//
impl Ord for PriorizedPath {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(&other).unwrap()
    }
}
