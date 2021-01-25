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

/// Approximate limit on the number of stored paths in PriorizedPartialPaths
///
/// At any point in time, we must arbitrate between two competing strategies:
///
/// - Explore shortest paths first, which gives us more information on longer
///   paths and allows us to make a more informed choice between them.
/// - Finish exploring ongoing paths in order to free up RAM and possibly update
///   our best path model too, which in turn enables massive pruning of old
///   paths which are now considered unviable.
///
/// This algorithm tuning parameter controls the threshold of accumulated paths
/// above which we start exploring longer paths.
///
const MAX_STORED_PATHS: usize = 1_000_000;

/// PartialPath container that enables priorization and randomization
pub struct PriorizedPartialPaths {
    /// Priorized partial paths, grouped by length
    ///
    /// - The slot at index i contains paths of length self.min_path_len + i
    /// - The first slot at index 0 is guaranteed to contain some paths.
    //
    paths_by_len: Vec<BinaryHeap<PriorizedPath>>,

    /// Minimal path length that hasn't been fully explored
    min_path_len: usize,

    /// Path length slot which we are currently processing
    curr_path_len: usize,

    /// Direction in which we're going once done exploring curr_path_len
    going_forward: bool,

    /// Maximal number of paths of each length that we allow ourselves to store
    max_paths_per_len: usize,

    /// Mechanism to periodically leave path selection to chance
    iters_since_last_rng: usize,
}
//
impl PriorizedPartialPaths {
    /// Create the collection
    pub fn new(full_path_len: usize) -> Self {
        let max_path_len = full_path_len - 1;
        let max_paths_per_len = MAX_STORED_PATHS / max_path_len;
        Self {
            paths_by_len: vec![BinaryHeap::with_capacity(max_paths_per_len); max_path_len],
            min_path_len: 1,
            curr_path_len: 0,
            going_forward: true,
            max_paths_per_len,
            iters_since_last_rng: 0,
        }
    }

    /// Record a new partial path
    #[inline(always)]
    pub fn push(&mut self, path: PartialPath) {
        debug_assert!(
            path.len() == self.min_path_len + 1 // Initial algorithm seeding
            || path.len() == self.min_path_len + 2 // Subsequent iterations
        );
        let relative_len = path.len() - self.min_path_len + 1;
        self.paths_by_len[relative_len - 1].push(PriorizedPath(path));
    }

    /// Extract one of the highest-priority paths
    #[inline(always)]
    pub fn pop(&mut self, mut rng: impl Rng) -> Option<PartialPath> {
        // Handle edge case where all paths have already been processed
        if self.paths_by_len.is_empty() {
            return None;
        }

        // Select a path length slot according to availability of paths of the
        // currently selected length and memory pressure on longer paths
        loop {
            // Common case where we still have paths available and the next
            // path length slot isn't fully filled up.
            let max_path_len = self.paths_by_len.len() - 1;
            let no_paths_available = self.paths_by_len[self.curr_path_len].is_empty();
            let next_slot_full = self.curr_path_len < max_path_len
                && self.paths_by_len[self.curr_path_len + 1].len() >= self.max_paths_per_len;
            if !no_paths_available && !next_slot_full {
                break;
            }

            // If the problem is availability of paths, check if we're
            // processing the first path length, and if so shrink our path
            // storage: we won't be observing more paths of that length.
            if no_paths_available && self.curr_path_len == 0 {
                self.paths_by_len.remove(0);
                if self.paths_by_len.is_empty() {
                    return None;
                } else {
                    self.min_path_len += 1;
                    self.max_paths_per_len = MAX_STORED_PATHS / self.paths_by_len.len();
                    continue;
                }
            }

            // Otherwise, move to the next path length
            if self.going_forward {
                if self.curr_path_len < max_path_len {
                    self.curr_path_len += 1;
                } else {
                    self.going_forward = false;
                    self.curr_path_len -= 1;
                }
            } else {
                if self.curr_path_len > 0 {
                    self.curr_path_len -= 1;
                } else {
                    self.going_forward = true;
                    self.curr_path_len += 1;
                }
            }
        }

        // Pop a path from the current path length slot
        //
        // TODO: Try again to use some intermittent randomization, but this time
        //       use weighted index sampling (and do it more rarely)
        //
        Some(
            self.paths_by_len[self.curr_path_len]
                .pop()
                .expect("Should work after above loop")
                .0,
        )
    }

    /// Count the total number of paths of each length that are currently stored
    pub fn num_paths_by_len(&self) -> Vec<usize> {
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
