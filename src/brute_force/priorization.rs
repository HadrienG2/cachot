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
use slotmap::SlotMap;
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
const MAX_STORED_PATHS: usize = 100_000;

/// PartialPath container that enables priorization and randomization
pub struct PriorizedPartialPaths {
    /// Priorized partial paths, grouped by length
    ///
    /// - The slot at index i contains paths of length self.min_path_len + i
    /// - The first slot at index 0 is guaranteed to contain some paths.
    //
    priorized_paths_by_len: Vec<BinaryHeap<PriorizedPath>>,

    /// Actual PartialPath storage
    ///
    /// The internal bookkeeping of BinaryHeap comes with some data movement,
    /// which doesn't play well with large-ish data structures like
    /// PartialPath. To eliminate this problem, PriorizedPath is only a
    /// small-ish handle into the actual PartialPath storage, which is the
    /// following slotmap.
    ///
    path_storage: SlotMap<slotmap::DefaultKey, PartialPath>,

    /// Minimal path length that hasn't been fully explored
    min_path_len: usize,

    /// Path length slot which we are currently processing
    curr_path_len: usize,

    /// Threshold of next slot occupancy above which we move to the next slot
    high_water_mark: usize,

    /// Mechanism to periodically leave path selection to chance
    iters_since_last_rng: usize,
}
//
impl PriorizedPartialPaths {
    /// Create the collection
    pub fn new(full_path_len: usize) -> Self {
        let max_path_len = full_path_len - 1;
        let high_water_mark = Self::high_water_mark(max_path_len);
        Self {
            priorized_paths_by_len: vec![BinaryHeap::with_capacity(high_water_mark); max_path_len],
            path_storage: SlotMap::with_capacity(MAX_STORED_PATHS),
            min_path_len: 1,
            curr_path_len: 0,
            high_water_mark,
            iters_since_last_rng: 0,
        }
    }

    /// Record a new partial path
    #[inline(always)]
    pub fn push(&mut self, path: PartialPath) {
        debug_assert!(
            // Initial seeding
            (self.min_path_len == 1 && self.curr_path_len == 0 && path.len() == 1)
            // Subsequent iterations
            || (path.len() == self.min_path_len + self.curr_path_len + 1)
        );
        let relative_len = path.len() - self.min_path_len + 1;
        let priority = PriorizedPath::priority(&path);
        let slot = self.path_storage.insert(path);
        self.priorized_paths_by_len[relative_len - 1].push(PriorizedPath { priority, slot });
    }

    /// Extract one of the highest-priority paths
    #[inline(always)]
    pub fn pop(&mut self, mut rng: impl Rng) -> Option<PartialPath> {
        // Handle edge case where all paths have already been processed
        if self.priorized_paths_by_len.is_empty() {
            return None;
        }

        // Select a path length slot according to availability of paths of the
        // currently selected length and memory pressure on longer paths
        loop {
            // Compute threshold for moving to a previous path length
            // The lower this is, the least frequently we switch between path
            // length slots, but the more we tap into low-priority paths.
            let low_water_mark = self.high_water_mark / 2;

            // We stay on the same slot as long as...
            // - Next slot (if any) isn't above high water mark
            // - Current slot is above low water mark, for "middle" slots only
            //   * Lowest-length slot may only go down, this is expected
            //   * Highest-length slot should be fully flushed as this is what
            //     feeds back information into the algorithm.
            // - Current slot isn't empty
            let max_len_slot = self.priorized_paths_by_len.len() - 1;
            let on_first_slot = self.curr_path_len == 0;
            let on_last_slot = self.curr_path_len == max_len_slot;
            let curr_slot_empty = self.priorized_paths_by_len[self.curr_path_len].is_empty();
            let curr_slot_low = !on_first_slot
                && !on_last_slot
                && self.priorized_paths_by_len[self.curr_path_len].len() < low_water_mark;
            let next_slot_full = !on_last_slot
                && self.priorized_paths_by_len[self.curr_path_len + 1].len()
                    >= self.high_water_mark;
            if !next_slot_full && !curr_slot_low && !curr_slot_empty {
                break;
            }

            // Handle edge case where we're done processing lowest-length paths
            if on_first_slot && curr_slot_empty {
                self.priorized_paths_by_len.remove(0);
                if self.priorized_paths_by_len.is_empty() {
                    return None;
                } else {
                    self.min_path_len += 1;
                    self.high_water_mark = Self::high_water_mark(self.priorized_paths_by_len.len());
                    continue;
                }
            }

            // Else move to the next/previous path length slot as appropriate
            if curr_slot_low || curr_slot_empty {
                self.curr_path_len -= 1;
            } else {
                debug_assert!(next_slot_full);
                self.curr_path_len += 1;
            }
        }

        // Pop a path from the current path length slot
        //
        // TODO: Try again to use some intermittent randomization, but this time
        //       use weighted index sampling (and do it more rarely)
        //
        let priorized_path = self.priorized_paths_by_len[self.curr_path_len]
            .pop()
            .expect("Should work after above loop");
        debug_assert!(self.path_storage.contains_key(priorized_path.slot));
        if !self.path_storage.contains_key(priorized_path.slot) {
            unsafe { core::hint::unreachable_unchecked() };
        }
        Some(
            self.path_storage
                .remove(priorized_path.slot)
                .expect("path_storage became inconsistent with priorized paths"),
        )
    }

    /// Drop all stored paths which don't match a certain predicate
    ///
    /// This is very expensive, but only meant to be done when a new cache cost
    /// record is achieved, which doesn't happen very often.
    ///
    pub fn prune(&mut self, mut should_prune: impl FnMut(&PartialPath) -> bool) {
        let mut new_paths = BinaryHeap::with_capacity(self.high_water_mark);
        for old_paths in &mut self.priorized_paths_by_len {
            for path in old_paths.drain() {
                if should_prune(&self.path_storage[path.slot]) {
                    self.path_storage.remove(path.slot);
                } else {
                    new_paths.push(path);
                }
            }
            std::mem::swap(old_paths, &mut new_paths);
        }
        self.curr_path_len = 0;
    }

    /// Count the total number of paths of each length that are currently stored
    pub fn num_paths_by_len(&self) -> Vec<usize> {
        std::iter::repeat(0)
            .take(self.min_path_len - 1)
            .chain(self.priorized_paths_by_len.iter().map(|paths| paths.len()))
            .collect()
    }

    /// Compute the high water mark for a certain number of paths
    const fn high_water_mark(path_len_slots: usize) -> usize {
        MAX_STORED_PATHS / path_len_slots
    }
}
//
#[derive(Clone)]
struct PriorizedPath {
    priority: Priority,
    slot: slotmap::DefaultKey,
}
//
/// Priorize cache cost, then given equal cache cost priorize simplest path
type Priority = (cache::Cost, StepDistance);
//
impl PriorizedPath {
    /// Prioritize a certain path with respect to other paths of equal length.
    /// A higher priority means that a path will be processed first.
    fn priority(path: &PartialPath) -> Priority {
        (-path.cost_so_far(), -path.extra_distance())
    }
}
//
impl PartialEq for PriorizedPath {
    fn eq(&self, other: &Self) -> bool {
        self.priority.eq(&other.priority)
    }
}
//
impl PartialOrd for PriorizedPath {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.priority.partial_cmp(&other.priority)
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
