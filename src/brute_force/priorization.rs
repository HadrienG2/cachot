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

use super::PartialPath;
use rand::prelude::*;
use std::collections::VecDeque;

/// PartialPath container that enables priorization and randomization
#[derive(Default)]
pub struct PriorizedPartialPaths {
    /// Sorted list of prioritized ongoing paths
    ///
    /// I tried using a BTreeMap before, but since we're always popping the last
    /// element and inserting close to the end, a sorted list is better.
    ///
    storage: Vec<(Priority, Vec<PartialPath>)>,

    /// Fast access to the number of paths in storage
    num_paths: usize,

    /// Mechanism to reuse inner Vec allocations
    storage_morgue: VecDeque<Vec<PartialPath>>,

    /// Mechanism to periodically leave path selection to chance
    iters_since_last_rng: usize,
}
//
type Priority = f32;
//
impl PriorizedPartialPaths {
    /// Create the collection
    pub fn new() -> Self {
        Self::default()
    }

    /// Prioritize a certain path wrt others, higher is more important
    pub fn priorize(&self, path: &PartialPath) -> Priority {
        // * The main goal is to avoid cache misses
        // * In doing so, however, we must be careful to finish paths as 1/that
        //   keeps our RAM usage bounded and 2/that updates our best path model,
        //   which in turns allows us to prune bad paths.
        // * Let's keep the path as close to nice (0, 1) steps as possible,
        //   given the aforementioned constraints.
        -path.cost_so_far() + (self.num_paths as f32 * 1e-3 * path.len() as f32)
            - path.extra_distance()
    }

    /// Record a new partial path
    #[inline(always)]
    pub fn push(&mut self, path: PartialPath) {
        // Count the input path, priorize it, and round the priority so that we
        // tend to have a small number of priority classes.
        self.num_paths += 1;
        let priority = (self.priorize(&path) * 2.0).round() * 0.5;

        // If that priority is new to us, we'll need to make a new list for it,
        // reusing allocations from past lists if we have some spare ones
        let storage_morgue = &mut self.storage_morgue;
        let mut make_new_path_list = |path| {
            let mut new_paths = storage_morgue.pop_front().unwrap_or_default();
            new_paths.push(path);
            (priority, new_paths)
        };

        // Inject the priorized paths into our internal storage
        for (idx, (curr_priority, curr_paths)) in self.storage.iter_mut().enumerate().rev() {
            if *curr_priority == priority {
                curr_paths.push(path);
                return;
            } else if *curr_priority < priority {
                self.storage.insert(idx + 1, make_new_path_list(path));
                return;
            }
        }
        self.storage.insert(0, make_new_path_list(path));
    }

    /// Extract one of the highest-priority paths
    #[inline(always)]
    pub fn pop(&mut self, mut rng: impl Rng) -> Option<PartialPath> {
        // Find the set of highest priority paths
        let (_priority, highest_priority_paths) = self.storage.last_mut()?;
        debug_assert!(!highest_priority_paths.is_empty());

        // If there are multiple paths th choose from...
        let path = if highest_priority_paths.len() != 1 {
            // ...periodically pick a random high-priority path in order to
            // reduce the odd of the algorithm getting stuck in a bad region of
            // the search space as a result of following an overly regular
            // search pattern. But don't do it too often as it gets expensive,
            // especially when there are lots of high-priority path. Instead,
            // favor picking the most accessible high-priority path at the end.
            const RNG_AVERSION: f32 = 0.1;
            let rng_threshold = (highest_priority_paths.len() as f32) * RNG_AVERSION;
            if self.iters_since_last_rng as f32 > rng_threshold {
                self.iters_since_last_rng = 0;
                let path_idx = rng.gen_range(0..highest_priority_paths.len());
                highest_priority_paths.remove(path_idx)
            } else {
                self.iters_since_last_rng += 1;
                highest_priority_paths.pop().unwrap()
            }
        } else {
            // If there's only one path, we must take that one
            highest_priority_paths.pop().unwrap()
        };

        // If the set of highest priority paths is now empty, we remove it from
        // the set of priorized paths, but keep some allocations around.
        if highest_priority_paths.is_empty() {
            let (_priority, mut highest_priority_paths) = self.storage.pop().unwrap();
            highest_priority_paths.clear();
            const MAX_MORGUE_SIZE: usize = 1 << 5;
            if self.storage_morgue.len() < MAX_MORGUE_SIZE {
                self.storage_morgue.push_back(highest_priority_paths);
            }
        }

        // Finally, we return the chosen path
        self.num_paths -= 1;
        Some(path)
    }

    /// Count the number of paths of each length within a vector of path
    fn count_by_len(paths: &Vec<PartialPath>) -> Vec<usize> {
        let mut histogram = Vec::new();
        for path in paths {
            if histogram.len() < path.len() {
                histogram.resize(path.len(), 0);
            }
            histogram[path.len() - 1] += 1;
        }
        histogram
    }

    /// Merge a result of count_by_len() with another
    fn merge_counts(src1: Vec<usize>, src2: Vec<usize>) -> Vec<usize> {
        let (mut target, mut src) = (src1, src2);
        if target.len() < src.len() {
            target.extend_from_slice(&src[target.len()..]);
        }
        src.truncate(target.len());
        for (idx, src_count) in src.into_iter().enumerate() {
            target[idx] += src_count;
        }
        target
    }

    /// Count the total number of paths of each length that are currently stored
    pub fn paths_by_len(&self) -> Vec<usize> {
        self.storage
            .iter()
            .map(|(_priority, path_vec)| Self::count_by_len(path_vec))
            .fold(Vec::new(), |src1, src2| Self::merge_counts(src1, src2))
    }

    /// Like paths_by_len, but only counts the number of high-priority paths
    pub fn high_priority_paths_by_len(&self) -> Vec<usize> {
        self.storage
            .last()
            .map(|(_priority, path_vec)| Self::count_by_len(path_vec))
            .unwrap_or(Vec::new())
    }
}
