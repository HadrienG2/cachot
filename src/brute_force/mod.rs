//! Mechanism for searching a better pair iterator than state-of-the-art 2D
//! iteration schemes designed for square lattices, via brute force search.

mod partial_path;
mod priorization;

pub use self::partial_path::PartialPath;

use self::priorization::PriorizedPartialPaths;
use crate::{
    cache::{self, CacheModel, L1_MISS_COST, NEW_ENTRY_COST},
    FeedIdx, MAX_FEEDS,
};
use std::{
    fmt::Write,
    time::{Duration, Instant},
};

/// Configure the level of debugging features from brute force path search.
///
/// This must be a const because brute force search is a CPU intensive process
/// that cannot afford to be constantly testing run-time variables and examining
/// all the paths that achieve a certain cache cost.
///
/// 0 = Don't log anything.
/// 1 = Log search goals, top-level search progress, and the first path that
///     achieves a new total cache cost record.
/// 2 = Log per-step cumulative cache costs when a record is achieved.
/// 3 = Log every time we take a step on a path.
/// 4 = Log the process of searching for a next step on a path.
///
const BRUTE_FORCE_DEBUG_LEVEL: u8 = 2;

/// Pair of feeds
pub type FeedPair = [FeedIdx; 2];

/// Type for storing paths through the 2D pair space
pub type Path = Box<[FeedPair]>;

/// Use brute force to find a path which is better than our best strategy so far
/// according to our cache simulation.
pub fn search_best_path(
    num_feeds: FeedIdx,
    entry_size: usize,
    max_radius: FeedIdx,
    best_cumulative_cost: &mut [cache::Cost],
    tolerance: cache::Cost,
    timeout: Duration,
) -> Option<Path> {
    // Let's be reasonable here
    let mut total_cost_target = *best_cumulative_cost.last().unwrap() - 1.0;
    assert!(
        num_feeds > 1
            && num_feeds <= MAX_FEEDS
            && max_radius > 0
            && entry_size > 0
            && total_cost_target >= 0.0
    );

    // Set up the cache model
    let cache_model = CacheModel::new(entry_size);
    debug_assert!(
        cache_model.max_l1_entries() >= 3,
        "Cache is unreasonably small"
    );

    // A path should go through every point of the 2D half-square defined by
    // x and y belonging to 0..num_feeds and y >= x. From this, we know exactly
    // how long the best path (assuming it exists) will be.
    let path_length = ((num_feeds as usize) * ((num_feeds as usize) + 1)) / 2;
    debug_assert_eq!(best_cumulative_cost.len(), path_length);

    // We seed the path search algorithm by enumerating every possible starting
    // point for a path, under the following contraints:
    //
    // - To match the output of other algorithms, we want y >= x.
    // - Starting from a point (x, y) is geometrically equivalent to starting
    //   from the symmetric point (num_points-y, num_points-x), so we don't need
    //   to explore both of these starting points to find the optimal solution.
    //
    let mut priorized_partial_paths = PriorizedPartialPaths::new();
    for start_y in 0..num_feeds {
        for start_x in 0..=start_y.min(num_feeds - start_y - 1) {
            priorized_partial_paths.push(PartialPath::new(&cache_model, [start_x, start_y]));
        }
    }

    // Next we iterate as long as we have incomplete paths by taking the most
    // promising path so far, considering all the next steps that can be taken
    // on that path, and pushing any further incomplete path that this creates
    // into our list of next actions.
    let mut best_path = None;
    let mut rng = rand::thread_rng();
    let mut progress_monitor = ProgressMonitor::new(path_length, &priorized_partial_paths);
    while let Some(partial_path) = priorized_partial_paths.pop(&mut rng) {
        // Check the watchdog timer
        if progress_monitor.watchdog_timer() > timeout {
            if BRUTE_FORCE_DEBUG_LEVEL >= 1 {
                println!(
                    "    - No significant progress in {:.1}s, aborting...",
                    progress_monitor.watchdog_timer().as_secs_f32()
                );
            }
            break;
        }

        // Indicate which partial path was chosen
        if BRUTE_FORCE_DEBUG_LEVEL >= 3 {
            let mut path_display = String::new();
            for step_and_cost in partial_path.iter_rev() {
                write!(path_display, "{:?} <- ", step_and_cost).unwrap();
            }
            path_display.push_str("START");
            println!("    - Currently on partial path {}", path_display);
        }

        // Ignore that path if we found another solution which is so good that
        // it's not worth exploring anymore.
        let best_current_cost = best_cumulative_cost[partial_path.len() - 1];
        if partial_path.cost_so_far() > (best_current_cost + tolerance).min(total_cost_target) {
            if BRUTE_FORCE_DEBUG_LEVEL >= 4 {
                println!(
                    "      * That exceeds cache cost tolerance with only {}/{} steps, ignore it.",
                    partial_path.len(),
                    path_length
                );
            }
            continue;
        }

        // Enumerate all possible next points, the constraints on them being...
        // - Next point should not be any point we've previously been through
        // - The total path cache cost is not allowed to go above the best path
        //   cache cost that we've observed so far (otherwise that path is less
        //   interesting than the best path).
        for next_x in 0..num_feeds {
            for next_y in next_x..num_feeds {
                let next_step = [next_x, next_y];

                // Trim candidates according to the max_radius criterion
                let &[curr_x, curr_y] = partial_path.last_step();
                let radius = (next_x as isize - curr_x as isize)
                    .abs()
                    .max((next_y as isize - curr_y as isize).abs())
                    as FeedIdx;
                if radius > max_radius {
                    continue;
                }

                // Log which neighbor we're looking at in verbose mode
                if BRUTE_FORCE_DEBUG_LEVEL >= 4 {
                    println!("      * Trying {:?}...", next_step);
                }

                // Check if we've been on that neighbor before
                if partial_path.contains(&next_step) {
                    if BRUTE_FORCE_DEBUG_LEVEL >= 4 {
                        println!("      * That's going circles, forget it.");
                    }
                    continue;
                }

                // Monitor progress
                progress_monitor.record_step(&priorized_partial_paths);

                // Does it seem worthwhile to try to go there?
                let next_step_eval = partial_path.evaluate_next_step(&cache_model, &next_step);
                let next_cost = next_step_eval.next_cost;
                let best_next_cost = best_cumulative_cost[partial_path.len()];
                if next_cost > (best_next_cost + tolerance).min(total_cost_target) {
                    if BRUTE_FORCE_DEBUG_LEVEL >= 4 {
                        println!(
                        "      * That exceeds cache cost tolerance with only {}/{} steps, ignore it.",
                        partial_path.len() + 1,
                        path_length
                    );
                    }
                    continue;
                }

                // Are we finished ?
                if partial_path.len() + 1 == path_length {
                    // Is this path better than what was observed before?
                    if next_cost <= total_cost_target {
                        // If so, materialize the path into a vector
                        let mut final_path =
                            vec![FeedPair::default(); path_length].into_boxed_slice();
                        final_path[path_length - 1] = next_step_eval.next_step;
                        best_cumulative_cost[path_length - 1] = next_step_eval.next_cost;
                        for (i, (step, cost)) in
                            (0..partial_path.len()).rev().zip(partial_path.iter_rev())
                        {
                            final_path[i] = step;
                            best_cumulative_cost[i] = cost;
                        }

                        // Announce victory
                        let new_entries_cost =
                            num_feeds as cache::Cost * NEW_ENTRY_COST / L1_MISS_COST;
                        if BRUTE_FORCE_DEBUG_LEVEL == 1 {
                            println!(
                                "  * Reached new total cache cost record {} ({} w/o new entries) with path {:?}",
                                next_cost, next_cost - new_entries_cost, final_path
                            );
                        }
                        if BRUTE_FORCE_DEBUG_LEVEL >= 2 {
                            let path_cost = final_path
                                .iter()
                                .zip(best_cumulative_cost.iter())
                                .collect::<Box<[_]>>();
                            println!(
                                "  * Reached new total cache cost record {} ({} w/o new entries) with path and cumulative cost {:?}",
                                next_cost, next_cost - new_entries_cost, path_cost
                            );
                        }

                        // Reset the watchdog timer
                        progress_monitor.reset_watchdog();

                        // Now record that path and look for a better one
                        best_path = Some(final_path);
                        total_cost_target = next_cost - 1.0;
                    }
                    continue;
                }

                // Otherwise, schedule searching further down this path
                if BRUTE_FORCE_DEBUG_LEVEL >= 4 {
                    println!("      * That seems reasonable, we'll explore that path further...");
                }
                priorized_partial_paths.push(partial_path.commit_next_step(next_step_eval));
            }
        }
        if BRUTE_FORCE_DEBUG_LEVEL >= 3 {
            println!("    - Done exploring possibilities from current path");
        }
    }

    // Return the optimal path, if any, along with its cache cost
    best_path
}

// ---

/// Mechanism to track brute force search progress
struct ProgressMonitor {
    path_length: usize,
    total_path_steps: u64,
    last_path_steps: u64,
    initial_time: Instant,
    last_report: Instant,
    last_watchdog_reset: Instant,
    initial_exhaustive_steps: f64,
    last_exhaustive_steps: f64,
}
//
impl ProgressMonitor {
    /// Set up a way to monitor brute force search progress
    pub fn new(path_length: usize, initial_paths: &PriorizedPartialPaths) -> Self {
        let initial_time = Instant::now();
        let initial_exhaustive_steps =
            Self::check_exhaustive_steps(path_length, initial_paths, false);
        Self {
            path_length,
            total_path_steps: 0,
            last_path_steps: 0,
            initial_time,
            last_report: initial_time,
            last_watchdog_reset: initial_time,
            initial_exhaustive_steps,
            last_exhaustive_steps: initial_exhaustive_steps,
        }
    }

    /// Record that a path step has been taken, print periodical reports
    pub fn record_step(&mut self, paths: &PriorizedPartialPaths) {
        // Only report on progress infrequently so that search isn't slowed down
        self.total_path_steps += 1;
        const CLOCK_CHECK_RATE: u64 = 1 << 18;
        const REPORT_RATE: Duration = Duration::from_secs(5);
        if self.total_path_steps % CLOCK_CHECK_RATE == 0 && self.last_report.elapsed() > REPORT_RATE
        {
            let elapsed_time = self.last_report.elapsed();
            let new_path_steps = self.total_path_steps - self.last_path_steps;

            // In verbose mode, display progress information
            if BRUTE_FORCE_DEBUG_LEVEL >= 1 {
                let new_million_steps = new_path_steps as f32 / 1_000_000.0;
                println!(
                    "  * Processed {}M new path steps ({:.1}M steps/s)",
                    new_million_steps.round(),
                    (new_million_steps as f32 / elapsed_time.as_secs_f32())
                );
            }

            // Check how much of the initial search space we covered
            let remaining_exhaustive_steps =
                Self::check_exhaustive_steps(self.path_length, paths, BRUTE_FORCE_DEBUG_LEVEL >= 1);
            if BRUTE_FORCE_DEBUG_LEVEL >= 1 {
                println!(
                    "    - Remaining search space: 10^{:.2} path steps ({:.2}% processed)",
                    remaining_exhaustive_steps.log10(),
                    (1.0 - (remaining_exhaustive_steps / self.initial_exhaustive_steps)) * 100.0
                );
            }

            // Check if we covered a significant fraction of the search space
            // that we last observed.
            const SIGNIFICANT_PROGRESS_THRESHOLD: f64 = 0.01;
            let last_processed_steps = self.last_exhaustive_steps - remaining_exhaustive_steps;
            let rel_processed_steps = last_processed_steps / self.last_exhaustive_steps;
            if rel_processed_steps > SIGNIFICANT_PROGRESS_THRESHOLD {
                self.reset_watchdog();
            }

            // Check how fast we are covering the search space
            let total_time = self.initial_time.elapsed();
            let steps_per_sec = last_processed_steps / total_time.as_secs_f64();
            if BRUTE_FORCE_DEBUG_LEVEL >= 1 {
                println!(
                    "    - Current search speed: 10^{:.2} path steps/s (10^{:.0}x)",
                    steps_per_sec.log10(),
                    (last_processed_steps / new_path_steps as f64)
                        .log10()
                        .round()
                );
                println!(
                    "    - Remaining time at that speed: {}s",
                    (remaining_exhaustive_steps / steps_per_sec).ceil()
                );
            }

            self.last_path_steps = self.total_path_steps;
            self.last_report = Instant::now();
            self.last_exhaustive_steps = remaining_exhaustive_steps;
        }
    }

    /// Check out how much time has elapsed since the last major event
    pub fn watchdog_timer(&self) -> Duration {
        self.last_watchdog_reset.elapsed()
    }

    /// Reset the watchdog timer to signal that an important event has occured,
    /// which suggests that it might be worthwhile to continue the search.
    pub fn reset_watchdog(&mut self) {
        self.last_watchdog_reset = Instant::now();
    }

    /// Check number of steps that would be remaining in an exhaustive search
    ///
    /// Our search is not exhaustive, but that's a good figure of merit of how
    /// much of the path search space we have covered.
    ///
    fn check_exhaustive_steps(
        path_length: usize,
        paths: &PriorizedPartialPaths,
        verbose: bool,
    ) -> f64 {
        // Compute histogram of number of paths by path length
        let paths_by_len = paths.paths_by_len();

        // In verbose mode, display that + high priority paths
        if verbose {
            let display_path_counts = |header, paths_by_len: &Vec<usize>| {
                print!("    - {:<25}: ", header);
                for partial_length in 1..path_length {
                    print!("{:>4} ", paths_by_len.get(partial_length - 1).unwrap_or(&0));
                }
                println!();
            };
            display_path_counts("Partial paths by length", &paths_by_len);
            display_path_counts(
                "Priorized paths by length",
                &paths.high_priority_paths_by_len(),
            );
        }

        // Compute remaining exhaustive search space
        let mut max_next_steps = 1.0f64;
        let mut max_total_steps = 0.0f64;
        for partial_length in (1..path_length).rev() {
            max_next_steps *= (path_length - partial_length) as f64;
            let num_paths = *paths_by_len.get(partial_length - 1).unwrap_or(&0);
            max_total_steps += num_paths as f64 * max_next_steps;
        }

        // Return exhaustive search space
        max_total_steps
    }
}
