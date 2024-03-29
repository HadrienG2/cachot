//! Mechanism for searching a better pair iterator than state-of-the-art 2D
//! iteration schemes designed for square lattices, via brute force search.

pub(self) mod partial_path;
pub(self) mod priorization;
mod progress;

pub use self::partial_path::{PartialPath, PathElemStorage};

use self::{
    partial_path::StepDistance, priorization::PriorizedPartialPaths, progress::ProgressMonitor,
};
use crate::{
    cache::{self, CacheModel, L1_MISS_COST, NEW_ENTRY_COST},
    FeedIdx, MAX_FEEDS,
};
use num_traits::identities::Zero;
use std::{cell::RefCell, fmt::Write, time::Duration};

/// Configure the level of debugging features from brute force path search.
///
/// This must be a const because brute force search is a CPU intensive process
/// that cannot afford to be constantly testing run-time variables and examining
/// all the paths that achieve a certain cache cost.
///
/// 0 = Don't log anything.
/// 1 = Log search goals, top-level search progress, and the first path that
///     achieves a new total cache cost record.
/// 2 = Log per-step cumulative cache costs when a record is achieved, and
///     print regular progress reports.
/// 3 = Log every time we take a step on a path.
/// 4 = Log the process of searching for a next step on a path.
///
pub(self) const BRUTE_FORCE_DEBUG_LEVEL: u8 = 2;

/// Pair of feeds
pub type FeedPair = [FeedIdx; 2];

/// Type for storing paths through the 2D pair space
pub type Path = Box<[FeedPair]>;

/// Use brute force to find a path which is better than our best strategy so far
/// according to our cache simulation.
pub fn search_best_path(
    num_feeds: FeedIdx,
    entry_size: usize,
    best_cumulative_cost: &mut [cache::Cost],
    iteration_timeout: Duration,
) -> Option<Path> {
    // Seed simplest path record tracker
    let mut best_extra_distance = StepDistance::MAX;
    let mut best_path = None;

    // We'll do multiple algorithmic passes with increasing tolerance, starting
    // with zero tolerance in order to quickly improve our best path estimate in
    // a minimal search space.
    let mut tolerance = cache::Cost::zero();
    'tolerance: loop {
        // For the same reason, for each tolerance, we'll start by only
        // considering very close neighbors of the current path element, and
        // increasing the neighbor search radius gradually.
        let mut search_radius = 1;
        'radius: loop {
            // Notify of loop progression
            if BRUTE_FORCE_DEBUG_LEVEL >= 1 {
                println!(
                    "- Using cumulative cost tolerance {tolerance} \
                       and neighbor search radius {search_radius}"
                );
            }

            // Perform a brute force search iteration
            if let Some(path) = search_best_path_iteration(
                num_feeds,
                entry_size,
                search_radius,
                &mut best_cumulative_cost[..],
                &mut best_extra_distance,
                tolerance,
                iteration_timeout,
            ) {
                if BRUTE_FORCE_DEBUG_LEVEL >= 1 {
                    println!(
                        "  * Found a better path \
                             at cumulative cost tolerance {tolerance} \
                             and search radius {search_radius}:"
                    );
                    println!("    - Path: {path:?}");
                    println!(
                        "    - Cache cost w/o first accesses: {}",
                        best_cumulative_cost.last().unwrap() - cache::min_cache_cost(num_feeds),
                    );
                    println!("    - Deviation from unit steps: {best_extra_distance:.1}");
                }
                best_path = Some(path);
            } else {
                if BRUTE_FORCE_DEBUG_LEVEL >= 1 {
                    println!("  * Did not find any better path");
                }
            }

            // Detect if we found one of the best possible paths, in which case
            // increasing the size of the search space any further is useless.
            if *best_cumulative_cost.last().unwrap()
                == L1_MISS_COST + cache::min_cache_cost(num_feeds)
            {
                if BRUTE_FORCE_DEBUG_LEVEL >= 1 {
                    println!("  * We won't be able to do any better than this cache cost.");
                }
                break 'tolerance;
            }

            // Increase search radius
            let max_radius = num_feeds - 1;
            if search_radius == max_radius {
                break 'radius;
            } else {
                search_radius = (2 * search_radius).min(max_radius);
                continue 'radius;
            }
        }

        // Increase cache cost tolerance
        let max_tolerance = *best_cumulative_cost.last().unwrap();
        if tolerance >= max_tolerance {
            break 'tolerance;
        } else if tolerance == 0.0 {
            tolerance = L1_MISS_COST
        } else {
            tolerance = (2 * tolerance).min(max_tolerance);
        }
    }

    // Announce results
    if BRUTE_FORCE_DEBUG_LEVEL >= 1 {
        if let Some(ref path) = best_path {
            println!("- Overall, the best path was: {path:?}");
            println!(
                "  * Cache cost w/o first accesses: {}",
                best_cumulative_cost.last().unwrap() - cache::min_cache_cost(num_feeds),
            );
            println!("  * Deviation from unit steps: {best_extra_distance:.1}");
        } else {
            println!("- Did not find any better path than the original seed!");
        }
    }
    best_path
}

// ---

/// Iteration of best path brute force search
///
/// Like all brute force searches, our brute force search for an optimal path is
/// a constant struggle against combinatorial explosion. The number of paths is
/// Npair! = [(Nfeeds)*(Nfeeds+1)/2]!, which for 8 feeds is 36! ~ 10^41 paths.
/// There is no way we can exhaustively explore all of these paths, so we must
/// operate under a finite time budget in which we use heuristics to...
///
/// - Explore most promising paths first, leaving paths which are less likely to
///   beat our cache cost record for later.
/// - Entirely eliminate paths which are particularly unlikely to beat the cache
///   cost record, so that we have more RAM available to store more promising
///   paths and make an informed choice between them.
///
/// The latter heuristic performs best when seeded with a good initial path,
/// since we can use that path's cumulative cache cost as a guide to eliminate
/// other paths with higher cumulative cache costs (modulo a certain tolerance,
/// since sometimes losing some cache cost early in the path can help us pay a
/// smaller cache cost later on.
///
/// Therefore, we start with a basic search with a strict cutoff (only consider
/// nearest neighbors for path propagation, don't consider path candidates whose
/// cumulative cache cost is any worse than the best path so far), which
/// converges quickly, and then progressively enlarge the search space once we
/// have a good initial guess that we can use to prune lots of combinatorics.
///
/// This function is the basis of that iterative behavior.
///
fn search_best_path_iteration(
    num_feeds: FeedIdx,
    entry_size: usize,
    max_radius: FeedIdx,
    best_cumulative_cost: &mut [cache::Cost],
    best_extra_distance: &mut StepDistance,
    tolerance: cache::Cost,
    timeout: Duration,
) -> Option<Path> {
    // Let's be reasonable here
    let mut last_cost_record = *best_cumulative_cost.last().unwrap();
    assert!(
        num_feeds > 1
            && num_feeds <= MAX_FEEDS
            && max_radius > 0
            && entry_size > 0
            && last_cost_record >= L1_MISS_COST + cache::min_cache_cost(num_feeds)
    );

    // Set up the cache model
    let cache_model = CacheModel::new(entry_size);
    assert!(
        cache_model.max_l1_entries() >= 3,
        "Cache is unreasonably small"
    );

    // A path should go through every point of the 2D half-square defined by
    // x and y belonging to 0..num_feeds and y >= x. From this, we know exactly
    // how long the best path (assuming it exists) will be.
    let path_length = ((num_feeds as usize) * ((num_feeds as usize) + 1)) / 2;
    assert_eq!(best_cumulative_cost.len(), path_length);

    // Set up storage for paths throughout the space of feed pairs
    let path_elem_storage = RefCell::new(PathElemStorage::new());
    let mut priorized_partial_paths =
        PriorizedPartialPaths::new(&cache_model, &path_elem_storage, path_length);

    // We seed the path search algorithm by enumerating every possible starting
    // point for a path, under the following contraints:
    //
    // - To match the output of other algorithms, we want y >= x.
    // - Starting from a point (x, y) is geometrically equivalent to starting
    //   from the symmetric point (num_points-y, num_points-x), so we don't need
    //   to explore both of these starting points to find the optimal solution.
    //
    for start_y in 0..num_feeds {
        for start_x in 0..=start_y.min(num_feeds - start_y - 1) {
            priorized_partial_paths.create([start_x, start_y]);
        }
    }

    // We stop exploring paths when their cumulative cache cost has risen too
    // too far above the best cumulative cache cost that was observed so far, as
    // it's unlikely that they will outperform the best path that way.
    let should_prune_path = |curr_path_len: usize,
                             cost_so_far: cache::Cost,
                             best_cumulative_cost: &[cache::Cost],
                             last_cost_record: cache::Cost|
     -> bool {
        let best_current_cost = best_cumulative_cost[curr_path_len - 1];
        let should_prune = cost_so_far > (best_current_cost + tolerance).min(last_cost_record);
        if should_prune && BRUTE_FORCE_DEBUG_LEVEL >= 4 {
            println!(
                "      * That exceeds cache cost tolerance with only \
                         {curr_path_len}/{path_length} steps, ignore it."
            );
        }
        should_prune
    };

    // Next we iterate as long as we have incomplete paths by taking the most
    // promising path so far, considering all the next steps that can be taken
    // on that path, and pushing any further incomplete path that this creates
    // into our list of next actions.
    let mut best_path = None;
    let mut rng = rand::thread_rng();
    let mut progress_monitor = ProgressMonitor::new();
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
                write!(path_display, "{step_and_cost:?} <- ").unwrap();
            }
            path_display.push_str("START");
            println!("    - Currently on partial path {path_display}");
        }

        // Enumerate all possible next points, the constraints on them being...
        // - Next point should be within max_radius of last point
        // - Next point should be within the iteration domain (x and y between
        //   0 and num_feeds and y >= x).
        // - Next point should not be any point we've previously been through
        // - The total path cache cost is not allowed to go above the best path
        //   cache cost that we've observed so far (otherwise that path is less
        //   interesting than the best path).
        let [curr_x, curr_y] = partial_path.last_step();
        for next_x in curr_x.saturating_sub(max_radius)..(curr_x + max_radius + 1).min(num_feeds) {
            for next_y in curr_y.saturating_sub(max_radius).max(next_x)
                ..(curr_y + max_radius + 1).min(num_feeds)
            {
                let next_step = [next_x, next_y];

                // Log which neighbor we're looking at in verbose mode
                if BRUTE_FORCE_DEBUG_LEVEL >= 4 {
                    println!("      * Trying {next_step:?}...");
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
                let next_step_eval = partial_path.evaluate_next_step(&next_step);
                let next_cost = next_step_eval.next_cost;
                if should_prune_path(
                    partial_path.len() + 1,
                    next_cost,
                    best_cumulative_cost,
                    last_cost_record,
                ) {
                    continue;
                }

                // Are we finished ?
                if partial_path.len() + 1 == path_length {
                    // Is this path better than what was observed before?
                    if next_cost < last_cost_record
                        || (next_cost == last_cost_record
                            && partial_path.extra_distance() < *best_extra_distance)
                    {
                        // If so, materialize the path into a vector and update
                        // best cumulative cost figure of merit.
                        let mut final_path =
                            vec![FeedPair::default(); path_length].into_boxed_slice();
                        final_path[path_length - 1] = next_step;
                        best_cumulative_cost[path_length - 1] = next_step_eval.next_cost;
                        for (i, (step, cost)) in
                            (0..partial_path.len()).rev().zip(partial_path.iter_rev())
                        {
                            final_path[i] = step;
                            best_cumulative_cost[i] = cost;
                        }

                        // Announce victory
                        let new_entries_cost =
                            cache::Cost::from_num(num_feeds) * NEW_ENTRY_COST / L1_MISS_COST;
                        if BRUTE_FORCE_DEBUG_LEVEL >= 1 {
                            println!("  * Reached a new cache cost or extra distance record!");
                            println!(
                                "    - Total cache cost was {next_cost} \
                                       ({cost_wo_new_entries} w/o new entries), \
                                       extra distance was {extra_distance:.1}",
                                cost_wo_new_entries = next_cost - new_entries_cost,
                                extra_distance = partial_path.extra_distance()
                            );
                        }
                        if BRUTE_FORCE_DEBUG_LEVEL == 1 {
                            println!("    - Path was {final_path:?}");
                        } else if BRUTE_FORCE_DEBUG_LEVEL >= 2 {
                            let path_cost = final_path
                                .iter()
                                .zip(best_cumulative_cost.iter())
                                .collect::<Box<[_]>>();
                            println!("    - Path and cumulative cost was {path_cost:?}");
                        }

                        // If a cache cost record has been achieved, prune stored
                        // paths which are no longer worthwhile according to the
                        // new best cumulative cost.
                        if next_cost < last_cost_record {
                            if BRUTE_FORCE_DEBUG_LEVEL >= 2 {
                                println!(
                                    "    - Pruning paths which are no longer considered viable..."
                                );
                            }
                            priorized_partial_paths.prune(|path| {
                                should_prune_path(
                                    path.len(),
                                    path.cost_so_far(),
                                    best_cumulative_cost,
                                    next_cost,
                                )
                            });
                        }

                        // Update best path tracking variables
                        best_path = Some(final_path);
                        *best_extra_distance = partial_path.extra_distance();
                        last_cost_record = next_cost;

                        // Reset the watchdog timer
                        progress_monitor.reset_watchdog();
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
