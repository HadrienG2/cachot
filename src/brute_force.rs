//! Mechanism for searching a better pair iterator than state-of-the-art 2D
//! iteration schemes designed for square lattices, via brute force search.

use crate::{
    cache::{self, CacheModel},
    FeedIdx,
};
use rand::prelude::*;
use std::collections::BTreeMap;

/// Configure the level of debugging features from brute force path search.
///
/// This must be a const because brute force search is a CPU intensive process
/// that cannot afford to be constantly testing run-time variables and examining
/// all the paths that achieve a certain cache cost.
///
/// 0 = Don't log anything.
/// 1 = Log search goals, top-level search progress, and the first path that
///     achieves a new cache cost record.
/// 2 = Search and enumerate all the paths that match a certain cache cost record.
/// 3 = Log every time we take a step on a path.
/// 4 = Log the process of searching for a next step on a path.
///
const BRUTE_FORCE_DEBUG_LEVEL: u8 = 1;

/// Type for storing paths through the 2D pair space
pub type Path = Vec<[FeedIdx; 2]>;

/// Use brute force to find a path which is better than our best strategy so far
/// according to our cache simulation.
pub fn search_best_path(
    num_feeds: FeedIdx,
    entry_size: usize,
    max_radius: FeedIdx,
    mut best_cost: cache::Cost,
) -> Option<(cache::Cost, Path)> {
    // Let's be reasonable here
    assert!(num_feeds > 1 && entry_size > 0 && max_radius >= 1 && best_cost > 0.0);

    // In exhaustive mode, make sure that at least we don't re-discover one of
    // the previously discovered strategies.
    if BRUTE_FORCE_DEBUG_LEVEL >= 2 {
        best_cost -= 1.0;
    }

    // A path should go through every point of the 2D half-square defined by
    // x and y belonging to 0..num_feeds and y >= x. From this, we know exactly
    // how long the best path (assuming it exists) will be.
    let path_length = ((num_feeds as usize) * ((num_feeds as usize) + 1)) / 2;

    // We seed the path search algorithm by enumerating every possible starting
    // point for a path, under the following contraints:
    //
    // - To match the output of other algorithms, we want y >= x.
    // - Starting from a point (x, y) is geometrically equivalent to starting
    //   from the symmetric point (num_points-y, num_points-x), so we don't need
    //   to explore both of these starting points to find the optimal solution.
    //
    let mut partial_paths = PartialPaths::new();
    for start_y in 0..num_feeds {
        for start_x in 0..=start_y.min(num_feeds - start_y - 1) {
            let path = vec![[start_x, start_y]];
            let mut cache_model = CacheModel::new(entry_size);
            let mut cost_so_far = cache_model.simulate_access(start_x);
            cost_so_far += cache_model.simulate_access(start_y);
            // TODO: Should check that cache capacity is at least 3 feeds
            debug_assert_eq!(cost_so_far, 0.0, "Cache is unreasonably small");
            partial_paths.push(PartialPath {
                path,
                cache_model,
                cost_so_far,
            });
        }
    }

    // Precompute the neighbors of every point of the [x, y] domain
    //
    // The constraints on them being...
    // - Next point should be within max_radius of current [x, y] position
    // - Next point should remain within the iteration domain (no greater
    //   than num_feeds, and y >= x).
    //
    // TODO: Only store the starting x and, for every x after that, the sequence
    //       of ranges of Y that we must go through. This minimizes memory
    //       traffic and compiler unknowns while achieving the intended goal of
    //       reducing the complexity of the inner loop's iteration logic.
    //
    //       We'll need to drop the test for x=y in exchange for this.
    //
    //       In PartialPath, store a table of all points which a path has not
    //       yet been through in a bit-packed format where every word represents
    //       a sets of packed x's words and the y's are bits.
    //
    //       During the neighbor search loop, take every x and y in the
    //       specified range, and test the corresponding bit of the packed
    //       table described above.
    //
    //       This should speed up the compiler work of testing whether a path
    //       has been through a certain point, while using minimal space (64
    //       bits per paths for 8 feeds).
    //
    let mut neighbors = vec![vec![]; num_feeds as usize * num_feeds as usize];
    let linear_idx = |curr_x, curr_y| curr_y as usize * num_feeds as usize + curr_x as usize;
    for curr_x in 0..num_feeds {
        for curr_y in curr_x..num_feeds {
            for next_x in
                curr_x.saturating_sub(max_radius)..(curr_x + max_radius + 1).min(num_feeds)
            {
                for next_y in curr_y.saturating_sub(max_radius).max(next_x)
                    ..(curr_y + max_radius + 1).min(num_feeds)
                {
                    // Loop invariants
                    debug_assert!(next_x < num_feeds);
                    debug_assert!(next_y < num_feeds);
                    debug_assert!(
                        (next_x as isize - curr_x as isize).abs() as FeedIdx <= max_radius
                    );
                    debug_assert!(
                        (next_y as isize - curr_y as isize).abs() as FeedIdx <= max_radius
                    );
                    debug_assert!(next_y >= next_x);
                    if [next_x, next_y] != [curr_x, curr_y] {
                        neighbors[linear_idx(curr_x, curr_y)].push([next_x, next_y]);
                    }
                }
            }
        }
    }
    let neighborhood = |curr_x, curr_y| neighbors[linear_idx(curr_x, curr_y)].iter().copied();

    // Next we iterate as long as we have incomplete paths by taking the most
    // promising path so far, considering all the next steps that can be taken
    // on that path, and pushing any further incomplete path that this creates
    // into our list of next actions.
    let mut best_path = Path::new();
    let mut rng = rand::thread_rng();
    while let Some(PartialPath {
        path,
        cache_model,
        cost_so_far,
    }) = partial_paths.pop(&mut rng)
    {
        // Indicate which partial path was chosen
        if BRUTE_FORCE_DEBUG_LEVEL >= 3 {
            println!(
                "    - Currently on partial path {:?} with cache cost {}",
                path, cost_so_far
            );
        }

        // Ignore that path if we found another solution which is so good that
        // it's not worth exploring anymore.
        if cost_so_far > best_cost || ((BRUTE_FORCE_DEBUG_LEVEL < 2) && (cost_so_far == best_cost))
        {
            if BRUTE_FORCE_DEBUG_LEVEL >= 4 {
                println!(
                    "      * That exceeds cache cost goal with only {}/{} steps, ignore it.",
                    path.len(),
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
        let &[curr_x, curr_y] = path.last().unwrap();
        for [next_x, next_y] in neighborhood(curr_x, curr_y) {
            // Enumeration tracking
            if BRUTE_FORCE_DEBUG_LEVEL >= 4 {
                println!("      * Trying [{}, {}]...", next_x, next_y);
            }

            // Have we been there before ?
            //
            // TODO: This happens to be a performance bottleneck in profiles,
            //       speed it up via the above strategy.
            //
            if path
                .iter()
                .find(|[prev_x, prev_y]| *prev_x == next_x && *prev_y == next_y)
                .is_some()
            {
                if BRUTE_FORCE_DEBUG_LEVEL >= 4 {
                    println!("      * That's going circles, forget it.");
                }
                continue;
            }

            // Is it worthwhile to go there?
            //
            // TODO: We could consider introducing a stricter cutoff here,
            //       based on the idea that if your partial cache cost is
            //       already X and you have still N steps left to perform,
            //       you're unlikely to beat the best cost.
            //
            //       But that's hard to do due to how chaotically the cache
            //       performs, with most cache misses being at the end of
            //       the curve.
            //
            //       Maybe we could at least track how well our best curve
            //       so far performed at each step, and have a quality
            //       cutoff based on that + a tolerance.
            //
            //       We could then have the search loop start with a fast
            //       low-tolerance search, and resume with a slower
            //       high-tolerance search, ultimately getting to the point
            //       where we can search without tolerance if we truly want the
            //       best of the best curves.
            //
            //       This requires a way to propagate the "best cost at every
            //       step" to the caller, instead of just the the best cost at
            //       the last step, which anyway would be useful once we get to
            //       searching at multiple radii.
            //
            // TODO: Also, we should introduce a sort of undo mechanism (e.g.
            //       an accessor that tells the cache position of a variable and
            //       a mutator that allows us to reset it) in order to delay
            //       memory allocation until the point where we're sure that we
            //       do need to do the cloning.
            //
            let mut next_cache = cache_model.clone();
            let mut next_cost = cost_so_far + next_cache.simulate_access(next_x);
            next_cost += next_cache.simulate_access(next_y);
            if next_cost > best_cost || ((BRUTE_FORCE_DEBUG_LEVEL < 2) && (next_cost == best_cost))
            {
                if BRUTE_FORCE_DEBUG_LEVEL >= 4 {
                    println!(
                        "      * That exceeds cache cost goal with only {}/{} steps, ignore it.",
                        path.len() + 1,
                        path_length
                    );
                }
                continue;
            }

            // Are we finished ?
            let next_path_len = path.len() + 1;
            let make_next_path = || {
                let mut next_path = path.clone();
                next_path.push([next_x, next_y]);
                next_path
            };
            if next_path_len == path_length {
                if next_cost < best_cost {
                    best_path = make_next_path();
                    best_cost = next_cost;
                    if BRUTE_FORCE_DEBUG_LEVEL >= 1 {
                        println!(
                            "  * Reached new cache cost record {} with path {:?}",
                            best_cost, best_path
                        );
                    }
                } else {
                    debug_assert_eq!(next_cost, best_cost);
                    if BRUTE_FORCE_DEBUG_LEVEL >= 2 {
                        println!(
                            "  * Found a path that matches current cache cost constraint: {:?}",
                            make_next_path(),
                        );
                    }
                }
                continue;
            }

            // Otherwise, schedule searching further down this path
            if BRUTE_FORCE_DEBUG_LEVEL >= 4 {
                println!("      * That seems reasonable, we'll explore that path further...");
            }
            partial_paths.push(PartialPath {
                path: make_next_path(),
                cache_model: next_cache,
                cost_so_far: next_cost,
            });
        }
        if BRUTE_FORCE_DEBUG_LEVEL >= 3 {
            println!("    - Done exploring possibilities from current path");
        }
    }

    // Return the optimal path, if any, along with its cache cost
    if best_path.is_empty() {
        None
    } else {
        Some((best_cost, best_path))
    }
}

// The amount of possible paths is ridiculously high (of the order of the
// factorial of path_length), so it's extremely important to...
//
// - Finish exploring paths reasonably quickly, to free up RAM and update
//   the "best cost" figure of merit, which in turn allow us to...
// - Prune paths as soon as it becomes clear that they won't beat the
//   current best cost.
// - Explore promising paths first, and make sure we explore large areas of the
//   path space quickly instead of perpetually staying in the same region of the
//   space of possible paths like basic depth-first search would have us do.
//
// To help us with these goals, we store information about the paths which
// we are in the process of exploring in a data structure which is allows
// priorizing the most promising tracks over others.
//
struct PartialPath {
    path: Path,
    cache_model: CacheModel,
    cost_so_far: cache::Cost,
}
//
type RoundedPriority = usize;
//
impl PartialPath {
    // Relative priority of exploring this path, higher is more important
    fn priority(&self) -> RoundedPriority {
        // Increasing path length weight means that the highest priority is
        // put on seeing paths through the end (which allows discarding
        // them), decreasing means that the highest priority is put on
        // following through the paths that are most promizing in terms of
        // cache cost (which tends to favor a more breadth-first approach as
        // the first curve points are free of cache costs).
        (1.3 * self.path.len() as f32 - self.cost_so_far).round() as _
    }
}
//
#[derive(Default)]
struct PartialPaths {
    storage: BTreeMap<RoundedPriority, Vec<PartialPath>>,
}
//
impl PartialPaths {
    fn new() -> Self {
        Self::default()
    }

    fn push(&mut self, path: PartialPath) {
        let same_priority_paths = self.storage.entry(path.priority()).or_default();
        same_priority_paths.push(path);
    }

    fn pop(&mut self, mut rng: impl Rng) -> Option<PartialPath> {
        let highest_priority_paths = self.storage.values_mut().rev().next()?;
        debug_assert!(!highest_priority_paths.is_empty());
        let path_idx = rng.gen_range(0..highest_priority_paths.len());
        let path = highest_priority_paths.remove(path_idx);
        if highest_priority_paths.is_empty() {
            self.storage.remove(&path.priority());
        }
        Some(path)
    }
}
