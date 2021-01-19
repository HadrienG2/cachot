mod cache;

use crate::cache::CacheModel;

use genawaiter::{stack::let_gen, yield_};
use space_filler::{hilbert, morton, CurveIdx};

fn main() {
    #[rustfmt::skip]
    const TESTED_NUM_FEEDS: &'static [FeedIdx] = &[
        // Minimal useful test (any iteration scheme is optimal with 2 feeds)
        // Useful for manual inspection of detailed execution traces
        4,
        // Actual PAON-4 configuration
        8,
        // What would happen with more feeds?
        16
    ];
    let mut debug_level = 2;
    for num_feeds in TESTED_NUM_FEEDS.iter().copied() {
        println!("=== Testing with {} feeds ===\n", num_feeds);

        // The L1 cache must be able to hold data for at least 3 feeds,
        // otherwise every access to a new pair will be a cache miss.
        //
        // When you're so much starved for cache, no smart iteration scheme will
        // save you and the basic iteration order will be the least bad one
        // because it has optimal locality for one of the feeds in the pair.
        //
        // Conversely, when your cache is large enough to hold all feeds, the
        // iteration order that you use across feeds doesn't matter.
        //
        // Interesting things happen between these two extreme case. Iteration
        // orders with good locality properties can live with a "smaller" cache,
        // but also with larger chunks of feed data, which are potentially more
        // efficient to process.
        //
        for num_l1_entries in 3..num_feeds {
            let entry_size = cache::L1_CAPACITY / num_l1_entries as usize;
            let mut locality_tester = PairLocalityTester::new(debug_level, entry_size);
            println!("--- Testing L1 capacity of {} feeds ---", num_l1_entries);
            if debug_level == 0 {
                println!();
            }

            // Naive iteration scheme
            let_gen!(naive, {
                for feed1 in 0..num_feeds {
                    for feed2 in feed1..num_feeds {
                        yield_!([feed1, feed2]);
                    }
                }
            });
            locality_tester.test_feed_pair_locality("Naive", naive.into_iter());

            // Block-wise iteration scheme
            let mut block_size = 2;
            while block_size < num_feeds {
                let_gen!(blocked_basic, {
                    for feed1_block in (0..num_feeds).step_by(block_size.into()) {
                        for feed2_block in (feed1_block..num_feeds).step_by(block_size.into()) {
                            for feed1 in feed1_block..(feed1_block + block_size).min(num_feeds) {
                                for feed2 in feed1.max(feed2_block)
                                    ..(feed2_block + block_size).min(num_feeds)
                                {
                                    yield_!([feed1, feed2]);
                                }
                            }
                        }
                    }
                });
                locality_tester.test_feed_pair_locality(
                    &format!("{0}x{0} blocks", block_size),
                    blocked_basic.into_iter(),
                );
                block_size += 1;
            }

            // Morton curve ("Z order") iteration
            let morton = morton::iter_2d()
                .take(num_feeds as usize * num_feeds as usize)
                .filter(|[feed1, feed2]| feed2 >= feed1);
            locality_tester.test_feed_pair_locality("Morton curve", morton);

            // Hilbert curve iteration
            let hilbert = (0..(num_feeds as CurveIdx * num_feeds as CurveIdx))
                .map(hilbert::decode_2d)
                .filter(|[feed1, feed2]| feed2 >= feed1);
            locality_tester.test_feed_pair_locality("Hilbert curve", hilbert);

            // Tell which iterator got the best results
            let mut best_cost = locality_tester.announce_best_iterator();

            // Now, let's try to brute-force a better iterator. First, evaluate
            // all possible paths through the iteration domain where we don't
            // step by more than (for now) [1, 1]...
            println!("\nPerforming brute force search for a better path...");
            for max_radius in 1..num_feeds {
                println!("- Using next step search radius {}", max_radius);
                if let Some((cost, _path)) =
                    search_best_path(num_feeds, entry_size, max_radius, best_cost)
                {
                    println!("  * Found better paths with cache cost {}", cost);
                    best_cost = cost;
                } else {
                    println!("  * Did not find any better path at that search radius");
                }
            }

            debug_level = debug_level.saturating_sub(1);
            println!();
        }
        debug_level = (num_feeds < 8).into();
    }
}

// ---

/// Integer type used for counting radio feeds
type FeedIdx = space_filler::Coordinate;

/// Test harness for evaluating the locality of several feed pair iterators and
/// picking the best of them.
struct PairLocalityTester {
    debug_level: usize,
    entry_size: usize,
    best_iterator: Option<(String, cache::Cost)>,
}
//
impl PairLocalityTester {
    /// Build the test harness
    pub fn new(debug_level: usize, entry_size: usize) -> Self {
        Self {
            debug_level,
            entry_size,
            best_iterator: None,
        }
    }

    /// Test the locality of one feed pair iterator, with diagnostics
    pub fn test_feed_pair_locality(
        &mut self,
        name: &str,
        feed_pair_iterator: impl Iterator<Item = [FeedIdx; 2]>,
    ) {
        if self.debug_level > 0 {
            println!("\nTesting feed pair iterator \"{}\"...", name);
        }
        let mut cache_model = CacheModel::new(self.entry_size);
        let mut total_cost = 0.0;
        let mut feed_load_count = 0;
        for feed_pair in feed_pair_iterator {
            if self.debug_level >= 2 {
                println!("- Accessing feed pair {:?}...", feed_pair)
            }
            let mut pair_cost = 0.0;
            for feed in feed_pair.iter().copied() {
                let feed_cost = cache_model.simulate_access(feed);
                if self.debug_level >= 2 {
                    println!("  * Accessed feed {} for cache cost {}", feed, feed_cost)
                }
                pair_cost += feed_cost;
            }
            match self.debug_level {
                0 => {}
                1 => println!(
                    "- Accessed feed pair {:?} for cache cost {}",
                    feed_pair, pair_cost
                ),
                _ => println!("  * Total cache cost of this pair is {}", pair_cost),
            }
            total_cost += pair_cost;
            feed_load_count += 2;
        }
        match self.debug_level {
            0 => println!(
                "Total cache cost of iterator \"{}\" is {} ({:.2} per feed load)",
                name,
                total_cost,
                total_cost / (feed_load_count as cache::Cost)
            ),
            _ => println!(
                "- Total cache cost of this iterator is {} ({:.2} per feed load)",
                total_cost,
                total_cost / (feed_load_count as cache::Cost)
            ),
        }
        let best_cost = self
            .best_iterator
            .as_ref()
            .map(|(_name, cost)| *cost)
            .unwrap_or(cache::Cost::MAX);
        if total_cost < best_cost {
            self.best_iterator = Some((name.to_owned(), total_cost));
        }
    }

    /// Tell which of the iterators that were tested so far got the best results
    ///
    /// If a tie occurs, pick the first iterator, as we're testing designs from
    /// the simplest to the most complex ones.
    ///
    pub fn announce_best_iterator(&self) -> cache::Cost {
        if self.debug_level > 0 {
            println!();
        }
        let (best_name, best_cost) = self
            .best_iterator
            .as_ref()
            .expect("No iterator has been tested yet");
        println!(
            "The best iterator so far is \"{}\" with cost {}",
            best_name, best_cost
        );
        *best_cost
    }
}

// ---

use std::{
    cmp::{Ord, Ordering},
    collections::BinaryHeap,
};

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

/// Use brute force to find a path which is better than our best strategy so far
/// according to our cache simulation.
pub fn search_best_path(
    num_feeds: FeedIdx,
    entry_size: usize,
    max_radius: FeedIdx,
    mut best_cost: cache::Cost,
) -> Option<(cache::Cost, Vec<[FeedIdx; 2]>)> {
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

    // The amount of possible paths is ridiculously high (of the order of the
    // factorial of path_length), so it's extremely important to...
    //
    // - Finish exploring paths reasonably quickly, to free up RAM and update
    //   the "best cost" figure of merit, which in turn allow us to...
    // - Prune paths as soon as it becomes clear that they won't beat the
    //   current best cost.
    // - Explore promising paths first, and don't be afraid to be creative at
    //   times, instead of perpetually staying in the same region of the space
    //   of possible paths.
    //
    // To help us with these goals, we store information about the paths which
    // we are in the process of exploring in a data structure which is amenable
    // to priorization.
    //
    struct PartialPath {
        path: Vec<[FeedIdx; 2]>,
        cache_model: CacheModel,
        cost_so_far: cache::Cost,
    }
    //
    impl PartialPath {
        // Relative priority of exploring this path, higher is more important
        fn priority(&self) -> f32 {
            self.path.len() as f32 - self.cost_so_far
        }
    }
    //
    impl PartialEq for PartialPath {
        fn eq(&self, other: &Self) -> bool {
            self.priority().eq(&other.priority())
        }
    }
    //
    impl PartialOrd for PartialPath {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            self.priority().partial_cmp(&other.priority())
        }
    }
    //
    impl Eq for PartialPath {}
    //
    impl Ord for PartialPath {
        fn cmp(&self, other: &Self) -> Ordering {
            self.priority().partial_cmp(&other.priority()).unwrap()
        }
    }
    //
    // TODO: Then we can switch to an ordered map from (path length, path cost)
    //       to a list of (path, path cache), based on the same ordering
    //       heuristic, which would allow us to pick one of the best paths
    //       randomly instead of deterministcally. This seems like a good
    //       strategy to avoid unnecessary regularity in the logic (which
    //       reduces the odds of finding an original solution quickly).
    //
    //       Finally, if we really want the algorithm to think outside the box,
    //       we could use an unordered map from (path length, path cost,
    //       number of paths) to a list of (path, path_cache), which would allow
    //       allow us to pick the (path length, path cost) at random through
    //       weighted index sampling, using number of paths as part of our
    //       weight since when all other things are equal, more paths mean more
    //       possibilities.
    //
    let mut partial_paths = BinaryHeap::new();

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
            let path = vec![[start_x, start_y]];
            let mut cache_model = CacheModel::new(entry_size);
            let mut cost_so_far = cache_model.simulate_access(start_x);
            cost_so_far += cache_model.simulate_access(start_y);
            debug_assert_eq!(cost_so_far, 0.0, "Cache is unreasonably small");
            partial_paths.push(PartialPath {
                path,
                cache_model,
                cost_so_far,
            });
        }
    }

    // Next we iterate as long as we have incomplete paths by taking the most
    // promising path so far, considering all the next steps that can be taken
    // on that path, and pushing any further incomplete path that this creates
    // into our list of next actions.
    let mut best_path = Vec::new();
    while let Some(PartialPath {
        path,
        cache_model,
        cost_so_far,
    }) = partial_paths.pop()
    {
        // Indicate which partial path was chosen
        if BRUTE_FORCE_DEBUG_LEVEL >= 3 {
            println!(
                "    - Currently on partial path {:?} with cache cost {}",
                path, cost_so_far
            );
        }

        // Enumerate all possible next points, the constraints on them being...
        // - Next point should be within max_radius of current [x, y] position
        // - Next point should remain within the iteration domain (no greater
        //   than num_feeds, and y >= x).
        // - Next point should not be any point we've previously been through
        // - The total path cache cost is not allowed to go above the best path
        //   cache cost that we've observed so far (otherwise that path is less
        //   interesting than the best path).
        //
        // TODO: If there is a performance bottleneck in enumerating those
        //       indices, we could memoize a list of neighbours of each point of
        //       the iteration domain, as they are always the same.
        //
        let &[curr_x, curr_y] = path.last().unwrap();
        for next_x in curr_x.saturating_sub(max_radius)..(curr_x + max_radius + 1).min(num_feeds) {
            for next_y in curr_y.saturating_sub(max_radius).max(next_x)
                ..(curr_y + max_radius + 1).min(num_feeds)
            {
                // Loop invariants
                debug_assert!(next_x < num_feeds);
                debug_assert!(next_y < num_feeds);
                debug_assert!((next_x as isize - curr_x as isize).abs() as FeedIdx <= max_radius);
                debug_assert!((next_y as isize - curr_y as isize).abs() as FeedIdx <= max_radius);
                debug_assert!(next_y >= next_x);

                // Loop tracking
                if BRUTE_FORCE_DEBUG_LEVEL >= 4 {
                    println!("      * Trying [{}, {}]...", next_x, next_y);
                }

                // Have we been there before ?
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
                let mut next_cache = cache_model.clone();
                let mut next_cost = cost_so_far + next_cache.simulate_access(next_x);
                next_cost += next_cache.simulate_access(next_y);
                if next_cost > best_cost
                    || ((BRUTE_FORCE_DEBUG_LEVEL < 2) && (next_cost == best_cost))
                {
                    if BRUTE_FORCE_DEBUG_LEVEL >= 4 {
                        println!(
                            "      * That exceeds cache cost goal with only {}/{} steps, ignore it.",
                            path.len() + 1,
                            path.capacity()
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
