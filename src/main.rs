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
                if let Some((cost, _paths)) =
                    search_best_paths(num_feeds, entry_size, max_radius, best_cost)
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

/// Configure the level of debugging output from brute force path search.
///
/// This must be a const because brute force search is a CPU intensive process
/// that cannot afford to be constantly testing run-time variables.
///
/// 0 = Don't log anything.
/// 1 = Log search goals, top-level search progress, and the first path that
///     achieves a new cache cost record.
/// 2 = Log every path that successfully matches the current cache cost record.
/// 3 = Log every time we take a step on a path.
/// 4 = Log the process of searching for a next step on a path.
///
const BRUTE_FORCE_DEBUG_LEVEL: u8 = 1;

/// Enumerate the possible paths through a 2D iteration domain where the
/// distance between two points is no greater than max_radius, using brute force
/// to look for a strategy which is better than our best strategy so far
/// according to cache simulation.
fn search_best_paths(
    num_feeds: FeedIdx,
    entry_size: usize,
    max_radius: FeedIdx,
    mut best_cost: cache::Cost,
) -> Option<(cache::Cost, Vec<Vec<[FeedIdx; 2]>>)> {
    // Make sure that brute force search doesn't re-discover our previous strategy
    best_cost -= 1.0;
    if BRUTE_FORCE_DEBUG_LEVEL >= 1 {
        println!(
            "  * We must be better than the previous search so the cost must be <={}",
            best_cost
        );
    }

    // A path goes through every point of the 2D half-square defined by [x, y], y >= x
    let path_length = ((num_feeds as usize) * ((num_feeds as usize) + 1)) / 2;

    // We start by enumerating every possible starting point, accounting for the
    // facts that starting from (x, y) is geometrically equivalent to starting
    // from (y, x) and that we want y >= x.
    let mut paths = Vec::with_capacity(path_length);
    for start_y in 0..num_feeds {
        for start_x in 0..=start_y.min(num_feeds - start_y - 1) {
            if BRUTE_FORCE_DEBUG_LEVEL >= 1 {
                println!("  * Searching paths from [{}, {}]", start_x, start_y);
            }
            // We start a path at that point, along with matching cache simulation
            let mut path = Vec::with_capacity(path_length);
            debug_assert_eq!(path.capacity(), path_length);
            path.push([start_x, start_y]);
            let mut path_cache = CacheModel::new(entry_size);
            let mut path_cost = path_cache.simulate_access(start_x);
            path_cost += path_cache.simulate_access(start_y);
            debug_assert_eq!(path_cost, 0.0, "Cache is unreasonably small");
            // ...and we recursively explore all paths from that point
            enumerate_paths_impl(
                num_feeds,
                max_radius,
                path,
                &mut path_cache,
                path_cost,
                &mut paths,
                &mut best_cost,
            );
        }
    }

    // Return the list of enumerated optimal paths, with their cache cost
    if paths.len() > 0 {
        Some((best_cost, paths))
    } else {
        None
    }
}
//
fn enumerate_paths_impl(
    num_feeds: FeedIdx,
    max_radius: FeedIdx,
    path: Vec<[FeedIdx; 2]>,
    path_cache: &CacheModel,
    path_cost: cache::Cost,
    paths: &mut Vec<Vec<[FeedIdx; 2]>>,
    best_cost: &mut cache::Cost,
) {
    // Check if we reached a full path yet
    if BRUTE_FORCE_DEBUG_LEVEL >= 3 {
        println!(
            "    - Currently on path {:?} with partial cache cost {}",
            path, path_cost
        );
    }
    let path_length = path.capacity();
    if path.len() == path_length {
        if path_cost < *best_cost {
            if BRUTE_FORCE_DEBUG_LEVEL >= 1 {
                println!(
                    "  * Reached new cache cost record {} with path {:?}",
                    path_cost, path
                );
            }
            paths.clear();
            *best_cost = path_cost;
        } else {
            debug_assert_eq!(path_cost, *best_cost);
            if ((BRUTE_FORCE_DEBUG_LEVEL >= 1) && (paths.len() == 0))
                || (BRUTE_FORCE_DEBUG_LEVEL >= 2)
            {
                println!(
                    "  * Found a path that matches the cache cost constraint: {:?}",
                    path
                );
            }
        }
        paths.push(path);
        return;
    }

    // Otherwise, enumerate all possible next points, the constraints on these
    // being that...
    // - Next point should be within max_radius of the current [x, y] position
    // - Next point should remain within the iteration domain (no greater than
    //   num_feeds and y >= x).
    // - Next point should not be any point we've previously been through
    // - The total path cache cost is not allowed to go above the best path
    //   cache cost that we've observed so far (otherwise that path is less
    //   interesting than the best path).
    //
    // TODO: We could probably apply some memoization at that stage, what we
    //       should memoize depends on what it is exactly that takes time.
    //
    if BRUTE_FORCE_DEBUG_LEVEL >= 3 {
        println!("    - Not done yet, let's investigate next steps");
    }
    let &[curr_x, curr_y] = path.last().unwrap();
    for next_x in curr_x.saturating_sub(max_radius)..(curr_x + max_radius + 1).min(num_feeds) {
        for next_y in
            curr_y.saturating_sub(max_radius).max(next_x)..(curr_y + max_radius + 1).min(num_feeds)
        {
            // Loop invariants
            debug_assert!(next_x < num_feeds);
            debug_assert!(next_y < num_feeds);
            debug_assert!((next_x as isize - curr_x as isize).abs() as FeedIdx <= max_radius);
            debug_assert!((next_y as isize - curr_y as isize).abs() as FeedIdx <= max_radius);
            debug_assert!(next_y >= next_x);
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
            let mut next_cache = path_cache.clone();
            let mut next_cost = path_cost + next_cache.simulate_access(next_x);
            next_cost += next_cache.simulate_access(next_y);
            if next_cost > *best_cost {
                if BRUTE_FORCE_DEBUG_LEVEL >= 4 {
                    println!(
                        "      * That exceeds cache cost goal with only {}/{} steps, ignore it.",
                        path.len() + 1,
                        path.capacity()
                    );
                }
                continue;
            }

            // If so, continue recursively searching more paths from that point
            if BRUTE_FORCE_DEBUG_LEVEL >= 4 {
                println!("      * That seems reasonable, let's explore that path further...");
            }
            let mut next_path = path.clone();
            debug_assert_eq!(next_path.capacity(), next_path.len());
            next_path.reserve_exact(path_length - next_path.len());
            debug_assert_eq!(next_path.capacity(), path_length);
            next_path.push([next_x, next_y]);
            enumerate_paths_impl(
                num_feeds,
                max_radius,
                next_path,
                &next_cache,
                next_cost,
                paths,
                best_cost,
            );
            if BRUTE_FORCE_DEBUG_LEVEL >= 3 {
                println!("    - Back to path {:?}", path);
            }
        }
    }
    if BRUTE_FORCE_DEBUG_LEVEL >= 3 {
        println!("    - Done exploring possibilities from current path");
    }
}
