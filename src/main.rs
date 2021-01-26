mod brute_force;
pub(crate) mod cache;
mod pair_locality;

use crate::{brute_force::StepDistance, pair_locality::PairLocalityTester};
use genawaiter::{stack::let_gen, yield_};
use space_filler::{hilbert, morton, CurveIdx};
use std::time::Duration;

/// Integer type used for counting radio feeds
type FeedIdx = space_filler::Coordinate;

/// Upper bound on the number of feeds that will be used
///
/// To reduce memory allocation and improve data locality, we would like to use
/// fixed-sized data structures when the size depends on the number of feeds, as
/// this parameter is known at compile time.
///
/// Unfortunately, Rust does not have const generics yet, and even the
/// `min_const_generics` version that will be stabilized soon-ish is not
/// powerful enough as it does not allow us to have a data structure whose size
/// is a function of the number of feeds (such as, for example, a bitvec whose
/// number of bits is equal to the number of feed pairs).
///
/// As a compromise until we get there, we will use an upper bound on the number
/// of feeds that will be used in tests.
///
pub(crate) const MAX_FEEDS: FeedIdx = 8 /* 16 */;

/// Maximum number of feed pairs
const MAX_PAIRS: usize = MAX_FEEDS as usize * MAX_FEEDS as usize;

/// Maximum number of ordered feed pairs
const MAX_ORDERED_PAIRS: usize = MAX_FEEDS as usize * (MAX_FEEDS as usize + 1) / 2;

fn main() {
    #[rustfmt::skip]
    const TESTED_NUM_FEEDS: &'static [FeedIdx] = &[
        // Minimal useful test (any iteration scheme is optimal with 2 feeds)
        // Useful for manual inspection of detailed execution traces
        4,
        // Actual PAON-4 configuration
        8,
        /* // TODO: What would happen with more feeds?
        16 */
    ];
    assert!(*TESTED_NUM_FEEDS.iter().max().unwrap() <= MAX_FEEDS);

    let mut debug_level = 2;
    for num_feeds in TESTED_NUM_FEEDS.iter().copied() {
        println!("=== Testing with {} feeds ===\n", num_feeds);
        assert!(
            num_feeds <= MAX_FEEDS,
            "Please update MAX_FEEDS for this configuration"
        );

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
            let mut best_cumulative_cost = locality_tester.announce_best_iterator().to_owned();

            // Now, let's try to brute-force a better iterator. First, evaluate
            // all possible paths through the iteration domain where we don't
            // step by more than (for now) [1, 1]...
            println!("\nPerforming brute force search for a better path...");
            let mut tolerance = 0.0;
            let mut best_extra_distance = StepDistance::MAX;
            while tolerance < *best_cumulative_cost.last().unwrap() {
                'radius: for max_radius in 1..num_feeds {
                    println!(
                        "- Using cumulative cost tolerance {} and search radius {}",
                        tolerance, max_radius
                    );
                    if let Some(path) = brute_force::search_best_path(
                        num_feeds,
                        entry_size,
                        max_radius,
                        &mut best_cumulative_cost[..],
                        &mut best_extra_distance,
                        tolerance,
                        Duration::from_secs(60),
                    ) {
                        println!("  * Found a better path at cumulative cost tolerance {} and search radius {}:", tolerance, max_radius);
                        println!(
                            "    - Cache cost w/o first accesses: {}",
                            best_cumulative_cost.last().unwrap() - cache::min_cache_cost(num_feeds),
                        );
                        println!(
                            "    - Deviation from unit steps: {:.2}",
                            best_extra_distance
                        );
                        println!("    - Path: {:?}", path,);
                    } else {
                        println!("  * Did not find any better path at that tolerance");
                    }
                    if *best_cumulative_cost.last().unwrap()
                        == 1.0 + cache::min_cache_cost(num_feeds)
                    {
                        println!("  * We won't be able to do any better by increasing the radius.");
                        break 'radius;
                    }
                }
                tolerance += 1.0;
            }

            debug_level = debug_level.saturating_sub(1);
            println!();
        }
        debug_level = (num_feeds < 8).into();
    }
}
