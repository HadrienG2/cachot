mod brute_force;
pub(crate) mod cache;
mod pair_locality;

use crate::pair_locality::PairLocalityTester;
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
const _MAX_UNORDERED_PAIRS: usize = MAX_FEEDS as usize * (MAX_FEEDS as usize + 1) / 2;

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
        for num_l1_entries in 2..num_feeds {
            // Announce test
            println!("--- Testing L1 capacity of {} feeds ---", num_l1_entries);
            if debug_level == 0 {
                println!();
            }

            // Set up test
            let entry_size = cache::L1_CAPACITY / num_l1_entries as usize;
            let mut locality_tester = PairLocalityTester::new(debug_level, entry_size);

            // Naive iteration scheme
            let_gen!(naive, {
                for feed1 in 0..num_feeds {
                    for feed2 in feed1..num_feeds {
                        yield_!([feed1, feed2]);
                    }
                }
            });
            locality_tester.test_feed_pair_locality("Naive", naive.into_iter());

            // Iteration scheme that goes from top to bottom, and for each row
            // alternatively goes from right to left and from left to right
            let_gen!(zigzag, {
                let mut reverse = true;
                for feed2 in 0..num_feeds {
                    if reverse {
                        for feed1 in (0..=feed2).rev() {
                            yield_!([feed1, feed2]);
                        }
                    } else {
                        for feed1 in 0..=feed2 {
                            yield_!([feed1, feed2]);
                        }
                    }
                    reverse = !reverse;
                }
            });
            locality_tester.test_feed_pair_locality("Zig-zag", zigzag.into_iter());

            // Variation of the "zig-zag" scheme that also switches the
            // iteration direction from vertical to horizontal and back whenever
            // the end of a line is reached
            let_gen!(zigzag_corner, {
                let mut reverse = false;
                let mut offset = 0;
                while offset < num_feeds - offset {
                    if reverse {
                        for feed1 in ((offset + 1)..(num_feeds - offset)).rev() {
                            yield_!([feed1, num_feeds - offset - 1]);
                        }
                        for feed2 in (offset..(num_feeds - offset)).rev() {
                            yield_!([offset, feed2]);
                        }
                    } else {
                        for feed2 in offset..(num_feeds - offset) {
                            yield_!([offset, feed2]);
                        }
                        for feed1 in (offset + 1)..(num_feeds - offset) {
                            yield_!([feed1, num_feeds - offset - 1]);
                        }
                    }
                    reverse = !reverse;
                    offset += 1;
                }
            });
            locality_tester.test_feed_pair_locality("Zig-zag corner", zigzag_corner.into_iter());

            // Iteration schemes that gradually shrinks the triangular domain of
            // radio feed pairs into smaller triangular domains by progressing
            // in diagonal stripes
            for stripe_width in 1..num_feeds {
                // Minimal version, all stripes taken from top-left to bottom-right
                let_gen!(striped_minimal, {
                    let mut stripe_offset = 0;
                    while stripe_offset < num_feeds {
                        for feed2 in stripe_offset..num_feeds {
                            let stripe_end = feed2.saturating_sub(stripe_offset);
                            let stripe_start = stripe_end.saturating_sub(stripe_width - 1);
                            for feed1 in stripe_start..=stripe_end {
                                yield_!([feed1, feed2]);
                            }
                        }
                        stripe_offset += stripe_width;
                    }
                });
                locality_tester.test_feed_pair_locality(
                    &format!("{0}-wide stripes (minimal)", stripe_width),
                    striped_minimal.into_iter(),
                );

                // Slightly more elaborate iteration order that goes from top
                // to bottom on the first iteration, from bottom to top on the
                // second iteration, then back from top to bottom...
                let_gen!(striped_vertical_zigzag, {
                    let mut stripe_offset = 0;
                    let mut reverse = false;
                    while stripe_offset < num_feeds {
                        if reverse {
                            for feed2 in (stripe_offset..num_feeds).rev() {
                                let stripe_end = feed2.saturating_sub(stripe_offset);
                                let stripe_start = stripe_end.saturating_sub(stripe_width - 1);
                                for feed1 in (stripe_start..=stripe_end).rev() {
                                    yield_!([feed1, feed2]);
                                }
                            }
                        } else {
                            for feed2 in stripe_offset..num_feeds {
                                let stripe_end = feed2.saturating_sub(stripe_offset);
                                let stripe_start = stripe_end.saturating_sub(stripe_width - 1);
                                for feed1 in stripe_start..=stripe_end {
                                    yield_!([feed1, feed2]);
                                }
                            }
                        }
                        stripe_offset += stripe_width;
                        reverse = !reverse;
                    }
                });
                locality_tester.test_feed_pair_locality(
                    &format!("{0}-wide stripes (vertical zig-zag)", stripe_width),
                    striped_vertical_zigzag.into_iter(),
                );

                // Further elaboration on top of the "vertical zig-zag" version,
                // which also goes back and forth in the horizontal direction
                let_gen!(striped_double_zigzag, {
                    let mut stripe_offset = 0;
                    let mut vertical_reverse = false;
                    while stripe_offset < num_feeds {
                        let mut horizontal_reverse = !vertical_reverse;
                        if vertical_reverse {
                            for feed2 in (stripe_offset..num_feeds).rev() {
                                let stripe_end = feed2.saturating_sub(stripe_offset);
                                let stripe_start = stripe_end.saturating_sub(stripe_width - 1);
                                if horizontal_reverse {
                                    for feed1 in stripe_start..=stripe_end {
                                        yield_!([feed1, feed2]);
                                    }
                                } else {
                                    for feed1 in (stripe_start..=stripe_end).rev() {
                                        yield_!([feed1, feed2]);
                                    }
                                }
                                horizontal_reverse = !horizontal_reverse;
                            }
                        } else {
                            for feed2 in stripe_offset..num_feeds {
                                let stripe_end = feed2.saturating_sub(stripe_offset);
                                let stripe_start = stripe_end.saturating_sub(stripe_width - 1);
                                if horizontal_reverse {
                                    for feed1 in (stripe_start..=stripe_end).rev() {
                                        yield_!([feed1, feed2]);
                                    }
                                } else {
                                    for feed1 in stripe_start..=stripe_end {
                                        yield_!([feed1, feed2]);
                                    }
                                }
                                horizontal_reverse = !horizontal_reverse;
                            }
                        }
                        stripe_offset += stripe_width;
                        vertical_reverse = !vertical_reverse;
                    }
                });
                locality_tester.test_feed_pair_locality(
                    &format!("{0}-wide stripes (double zig-zag)", stripe_width),
                    striped_double_zigzag.into_iter(),
                );

                // TODO: Add "mirrorred" strategy that flips the coordinate
                //       iteration order
            }

            // Block-wise iteration scheme
            for block_size in 2..num_feeds {
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

            // Now, let's try to brute-force a better iterator
            println!("\nPerforming brute force search for a better path...");
            brute_force::search_best_path(
                num_feeds,
                entry_size,
                &mut best_cumulative_cost[..],
                Duration::from_secs(60),
            );

            debug_level = debug_level.saturating_sub(1);
            println!();
        }
        debug_level = (num_feeds < 8).into();
    }
}
