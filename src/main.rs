mod cache;

use crate::cache::CacheModel;

use genawaiter::{stack::let_gen, yield_};
use space_filler::{hilbert, morton, Coordinate, CurveIdx};

type FeedIdx = Coordinate;

fn test_feed_pair_locality(
    debug_level: usize,
    entry_size: usize,
    name: &str,
    feed_pair_iterator: impl Iterator<Item = [FeedIdx; 2]>,
) {
    if debug_level > 0 {
        println!("\nTesting feed pair iterator \"{}\"...", name);
    }
    let mut cache_model = CacheModel::new(entry_size);
    let mut total_cost = 0.0;
    let mut feed_load_count = 0;
    for feed_pair in feed_pair_iterator {
        if debug_level >= 2 {
            println!("- Accessing feed pair {:?}...", feed_pair)
        }
        let mut pair_cost = 0.0;
        for feed in feed_pair.iter().copied() {
            let feed_cost = cache_model.simulate_access(feed);
            if debug_level >= 2 {
                println!("  * Accessed feed {} for cache cost {}", feed, feed_cost)
            }
            pair_cost += feed_cost;
        }
        match debug_level {
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
    match debug_level {
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
}

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
            println!("--- Testing L1 capacity of {} feeds ---", num_l1_entries);
            if debug_level == 0 {
                println!();
            }

            // Naive iteration scheme
            let_gen!(basic, {
                for feed1 in 0..num_feeds {
                    for feed2 in feed1..num_feeds {
                        yield_!([feed1, feed2]);
                    }
                }
            });
            test_feed_pair_locality(debug_level, entry_size, "Naive", basic.into_iter());

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
                test_feed_pair_locality(
                    debug_level,
                    entry_size,
                    &format!("{0}x{0} blocks", block_size),
                    blocked_basic.into_iter(),
                );
                block_size += 1;
            }

            // Morton curve ("Z order") iteration
            let morton = (0..(num_feeds as CurveIdx * num_feeds as CurveIdx))
                .map(morton::decode_2d)
                .filter(|[feed1, feed2]| feed2 >= feed1);
            test_feed_pair_locality(debug_level, entry_size, "Morton curve", morton);

            // Hilbert curve iteration
            let hilbert = (0..(num_feeds as CurveIdx * num_feeds as CurveIdx))
                .map(hilbert::decode_2d)
                .filter(|[feed1, feed2]| feed2 >= feed1);
            test_feed_pair_locality(debug_level, entry_size, "Hilbert curve", hilbert);

            debug_level = debug_level.saturating_sub(1);
            println!();
        }
        debug_level = (num_feeds < 8).into();
    }
}
