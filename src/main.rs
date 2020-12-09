mod morton;

use genawaiter::{stack::let_gen, yield_};

type FeedIdx = usize;
type Entry = FeedIdx;
type Cost = f32;

// Numbers stolen from the latency plot of Anandtech's Zen3 review, not very
// precise but we only care about the orders of magnitude on recent CPUs...
//
// We're using numbers from the region where most of AnandTech's tests move out
// of cache. The "full random" test is probably too pessimistic here.
//
// We're taking the height of cache latencies plateaux as our cost figure and
// the abscissa of half-plateau as our capacity figure.
//
const L1_CAPACITY: usize = 32 * 1024;
const L1_MISS_COST: Cost = 2.0;
const L2_CAPACITY: usize = 512 * 1024;
const L2_MISS_COST: Cost = 10.0;
const L3_CAPACITY: usize = 32 * 1024 * 1024;
const L3_MISS_COST: Cost = 60.0;

/// CPU cache model, used for evaluating locality merits of 2D iteration schemes
#[derive(Debug)]
struct CacheModel {
    // Entries ordered by access date, most recently accessed entry goes last
    entries: Vec<Entry>,

    // L1 capacity in entries
    l1_entries: usize,

    // L2 capacity in entries
    l2_entries: usize,

    // L3 capacity in entries
    l3_entries: usize,
}

impl CacheModel {
    /// Set up the cache model by telling the size of individual cache entries
    pub fn new(entry_size: usize) -> Self {
        Self {
            entries: Vec::new(),
            l1_entries: L1_CAPACITY / entry_size,
            l2_entries: L2_CAPACITY / entry_size,
            l3_entries: L3_CAPACITY / entry_size,
        }
    }

    /// Tell how expensive it would be to access an entry (in units of L1 cache
    /// miss costs, with L1 hits considered free), given how many other entries
    /// have been accessed since the last time this entry was accessed.
    fn cost_model(&self, age: usize) -> Cost {
        if age < self.l1_entries {
            0.0
        } else if age < self.l2_entries {
            1.0
        } else if age < self.l3_entries {
            L2_MISS_COST / L1_MISS_COST
        } else {
            L3_MISS_COST / L1_MISS_COST
        }
    }

    /// Simulate loading an entry and return the cost in units of L1 cache miss
    pub fn simulate_access(&mut self, entry: Entry) -> Cost {
        // Look up the entry in the cache
        let entry_pos = self.entries.iter().rposition(|&item| item == entry);

        // Was it found?
        if let Some(entry_pos) = entry_pos {
            // If so, compute entry age and deduce access cost
            let entry_age = self.entries.len() - entry_pos - 1;
            let access_cost = self.cost_model(entry_age);

            // Move the entry back to the front of the cache
            self.entries.remove(entry_pos);
            self.entries.push(entry);

            // Return the access cost
            access_cost
        } else {
            // If not, insert the entry in the cache
            self.entries.push(entry);

            // Report a zero cost. We don't want to penalize the first access in
            // our cost model since it will have to happen no matter how good we
            // are in our cache access pattern...
            0.0
        }
    }
}

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
            total_cost / (feed_load_count as Cost)
        ),
        _ => println!(
            "- Total cache cost of this iterator is {} ({:.2} per feed load)",
            total_cost,
            total_cost / (feed_load_count as Cost)
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
            let entry_size = L1_CAPACITY / num_l1_entries;
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
                    for feed1_block in (0..num_feeds).step_by(block_size) {
                        for feed2_block in (feed1_block..num_feeds).step_by(block_size) {
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
            let_gen!(morton, {
                // Iterate over Morton curve indices
                for morton_idx in 0..(num_feeds * num_feeds) {
                    // Translate back into grid indices
                    let [feed1, feed2] = morton::decode_2d(morton_idx);
                    // Only yield each feed pair once
                    if feed2 >= feed1 {
                        yield_!([feed1, feed2]);
                    }
                }
            });
            test_feed_pair_locality(debug_level, entry_size, "Morton curve", morton.into_iter());

            // TODO: Maybe test Hilbert iteration

            debug_level = debug_level.saturating_sub(1);
            println!();
        }
        debug_level = (num_feeds < 8).into();
    }
}
