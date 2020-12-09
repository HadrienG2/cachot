use genawaiter::{stack::let_gen, yield_};

type Feed = usize;
type Entry = Feed;
type Cost = usize;

#[derive(Debug, Default)]
struct CacheModel {
    // Entries ordered by access date, most recently accessed entry goes last
    entries: Vec<Entry>,
}

impl CacheModel {
    // Set up a cache model
    pub fn new() -> Self {
        Self::default()
    }

    // Model of how expensive it is to access an entry with respect to how many
    // other entries have been accessed since the last time it was accessed.
    fn cost_model(age: usize) -> Cost {
        // We assume that the cache is large enough to store a single pair of
        // entries, and anything beyond that gets more and more costly.
        // TODO: Try a more realistic multi-step function model that represents
        //       the L1/L2/L3/RAM cache levels of Intel CPUs.
        age.saturating_sub(1)
    }

    pub fn simulate_access(&mut self, entry: Entry) -> Cost {
        // Look up the entry in the cache
        let entry_pos = self.entries.iter().rposition(|&item| item == entry);

        // Was it found?
        if let Some(entry_pos) = entry_pos {
            // If so, compute entry age and deduce access cost
            let entry_age = self.entries.len() - entry_pos - 1;
            let access_cost = Self::cost_model(entry_age);

            // Move the entry to the front of the cache
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
            0
        }
    }
}

fn test_feed_pair_locality(
    debug_level: usize,
    name: &str,
    feed_pair_iterator: impl Iterator<Item = [Feed; 2]>,
) {
    println!("Testing feed pair iterator \"{}\"...", name);
    let mut cache_model = CacheModel::new();
    let mut total_cost = 0;
    for feed_pair in feed_pair_iterator {
        if debug_level >= 2 {
            println!("- Accessing feed pair {:?}...", feed_pair)
        }
        let mut pair_cost = 0;
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
            2 => println!("  * Cache cost of this pair is {}", pair_cost),
            _ => unreachable!(),
        }
        total_cost += pair_cost;
    }
    println!("- Cache cost of this iterator is {}\n", total_cost);
}

fn main() {
    const TESTED_NUM_FEEDS: [Feed; 4] = [
        4, // Minimal useful test (any scheme is optimal with two feeds)
        8, // PAON-4 configuration
        16, 32,
    ];
    let mut debug_level = 2;
    for num_feeds in TESTED_NUM_FEEDS.iter().copied() {
        println!("=== TESTING WITH {} FEEDS ===\n", num_feeds);

        // Current iteration scheme
        let_gen!(basic_iterator, {
            for feed1 in 0..num_feeds {
                for feed2 in feed1..num_feeds {
                    yield_!([feed1, feed2]);
                }
            }
        });
        test_feed_pair_locality(debug_level, "basic", basic_iterator.into_iter());

        // Block-wise iteration scheme
        let mut block_size = 2;
        while block_size < num_feeds {
            let_gen!(blocked_basic_iterator, {
                for feed1_block in (0..num_feeds).step_by(block_size) {
                    for feed2_block in (feed1_block..num_feeds).step_by(block_size) {
                        for feed1 in feed1_block..feed1_block + block_size {
                            for feed2 in feed1.max(feed2_block)..feed2_block + block_size {
                                yield_!([feed1, feed2]);
                            }
                        }
                    }
                }
            });
            test_feed_pair_locality(
                debug_level,
                &format!("blocked basic (block size {})", block_size),
                blocked_basic_iterator.into_iter(),
            );
            block_size *= 2;
        }

        debug_level = debug_level.saturating_sub(1);
    }
}
