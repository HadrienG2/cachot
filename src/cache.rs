//! Minimal cache simulator for 2D iteration locality studies

use crate::{FeedIdx, MAX_FEEDS, MAX_ORDERED_PAIRS};

pub type Entry = FeedIdx;
pub type Cost = f32;

// Numbers stolen from the latency plot of Anandtech's Zen3 review, not very
// precise but we only care about the orders of magnitude on recent CPUs...
//
// We're using numbers from the region where most of AnandTech's tests move out
// of cache. The "full random" test is probably too pessimistic here.
//
// We're taking the height of cache latencies plateaux as our cost figure and
// the abscissa of half-plateau as our capacity figure.
//
pub const L1_CAPACITY: usize = 32 * 1024;
pub const L1_MISS_COST: Cost = 2.0;
pub const L2_CAPACITY: usize = 512 * 1024;
pub const L2_MISS_COST: Cost = 10.0;
pub const L3_CAPACITY: usize = 32 * 1024 * 1024;
pub const L3_MISS_COST: Cost = 60.0;

/// CPU cache model, used for evaluating locality merits of 2D iteration schemes
#[derive(Debug)]
pub struct CacheModel {
    // L1 capacity in entries
    l1_entries: usize,

    // L2 capacity in entries
    l2_entries: usize,

    // L3 capacity in entries
    l3_entries: usize,
}
//
impl CacheModel {
    /// Set up the cache model by telling the size of individual cache entries
    pub fn new(entry_size: usize) -> Self {
        Self {
            l1_entries: L1_CAPACITY / entry_size,
            l2_entries: L2_CAPACITY / entry_size,
            l3_entries: L3_CAPACITY / entry_size,
        }
    }

    /// Query the number of L1 cache entries
    pub(crate) fn max_l1_entries(&self) -> usize {
        self.l1_entries
    }

    /// Tell how expensive it would be to access an entry (in units of L1 cache
    /// miss costs, with L1 hits considered free), given how many other entries
    /// have been accessed since the last time this entry was accessed.
    //
    // TODO: Entry size should probably also play a role here
    //
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

    /// Start a cache simulation
    pub fn start_simulation(&self) -> CacheSimulation {
        CacheSimulation::new()
    }
}

/// CPU cache simulation
///
/// Split from the main CacheModel so that we can efficiently have multiple
/// cache simulations that efficiently follow the same cache model.
///
#[derive(Clone)]
pub struct CacheSimulation {
    clock: CacheClock,
    last_accesses: [CacheClock; MAX_FEEDS as usize],
}
//
/// Clock type should be able to hold the max cache timestamp, which is twice
/// the number of pairs, which itself is MAX_FEEDS * (MAX_FEEDS + 1) / 2
type CacheClock = u16;
//
impl CacheSimulation {
    /// Set up some cache entries and a clock
    fn new() -> Self {
        assert!(CacheClock::MAX as usize >= 2 * MAX_ORDERED_PAIRS + 1);
        Self {
            clock: 1,
            last_accesses: [0; MAX_FEEDS as usize],
        }
    }

    /// Check out how many other entries have been accessed since a cache entry
    /// was last accessed, return 0 if the entry was never accessed.
    fn age(&self, entry: Entry) -> usize {
        let last_access_time = self.last_accesses[entry as usize];
        if last_access_time == 0 {
            0
        } else {
            self.last_accesses
                .iter()
                .filter(|&&access_time| access_time > last_access_time)
                .count()
        }
    }

    /// Simulate a cache access and return previous entry age
    fn access(&mut self, entry: Entry) -> usize {
        let age = self.age(entry);
        self.last_accesses[entry as usize] = self.clock;
        self.clock += 1;
        age
    }

    /// Simulate a cache access and return its cost
    pub fn simulate_access(&mut self, model: &CacheModel, entry: Entry) -> Cost {
        model.cost_model(self.access(entry))
    }
}
