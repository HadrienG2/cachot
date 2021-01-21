//! Minimal cache simulator for 2D iteration locality studies

use crate::{FeedIdx, MAX_FEEDS};

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

/// CPU cache entries
///
/// Split from the main CacheModel so that we can have multiple cache
/// simulations that efficiently follow the main cache model.
//
// Entries are ordered by access date, most recently accessed entry goes last
//
pub type CacheEntries = Vec<Entry>;

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

    /// Simulate loading an entry and return the cost in units of L1 cache miss
    pub fn simulate_access(&self, sim: &mut CacheSimulation, entry: Entry) -> Cost {
        self.cost_model(sim.access(entry))
    }
}

/// CPU cache entries
///
/// Split from the main CacheModel so that we can have multiple cache
/// simulations that efficiently follow the main cache model.
///
#[derive(Clone)]
pub struct CacheSimulation {
    clock: usize,
    last_accesses: [usize; MAX_FEEDS as usize],
}
//
impl CacheSimulation {
    /// Set up some cache entries and a clock
    fn new() -> Self {
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

    // Simulate a cache access and if the entry was accessed
    fn access(&mut self, entry: Entry) -> usize {
        let age = self.age(entry);
        self.last_accesses[entry as usize] = self.clock;
        self.clock += 1;
        age
    }
}
