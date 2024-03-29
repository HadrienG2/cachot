//! Mechanism for testing the cache locality of pair iterators

use crate::{
    cache::{self, CacheModel, L1_MISS_COST, NEW_ENTRY_COST},
    FeedIdx,
};
use num_traits::identities::Zero;

/// Test harness for evaluating the locality of several feed pair iterators and
/// picking the best of them.
pub struct PairLocalityTester {
    debug_level: usize,
    cache_model: CacheModel,
    best_iterator: Option<(String, Box<[cache::Cost]>)>,
}
//
impl PairLocalityTester {
    /// Build the test harness
    pub fn new(debug_level: usize, entry_size: usize) -> Self {
        if debug_level == 0 {
            println!("┌──────────────────────────────────────────┬──────────────────┬────────────────────┬───────────────┐");
            println!("│ Iterator name                            │ Total cache cost │ w/o first accesses │ Per feed load │");
            println!("├──────────────────────────────────────────┼──────────────────┼────────────────────┼───────────────┤");
        }
        Self {
            debug_level,
            cache_model: CacheModel::new(entry_size),
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
            println!("\nTesting feed pair iterator \"{name}\"...");
        }
        let mut cache_sim = self.cache_model.start_simulation();
        let mut total_cost = cache::Cost::zero();
        let mut new_entries_cost = cache::Cost::zero();
        let mut cumulative_cost = Vec::new();
        let mut feed_load_count = 0;
        for feed_pair in feed_pair_iterator {
            if self.debug_level >= 2 {
                println!("- Accessing feed pair {feed_pair:?}...")
            }
            let mut pair_cost = cache::Cost::zero();
            let mut pair_entries_cost = cache::Cost::zero();
            for feed in feed_pair.iter().copied() {
                let prev_accessed_entries = cache_sim.num_accessed_entries();
                let feed_cost = cache_sim.simulate_access(&self.cache_model, feed);
                let is_new_entry = cache_sim.num_accessed_entries() != prev_accessed_entries;
                let new_entry_str = if is_new_entry { " (first access)" } else { "" };
                if self.debug_level >= 2 {
                    println!(
                        "  * Accessed feed {feed} \
                             for cache cost {feed_cost}{new_entry_str}",
                    )
                }
                pair_cost += feed_cost;
                pair_entries_cost +=
                    cache::Cost::from_num(is_new_entry as u8) * NEW_ENTRY_COST / L1_MISS_COST;
            }
            let new_entries_str = if pair_entries_cost != 0.0 {
                format!(" ({pair_entries_cost} from first accesses)")
            } else {
                String::new()
            };
            match self.debug_level {
                0 => {}
                1 => println!(
                    "- Accessed feed pair {feed_pair:?} \
                       for cache cost {pair_cost}{new_entries_str}"
                ),
                _ => println!("  * Total cache cost of this pair is {pair_cost}{new_entries_str}"),
            }
            total_cost += pair_cost;
            new_entries_cost += pair_entries_cost;
            cumulative_cost.push(total_cost);
            feed_load_count += 2;
        }
        match self.debug_level {
            0 => {
                println!(
                    "│ {:<40} │ {:<16} │ {:<18} │ {:<13.2} │",
                    name,
                    total_cost,
                    total_cost - new_entries_cost,
                    (total_cost - new_entries_cost).to_num::<f32>() / (feed_load_count as f32)
                )
            }
            _ => println!(
                "- Total cache cost of this iterator is {}, {} w/o first accesses, {:.2} per feed load",
                total_cost,
                total_cost - new_entries_cost,
                (total_cost - new_entries_cost).to_num::<f32>() / (feed_load_count as f32)
            ),
        }
        let best_cost = self
            .best_iterator
            .as_ref()
            .map(|(_name, cumulative_cost)| *cumulative_cost.last().unwrap())
            .unwrap_or(cache::Cost::MAX);
        if total_cost < best_cost {
            self.best_iterator = Some((name.to_owned(), cumulative_cost.into()));
        }
    }

    /// Tell which of the iterators that were tested so far got the best results
    ///
    /// If a tie occurs, pick the first iterator, as we're testing designs from
    /// the simplest to the most complex ones.
    ///
    pub fn announce_best_iterator(&self) -> &[cache::Cost] {
        if self.debug_level > 0 {
            println!();
        } else {
            println!("└──────────────────────────────────────────┴──────────────────┴────────────────────┴───────────────┘");
        }
        let (best_name, cumulative_cost) = self
            .best_iterator
            .as_ref()
            .expect("No iterator has been tested yet");
        println!(
            "The best iterator so far is \"{best_name}\" \
             with cumulative cost at each step {cumulative_cost:?}"
        );
        &cumulative_cost[..]
    }
}
