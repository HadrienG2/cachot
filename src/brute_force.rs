//! Mechanism for searching a better pair iterator than state-of-the-art 2D
//! iteration schemes designed for square lattices, via brute force search.

use crate::{
    cache::{self, CacheModel, CacheSimulation},
    FeedIdx, MAX_FEEDS, MAX_PAIRS,
};
use rand::prelude::*;
use std::{fmt::Write, rc::Rc};

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

/// Pair of feeds
pub type FeedPair = [FeedIdx; 2];

/// Type for storing paths through the 2D pair space
pub type Path = Vec<FeedPair>;

/// Use brute force to find a path which is better than our best strategy so far
/// according to our cache simulation.
pub fn search_best_path(
    num_feeds: FeedIdx,
    entry_size: usize,
    max_radius: FeedIdx,
    mut best_cost: cache::Cost,
) -> Option<(cache::Cost, Path)> {
    // Let's be reasonable here
    assert!(
        num_feeds > 1
            && num_feeds <= MAX_FEEDS
            && entry_size > 0
            && max_radius >= 1
            && best_cost > 0.0
    );

    // Set up the cache model
    let cache_model = CacheModel::new(entry_size);
    debug_assert!(
        cache_model.max_l1_entries() >= 3,
        "Cache is unreasonably small"
    );

    // In exhaustive mode, make sure that at least we don't re-discover one of
    // the previously discovered strategies.
    if BRUTE_FORCE_DEBUG_LEVEL >= 2 {
        best_cost -= 1.0;
    }

    // A path should go through every point of the 2D half-square defined by
    // x and y belonging to 0..num_feeds and y >= x. From this, we know exactly
    // how long the best path (assuming it exists) will be.
    let path_length = ((num_feeds as usize) * ((num_feeds as usize) + 1)) / 2;

    // We seed the path search algorithm by enumerating every possible starting
    // point for a path, under the following contraints:
    //
    // - To match the output of other algorithms, we want y >= x.
    // - Starting from a point (x, y) is geometrically equivalent to starting
    //   from the symmetric point (num_points-y, num_points-x), so we don't need
    //   to explore both of these starting points to find the optimal solution.
    //
    let mut priorized_partial_paths = PriorizedPartialPaths::new();
    for start_y in 0..num_feeds {
        for start_x in 0..=start_y.min(num_feeds - start_y - 1) {
            priorized_partial_paths.push(PartialPath::new(&cache_model, [start_x, start_y]));
        }
    }

    // Precompute the neighbors of every point of the [x, y] domain
    //
    // The constraints on them being...
    //
    // - Next point should be within max_radius of current [x, y] position
    // - Next point should remain within the iteration domain (no greater
    //   than num_feeds, and y >= x).
    //
    // For each point, we store...
    //
    // - The x coordinate of the first neighbor
    // - For this x coordinate and all subsequent ones, the range of y
    //   coordinates of all neighbors that have this x coordinate.
    //
    // This should achieve the indended goal of moving the neighbor constraint
    // logic out of the hot loop, without generating too much memory traffic
    // associated with reading out neighbor coordinates, nor hiding valuable
    // information about the next_x/next_y iteration pattern from the compiler.
    //
    // We also provide a convenient iteration function that produces the
    // iterator of neighbors associated with a certain point from this storage.
    //
    let mut neighbors = vec![(0, vec![]); num_feeds as usize * num_feeds as usize];
    let linear_idx =
        |&[curr_x, curr_y]: &FeedPair| curr_y as usize * num_feeds as usize + curr_x as usize;
    for curr_x in 0..num_feeds {
        for curr_y in curr_x..num_feeds {
            let next_x_range =
                curr_x.saturating_sub(max_radius)..(curr_x + max_radius + 1).min(num_feeds);
            debug_assert!(next_x_range.end <= num_feeds);
            debug_assert!((curr_x as isize - next_x_range.start as isize) <= max_radius as isize);
            debug_assert!((next_x_range.end as isize - curr_x as isize) <= max_radius as isize + 1);

            let (first_next_x, next_y_ranges) = &mut neighbors[linear_idx(&[curr_x, curr_y])];
            *first_next_x = next_x_range.start;

            for next_x in next_x_range {
                let next_y_range = curr_y.saturating_sub(max_radius).max(next_x)
                    ..(curr_y + max_radius + 1).min(num_feeds);
                debug_assert!(next_y_range.end <= num_feeds);
                debug_assert!(
                    (curr_y as isize - next_y_range.start as isize) <= max_radius as isize
                );
                debug_assert!(
                    (next_y_range.end as isize - curr_y as isize) <= max_radius as isize + 1
                );
                debug_assert!(next_y_range.start >= next_x);

                next_y_ranges.push(next_y_range);
            }
        }
    }
    let neighborhood = |&[curr_x, curr_y]: &FeedPair| {
        debug_assert!(curr_y >= curr_x);
        let (first_next_x, ref next_y_ranges) = &neighbors[linear_idx(&[curr_x, curr_y])];
        next_y_ranges.into_iter().cloned().enumerate().flat_map(
            move |(next_x_offset, next_y_range)| {
                next_y_range.map(move |next_y| [first_next_x + next_x_offset as u8, next_y])
            },
        )
    };

    // Next we iterate as long as we have incomplete paths by taking the most
    // promising path so far, considering all the next steps that can be taken
    // on that path, and pushing any further incomplete path that this creates
    // into our list of next actions.
    let mut best_path = Path::new();
    let mut rng = rand::thread_rng();
    while let Some(partial_path) = priorized_partial_paths.pop(&mut rng) {
        // Indicate which partial path was chosen
        if BRUTE_FORCE_DEBUG_LEVEL >= 3 {
            let mut path_display = String::new();
            for step in partial_path.iter_rev() {
                write!(path_display, "{:?} <- ", step).unwrap();
            }
            path_display.push_str("START");
            println!(
                "    - Currently on partial path {} with cache cost {}",
                path_display,
                partial_path.cost_so_far()
            );
        }

        // Ignore that path if we found another solution which is so good that
        // it's not worth exploring anymore.
        if partial_path.cost_so_far() > best_cost
            || ((BRUTE_FORCE_DEBUG_LEVEL < 2) && (partial_path.cost_so_far() == best_cost))
        {
            if BRUTE_FORCE_DEBUG_LEVEL >= 4 {
                println!(
                    "      * That exceeds cache cost goal with only {}/{} steps, ignore it.",
                    partial_path.len(),
                    path_length
                );
            }
            continue;
        }

        // Enumerate all possible next points, the constraints on them being...
        // - Next point should not be any point we've previously been through
        // - The total path cache cost is not allowed to go above the best path
        //   cache cost that we've observed so far (otherwise that path is less
        //   interesting than the best path).
        for next_step in neighborhood(partial_path.last_step()) {
            // Log which neighbor we're looking at in verbose mode
            if BRUTE_FORCE_DEBUG_LEVEL >= 4 {
                println!("      * Trying {:?}...", next_step);
            }

            // Have we been there before ?
            if partial_path.contains(&next_step) {
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
            //       We could then have the search loop start with a fast
            //       low-tolerance search, and resume with a slower
            //       high-tolerance search, ultimately getting to the point
            //       where we can search with infinite tolerance if we truly
            //       want the best of the best curves.
            //
            //       (note: for pairwise iteration that fits in L2 cache, a
            //       tolerance of 2 per step is an infinite tolerance).
            //
            //       This requires a way to propagate the "best cost at every
            //       step" to the caller, instead of just the the best cost at
            //       the last step, which anyway would be useful once we get to
            //       searching at multiple radii.
            //
            let next_step_eval = partial_path.evaluate_next_step(&cache_model, &next_step);
            let next_cost = next_step_eval.next_cost;
            if next_cost > best_cost || ((BRUTE_FORCE_DEBUG_LEVEL < 2) && (next_cost == best_cost))
            {
                if BRUTE_FORCE_DEBUG_LEVEL >= 4 {
                    println!(
                        "      * That exceeds cache cost goal with only {}/{} steps, ignore it.",
                        partial_path.len() + 1,
                        path_length
                    );
                }
                continue;
            }

            // Are we finished ?
            let next_path_len = partial_path.len() + 1;
            if next_path_len == path_length {
                if next_cost < best_cost {
                    best_path = partial_path.finish_path(next_step);
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
                            partial_path.finish_path(next_step),
                        );
                    }
                }
                continue;
            }

            // Otherwise, schedule searching further down this path
            if BRUTE_FORCE_DEBUG_LEVEL >= 4 {
                println!("      * That seems reasonable, we'll explore that path further...");
            }
            priorized_partial_paths.push(partial_path.commit_next_step(next_step_eval));
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

// ---

/// Machine word size in bits
const WORD_SIZE: u32 = (std::mem::size_of::<usize>() * 8) as u32;

/// Integer division with upper rounding
const fn div_round_up(num: usize, denom: usize) -> usize {
    (num / denom) + (num % denom != 0) as usize
}

/// Maximum number of machine words needed to store one bit per feed pair
const MAX_PAIR_WORDS: usize = div_round_up(MAX_PAIRS, WORD_SIZE as usize);

/// Path which we are in the process of exploring
struct PartialPath {
    // TODO: Consider replacing Rc with something that allows allocation reuse
    path: Rc<PathElems>,
    path_len: usize,
    visited_pairs: [usize; MAX_PAIR_WORDS],
    cache_sim: CacheSimulation,
    cost_so_far: cache::Cost,
}
//
/// Path elements are stored as a linked list to enable sharing of common nodes.
///
/// This should enable tremendous memory savings and faster path forking, both
/// of which are very important, at the cost of...
///
/// - Slowing down path iteration, but outside of debug logging scenarios we
///   only need that at the end of a path, and we don't reach the end of a path
///   very often as most paths are discarded due to excessive cost.
/// - Slowing down path length counting, which is more important. We address
///   this by double-counting the length so that we don't need to traverse the
///   list in order to know the length.
///
struct PathElems {
    curr_step: FeedPair,
    prev_steps: Option<Rc<PathElems>>,
}
//
impl PartialPath {
    /// Index of a certain coordinate in the visited_pairs bitvec
    //
    // NOTE: This operation is super hot and must be very fast
    //
    const fn coord_to_bit_index(&[x, y]: &FeedPair) -> (usize, u32) {
        let linear_index = y as usize * MAX_FEEDS as usize + x as usize;
        let word_index = linear_index / WORD_SIZE as usize;
        let bit_index = (linear_index % WORD_SIZE as usize) as u32;
        (word_index, bit_index)
    }

    // Inverse of to_bit_index
    //
    // NOTE: This operation is very rare and can be slow
    //
    const fn bit_index_to_coord(word: usize, bit: u32) -> FeedPair {
        let linear_index = word * WORD_SIZE as usize + bit as usize;
        let y = (linear_index / (MAX_FEEDS as usize)) as FeedIdx;
        let x = (linear_index % (MAX_FEEDS as usize)) as FeedIdx;
        [x, y]
    }

    /// Start a path
    //
    // NOTE: This operation is very rare and can be slow
    //
    pub fn new(cache_model: &CacheModel, start: FeedPair) -> Self {
        let path = Rc::new(PathElems {
            curr_step: start,
            prev_steps: None,
        });

        let mut cache_sim = cache_model.start_simulation();
        for &feed in start.iter() {
            let access_cost = cache_sim.simulate_access(&cache_model, feed);
            debug_assert_eq!(access_cost, 0.0);
        }

        let mut visited_pairs = [0; MAX_PAIR_WORDS];
        for word in 0..MAX_PAIR_WORDS {
            let mut current_word = 0;
            for bit in (0..WORD_SIZE).rev() {
                let [x, y] = Self::bit_index_to_coord(word, bit);
                let visited = (y < x) || ([x, y] == start);
                current_word = (current_word << 1) | (visited as usize);
            }
            visited_pairs[word] = current_word;
        }

        Self {
            path,
            path_len: 1,
            visited_pairs,
            cache_sim,
            cost_so_far: 0.0,
        }
    }

    /// Tell how long the path is
    //
    // NOTE: This operation is hot and must be fast
    //
    pub fn len(&self) -> usize {
        self.path_len
    }

    /// Get the last path entry
    //
    // NOTE: This operation is hot and must be fast
    //
    pub fn last_step(&self) -> &FeedPair {
        &self.path.curr_step
    }

    /// Iterate over the path in reverse step order
    //
    // NOTE: This operation can be slow, it is only intended for debug output.
    //
    pub fn iter_rev(&self) -> impl Iterator<Item = &FeedPair> {
        let mut next_node = Some(&*self.path);
        std::iter::from_fn(move || {
            let node = next_node?;
            next_node = node.prev_steps.as_ref().map(|rc| &**rc);
            Some(&node.curr_step)
        })
    }

    /// Tell whether a path contains a certain feed pair
    //
    // NOTE: This operation is super hot and must be very fast
    //
    pub fn contains(&self, pair: &FeedPair) -> bool {
        let (word, bit) = Self::coord_to_bit_index(pair);
        debug_assert!(word < self.visited_pairs.len());
        (unsafe { self.visited_pairs.get_unchecked(word) } & (1 << bit)) != 0
    }

    /// Get the accumulated cache cost of following this path so far
    //
    // NOTE: This operation is hot and must be fast
    //
    pub fn cost_so_far(&self) -> cache::Cost {
        self.cost_so_far
    }

    /// Given an extra feed pair, tell what the accumulated cache cost would
    /// become if the path was completed by this pair, and what the cache
    /// entries would then be.
    //
    // NOTE: This operation is super hot and must be very fast
    //
    pub fn evaluate_next_step(
        &self,
        cache_model: &CacheModel,
        &next_step: &FeedPair,
    ) -> NextStepEvaluation {
        let mut next_cache = self.cache_sim.clone();
        let next_cost = self.cost_so_far
            + next_step
                .iter()
                .map(|&feed| next_cache.simulate_access(&cache_model, feed))
                .sum::<f32>();
        NextStepEvaluation {
            next_step,
            next_cost,
            next_cache,
        }
    }

    /// Create a new partial path which follows all the steps from this one,
    /// plus an extra step for which the new cache cost and cache entries are
    /// provided.
    //
    // NOTE: This operation is relatively hot and must be quite fast
    //
    pub fn commit_next_step(&self, next_step_eval: NextStepEvaluation) -> Self {
        let NextStepEvaluation {
            next_step,
            next_cost,
            next_cache,
        } = next_step_eval;

        let next_path = Rc::new(PathElems {
            curr_step: next_step,
            prev_steps: Some(self.path.clone()),
        });

        let mut next_visited_pairs = self.visited_pairs.clone();
        let (word, bit) = Self::coord_to_bit_index(&next_step);
        // TODO: Make sure the bound check is elided or has negligible cost, if
        //       it is too expensive use get_unchecked.
        next_visited_pairs[word] |= 1 << bit;

        Self {
            path: next_path,
            path_len: self.path_len + 1,
            visited_pairs: next_visited_pairs,
            cache_sim: next_cache,
            cost_so_far: next_cost,
        }
    }

    /// Finish this path with a last step
    //
    // NOTE: This operation is rare (unless extra debugging output is enabled)
    //       and can therefore be quite slow.
    //
    pub fn finish_path(&self, last_step: FeedPair) -> Path {
        let mut final_path = vec![FeedPair::default(); self.path_len];
        final_path[self.path_len - 1] = last_step;
        for (i, step) in (0..self.path_len - 1).rev().zip(self.iter_rev()) {
            final_path[i] = *step;
        }
        final_path
    }
}

/// Result of `PartialPath::evaluate_next_step()`
struct NextStepEvaluation {
    next_step: FeedPair,
    next_cost: cache::Cost,
    next_cache: CacheSimulation,
}

// ---

/// PartialPath container that enables priorization and randomization
///
/// The amount of possible paths is ridiculously high (of the order of the
/// factorial of path_length, which itself grows like the square of num_feeds),
/// so it's extremely important to...
///
/// - Finish exploring paths reasonably quickly, to free up RAM and update
///   the "best cost" figure of merit, which in turn allow us to...
/// - Prune paths as soon as it becomes clear that they won't beat the
///   current best cost.
/// - Explore promising paths first, and make sure we explore large areas of the
///   path space quickly instead of perpetually staying in the same region of
///   the space of possible paths like depth-first search would have us do.
///
/// To help us with these goals, we store information about the paths which
/// we are in the process of exploring in a data structure which allows
/// priorizing the most promising tracks over others.
///
#[derive(Default)]
struct PriorizedPartialPaths {
    /// Sorted list of prioritized ongoing paths
    ///
    /// I tried using a BTreeMap before, but since we're always popping the last
    /// element and inserting close to the end, a sorted list is better.
    ///
    storage: Vec<(RoundedPriority, Vec<PartialPath>)>,

    /// Mechanism to reuse inner Vec allocations
    storage_morgue: Vec<Vec<PartialPath>>,

    /// Mechanism to periodically trigger path selection randomization
    randomness_clock: usize,
}
//
type RoundedPriority = usize;
//
impl PriorizedPartialPaths {
    /// Create the collection
    pub fn new() -> Self {
        Self::default()
    }

    /// Prioritize a certain path wrt others, higher is more important
    pub fn priorize(path: &PartialPath) -> RoundedPriority {
        // Increasing path length weight means that the highest priority is
        // put on seeing paths through the end (which allows discarding
        // them), decreasing means that the highest priority is put on
        // following through the paths that are most promizing in terms of
        // cache cost (which tends to favor a more breadth-first approach as
        // the first curve points are free of cache costs).
        (1.3 * path.len() as f32 - path.cost_so_far()).round() as _
    }

    /// Record a new partial path
    #[inline(always)]
    pub fn push(&mut self, path: PartialPath) {
        let priority = Self::priorize(&path);
        let storage_morgue = &mut self.storage_morgue;
        let mut make_new_path_list = |path| {
            let mut new_paths = storage_morgue.pop().unwrap_or_default();
            new_paths.push(path);
            (priority, new_paths)
        };
        for (idx, (curr_priority, curr_paths)) in self.storage.iter_mut().enumerate().rev() {
            if *curr_priority == priority {
                curr_paths.push(path);
                return;
            } else if *curr_priority < priority {
                self.storage.insert(idx + 1, make_new_path_list(path));
                return;
            }
        }
        self.storage.insert(0, make_new_path_list(path));
    }

    /// Extract one of the highest-priority paths
    #[inline(always)]
    pub fn pop(&mut self, mut rng: impl Rng) -> Option<PartialPath> {
        // Find the set of highest priority paths
        let (_priority, highest_priority_paths) = self.storage.last_mut()?;
        debug_assert!(!highest_priority_paths.is_empty());

        // Normally, we pick the last path in the list, which is most efficient,
        // but we regularly allow ourselves to pick a random path instead in
        // order to make sure that the algorithm doesn't get stuck perpetually
        // exploring the same region of the decision tree.
        const RANDOM_PICK_RATE: usize = 1 << 5;
        self.randomness_clock += 1;
        let path = if self.randomness_clock % RANDOM_PICK_RATE != 0 {
            highest_priority_paths.pop().unwrap()
        } else {
            let path_idx = rng.gen_range(0..highest_priority_paths.len());
            highest_priority_paths.remove(path_idx)
        };

        // If the set of highest priority paths is now empty, we remove it, but
        // keep the allocation around for re-use
        if highest_priority_paths.is_empty() {
            let (_priority, mut highest_priority_paths) = self.storage.pop().unwrap();
            highest_priority_paths.clear();
            self.storage_morgue.push(highest_priority_paths);
        }

        // Finally, we return the chosen path
        Some(path)
    }
}
