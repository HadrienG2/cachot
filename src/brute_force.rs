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
///     achieves a new total cache cost record.
/// 2 = Log per-step cumulative cache costs when a record is achieved.
/// 3 = Log every time we take a step on a path.
/// 4 = Log the process of searching for a next step on a path.
///
const BRUTE_FORCE_DEBUG_LEVEL: u8 = 2;

/// Pair of feeds
pub type FeedPair = [FeedIdx; 2];

/// Type for storing paths through the 2D pair space
pub type Path = Box<[FeedPair]>;

/// Use brute force to find a path which is better than our best strategy so far
/// according to our cache simulation.
pub fn search_best_path(
    num_feeds: FeedIdx,
    entry_size: usize,
    best_cumulative_cost: &mut [cache::Cost],
    tolerance: cache::Cost,
) -> Option<Path> {
    // Let's be reasonable here
    let mut total_cost_target = *best_cumulative_cost.last().unwrap() - 1.0;
    assert!(num_feeds > 1 && num_feeds <= MAX_FEEDS && entry_size > 0 && total_cost_target >= 0.0);

    // Set up the cache model
    let cache_model = CacheModel::new(entry_size);
    debug_assert!(
        cache_model.max_l1_entries() >= 3,
        "Cache is unreasonably small"
    );

    // A path should go through every point of the 2D half-square defined by
    // x and y belonging to 0..num_feeds and y >= x. From this, we know exactly
    // how long the best path (assuming it exists) will be.
    let path_length = ((num_feeds as usize) * ((num_feeds as usize) + 1)) / 2;
    debug_assert_eq!(best_cumulative_cost.len(), path_length);

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

    // Next we iterate as long as we have incomplete paths by taking the most
    // promising path so far, considering all the next steps that can be taken
    // on that path, and pushing any further incomplete path that this creates
    // into our list of next actions.
    let mut best_path = None;
    let mut rng = rand::thread_rng();
    let mut path_counter = 0u64;
    while let Some(partial_path) = priorized_partial_paths.pop(&mut rng) {
        // Indicate which partial path was chosen
        if BRUTE_FORCE_DEBUG_LEVEL >= 3 {
            let mut path_display = String::new();
            for step_and_cost in partial_path.iter_rev() {
                write!(path_display, "{:?} <- ", step_and_cost).unwrap();
            }
            path_display.push_str("START");
            println!("    - Currently on partial path {}", path_display);
        }

        // Ignore that path if we found another solution which is so good that
        // it's not worth exploring anymore.
        let best_current_cost = best_cumulative_cost[partial_path.len() - 1];
        if partial_path.cost_so_far() > (best_current_cost + tolerance).min(total_cost_target) {
            if BRUTE_FORCE_DEBUG_LEVEL >= 4 {
                println!(
                    "      * That exceeds cache cost tolerance with only {}/{} steps, ignore it.",
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
        for next_x in 0..num_feeds {
            for next_y in next_x..num_feeds {
                let next_step = [next_x, next_y];

                // Log which neighbor we're looking at in verbose mode
                if BRUTE_FORCE_DEBUG_LEVEL >= 4 {
                    println!("      * Trying {:?}...", next_step);
                }

                // Display progress
                if BRUTE_FORCE_DEBUG_LEVEL >= 1 && path_counter % 300_000_000 == 0 {
                    println!("  * Processed {}M path steps", path_counter / 1_000_000,);

                    let paths_by_len = priorized_partial_paths.paths_by_len();
                    let display_path_counts = |header, paths_by_len: &Vec<usize>| {
                        print!("    - {:<25}: ", header);
                        for partial_length in 1..path_length {
                            print!("{:>4} ", paths_by_len.get(partial_length - 1).unwrap_or(&0));
                        }
                        println!();
                    };
                    display_path_counts("Partial paths by length", &paths_by_len);
                    display_path_counts(
                        "Priorized paths by length",
                        &priorized_partial_paths.high_priority_paths_by_len(),
                    );

                    let mut max_next_steps = 1.0f64;
                    let mut max_total_steps = 0.0f64;
                    for partial_length in (1..path_length).rev() {
                        max_next_steps *= (path_length - partial_length) as f64;
                        max_total_steps +=
                            paths_by_len.get(partial_length - 1).copied().unwrap_or(0) as f64
                                * max_next_steps;
                    }
                    println!(
                        "    - Exhaustive search space  : 10^{} paths",
                        max_total_steps.log10()
                    );
                }
                path_counter += 1;

                // Have we been there before ?
                if partial_path.contains(&next_step) {
                    if BRUTE_FORCE_DEBUG_LEVEL >= 4 {
                        println!("      * That's going circles, forget it.");
                    }
                    continue;
                }

                // Does it seem worthwhile to try to go there?
                let next_step_eval = partial_path.evaluate_next_step(&cache_model, &next_step);
                let next_cost = next_step_eval.next_cost;
                let best_next_cost = best_cumulative_cost[partial_path.len()];
                if next_cost > (best_next_cost + tolerance).min(total_cost_target) {
                    if BRUTE_FORCE_DEBUG_LEVEL >= 4 {
                        println!(
                        "      * That exceeds cache cost tolerance with only {}/{} steps, ignore it.",
                        partial_path.len() + 1,
                        path_length
                    );
                    }
                    continue;
                }

                // Are we finished ?
                if partial_path.len() + 1 == path_length {
                    // Is this path better than what was observed before?
                    if next_cost <= total_cost_target {
                        // If so, materialize the path into a vector
                        let mut final_path =
                            vec![FeedPair::default(); path_length].into_boxed_slice();
                        final_path[path_length - 1] = next_step_eval.next_step;
                        best_cumulative_cost[path_length - 1] = next_step_eval.next_cost;
                        for (i, (step, cost)) in
                            (0..partial_path.len()).rev().zip(partial_path.iter_rev())
                        {
                            final_path[i] = step;
                            best_cumulative_cost[i] = cost;
                        }

                        // Announce victory
                        if BRUTE_FORCE_DEBUG_LEVEL == 1 {
                            println!(
                                "  * Reached new total cache cost record {} with path {:?}",
                                next_cost, final_path
                            );
                        }
                        if BRUTE_FORCE_DEBUG_LEVEL >= 2 {
                            let path_cost = final_path
                                .iter()
                                .zip(best_cumulative_cost.iter())
                                .collect::<Box<[_]>>();
                            println!(
                            "  * Reached new total cache cost record {} with path and cumulative cost {:?}",
                            next_cost, path_cost
                        );
                        }

                        // Now record that path and look for a better one
                        best_path = Some(final_path);
                        total_cost_target = next_cost - 1.0;
                    }
                    continue;
                }

                // Otherwise, schedule searching further down this path
                if BRUTE_FORCE_DEBUG_LEVEL >= 4 {
                    println!("      * That seems reasonable, we'll explore that path further...");
                }
                priorized_partial_paths.push(partial_path.commit_next_step(next_step_eval));
            }
        }
        if BRUTE_FORCE_DEBUG_LEVEL >= 3 {
            println!("    - Done exploring possibilities from current path");
        }
    }

    // Return the optimal path, if any, along with its cache cost
    best_path
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
    extra_distance: StepDistance,
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
    curr_cost: cache::Cost,
    prev_steps: Option<Rc<PathElems>>,
}
//
/// Total distance that was "walked" across a path step
type StepDistance = f32;
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
        let mut cache_sim = cache_model.start_simulation();
        for &feed in start.iter() {
            let access_cost = cache_sim.simulate_access(&cache_model, feed);
            debug_assert_eq!(access_cost, 0.0);
        }

        let path = Rc::new(PathElems {
            curr_step: start,
            curr_cost: 0.0,
            prev_steps: None,
        });

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
            extra_distance: 0.0,
        }
    }

    /// Tell how long the path is
    //
    // NOTE: This operation is hot and must be fast
    //
    pub fn len(&self) -> usize {
        self.path_len
    }

    /// Tell how much excess distance was covered through path stepping
    pub fn extra_distance(&self) -> StepDistance {
        self.extra_distance
    }

    /// Get the last path entry
    //
    // NOTE: This operation is hot and must be fast
    //
    pub fn last_step(&self) -> &FeedPair {
        &self.path.curr_step
    }

    /// Iterate over the path steps and cumulative costs in reverse step order
    //
    // NOTE: This operation can be slow, it is only called when a better path
    //       has been found (very rare) or when displaying debug output.
    //
    pub fn iter_rev(&self) -> impl Iterator<Item = (FeedPair, cache::Cost)> + '_ {
        let mut next_node = Some(&self.path);
        std::iter::from_fn(move || {
            let node = next_node?;
            next_node = node.prev_steps.as_ref();
            Some((node.curr_step, node.curr_cost))
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
        self.path.curr_cost
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
        let next_cost = self.cost_so_far()
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
    /// plus an extra step for which the new cache cost and updated cache model
    /// are provided.
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
            curr_cost: next_cost,
            prev_steps: Some(self.path.clone()),
        });

        let mut next_visited_pairs = self.visited_pairs.clone();
        let (word, bit) = Self::coord_to_bit_index(&next_step);
        // TODO: Make sure the bound check is elided or has negligible cost, if
        //       it is too expensive use get_unchecked.
        next_visited_pairs[word] |= 1 << bit;

        let step_length = self
            .last_step()
            .iter()
            .zip(next_step.iter())
            .map(|(&curr_coord, &next_coord)| {
                ((next_coord as StepDistance) - (curr_coord as StepDistance)).powi(2)
            })
            .sum::<StepDistance>()
            .sqrt();

        Self {
            path: next_path,
            path_len: self.path_len + 1,
            visited_pairs: next_visited_pairs,
            cache_sim: next_cache,
            extra_distance: self.extra_distance() + (step_length - 1.0),
        }
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
    storage: Vec<(Priority, Vec<PartialPath>)>,

    /// Mechanism to reuse inner Vec allocations
    storage_morgue: Vec<Vec<PartialPath>>,
}
//
type Priority = f32;
//
impl PriorizedPartialPaths {
    /// Create the collection
    pub fn new() -> Self {
        Self::default()
    }

    /// Prioritize a certain path wrt others, higher is more important
    pub fn priorize(path: &PartialPath) -> Priority {
        // * The main goal is to avoid cache misses
        // * In doing so, however, we must be careful to finish paths as 1/that
        //   frees up memory and 2/that updates our best path model, which in
        //   turns allows us to prune bad paths.
        // * Let's keep the path as close to (0, 1) steps as possible, given the
        //   aforementioned constraints.
        0.99 * path.len() as f32 - path.cost_so_far() - 0.2 * path.extra_distance()
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

        // Pick a random high-priority path
        let path_idx = rng.gen_range(0..highest_priority_paths.len());
        let path = highest_priority_paths.remove(path_idx);

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

    /// Count the number of paths of each length within a vector of path
    fn count_by_len(paths: &Vec<PartialPath>) -> Vec<usize> {
        let mut histogram = Vec::new();
        for path in paths {
            if histogram.len() < path.len() {
                histogram.resize(path.len(), 0);
            }
            histogram[path.len() - 1] += 1;
        }
        histogram
    }

    /// Merge a result of count_by_len() into another
    fn merge_counts(src1: Vec<usize>, src2: Vec<usize>) -> Vec<usize> {
        let (mut target, mut src) = (src1, src2);
        if target.len() < src.len() {
            target.extend_from_slice(&src[target.len()..]);
        }
        src.truncate(target.len());
        for (idx, src_count) in src.into_iter().enumerate() {
            target[idx] += src_count;
        }
        target
    }

    /// Count the total number of paths that are currently stored
    pub fn paths_by_len(&self) -> Vec<usize> {
        self.storage
            .iter()
            .map(|(_priority, path_vec)| Self::count_by_len(path_vec))
            .fold(Vec::new(), |src1, src2| Self::merge_counts(src1, src2))
    }

    /// Count the number of highest-priority paths that are currently stored
    pub fn high_priority_paths_by_len(&self) -> Vec<usize> {
        self.storage
            .last()
            .map(|(_priority, path_vec)| Self::count_by_len(path_vec))
            .unwrap_or(Vec::new())
    }
}
