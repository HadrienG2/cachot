//! Brute force search progress monitoring and reporting

use super::{priorization::PriorizedPartialPaths, BRUTE_FORCE_DEBUG_LEVEL};
use std::time::{Duration, Instant};

/// Mechanism to track and report brute force search progress
pub struct ProgressMonitor {
    /// Number of path steps that were explored since the start of the search
    total_path_steps: u64,

    /// Number of path steps that were already explored as of last report
    last_path_steps: u64,

    /// Time at which the last report occurred
    last_report_time: Instant,

    /// Number of paths of each length at the time of the last report.
    last_paths_by_len: Option<Box<[usize]>>,

    /// Number of paths of each length at time of the first report in a series.
    /// Reset whenever significant progress ("watchdog event") occurs.
    first_paths_by_len: Option<Box<[usize]>>,

    /// Time at which the last sign of significant progress occurred
    last_watchdog_reset: Instant,

    /// Cache of last_watchdog_reset.elapsed() that is updated at reporting time
    last_watchdog_timer: Duration,
}
//
impl ProgressMonitor {
    /// Set up a way to monitor brute force search progress
    pub fn new() -> Self {
        let initial_time = Instant::now();
        Self {
            total_path_steps: 0,
            last_path_steps: 0,
            last_report_time: initial_time,
            last_paths_by_len: None,
            first_paths_by_len: None,
            last_watchdog_reset: initial_time,
            last_watchdog_timer: Duration::new(0, 0),
        }
    }

    /// Record that a path step has been taken, print periodical reports
    pub fn record_step(&mut self, paths: &PriorizedPartialPaths<'_>) {
        // Only report on progress infrequently so that search isn't slowed down
        self.total_path_steps += 1;
        const CLOCK_CHECK_RATE: u64 = 1 << 17;
        const REPORT_RATE: Duration = Duration::from_secs(5);
        if self.total_path_steps % CLOCK_CHECK_RATE == 0
            && self.last_report_time.elapsed() > REPORT_RATE
        {
            // Display number of processed path steps
            let elapsed_time = self.last_report_time.elapsed();
            let new_path_steps = self.total_path_steps - self.last_path_steps;
            if BRUTE_FORCE_DEBUG_LEVEL >= 2 {
                let new_million_steps = new_path_steps as f32 / 1_000_000.0;
                println!(
                    "  * Processed {}M new path steps ({:.1}M steps/s)",
                    new_million_steps.round(),
                    (new_million_steps as f32 / elapsed_time.as_secs_f32())
                );
            }

            // Display number of remaining path steps
            let num_paths_by_len = paths.num_paths_by_len();
            if BRUTE_FORCE_DEBUG_LEVEL >= 2 {
                // ...as an absolute number...
                print!("    - Partial paths by length: ");
                for path_count in num_paths_by_len.iter() {
                    print!("{:>5} ", *path_count);
                }
                println!();

                // ...compared to the last progress report...
                if let Some(last_paths_by_len) = &self.last_paths_by_len {
                    print!("    - Diff. since last report: ");
                    for (curr_path_count, last_path_count) in
                        num_paths_by_len.iter().zip(last_paths_by_len.iter())
                    {
                        print!(
                            "{:>+5} ",
                            *curr_path_count as isize - *last_path_count as isize
                        );
                    }
                    println!();
                }

                // ...compared to the last major search event
                if let Some(first_paths_by_len) = &self.first_paths_by_len {
                    print!("    - Diff. since last event : ");
                    for (curr_path_count, first_path_count) in
                        num_paths_by_len.iter().zip(first_paths_by_len.iter())
                    {
                        print!(
                            "{:>+5} ",
                            *curr_path_count as isize - *first_path_count as isize
                        );
                    }
                    println!();
                } else {
                    self.first_paths_by_len = Some(num_paths_by_len.clone());
                }
            }

            // Update internal state
            self.last_path_steps = self.total_path_steps;
            self.last_report_time = Instant::now();
            self.last_paths_by_len = Some(num_paths_by_len);
            self.last_watchdog_timer = self.last_watchdog_reset.elapsed();
        }
    }

    /// Check out how much time has elapsed since the last major event
    pub fn watchdog_timer(&self) -> Duration {
        self.last_watchdog_timer
    }

    /// Reset the watchdog timer to signal that an important event has occured,
    /// which suggests that it might be worthwhile to continue the search.
    pub fn reset_watchdog(&mut self) {
        self.first_paths_by_len = None;
        self.last_watchdog_reset = Instant::now();
    }
}
