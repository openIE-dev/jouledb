//! Profiler report generation — function-level timing, call counts,
//! inclusive/exclusive time, percentage of total, sorted by various criteria,
//! flamegraph-compatible output, top-N report, and comparison between runs.

use std::collections::HashMap;

// ── Function Profile ─────────────────────────────────────────────

/// Profile data for a single function.
#[derive(Debug, Clone)]
pub struct FunctionProfile {
    pub name: String,
    pub call_count: u64,
    /// Inclusive time: total time including callees (microseconds).
    pub inclusive_time_us: u64,
    /// Exclusive time: self-time excluding callees (microseconds).
    pub exclusive_time_us: u64,
    /// Minimum single-call duration.
    pub min_time_us: u64,
    /// Maximum single-call duration.
    pub max_time_us: u64,
}

impl FunctionProfile {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            call_count: 0,
            inclusive_time_us: 0,
            exclusive_time_us: 0,
            min_time_us: u64::MAX,
            max_time_us: 0,
        }
    }

    /// Record a single call with inclusive and exclusive times.
    pub fn record(&mut self, inclusive_us: u64, exclusive_us: u64) {
        self.call_count += 1;
        self.inclusive_time_us += inclusive_us;
        self.exclusive_time_us += exclusive_us;
        if inclusive_us < self.min_time_us {
            self.min_time_us = inclusive_us;
        }
        if inclusive_us > self.max_time_us {
            self.max_time_us = inclusive_us;
        }
    }

    /// Average inclusive time per call.
    pub fn avg_inclusive_us(&self) -> f64 {
        if self.call_count == 0 {
            0.0
        } else {
            self.inclusive_time_us as f64 / self.call_count as f64
        }
    }

    /// Average exclusive time per call.
    pub fn avg_exclusive_us(&self) -> f64 {
        if self.call_count == 0 {
            0.0
        } else {
            self.exclusive_time_us as f64 / self.call_count as f64
        }
    }

    /// Inclusive time as a percentage of the total program time.
    pub fn inclusive_pct(&self, total_us: u64) -> f64 {
        if total_us == 0 {
            0.0
        } else {
            (self.inclusive_time_us as f64 / total_us as f64) * 100.0
        }
    }

    /// Exclusive time as a percentage of the total program time.
    pub fn exclusive_pct(&self, total_us: u64) -> f64 {
        if total_us == 0 {
            0.0
        } else {
            (self.exclusive_time_us as f64 / total_us as f64) * 100.0
        }
    }
}

// ── Sort Criteria ────────────────────────────────────────────────

/// Criteria for sorting profiler reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortBy {
    InclusiveTime,
    ExclusiveTime,
    CallCount,
    AvgInclusiveTime,
    AvgExclusiveTime,
    Name,
}

// ── Profiling Sample ─────────────────────────────────────────────

/// A single profiling sample: one invocation of a function.
#[derive(Debug, Clone)]
pub struct ProfilingSample {
    pub function_name: String,
    pub inclusive_us: u64,
    pub exclusive_us: u64,
    /// Stack trace leading to this call (for flamegraph output).
    pub stack: Vec<String>,
}

impl ProfilingSample {
    pub fn new(name: &str, inclusive_us: u64, exclusive_us: u64) -> Self {
        Self {
            function_name: name.to_string(),
            inclusive_us,
            exclusive_us,
            stack: Vec::new(),
        }
    }

    pub fn with_stack(mut self, stack: Vec<String>) -> Self {
        self.stack = stack;
        self
    }
}

// ── Comparison Result ────────────────────────────────────────────

/// Comparison of a function between two profiler runs.
#[derive(Debug, Clone)]
pub struct FunctionComparison {
    pub name: String,
    pub before_inclusive_us: u64,
    pub after_inclusive_us: u64,
    pub before_exclusive_us: u64,
    pub after_exclusive_us: u64,
    pub before_call_count: u64,
    pub after_call_count: u64,
    pub inclusive_change_pct: f64,
    pub exclusive_change_pct: f64,
}

impl FunctionComparison {
    /// True if the function got slower (inclusive time increased).
    pub fn is_regression(&self) -> bool {
        self.inclusive_change_pct > 0.0
    }

    /// True if the function got faster (inclusive time decreased).
    pub fn is_improvement(&self) -> bool {
        self.inclusive_change_pct < 0.0
    }

    /// True if the function is new (not in the "before" run).
    pub fn is_new(&self) -> bool {
        self.before_call_count == 0 && self.after_call_count > 0
    }

    /// True if the function was removed (not in the "after" run).
    pub fn is_removed(&self) -> bool {
        self.before_call_count > 0 && self.after_call_count == 0
    }
}

// ── Report Line ──────────────────────────────────────────────────

/// A single line in a text report.
#[derive(Debug, Clone)]
pub struct ReportLine {
    pub name: String,
    pub call_count: u64,
    pub inclusive_us: u64,
    pub exclusive_us: u64,
    pub inclusive_pct: f64,
    pub exclusive_pct: f64,
    pub avg_inclusive_us: f64,
}

// ── Profiler Report ──────────────────────────────────────────────

/// A profiler report built from profiling samples.
pub struct ProfilerReport {
    profiles: HashMap<String, FunctionProfile>,
    total_time_us: u64,
    sample_count: u64,
}

impl ProfilerReport {
    pub fn new() -> Self {
        Self {
            profiles: HashMap::new(),
            total_time_us: 0,
            sample_count: 0,
        }
    }

    /// Add a profiling sample.
    pub fn add_sample(&mut self, sample: &ProfilingSample) {
        let profile = self
            .profiles
            .entry(sample.function_name.clone())
            .or_insert_with(|| FunctionProfile::new(&sample.function_name));
        profile.record(sample.inclusive_us, sample.exclusive_us);
        self.sample_count += 1;
    }

    /// Set the total program time (the inclusive time of the root function).
    pub fn set_total_time(&mut self, total_us: u64) {
        self.total_time_us = total_us;
    }

    /// Compute total time from the sum of all exclusive times.
    pub fn compute_total_from_exclusive(&mut self) {
        self.total_time_us = self.profiles.values().map(|p| p.exclusive_time_us).sum();
    }

    /// Get the total program time.
    pub fn total_time_us(&self) -> u64 {
        self.total_time_us
    }

    /// Get a function profile by name.
    pub fn get_profile(&self, name: &str) -> Option<&FunctionProfile> {
        self.profiles.get(name)
    }

    /// Number of distinct functions profiled.
    pub fn function_count(&self) -> usize {
        self.profiles.len()
    }

    /// Total number of samples recorded.
    pub fn sample_count(&self) -> u64 {
        self.sample_count
    }

    /// Get all function profiles sorted by the given criteria.
    pub fn sorted_by(&self, criteria: SortBy) -> Vec<&FunctionProfile> {
        let mut profiles: Vec<&FunctionProfile> = self.profiles.values().collect();
        match criteria {
            SortBy::InclusiveTime => {
                profiles.sort_by(|a, b| {
                    b.inclusive_time_us
                        .cmp(&a.inclusive_time_us)
                        .then_with(|| a.name.cmp(&b.name))
                });
            }
            SortBy::ExclusiveTime => {
                profiles.sort_by(|a, b| {
                    b.exclusive_time_us
                        .cmp(&a.exclusive_time_us)
                        .then_with(|| a.name.cmp(&b.name))
                });
            }
            SortBy::CallCount => {
                profiles.sort_by(|a, b| {
                    b.call_count
                        .cmp(&a.call_count)
                        .then_with(|| a.name.cmp(&b.name))
                });
            }
            SortBy::AvgInclusiveTime => {
                profiles.sort_by(|a, b| {
                    b.avg_inclusive_us()
                        .partial_cmp(&a.avg_inclusive_us())
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.name.cmp(&b.name))
                });
            }
            SortBy::AvgExclusiveTime => {
                profiles.sort_by(|a, b| {
                    b.avg_exclusive_us()
                        .partial_cmp(&a.avg_exclusive_us())
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.name.cmp(&b.name))
                });
            }
            SortBy::Name => {
                profiles.sort_by(|a, b| a.name.cmp(&b.name));
            }
        }
        profiles
    }

    /// Get the top N functions by exclusive time.
    pub fn top_n(&self, n: usize) -> Vec<&FunctionProfile> {
        let mut sorted = self.sorted_by(SortBy::ExclusiveTime);
        sorted.truncate(n);
        sorted
    }

    /// Generate a text report sorted by the given criteria.
    pub fn text_report(&self, criteria: SortBy) -> Vec<ReportLine> {
        self.sorted_by(criteria)
            .iter()
            .map(|p| ReportLine {
                name: p.name.clone(),
                call_count: p.call_count,
                inclusive_us: p.inclusive_time_us,
                exclusive_us: p.exclusive_time_us,
                inclusive_pct: p.inclusive_pct(self.total_time_us),
                exclusive_pct: p.exclusive_pct(self.total_time_us),
                avg_inclusive_us: p.avg_inclusive_us(),
            })
            .collect()
    }

    /// Format the report as a table string.
    pub fn format_table(&self, criteria: SortBy) -> String {
        let lines = self.text_report(criteria);
        let mut out = String::new();
        out.push_str(&format!(
            "{:<30} {:>8} {:>12} {:>12} {:>8} {:>8}\n",
            "Function", "Calls", "Incl (us)", "Excl (us)", "Incl%", "Excl%"
        ));
        out.push_str(&"-".repeat(82));
        out.push('\n');

        for line in &lines {
            out.push_str(&format!(
                "{:<30} {:>8} {:>12} {:>12} {:>7.1}% {:>7.1}%\n",
                line.name,
                line.call_count,
                line.inclusive_us,
                line.exclusive_us,
                line.inclusive_pct,
                line.exclusive_pct,
            ));
        }
        out
    }

    /// Generate flamegraph-compatible folded stack output from samples.
    pub fn to_folded_stacks(samples: &[ProfilingSample]) -> String {
        let mut stacks: HashMap<String, u64> = HashMap::new();

        for sample in samples {
            let key = if sample.stack.is_empty() {
                sample.function_name.clone()
            } else {
                let mut path = sample.stack.clone();
                path.push(sample.function_name.clone());
                path.join(";")
            };
            *stacks.entry(key).or_insert(0) += sample.exclusive_us;
        }

        let mut lines: Vec<(String, u64)> = stacks.into_iter().collect();
        lines.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        lines
            .iter()
            .map(|(path, count)| format!("{} {}", path, count))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Compare two profiler reports. Returns a list of function comparisons.
    pub fn compare(before: &ProfilerReport, after: &ProfilerReport) -> Vec<FunctionComparison> {
        let mut all_names: Vec<String> = before
            .profiles
            .keys()
            .chain(after.profiles.keys())
            .cloned()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        all_names.sort();

        all_names
            .into_iter()
            .map(|name| {
                let before_p = before.profiles.get(&name);
                let after_p = after.profiles.get(&name);

                let bi = before_p.map_or(0, |p| p.inclusive_time_us);
                let ai = after_p.map_or(0, |p| p.inclusive_time_us);
                let be = before_p.map_or(0, |p| p.exclusive_time_us);
                let ae = after_p.map_or(0, |p| p.exclusive_time_us);
                let bc = before_p.map_or(0, |p| p.call_count);
                let ac = after_p.map_or(0, |p| p.call_count);

                let inclusive_change_pct = if bi == 0 {
                    if ai == 0 {
                        0.0
                    } else {
                        100.0
                    }
                } else {
                    ((ai as f64 - bi as f64) / bi as f64) * 100.0
                };

                let exclusive_change_pct = if be == 0 {
                    if ae == 0 {
                        0.0
                    } else {
                        100.0
                    }
                } else {
                    ((ae as f64 - be as f64) / be as f64) * 100.0
                };

                FunctionComparison {
                    name,
                    before_inclusive_us: bi,
                    after_inclusive_us: ai,
                    before_exclusive_us: be,
                    after_exclusive_us: ae,
                    before_call_count: bc,
                    after_call_count: ac,
                    inclusive_change_pct,
                    exclusive_change_pct,
                }
            })
            .collect()
    }
}

impl Default for ProfilerReport {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn build_report() -> ProfilerReport {
        let mut r = ProfilerReport::new();
        r.add_sample(&ProfilingSample::new("main", 1000, 100));
        r.add_sample(&ProfilingSample::new("process", 800, 200));
        r.add_sample(&ProfilingSample::new("process", 900, 300));
        r.add_sample(&ProfilingSample::new("compute", 500, 500));
        r.add_sample(&ProfilingSample::new("io_read", 300, 300));
        r.set_total_time(1000);
        r
    }

    #[test]
    fn test_add_sample() {
        let r = build_report();
        assert_eq!(r.function_count(), 4);
        assert_eq!(r.sample_count(), 5);
    }

    #[test]
    fn test_function_profile() {
        let r = build_report();
        let p = r.get_profile("process").unwrap();
        assert_eq!(p.call_count, 2);
        assert_eq!(p.inclusive_time_us, 1700);
        assert_eq!(p.exclusive_time_us, 500);
    }

    #[test]
    fn test_min_max_time() {
        let r = build_report();
        let p = r.get_profile("process").unwrap();
        assert_eq!(p.min_time_us, 800);
        assert_eq!(p.max_time_us, 900);
    }

    #[test]
    fn test_avg_times() {
        let r = build_report();
        let p = r.get_profile("process").unwrap();
        assert!((p.avg_inclusive_us() - 850.0).abs() < 0.01);
        assert!((p.avg_exclusive_us() - 250.0).abs() < 0.01);
    }

    #[test]
    fn test_percentages() {
        let r = build_report();
        let p = r.get_profile("compute").unwrap();
        assert!((p.inclusive_pct(1000) - 50.0).abs() < 0.01);
        assert!((p.exclusive_pct(1000) - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_sort_by_inclusive() {
        let r = build_report();
        let sorted = r.sorted_by(SortBy::InclusiveTime);
        assert_eq!(sorted[0].name, "process"); // 1700
    }

    #[test]
    fn test_sort_by_exclusive() {
        let r = build_report();
        let sorted = r.sorted_by(SortBy::ExclusiveTime);
        // process: 500, compute: 500, io_read: 300, main: 100
        // Tie broken by name
        assert_eq!(sorted[0].name, "compute");
        assert_eq!(sorted[1].name, "process");
    }

    #[test]
    fn test_sort_by_calls() {
        let r = build_report();
        let sorted = r.sorted_by(SortBy::CallCount);
        assert_eq!(sorted[0].name, "process"); // 2 calls
    }

    #[test]
    fn test_sort_by_name() {
        let r = build_report();
        let sorted = r.sorted_by(SortBy::Name);
        assert_eq!(sorted[0].name, "compute");
        assert_eq!(sorted[1].name, "io_read");
    }

    #[test]
    fn test_top_n() {
        let r = build_report();
        let top = r.top_n(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].name, "compute");
    }

    #[test]
    fn test_text_report() {
        let r = build_report();
        let lines = r.text_report(SortBy::ExclusiveTime);
        assert_eq!(lines.len(), 4);
        assert!((lines[0].exclusive_pct - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_format_table() {
        let r = build_report();
        let table = r.format_table(SortBy::ExclusiveTime);
        assert!(table.contains("Function"));
        assert!(table.contains("compute"));
        assert!(table.contains("process"));
    }

    #[test]
    fn test_folded_stacks() {
        let samples = vec![
            ProfilingSample::new("compute", 500, 500)
                .with_stack(vec!["main".to_string(), "process".to_string()]),
            ProfilingSample::new("io_read", 300, 300)
                .with_stack(vec!["main".to_string()]),
        ];
        let folded = ProfilerReport::to_folded_stacks(&samples);
        assert!(folded.contains("main;process;compute 500"));
        assert!(folded.contains("main;io_read 300"));
    }

    #[test]
    fn test_folded_stacks_no_stack() {
        let samples = vec![ProfilingSample::new("standalone", 100, 100)];
        let folded = ProfilerReport::to_folded_stacks(&samples);
        assert_eq!(folded, "standalone 100");
    }

    #[test]
    fn test_compare_reports() {
        let mut before = ProfilerReport::new();
        before.add_sample(&ProfilingSample::new("fast_fn", 100, 100));
        before.add_sample(&ProfilingSample::new("slow_fn", 500, 500));

        let mut after = ProfilerReport::new();
        after.add_sample(&ProfilingSample::new("fast_fn", 50, 50));
        after.add_sample(&ProfilingSample::new("slow_fn", 600, 600));
        after.add_sample(&ProfilingSample::new("new_fn", 200, 200));

        let comparisons = ProfilerReport::compare(&before, &after);
        assert_eq!(comparisons.len(), 3);

        let fast = comparisons.iter().find(|c| c.name == "fast_fn").unwrap();
        assert!(fast.is_improvement());

        let slow = comparisons.iter().find(|c| c.name == "slow_fn").unwrap();
        assert!(slow.is_regression());

        let new = comparisons.iter().find(|c| c.name == "new_fn").unwrap();
        assert!(new.is_new());
    }

    #[test]
    fn test_compare_removed_function() {
        let mut before = ProfilerReport::new();
        before.add_sample(&ProfilingSample::new("removed_fn", 100, 100));

        let after = ProfilerReport::new();

        let comparisons = ProfilerReport::compare(&before, &after);
        let removed = comparisons.iter().find(|c| c.name == "removed_fn").unwrap();
        assert!(removed.is_removed());
    }

    #[test]
    fn test_compute_total_from_exclusive() {
        let mut r = ProfilerReport::new();
        r.add_sample(&ProfilingSample::new("a", 100, 40));
        r.add_sample(&ProfilingSample::new("b", 80, 60));
        r.compute_total_from_exclusive();
        assert_eq!(r.total_time_us(), 100); // 40 + 60
    }

    #[test]
    fn test_empty_report() {
        let r = ProfilerReport::new();
        assert_eq!(r.function_count(), 0);
        assert_eq!(r.sample_count(), 0);
        assert!(r.top_n(10).is_empty());
    }

    #[test]
    fn test_zero_total_percentages() {
        let p = FunctionProfile::new("f");
        assert!((p.inclusive_pct(0) - 0.0).abs() < 0.001);
        assert!((p.exclusive_pct(0) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_zero_calls_avg() {
        let p = FunctionProfile::new("f");
        assert!((p.avg_inclusive_us() - 0.0).abs() < 0.001);
        assert!((p.avg_exclusive_us() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_sort_by_avg_inclusive() {
        let r = build_report();
        let sorted = r.sorted_by(SortBy::AvgInclusiveTime);
        assert_eq!(sorted[0].name, "main"); // 1000/1 = 1000
    }

    #[test]
    fn test_sort_by_avg_exclusive() {
        let r = build_report();
        let sorted = r.sorted_by(SortBy::AvgExclusiveTime);
        assert_eq!(sorted[0].name, "compute"); // 500/1 = 500
    }

    #[test]
    fn test_folded_merges_same_stack() {
        let samples = vec![
            ProfilingSample::new("leaf", 100, 100)
                .with_stack(vec!["root".to_string()]),
            ProfilingSample::new("leaf", 200, 200)
                .with_stack(vec!["root".to_string()]),
        ];
        let folded = ProfilerReport::to_folded_stacks(&samples);
        assert!(folded.contains("root;leaf 300"));
    }
}
