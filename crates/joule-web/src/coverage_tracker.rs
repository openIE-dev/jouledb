//! Code coverage tracking.
//!
//! Replaces `istanbul`, `c8`, `kcov`, and similar coverage tools with a
//! pure-Rust in-process coverage model. Tracks line/branch/function coverage,
//! generates reports (text and JSON), calculates percentages, identifies
//! uncovered lines, and merges coverage from multiple runs.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Errors ───────────────────────────────────────────────────────

/// Coverage errors.
#[derive(Debug, Clone, PartialEq)]
pub enum CoverageError {
    /// File not found in the report.
    FileNotFound(String),
    /// Coverage below threshold.
    BelowThreshold { actual: f64, required: f64 },
    /// Invalid line number.
    InvalidLine { file: String, line: u32 },
}

impl fmt::Display for CoverageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FileNotFound(path) => write!(f, "file not found: {path}"),
            Self::BelowThreshold { actual, required } => {
                write!(f, "coverage {actual:.1}% below threshold {required:.1}%")
            }
            Self::InvalidLine { file, line } => {
                write!(f, "invalid line {line} in {file}")
            }
        }
    }
}

impl std::error::Error for CoverageError {}

// ── Line Coverage ────────────────────────────────────────────────

/// Coverage data for a single line.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineCoverage {
    /// Line number (1-based).
    pub line: u32,
    /// Number of times this line was executed.
    pub hit_count: u64,
}

impl LineCoverage {
    pub fn new(line: u32) -> Self {
        Self { line, hit_count: 0 }
    }

    pub fn with_hits(line: u32, hit_count: u64) -> Self {
        Self { line, hit_count }
    }

    pub fn is_covered(&self) -> bool {
        self.hit_count > 0
    }

    pub fn record_hit(&mut self) {
        self.hit_count += 1;
    }

    pub fn record_hits(&mut self, count: u64) {
        self.hit_count += count;
    }
}

// ── Branch Coverage ──────────────────────────────────────────────

/// Coverage data for a branch point (e.g., if/else, match arm).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchCoverage {
    /// Line where the branch occurs.
    pub line: u32,
    /// Branch index (0 = first branch, 1 = second, etc.).
    pub branch_index: u32,
    /// Number of times this branch was taken.
    pub hit_count: u64,
    /// Label for the branch (e.g., "if-true", "else", "arm-0").
    pub label: String,
}

impl BranchCoverage {
    pub fn new(line: u32, branch_index: u32, label: &str) -> Self {
        Self {
            line,
            branch_index,
            hit_count: 0,
            label: label.to_string(),
        }
    }

    pub fn is_covered(&self) -> bool {
        self.hit_count > 0
    }

    pub fn record_hit(&mut self) {
        self.hit_count += 1;
    }
}

// ── Function Coverage ────────────────────────────────────────────

/// Coverage data for a function/method.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCoverage {
    /// Function name.
    pub name: String,
    /// Starting line number.
    pub start_line: u32,
    /// Ending line number.
    pub end_line: u32,
    /// Number of times the function was called.
    pub hit_count: u64,
}

impl FunctionCoverage {
    pub fn new(name: &str, start_line: u32, end_line: u32) -> Self {
        Self {
            name: name.to_string(),
            start_line,
            end_line,
            hit_count: 0,
        }
    }

    pub fn is_covered(&self) -> bool {
        self.hit_count > 0
    }

    pub fn record_hit(&mut self) {
        self.hit_count += 1;
    }

    /// Number of lines in this function.
    pub fn line_count(&self) -> u32 {
        if self.end_line >= self.start_line {
            self.end_line - self.start_line + 1
        } else {
            0
        }
    }
}

// ── File Coverage ────────────────────────────────────────────────

/// Aggregated coverage data for a single file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileCoverage {
    /// File path.
    pub path: String,
    /// Total number of lines in the file.
    pub total_lines: u32,
    /// Per-line coverage data.
    pub lines: Vec<LineCoverage>,
    /// Branch coverage data.
    pub branches: Vec<BranchCoverage>,
    /// Function coverage data.
    pub functions: Vec<FunctionCoverage>,
}

impl FileCoverage {
    pub fn new(path: &str, total_lines: u32) -> Self {
        Self {
            path: path.to_string(),
            total_lines,
            lines: Vec::new(),
            branches: Vec::new(),
            functions: Vec::new(),
        }
    }

    /// Add a line to track.
    pub fn add_line(&mut self, line: u32) {
        if !self.lines.iter().any(|l| l.line == line) {
            self.lines.push(LineCoverage::new(line));
        }
    }

    /// Record a hit on a specific line.
    pub fn record_line_hit(&mut self, line: u32) {
        if let Some(lc) = self.lines.iter_mut().find(|l| l.line == line) {
            lc.record_hit();
        } else {
            self.lines.push(LineCoverage::with_hits(line, 1));
        }
    }

    /// Add a branch to track.
    pub fn add_branch(&mut self, line: u32, branch_index: u32, label: &str) {
        self.branches.push(BranchCoverage::new(line, branch_index, label));
    }

    /// Record a hit on a branch.
    pub fn record_branch_hit(&mut self, line: u32, branch_index: u32) {
        if let Some(bc) = self
            .branches
            .iter_mut()
            .find(|b| b.line == line && b.branch_index == branch_index)
        {
            bc.record_hit();
        }
    }

    /// Add a function to track.
    pub fn add_function(&mut self, name: &str, start_line: u32, end_line: u32) {
        self.functions.push(FunctionCoverage::new(name, start_line, end_line));
    }

    /// Record a hit on a function.
    pub fn record_function_hit(&mut self, name: &str) {
        if let Some(fc) = self.functions.iter_mut().find(|f| f.name == name) {
            fc.record_hit();
        }
    }

    // ── Calculations ─────────────────────────────────────────────

    /// Number of tracked (executable) lines.
    pub fn executable_lines(&self) -> usize {
        self.lines.len()
    }

    /// Number of covered lines.
    pub fn covered_lines(&self) -> usize {
        self.lines.iter().filter(|l| l.is_covered()).count()
    }

    /// Line coverage percentage.
    pub fn line_coverage_pct(&self) -> f64 {
        if self.lines.is_empty() {
            return 100.0;
        }
        (self.covered_lines() as f64 / self.lines.len() as f64) * 100.0
    }

    /// Number of tracked branches.
    pub fn total_branches(&self) -> usize {
        self.branches.len()
    }

    /// Number of covered branches.
    pub fn covered_branches(&self) -> usize {
        self.branches.iter().filter(|b| b.is_covered()).count()
    }

    /// Branch coverage percentage.
    pub fn branch_coverage_pct(&self) -> f64 {
        if self.branches.is_empty() {
            return 100.0;
        }
        (self.covered_branches() as f64 / self.branches.len() as f64) * 100.0
    }

    /// Number of tracked functions.
    pub fn total_functions(&self) -> usize {
        self.functions.len()
    }

    /// Number of covered functions.
    pub fn covered_functions(&self) -> usize {
        self.functions.iter().filter(|f| f.is_covered()).count()
    }

    /// Function coverage percentage.
    pub fn function_coverage_pct(&self) -> f64 {
        if self.functions.is_empty() {
            return 100.0;
        }
        (self.covered_functions() as f64 / self.functions.len() as f64) * 100.0
    }

    /// Return uncovered line numbers.
    pub fn uncovered_lines(&self) -> Vec<u32> {
        self.lines
            .iter()
            .filter(|l| !l.is_covered())
            .map(|l| l.line)
            .collect()
    }

    /// Return uncovered function names.
    pub fn uncovered_functions(&self) -> Vec<String> {
        self.functions
            .iter()
            .filter(|f| !f.is_covered())
            .map(|f| f.name.clone())
            .collect()
    }

    /// Return uncovered branch labels.
    pub fn uncovered_branches(&self) -> Vec<String> {
        self.branches
            .iter()
            .filter(|b| !b.is_covered())
            .map(|b| format!("L{}:{} ({})", b.line, b.branch_index, b.label))
            .collect()
    }
}

// ── Coverage Report ──────────────────────────────────────────────

/// Aggregated coverage report across multiple files.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CoverageReport {
    /// Per-file coverage data.
    pub files: HashMap<String, FileCoverage>,
    /// Report name/identifier.
    pub name: String,
}

impl CoverageReport {
    pub fn new(name: &str) -> Self {
        Self {
            files: HashMap::new(),
            name: name.to_string(),
        }
    }

    /// Add file coverage data.
    pub fn add_file(&mut self, file_cov: FileCoverage) {
        self.files.insert(file_cov.path.clone(), file_cov);
    }

    /// Get coverage for a file.
    pub fn get_file(&self, path: &str) -> Result<&FileCoverage, CoverageError> {
        self.files
            .get(path)
            .ok_or_else(|| CoverageError::FileNotFound(path.to_string()))
    }

    /// Get mutable coverage for a file.
    pub fn get_file_mut(&mut self, path: &str) -> Result<&mut FileCoverage, CoverageError> {
        self.files
            .get_mut(path)
            .ok_or_else(|| CoverageError::FileNotFound(path.to_string()))
    }

    /// Number of files tracked.
    pub fn file_count(&self) -> usize {
        self.files.len()
    }

    /// File paths sorted.
    pub fn file_paths(&self) -> Vec<String> {
        let mut paths: Vec<String> = self.files.keys().cloned().collect();
        paths.sort();
        paths
    }

    // ── Aggregate Calculations ───────────────────────────────────

    /// Total executable lines across all files.
    pub fn total_executable_lines(&self) -> usize {
        self.files.values().map(|f| f.executable_lines()).sum()
    }

    /// Total covered lines across all files.
    pub fn total_covered_lines(&self) -> usize {
        self.files.values().map(|f| f.covered_lines()).sum()
    }

    /// Overall line coverage percentage.
    pub fn overall_line_coverage(&self) -> f64 {
        let total = self.total_executable_lines();
        if total == 0 {
            return 100.0;
        }
        (self.total_covered_lines() as f64 / total as f64) * 100.0
    }

    /// Total branches across all files.
    pub fn total_branches(&self) -> usize {
        self.files.values().map(|f| f.total_branches()).sum()
    }

    /// Total covered branches.
    pub fn total_covered_branches(&self) -> usize {
        self.files.values().map(|f| f.covered_branches()).sum()
    }

    /// Overall branch coverage percentage.
    pub fn overall_branch_coverage(&self) -> f64 {
        let total = self.total_branches();
        if total == 0 {
            return 100.0;
        }
        (self.total_covered_branches() as f64 / total as f64) * 100.0
    }

    /// Total functions across all files.
    pub fn total_functions(&self) -> usize {
        self.files.values().map(|f| f.total_functions()).sum()
    }

    /// Total covered functions.
    pub fn total_covered_functions(&self) -> usize {
        self.files.values().map(|f| f.covered_functions()).sum()
    }

    /// Overall function coverage percentage.
    pub fn overall_function_coverage(&self) -> f64 {
        let total = self.total_functions();
        if total == 0 {
            return 100.0;
        }
        (self.total_covered_functions() as f64 / total as f64) * 100.0
    }

    /// Check if coverage meets a threshold.
    pub fn check_threshold(&self, threshold: f64) -> Result<(), CoverageError> {
        let actual = self.overall_line_coverage();
        if actual >= threshold {
            Ok(())
        } else {
            Err(CoverageError::BelowThreshold {
                actual,
                required: threshold,
            })
        }
    }

    // ── Merge ────────────────────────────────────────────────────

    /// Merge another report into this one. Hit counts are summed.
    pub fn merge(&mut self, other: &CoverageReport) {
        for (path, other_file) in &other.files {
            if let Some(our_file) = self.files.get_mut(path) {
                // Merge line coverage
                for other_line in &other_file.lines {
                    if let Some(our_line) = our_file.lines.iter_mut().find(|l| l.line == other_line.line) {
                        our_line.hit_count += other_line.hit_count;
                    } else {
                        our_file.lines.push(other_line.clone());
                    }
                }
                // Merge branch coverage
                for other_branch in &other_file.branches {
                    if let Some(our_branch) = our_file.branches.iter_mut().find(|b| {
                        b.line == other_branch.line && b.branch_index == other_branch.branch_index
                    }) {
                        our_branch.hit_count += other_branch.hit_count;
                    } else {
                        our_file.branches.push(other_branch.clone());
                    }
                }
                // Merge function coverage
                for other_fn in &other_file.functions {
                    if let Some(our_fn) = our_file.functions.iter_mut().find(|f| f.name == other_fn.name) {
                        our_fn.hit_count += other_fn.hit_count;
                    } else {
                        our_file.functions.push(other_fn.clone());
                    }
                }
            } else {
                self.files.insert(path.clone(), other_file.clone());
            }
        }
    }

    // ── Report Generation ────────────────────────────────────────

    /// Generate a text summary report.
    pub fn text_report(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("Coverage Report: {}\n", self.name));
        out.push_str(&format!("{}\n", "=".repeat(60)));

        let mut paths = self.file_paths();
        paths.sort();

        for path in &paths {
            let fc = &self.files[path];
            out.push_str(&format!(
                "{}: lines {:.1}% ({}/{}), branches {:.1}% ({}/{}), fns {:.1}% ({}/{})\n",
                path,
                fc.line_coverage_pct(),
                fc.covered_lines(),
                fc.executable_lines(),
                fc.branch_coverage_pct(),
                fc.covered_branches(),
                fc.total_branches(),
                fc.function_coverage_pct(),
                fc.covered_functions(),
                fc.total_functions(),
            ));
        }

        out.push_str(&format!("{}\n", "-".repeat(60)));
        out.push_str(&format!(
            "Overall: lines {:.1}%, branches {:.1}%, functions {:.1}%\n",
            self.overall_line_coverage(),
            self.overall_branch_coverage(),
            self.overall_function_coverage(),
        ));
        out
    }

    /// Generate a JSON report.
    pub fn json_report(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }
}

impl fmt::Display for CoverageReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.text_report())
    }
}

// ── Coverage Tracker ─────────────────────────────────────────────

/// Live coverage tracker that records hits as code executes.
#[derive(Debug, Clone, Default)]
pub struct CoverageTracker {
    report: CoverageReport,
    enabled: bool,
}

impl CoverageTracker {
    pub fn new(name: &str) -> Self {
        Self {
            report: CoverageReport::new(name),
            enabled: true,
        }
    }

    /// Enable tracking.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable tracking (hits are ignored).
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Check if tracking is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Register a file for tracking.
    pub fn register_file(&mut self, path: &str, total_lines: u32) {
        self.report.add_file(FileCoverage::new(path, total_lines));
    }

    /// Register lines in a file.
    pub fn register_lines(&mut self, path: &str, lines: &[u32]) {
        if let Ok(fc) = self.report.get_file_mut(path) {
            for line in lines {
                fc.add_line(*line);
            }
        }
    }

    /// Register a function.
    pub fn register_function(&mut self, path: &str, name: &str, start: u32, end: u32) {
        if let Ok(fc) = self.report.get_file_mut(path) {
            fc.add_function(name, start, end);
        }
    }

    /// Register a branch.
    pub fn register_branch(&mut self, path: &str, line: u32, index: u32, label: &str) {
        if let Ok(fc) = self.report.get_file_mut(path) {
            fc.add_branch(line, index, label);
        }
    }

    /// Record a line hit.
    pub fn hit_line(&mut self, path: &str, line: u32) {
        if !self.enabled {
            return;
        }
        if let Ok(fc) = self.report.get_file_mut(path) {
            fc.record_line_hit(line);
        }
    }

    /// Record a function hit.
    pub fn hit_function(&mut self, path: &str, name: &str) {
        if !self.enabled {
            return;
        }
        if let Ok(fc) = self.report.get_file_mut(path) {
            fc.record_function_hit(name);
        }
    }

    /// Record a branch hit.
    pub fn hit_branch(&mut self, path: &str, line: u32, index: u32) {
        if !self.enabled {
            return;
        }
        if let Ok(fc) = self.report.get_file_mut(path) {
            fc.record_branch_hit(line, index);
        }
    }

    /// Get the current report.
    pub fn report(&self) -> &CoverageReport {
        &self.report
    }

    /// Take ownership of the report.
    pub fn into_report(self) -> CoverageReport {
        self.report
    }

    /// Reset all hit counts (keep registrations).
    pub fn reset_hits(&mut self) {
        for fc in self.report.files.values_mut() {
            for line in &mut fc.lines {
                line.hit_count = 0;
            }
            for branch in &mut fc.branches {
                branch.hit_count = 0;
            }
            for func in &mut fc.functions {
                func.hit_count = 0;
            }
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_file_coverage() -> FileCoverage {
        let mut fc = FileCoverage::new("src/main.rs", 100);
        for line in 1..=10 {
            fc.add_line(line);
        }
        // Cover lines 1-7, leave 8-10 uncovered
        for line in 1..=7 {
            fc.record_line_hit(line);
        }
        fc.add_branch(5, 0, "if-true");
        fc.add_branch(5, 1, "if-false");
        fc.record_branch_hit(5, 0);

        fc.add_function("main", 1, 50);
        fc.add_function("helper", 51, 80);
        fc.record_function_hit("main");
        fc
    }

    #[test]
    fn line_coverage_calculation() {
        let fc = sample_file_coverage();
        assert_eq!(fc.executable_lines(), 10);
        assert_eq!(fc.covered_lines(), 7);
        assert!((fc.line_coverage_pct() - 70.0).abs() < 0.01);
    }

    #[test]
    fn branch_coverage_calculation() {
        let fc = sample_file_coverage();
        assert_eq!(fc.total_branches(), 2);
        assert_eq!(fc.covered_branches(), 1);
        assert!((fc.branch_coverage_pct() - 50.0).abs() < 0.01);
    }

    #[test]
    fn function_coverage_calculation() {
        let fc = sample_file_coverage();
        assert_eq!(fc.total_functions(), 2);
        assert_eq!(fc.covered_functions(), 1);
        assert!((fc.function_coverage_pct() - 50.0).abs() < 0.01);
    }

    #[test]
    fn uncovered_lines_identified() {
        let fc = sample_file_coverage();
        let uncovered = fc.uncovered_lines();
        assert_eq!(uncovered, vec![8, 9, 10]);
    }

    #[test]
    fn uncovered_functions() {
        let fc = sample_file_coverage();
        let uncovered = fc.uncovered_functions();
        assert_eq!(uncovered, vec!["helper"]);
    }

    #[test]
    fn uncovered_branches() {
        let fc = sample_file_coverage();
        let uncovered = fc.uncovered_branches();
        assert_eq!(uncovered.len(), 1);
        assert!(uncovered[0].contains("if-false"));
    }

    #[test]
    fn coverage_report_aggregate() {
        let mut report = CoverageReport::new("test");
        report.add_file(sample_file_coverage());

        let mut fc2 = FileCoverage::new("src/lib.rs", 50);
        for line in 1..=5 {
            fc2.add_line(line);
            fc2.record_line_hit(line);
        }
        report.add_file(fc2);

        assert_eq!(report.file_count(), 2);
        assert_eq!(report.total_executable_lines(), 15);
        assert_eq!(report.total_covered_lines(), 12);
        assert!((report.overall_line_coverage() - 80.0).abs() < 0.01);
    }

    #[test]
    fn threshold_check_passes() {
        let mut report = CoverageReport::new("test");
        let mut fc = FileCoverage::new("x.rs", 10);
        for line in 1..=10 {
            fc.add_line(line);
            fc.record_line_hit(line);
        }
        report.add_file(fc);
        assert!(report.check_threshold(100.0).is_ok());
    }

    #[test]
    fn threshold_check_fails() {
        let mut report = CoverageReport::new("test");
        report.add_file(sample_file_coverage());
        let err = report.check_threshold(80.0).unwrap_err();
        assert!(matches!(err, CoverageError::BelowThreshold { .. }));
    }

    #[test]
    fn merge_reports() {
        let mut r1 = CoverageReport::new("run1");
        let mut fc1 = FileCoverage::new("a.rs", 10);
        fc1.add_line(1);
        fc1.record_line_hit(1);
        r1.add_file(fc1);

        let mut r2 = CoverageReport::new("run2");
        let mut fc2 = FileCoverage::new("a.rs", 10);
        fc2.add_line(1);
        fc2.record_line_hit(1);
        fc2.record_line_hit(1); // 2 more hits
        fc2.add_line(2);
        fc2.record_line_hit(2);
        r2.add_file(fc2);

        r1.merge(&r2);
        let merged = r1.get_file("a.rs").unwrap();
        let line1 = merged.lines.iter().find(|l| l.line == 1).unwrap();
        assert_eq!(line1.hit_count, 3); // 1 + 2
        assert_eq!(merged.covered_lines(), 2);
    }

    #[test]
    fn merge_adds_new_files() {
        let mut r1 = CoverageReport::new("run1");
        let fc1 = FileCoverage::new("a.rs", 10);
        r1.add_file(fc1);

        let mut r2 = CoverageReport::new("run2");
        let fc2 = FileCoverage::new("b.rs", 20);
        r2.add_file(fc2);

        r1.merge(&r2);
        assert_eq!(r1.file_count(), 2);
    }

    #[test]
    fn text_report_format() {
        let mut report = CoverageReport::new("test_suite");
        report.add_file(sample_file_coverage());
        let text = report.text_report();
        assert!(text.contains("test_suite"));
        assert!(text.contains("src/main.rs"));
        assert!(text.contains("Overall"));
    }

    #[test]
    fn json_report_roundtrip() {
        let mut report = CoverageReport::new("test");
        report.add_file(sample_file_coverage());
        let json = report.json_report();
        let parsed: CoverageReport = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.file_count(), 1);
    }

    #[test]
    fn tracker_records_hits() {
        let mut tracker = CoverageTracker::new("live");
        tracker.register_file("a.rs", 10);
        tracker.register_lines("a.rs", &[1, 2, 3, 4, 5]);
        tracker.hit_line("a.rs", 1);
        tracker.hit_line("a.rs", 2);
        tracker.hit_line("a.rs", 3);

        let report = tracker.report();
        let fc = report.get_file("a.rs").unwrap();
        assert_eq!(fc.covered_lines(), 3);
    }

    #[test]
    fn tracker_disabled_ignores_hits() {
        let mut tracker = CoverageTracker::new("test");
        tracker.register_file("a.rs", 5);
        tracker.register_lines("a.rs", &[1, 2]);
        tracker.disable();
        tracker.hit_line("a.rs", 1);
        tracker.enable();

        let fc = tracker.report().get_file("a.rs").unwrap();
        assert_eq!(fc.covered_lines(), 0);
    }

    #[test]
    fn tracker_reset_hits() {
        let mut tracker = CoverageTracker::new("test");
        tracker.register_file("a.rs", 5);
        tracker.register_lines("a.rs", &[1]);
        tracker.hit_line("a.rs", 1);
        tracker.reset_hits();

        let fc = tracker.report().get_file("a.rs").unwrap();
        assert_eq!(fc.covered_lines(), 0);
    }

    #[test]
    fn tracker_function_hits() {
        let mut tracker = CoverageTracker::new("test");
        tracker.register_file("a.rs", 50);
        tracker.register_function("a.rs", "run", 1, 25);
        tracker.hit_function("a.rs", "run");

        let fc = tracker.report().get_file("a.rs").unwrap();
        assert_eq!(fc.covered_functions(), 1);
    }

    #[test]
    fn tracker_branch_hits() {
        let mut tracker = CoverageTracker::new("test");
        tracker.register_file("a.rs", 50);
        tracker.register_branch("a.rs", 10, 0, "if-true");
        tracker.register_branch("a.rs", 10, 1, "if-false");
        tracker.hit_branch("a.rs", 10, 0);

        let fc = tracker.report().get_file("a.rs").unwrap();
        assert_eq!(fc.covered_branches(), 1);
        assert_eq!(fc.total_branches(), 2);
    }

    #[test]
    fn empty_file_coverage_is_100() {
        let fc = FileCoverage::new("empty.rs", 0);
        assert!((fc.line_coverage_pct() - 100.0).abs() < 0.01);
        assert!((fc.branch_coverage_pct() - 100.0).abs() < 0.01);
        assert!((fc.function_coverage_pct() - 100.0).abs() < 0.01);
    }

    #[test]
    fn file_not_found_error() {
        let report = CoverageReport::new("test");
        let err = report.get_file("missing.rs").unwrap_err();
        assert!(matches!(err, CoverageError::FileNotFound(_)));
    }

    #[test]
    fn error_display() {
        let err = CoverageError::BelowThreshold {
            actual: 65.0,
            required: 80.0,
        };
        let s = format!("{err}");
        assert!(s.contains("65.0"));
        assert!(s.contains("80.0"));
    }

    #[test]
    fn function_line_count() {
        let fc = FunctionCoverage::new("test", 10, 20);
        assert_eq!(fc.line_count(), 11);
    }

    #[test]
    fn file_paths_sorted() {
        let mut report = CoverageReport::new("test");
        report.add_file(FileCoverage::new("z.rs", 1));
        report.add_file(FileCoverage::new("a.rs", 1));
        let paths = report.file_paths();
        assert_eq!(paths, vec!["a.rs", "z.rs"]);
    }
}
