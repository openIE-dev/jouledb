//! Code coverage tracking model — regions, file coverage, LCOV output, and thresholds.
//!
//! Models coverage data for analysis and reporting. Supports merging reports,
//! computing percentages, LCOV format output, branch coverage, and threshold checks.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Errors ──────────────────────────────────────────────────────

/// Coverage domain errors.
#[derive(Debug, Clone, PartialEq)]
pub enum CoverageError {
    /// Coverage is below the required threshold.
    BelowThreshold { actual: f64, required: f64 },
    /// File not found in report.
    FileNotFound(String),
}

impl std::fmt::Display for CoverageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BelowThreshold { actual, required } => {
                write!(f, "coverage {actual:.1}% below threshold {required:.1}%")
            }
            Self::FileNotFound(path) => write!(f, "file not found: {path}"),
        }
    }
}

impl std::error::Error for CoverageError {}

// ── Coverage Region ─────────────────────────────────────────────

/// A contiguous region of code with coverage data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageRegion {
    pub file: String,
    pub start_line: u32,
    pub end_line: u32,
    pub hit_count: u64,
}

impl CoverageRegion {
    pub fn new(file: impl Into<String>, start_line: u32, end_line: u32, hit_count: u64) -> Self {
        Self {
            file: file.into(),
            start_line,
            end_line,
            hit_count,
        }
    }

    /// Number of lines in this region.
    pub fn line_count(&self) -> u32 {
        if self.end_line >= self.start_line {
            self.end_line - self.start_line + 1
        } else {
            0
        }
    }

    /// Whether this region was hit at least once.
    pub fn is_covered(&self) -> bool {
        self.hit_count > 0
    }
}

// ── Branch Coverage ─────────────────────────────────────────────

/// Branch coverage for a decision point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchCoverage {
    pub decision_id: String,
    pub file: String,
    pub line: u32,
    pub taken: u64,
    pub not_taken: u64,
}

impl BranchCoverage {
    pub fn new(
        decision_id: impl Into<String>,
        file: impl Into<String>,
        line: u32,
        taken: u64,
        not_taken: u64,
    ) -> Self {
        Self {
            decision_id: decision_id.into(),
            file: file.into(),
            line,
            taken,
            not_taken,
        }
    }

    /// Whether both branches were taken.
    pub fn is_fully_covered(&self) -> bool {
        self.taken > 0 && self.not_taken > 0
    }

    /// Whether at least one branch was taken.
    pub fn is_partially_covered(&self) -> bool {
        self.taken > 0 || self.not_taken > 0
    }
}

// ── File Coverage ───────────────────────────────────────────────

/// Coverage data for a single file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileCoverage {
    pub path: String,
    pub regions: Vec<CoverageRegion>,
    pub total_lines: u32,
    pub branches: Vec<BranchCoverage>,
}

impl FileCoverage {
    pub fn new(path: impl Into<String>, total_lines: u32) -> Self {
        Self {
            path: path.into(),
            regions: Vec::new(),
            total_lines,
            branches: Vec::new(),
        }
    }

    /// Add a coverage region.
    pub fn add_region(&mut self, region: CoverageRegion) {
        self.regions.push(region);
    }

    /// Add a branch coverage entry.
    pub fn add_branch(&mut self, branch: BranchCoverage) {
        self.branches.push(branch);
    }

    /// Count of lines covered (hit_count > 0).
    pub fn covered_lines(&self) -> u32 {
        let mut covered = std::collections::HashSet::new();
        for region in &self.regions {
            if region.is_covered() {
                for line in region.start_line..=region.end_line {
                    covered.insert(line);
                }
            }
        }
        covered.len() as u32
    }

    /// Count of instrumented lines (all lines in regions).
    pub fn instrumented_lines(&self) -> u32 {
        let mut lines = std::collections::HashSet::new();
        for region in &self.regions {
            for line in region.start_line..=region.end_line {
                lines.insert(line);
            }
        }
        lines.len() as u32
    }

    /// Line coverage percentage.
    pub fn coverage_percentage(&self) -> f64 {
        let instrumented = self.instrumented_lines();
        if instrumented == 0 {
            return 100.0;
        }
        self.covered_lines() as f64 / instrumented as f64 * 100.0
    }

    /// Branch coverage percentage.
    pub fn branch_coverage_percentage(&self) -> f64 {
        if self.branches.is_empty() {
            return 100.0;
        }
        let total = self.branches.len() * 2; // each branch has taken + not_taken
        let covered: usize = self
            .branches
            .iter()
            .map(|b| {
                let mut c = 0;
                if b.taken > 0 {
                    c += 1;
                }
                if b.not_taken > 0 {
                    c += 1;
                }
                c
            })
            .sum();
        covered as f64 / total as f64 * 100.0
    }

    /// Get ranges of uncovered lines.
    pub fn uncovered_ranges(&self) -> Vec<(u32, u32)> {
        let mut uncovered_lines: Vec<u32> = Vec::new();
        for region in &self.regions {
            if !region.is_covered() {
                for line in region.start_line..=region.end_line {
                    uncovered_lines.push(line);
                }
            }
        }
        uncovered_lines.sort();
        uncovered_lines.dedup();

        // Collapse into ranges
        let mut ranges = Vec::new();
        let mut iter = uncovered_lines.into_iter();
        if let Some(first) = iter.next() {
            let mut start = first;
            let mut end = first;
            for line in iter {
                if line == end + 1 {
                    end = line;
                } else {
                    ranges.push((start, end));
                    start = line;
                    end = line;
                }
            }
            ranges.push((start, end));
        }
        ranges
    }
}

// ── Coverage Report ─────────────────────────────────────────────

/// A coverage report spanning multiple files.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CoverageReport {
    pub files: Vec<FileCoverage>,
}

impl CoverageReport {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add file coverage.
    pub fn add_file(&mut self, file: FileCoverage) {
        self.files.push(file);
    }

    /// Get coverage for a specific file.
    pub fn get_file(&self, path: &str) -> Option<&FileCoverage> {
        self.files.iter().find(|f| f.path == path)
    }

    /// Overall line coverage percentage.
    pub fn overall_percentage(&self) -> f64 {
        let total_instrumented: u32 = self.files.iter().map(|f| f.instrumented_lines()).sum();
        let total_covered: u32 = self.files.iter().map(|f| f.covered_lines()).sum();
        if total_instrumented == 0 {
            return 100.0;
        }
        total_covered as f64 / total_instrumented as f64 * 100.0
    }

    /// Overall branch coverage percentage.
    pub fn overall_branch_percentage(&self) -> f64 {
        let total_branches: usize = self.files.iter().map(|f| f.branches.len() * 2).sum();
        if total_branches == 0 {
            return 100.0;
        }
        let covered: usize = self
            .files
            .iter()
            .flat_map(|f| &f.branches)
            .map(|b| {
                let mut c = 0usize;
                if b.taken > 0 {
                    c += 1;
                }
                if b.not_taken > 0 {
                    c += 1;
                }
                c
            })
            .sum();
        covered as f64 / total_branches as f64 * 100.0
    }

    /// Check if coverage meets a threshold.
    pub fn check_threshold(&self, threshold_pct: f64) -> Result<f64, CoverageError> {
        let actual = self.overall_percentage();
        if actual < threshold_pct {
            Err(CoverageError::BelowThreshold {
                actual,
                required: threshold_pct,
            })
        } else {
            Ok(actual)
        }
    }

    /// Per-file summary: (path, coverage_pct).
    pub fn per_file_summary(&self) -> Vec<(&str, f64)> {
        self.files
            .iter()
            .map(|f| (f.path.as_str(), f.coverage_percentage()))
            .collect()
    }

    /// Merge another report into this one.
    /// For files present in both, hit counts are summed.
    pub fn merge(&mut self, other: &CoverageReport) {
        let mut file_map: HashMap<String, usize> = self
            .files
            .iter()
            .enumerate()
            .map(|(i, f)| (f.path.clone(), i))
            .collect();

        for other_file in &other.files {
            if let Some(&idx) = file_map.get(&other_file.path) {
                // Merge regions: match by start/end lines, sum hit counts
                let existing = &mut self.files[idx];
                for other_region in &other_file.regions {
                    if let Some(existing_region) = existing.regions.iter_mut().find(|r| {
                        r.start_line == other_region.start_line
                            && r.end_line == other_region.end_line
                    }) {
                        existing_region.hit_count += other_region.hit_count;
                    } else {
                        existing.regions.push(other_region.clone());
                    }
                }
                // Merge branches
                for other_branch in &other_file.branches {
                    if let Some(existing_branch) = existing
                        .branches
                        .iter_mut()
                        .find(|b| b.decision_id == other_branch.decision_id)
                    {
                        existing_branch.taken += other_branch.taken;
                        existing_branch.not_taken += other_branch.not_taken;
                    } else {
                        existing.branches.push(other_branch.clone());
                    }
                }
            } else {
                file_map.insert(other_file.path.clone(), self.files.len());
                self.files.push(other_file.clone());
            }
        }
    }

    /// Generate LCOV format output.
    pub fn to_lcov(&self) -> String {
        let mut output = String::new();
        for file in &self.files {
            output.push_str("TN:\n");
            output.push_str(&format!("SF:{}\n", file.path));

            // Line data
            let mut line_hits: HashMap<u32, u64> = HashMap::new();
            for region in &file.regions {
                for line in region.start_line..=region.end_line {
                    *line_hits.entry(line).or_insert(0) += region.hit_count;
                }
            }
            let mut lines: Vec<u32> = line_hits.keys().copied().collect();
            lines.sort();
            for line in &lines {
                output.push_str(&format!("DA:{},{}\n", line, line_hits[line]));
            }

            // Branch data
            for (i, branch) in file.branches.iter().enumerate() {
                output.push_str(&format!("BRDA:{},{},0,{}\n", branch.line, i, branch.taken));
                output.push_str(&format!(
                    "BRDA:{},{},1,{}\n",
                    branch.line, i, branch.not_taken
                ));
            }
            if !file.branches.is_empty() {
                let brf = file.branches.len() * 2;
                let brh: usize = file
                    .branches
                    .iter()
                    .map(|b| (b.taken > 0) as usize + (b.not_taken > 0) as usize)
                    .sum();
                output.push_str(&format!("BRF:{brf}\n"));
                output.push_str(&format!("BRH:{brh}\n"));
            }

            output.push_str(&format!("LF:{}\n", lines.len()));
            let lh = lines.iter().filter(|l| line_hits[l] > 0).count();
            output.push_str(&format!("LH:{lh}\n"));
            output.push_str("end_of_record\n");
        }
        output
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_file() -> FileCoverage {
        let mut fc = FileCoverage::new("src/main.rs", 50);
        fc.add_region(CoverageRegion::new("src/main.rs", 1, 10, 5));
        fc.add_region(CoverageRegion::new("src/main.rs", 11, 20, 0));
        fc.add_region(CoverageRegion::new("src/main.rs", 21, 30, 3));
        fc
    }

    fn sample_report() -> CoverageReport {
        let mut report = CoverageReport::new();
        report.add_file(sample_file());
        report
    }

    #[test]
    fn region_line_count() {
        let r = CoverageRegion::new("f.rs", 5, 10, 1);
        assert_eq!(r.line_count(), 6);
    }

    #[test]
    fn region_is_covered() {
        assert!(CoverageRegion::new("f.rs", 1, 5, 1).is_covered());
        assert!(!CoverageRegion::new("f.rs", 1, 5, 0).is_covered());
    }

    #[test]
    fn file_coverage_percentage() {
        let fc = sample_file();
        // 10 covered (1-10) + 10 covered (21-30) = 20 out of 30 instrumented
        assert!((fc.coverage_percentage() - 66.666).abs() < 0.1);
    }

    #[test]
    fn uncovered_ranges() {
        let fc = sample_file();
        let ranges = fc.uncovered_ranges();
        assert_eq!(ranges, vec![(11, 20)]);
    }

    #[test]
    fn overall_percentage() {
        let report = sample_report();
        assert!((report.overall_percentage() - 66.666).abs() < 0.1);
    }

    #[test]
    fn threshold_pass() {
        let report = sample_report();
        assert!(report.check_threshold(50.0).is_ok());
    }

    #[test]
    fn threshold_fail() {
        let report = sample_report();
        let result = report.check_threshold(80.0);
        assert!(matches!(
            result,
            Err(CoverageError::BelowThreshold { .. })
        ));
    }

    #[test]
    fn merge_reports() {
        let mut r1 = sample_report();
        let mut r2 = CoverageReport::new();
        let mut fc2 = FileCoverage::new("src/main.rs", 50);
        fc2.add_region(CoverageRegion::new("src/main.rs", 11, 20, 2)); // was 0, now 2
        r2.add_file(fc2);

        r1.merge(&r2);
        let file = r1.get_file("src/main.rs").unwrap();
        // Region 11-20 should now have hit_count = 0 + 2 = 2
        let region = file
            .regions
            .iter()
            .find(|r| r.start_line == 11)
            .unwrap();
        assert_eq!(region.hit_count, 2);
    }

    #[test]
    fn merge_new_file() {
        let mut r1 = sample_report();
        let mut r2 = CoverageReport::new();
        r2.add_file(FileCoverage::new("src/lib.rs", 30));
        r1.merge(&r2);
        assert_eq!(r1.files.len(), 2);
    }

    #[test]
    fn per_file_summary() {
        let mut report = CoverageReport::new();
        let mut fc1 = FileCoverage::new("a.rs", 10);
        fc1.add_region(CoverageRegion::new("a.rs", 1, 10, 1));
        let mut fc2 = FileCoverage::new("b.rs", 10);
        fc2.add_region(CoverageRegion::new("b.rs", 1, 10, 0));
        report.add_file(fc1);
        report.add_file(fc2);
        let summary = report.per_file_summary();
        assert_eq!(summary.len(), 2);
        assert!((summary[0].1 - 100.0).abs() < 0.01);
        assert!((summary[1].1 - 0.0).abs() < 0.01);
    }

    #[test]
    fn lcov_output() {
        let report = sample_report();
        let lcov = report.to_lcov();
        assert!(lcov.contains("SF:src/main.rs"));
        assert!(lcov.contains("DA:1,5"));
        assert!(lcov.contains("DA:11,0"));
        assert!(lcov.contains("end_of_record"));
    }

    #[test]
    fn branch_coverage() {
        let mut fc = FileCoverage::new("b.rs", 20);
        fc.add_branch(BranchCoverage::new("br1", "b.rs", 5, 10, 3));
        fc.add_branch(BranchCoverage::new("br2", "b.rs", 10, 5, 0));
        // br1: both taken (2/2), br2: only taken (1/2) = 3/4 = 75%
        assert!((fc.branch_coverage_percentage() - 75.0).abs() < 0.01);
    }

    #[test]
    fn branch_fully_covered() {
        let b = BranchCoverage::new("br", "f.rs", 1, 5, 3);
        assert!(b.is_fully_covered());
    }

    #[test]
    fn branch_partially_covered() {
        let b = BranchCoverage::new("br", "f.rs", 1, 5, 0);
        assert!(!b.is_fully_covered());
        assert!(b.is_partially_covered());
    }

    #[test]
    fn lcov_with_branches() {
        let mut report = CoverageReport::new();
        let mut fc = FileCoverage::new("b.rs", 20);
        fc.add_region(CoverageRegion::new("b.rs", 1, 10, 5));
        fc.add_branch(BranchCoverage::new("br1", "b.rs", 5, 10, 3));
        report.add_file(fc);
        let lcov = report.to_lcov();
        assert!(lcov.contains("BRDA:5,0,0,10"));
        assert!(lcov.contains("BRDA:5,0,1,3"));
        assert!(lcov.contains("BRF:2"));
        assert!(lcov.contains("BRH:2"));
    }

    #[test]
    fn empty_report_full_coverage() {
        let report = CoverageReport::new();
        assert!((report.overall_percentage() - 100.0).abs() < f64::EPSILON);
    }
}
