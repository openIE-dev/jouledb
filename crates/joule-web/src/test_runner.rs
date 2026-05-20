//! Test suite runner.
//!
//! Replaces `jest`, `mocha`, and `vitest` with a pure-Rust test runner model.
//! Supports test cases with tags, suites with before/after hooks, filtering,
//! parallel partitioning, retry logic, and JUnit XML output.

use std::fmt;
use std::time::Instant;

// ── TestCase ────────────────────────────────────────────────────

/// Outcome of a single test.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TestOutcome {
    Passed,
    Failed(String),
    Skipped,
}

/// A single test case.
pub struct TestCase {
    pub name: String,
    pub tags: Vec<String>,
    func: Box<dyn Fn() -> Result<(), String>>,
}

impl TestCase {
    /// Create a new test case.
    pub fn new(
        name: &str,
        tags: &[&str],
        func: impl Fn() -> Result<(), String> + 'static,
    ) -> Self {
        Self {
            name: name.to_string(),
            tags: tags.iter().map(|s| s.to_string()).collect(),
            func: Box::new(func),
        }
    }

    /// Check if this test has a specific tag.
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|t| t == tag)
    }

    /// Check if the name matches a pattern (substring match).
    pub fn name_matches(&self, pattern: &str) -> bool {
        self.name.contains(pattern)
    }

    /// Run the test function.
    pub fn run(&self) -> TestOutcome {
        match (self.func)() {
            Ok(()) => TestOutcome::Passed,
            Err(msg) => TestOutcome::Failed(msg),
        }
    }
}

impl fmt::Debug for TestCase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TestCase")
            .field("name", &self.name)
            .field("tags", &self.tags)
            .finish()
    }
}

// ── TestResult ──────────────────────────────────────────────────

/// Result of running a single test case.
#[derive(Debug, Clone)]
pub struct TestResult {
    pub name: String,
    pub outcome: TestOutcome,
    pub duration_ms: f64,
    pub retries: u32,
}

// ── TestSuite ───────────────────────────────────────────────────

/// A collection of test cases with optional hooks.
pub struct TestSuite {
    pub name: String,
    cases: Vec<TestCase>,
    before_each: Option<Box<dyn Fn()>>,
    after_each: Option<Box<dyn Fn()>>,
}

impl TestSuite {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            cases: Vec::new(),
            before_each: None,
            after_each: None,
        }
    }

    /// Add a test case.
    pub fn add(&mut self, case: TestCase) {
        self.cases.push(case);
    }

    /// Set before_each hook.
    pub fn before_each(&mut self, hook: impl Fn() + 'static) {
        self.before_each = Some(Box::new(hook));
    }

    /// Set after_each hook.
    pub fn after_each(&mut self, hook: impl Fn() + 'static) {
        self.after_each = Some(Box::new(hook));
    }

    /// Number of test cases.
    pub fn len(&self) -> usize {
        self.cases.len()
    }

    /// Check if suite is empty.
    pub fn is_empty(&self) -> bool {
        self.cases.is_empty()
    }
}

// ── Filter ──────────────────────────────────────────────────────

/// Filter criteria for selecting which tests to run.
#[derive(Debug, Clone, Default)]
pub struct TestFilter {
    /// Substring match on test name.
    pub name_pattern: Option<String>,
    /// Only run tests with this tag.
    pub tag: Option<String>,
    /// Skip tests with this tag.
    pub skip_tag: Option<String>,
}

impl TestFilter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn name(mut self, pattern: &str) -> Self {
        self.name_pattern = Some(pattern.to_string());
        self
    }

    pub fn tag(mut self, tag: &str) -> Self {
        self.tag = Some(tag.to_string());
        self
    }

    pub fn skip(mut self, tag: &str) -> Self {
        self.skip_tag = Some(tag.to_string());
        self
    }

    fn should_run(&self, case: &TestCase) -> bool {
        if let Some(pattern) = &self.name_pattern {
            if !case.name_matches(pattern) {
                return false;
            }
        }
        if let Some(tag) = &self.tag {
            if !case.has_tag(tag) {
                return false;
            }
        }
        if let Some(skip) = &self.skip_tag {
            if case.has_tag(skip) {
                return false;
            }
        }
        true
    }
}

// ── TestReport ──────────────────────────────────────────────────

/// Aggregate results from running a suite.
#[derive(Debug, Clone)]
pub struct TestReport {
    pub suite_name: String,
    pub results: Vec<TestResult>,
    pub total_duration_ms: f64,
}

impl TestReport {
    pub fn passed(&self) -> usize {
        self.results
            .iter()
            .filter(|r| r.outcome == TestOutcome::Passed)
            .count()
    }

    pub fn failed(&self) -> usize {
        self.results
            .iter()
            .filter(|r| matches!(r.outcome, TestOutcome::Failed(_)))
            .count()
    }

    pub fn skipped(&self) -> usize {
        self.results
            .iter()
            .filter(|r| r.outcome == TestOutcome::Skipped)
            .count()
    }

    pub fn total(&self) -> usize {
        self.results.len()
    }

    pub fn all_passed(&self) -> bool {
        self.failed() == 0
    }

    /// Produce JUnit XML output.
    pub fn to_junit_xml(&self) -> String {
        let mut xml = String::new();
        xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        fmt::write(
            &mut xml,
            format_args!(
                "<testsuite name=\"{}\" tests=\"{}\" failures=\"{}\" skipped=\"{}\" time=\"{:.3}\">\n",
                escape_xml(&self.suite_name),
                self.total(),
                self.failed(),
                self.skipped(),
                self.total_duration_ms / 1000.0
            ),
        )
        .ok();

        for result in &self.results {
            fmt::write(
                &mut xml,
                format_args!(
                    "  <testcase name=\"{}\" time=\"{:.3}\"",
                    escape_xml(&result.name),
                    result.duration_ms / 1000.0
                ),
            )
            .ok();

            match &result.outcome {
                TestOutcome::Passed => {
                    xml.push_str(" />\n");
                }
                TestOutcome::Failed(msg) => {
                    xml.push_str(">\n");
                    fmt::write(
                        &mut xml,
                        format_args!(
                            "    <failure message=\"{}\">{}</failure>\n",
                            escape_xml(msg),
                            escape_xml(msg)
                        ),
                    )
                    .ok();
                    xml.push_str("  </testcase>\n");
                }
                TestOutcome::Skipped => {
                    xml.push_str(">\n    <skipped />\n  </testcase>\n");
                }
            }
        }

        xml.push_str("</testsuite>\n");
        xml
    }
}

impl fmt::Display for TestReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Suite: {}", self.suite_name)?;
        writeln!(
            f,
            "  {} passed, {} failed, {} skipped ({:.1}ms)",
            self.passed(),
            self.failed(),
            self.skipped(),
            self.total_duration_ms
        )?;
        for result in &self.results {
            let icon = match &result.outcome {
                TestOutcome::Passed => "PASS",
                TestOutcome::Failed(_) => "FAIL",
                TestOutcome::Skipped => "SKIP",
            };
            writeln!(f, "  [{icon}] {} ({:.1}ms)", result.name, result.duration_ms)?;
        }
        Ok(())
    }
}

fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

// ── Runner ──────────────────────────────────────────────────────

/// Run a test suite with optional filtering and retries.
pub fn run_suite(suite: &TestSuite, filter: &TestFilter, max_retries: u32) -> TestReport {
    let start = Instant::now();
    let mut results = Vec::new();

    for case in &suite.cases {
        if !filter.should_run(case) {
            results.push(TestResult {
                name: case.name.clone(),
                outcome: TestOutcome::Skipped,
                duration_ms: 0.0,
                retries: 0,
            });
            continue;
        }

        let mut outcome = TestOutcome::Failed("not run".to_string());
        let mut retries = 0u32;
        let case_start = Instant::now();

        for attempt in 0..=max_retries {
            if let Some(hook) = &suite.before_each {
                hook();
            }

            outcome = case.run();

            if let Some(hook) = &suite.after_each {
                hook();
            }

            if outcome == TestOutcome::Passed {
                break;
            }
            if attempt < max_retries {
                retries += 1;
            }
        }

        let duration_ms = case_start.elapsed().as_secs_f64() * 1000.0;

        results.push(TestResult {
            name: case.name.clone(),
            outcome,
            duration_ms,
            retries,
        });
    }

    let total_duration_ms = start.elapsed().as_secs_f64() * 1000.0;

    TestReport {
        suite_name: suite.name.clone(),
        results,
        total_duration_ms,
    }
}

/// Partition test cases into `n` groups for parallel execution simulation.
pub fn partition_cases(count: usize, groups: usize) -> Vec<(usize, usize)> {
    if groups == 0 || count == 0 {
        return Vec::new();
    }
    let per_group = count / groups;
    let remainder = count % groups;
    let mut partitions = Vec::with_capacity(groups);
    let mut start = 0;

    for i in 0..groups {
        let extra = if i < remainder { 1 } else { 0 };
        let end = start + per_group + extra;
        if start < count {
            partitions.push((start, end.min(count)));
        }
        start = end;
    }

    partitions
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn test_case_passes() {
        let case = TestCase::new("simple", &[], || Ok(()));
        assert_eq!(case.run(), TestOutcome::Passed);
    }

    #[test]
    fn test_case_fails() {
        let case = TestCase::new("fail", &[], || Err("broken".to_string()));
        match case.run() {
            TestOutcome::Failed(msg) => assert_eq!(msg, "broken"),
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    #[test]
    fn test_case_tags() {
        let case = TestCase::new("tagged", &["unit", "fast"], || Ok(()));
        assert!(case.has_tag("unit"));
        assert!(case.has_tag("fast"));
        assert!(!case.has_tag("slow"));
    }

    #[test]
    fn suite_run_all_pass() {
        let mut suite = TestSuite::new("math");
        suite.add(TestCase::new("add", &[], || Ok(())));
        suite.add(TestCase::new("sub", &[], || Ok(())));

        let report = run_suite(&suite, &TestFilter::new(), 0);
        assert_eq!(report.passed(), 2);
        assert_eq!(report.failed(), 0);
        assert!(report.all_passed());
    }

    #[test]
    fn suite_with_failure() {
        let mut suite = TestSuite::new("mixed");
        suite.add(TestCase::new("ok", &[], || Ok(())));
        suite.add(TestCase::new("bad", &[], || Err("oops".to_string())));

        let report = run_suite(&suite, &TestFilter::new(), 0);
        assert_eq!(report.passed(), 1);
        assert_eq!(report.failed(), 1);
        assert!(!report.all_passed());
    }

    #[test]
    fn filter_by_name() {
        let mut suite = TestSuite::new("filtered");
        suite.add(TestCase::new("test_add", &[], || Ok(())));
        suite.add(TestCase::new("test_sub", &[], || Ok(())));
        suite.add(TestCase::new("bench_mul", &[], || Ok(())));

        let filter = TestFilter::new().name("test_");
        let report = run_suite(&suite, &filter, 0);
        assert_eq!(report.passed(), 2);
        assert_eq!(report.skipped(), 1);
    }

    #[test]
    fn filter_by_tag() {
        let mut suite = TestSuite::new("tagged");
        suite.add(TestCase::new("a", &["unit"], || Ok(())));
        suite.add(TestCase::new("b", &["integration"], || Ok(())));
        suite.add(TestCase::new("c", &["unit"], || Ok(())));

        let filter = TestFilter::new().tag("unit");
        let report = run_suite(&suite, &filter, 0);
        assert_eq!(report.passed(), 2);
        assert_eq!(report.skipped(), 1);
    }

    #[test]
    fn filter_skip_tag() {
        let mut suite = TestSuite::new("skip");
        suite.add(TestCase::new("a", &["slow"], || Ok(())));
        suite.add(TestCase::new("b", &["fast"], || Ok(())));

        let filter = TestFilter::new().skip("slow");
        let report = run_suite(&suite, &filter, 0);
        assert_eq!(report.passed(), 1);
        assert_eq!(report.skipped(), 1);
    }

    #[test]
    fn retry_on_failure() {
        let counter = Cell::new(0u32);
        let mut suite = TestSuite::new("retry");
        suite.add(TestCase::new("flaky", &[], move || {
            let c = counter.get() + 1;
            counter.set(c);
            if c < 3 { Err("not yet".to_string()) } else { Ok(()) }
        }));

        let report = run_suite(&suite, &TestFilter::new(), 5);
        assert_eq!(report.passed(), 1);
        assert_eq!(report.results[0].retries, 2);
    }

    #[test]
    fn before_after_hooks() {
        let hook_count = Cell::new(0u32);
        let mut suite = TestSuite::new("hooks");

        // Use a raw pointer to share the Cell across closures
        let hook_ptr = &hook_count as *const Cell<u32>;

        suite.before_each(move || {
            unsafe { &*hook_ptr }.set(unsafe { &*hook_ptr }.get() + 1);
        });
        suite.after_each(move || {
            unsafe { &*hook_ptr }.set(unsafe { &*hook_ptr }.get() + 10);
        });

        suite.add(TestCase::new("t1", &[], || Ok(())));
        suite.add(TestCase::new("t2", &[], || Ok(())));

        let report = run_suite(&suite, &TestFilter::new(), 0);
        assert_eq!(report.passed(), 2);
        // 2 before_each (+2) + 2 after_each (+20) = 22
        assert_eq!(hook_count.get(), 22);
    }

    #[test]
    fn junit_xml_output() {
        let mut suite = TestSuite::new("xml_test");
        suite.add(TestCase::new("pass_test", &[], || Ok(())));
        suite.add(TestCase::new("fail_test", &[], || Err("boom".to_string())));

        let report = run_suite(&suite, &TestFilter::new(), 0);
        let xml = report.to_junit_xml();
        assert!(xml.contains("<?xml version"));
        assert!(xml.contains("testsuite name=\"xml_test\""));
        assert!(xml.contains("testcase name=\"pass_test\""));
        assert!(xml.contains("testcase name=\"fail_test\""));
        assert!(xml.contains("<failure"));
        assert!(xml.contains("boom"));
    }

    #[test]
    fn partition_even() {
        let parts = partition_cases(10, 3);
        assert_eq!(parts.len(), 3);
        // 10/3 = 3 remainder 1, so first group gets 4
        assert_eq!(parts[0], (0, 4));
        assert_eq!(parts[1], (4, 7));
        assert_eq!(parts[2], (7, 10));
    }

    #[test]
    fn partition_more_groups_than_items() {
        let parts = partition_cases(2, 5);
        // Only 2 non-empty groups
        assert_eq!(parts.len(), 2);
    }

    #[test]
    fn report_display() {
        let mut suite = TestSuite::new("display");
        suite.add(TestCase::new("a", &[], || Ok(())));
        let report = run_suite(&suite, &TestFilter::new(), 0);
        let s = format!("{report}");
        assert!(s.contains("Suite: display"));
        assert!(s.contains("PASS"));
    }
}
