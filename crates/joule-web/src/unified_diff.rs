//! Unified diff format.
//!
//! Generate and parse unified diff output (the `--- +++ @@` format used by
//! `diff -u`, Git, and patch files). Supports configurable context lines,
//! file headers, multi-file diffs, color-annotated output, and hunk merging.

use std::fmt;
use std::fmt::Write as FmtWrite;

// ── Types ──────────────────────────────────────────────────────────

/// Error type for unified diff operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnifiedDiffError {
    /// Parse error with description.
    ParseError(String),
    /// Invalid hunk header.
    InvalidHunkHeader(String),
    /// Patch application failure.
    ApplyError(String),
}

impl fmt::Display for UnifiedDiffError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ParseError(s) => write!(f, "parse error: {}", s),
            Self::InvalidHunkHeader(s) => write!(f, "invalid hunk header: {}", s),
            Self::ApplyError(s) => write!(f, "apply error: {}", s),
        }
    }
}

/// A line operation in a hunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiffLine {
    /// Context line (unchanged).
    Context(String),
    /// Added line.
    Addition(String),
    /// Removed line.
    Removal(String),
}

/// A single hunk in a unified diff.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hunk {
    pub old_start: usize,
    pub old_count: usize,
    pub new_start: usize,
    pub new_count: usize,
    pub section_header: Option<String>,
    pub lines: Vec<DiffLine>,
}

/// File header information.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileHeader {
    pub old_file: String,
    pub new_file: String,
    pub old_timestamp: Option<String>,
    pub new_timestamp: Option<String>,
}

/// A unified diff for a single file pair.
#[derive(Debug, Clone)]
pub struct FileDiff {
    pub header: FileHeader,
    pub hunks: Vec<Hunk>,
}

/// A multi-file unified diff.
#[derive(Debug, Clone)]
pub struct MultiFileDiff {
    pub files: Vec<FileDiff>,
}

/// Configuration for diff generation.
pub struct DiffConfig {
    /// Number of context lines around each change.
    pub context_lines: usize,
    /// Whether to merge adjacent hunks.
    pub merge_hunks: bool,
}

impl Default for DiffConfig {
    fn default() -> Self {
        Self {
            context_lines: 3,
            merge_hunks: true,
        }
    }
}

/// ANSI color codes for terminal output.
struct AnsiColors;

impl AnsiColors {
    const RED: &'static str = "\x1b[31m";
    const GREEN: &'static str = "\x1b[32m";
    const CYAN: &'static str = "\x1b[36m";
    const BOLD: &'static str = "\x1b[1m";
    const RESET: &'static str = "\x1b[0m";
}

// ── Internal diff engine ───────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
enum RawOp {
    Equal(String),
    Insert(String),
    Delete(String),
}

fn diff_sequences(old: &[&str], new: &[&str]) -> Vec<RawOp> {
    let n = old.len();
    let m = new.len();

    if n == 0 && m == 0 {
        return Vec::new();
    }
    if n == 0 {
        return new.iter().map(|s| RawOp::Insert(s.to_string())).collect();
    }
    if m == 0 {
        return old.iter().map(|s| RawOp::Delete(s.to_string())).collect();
    }

    let mut dp = vec![vec![0u32; m + 1]; n + 1];
    for i in 1..=n {
        for j in 1..=m {
            if old[i - 1] == new[j - 1] {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }

    let mut ops = Vec::new();
    let mut i = n;
    let mut j = m;
    while i > 0 || j > 0 {
        if i > 0 && j > 0 && old[i - 1] == new[j - 1] {
            ops.push(RawOp::Equal(old[i - 1].to_string()));
            i -= 1;
            j -= 1;
        } else if j > 0 && (i == 0 || dp[i][j - 1] >= dp[i - 1][j]) {
            ops.push(RawOp::Insert(new[j - 1].to_string()));
            j -= 1;
        } else {
            ops.push(RawOp::Delete(old[i - 1].to_string()));
            i -= 1;
        }
    }
    ops.reverse();
    ops
}

// ── Generation ─────────────────────────────────────────────────────

/// Generate a unified diff between two strings with default settings.
pub fn generate(old: &str, new: &str, old_name: &str, new_name: &str) -> String {
    generate_with_config(old, new, old_name, new_name, &DiffConfig::default())
}

/// Generate a unified diff with custom configuration.
pub fn generate_with_config(
    old: &str,
    new: &str,
    old_name: &str,
    new_name: &str,
    config: &DiffConfig,
) -> String {
    let file_diff = generate_file_diff(old, new, old_name, new_name, config);
    render_file_diff(&file_diff)
}

/// Generate a FileDiff structure.
pub fn generate_file_diff(
    old: &str,
    new: &str,
    old_name: &str,
    new_name: &str,
    config: &DiffConfig,
) -> FileDiff {
    let old_lines: Vec<&str> = if old.is_empty() {
        Vec::new()
    } else {
        old.lines().collect()
    };
    let new_lines: Vec<&str> = if new.is_empty() {
        Vec::new()
    } else {
        new.lines().collect()
    };

    let raw_ops = diff_sequences(&old_lines, &new_lines);
    let hunks = build_hunks(&raw_ops, config.context_lines, config.merge_hunks);

    FileDiff {
        header: FileHeader {
            old_file: old_name.to_string(),
            new_file: new_name.to_string(),
            old_timestamp: None,
            new_timestamp: None,
        },
        hunks,
    }
}

fn build_hunks(ops: &[RawOp], context: usize, merge: bool) -> Vec<Hunk> {
    if ops.is_empty() {
        return Vec::new();
    }

    // Track line numbers and mark change positions.
    struct IndexedOp {
        op: RawOp,
        old_line: usize,
        new_line: usize,
    }

    let mut indexed = Vec::new();
    let mut ol = 1usize;
    let mut nl = 1usize;
    for op in ops {
        let cur_ol = ol;
        let cur_nl = nl;
        match op {
            RawOp::Equal(_) => {
                ol += 1;
                nl += 1;
            }
            RawOp::Delete(_) => {
                ol += 1;
            }
            RawOp::Insert(_) => {
                nl += 1;
            }
        }
        indexed.push(IndexedOp {
            op: op.clone(),
            old_line: cur_ol,
            new_line: cur_nl,
        });
    }

    // Find change positions.
    let change_positions: Vec<usize> = indexed
        .iter()
        .enumerate()
        .filter(|(_, iop)| !matches!(&iop.op, RawOp::Equal(_)))
        .map(|(i, _)| i)
        .collect();

    if change_positions.is_empty() {
        return Vec::new();
    }

    // Group change positions into hunk ranges.
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    let total = indexed.len();
    let mut start = change_positions[0].saturating_sub(context);
    let mut end = (change_positions[0] + context + 1).min(total);

    for &pos in &change_positions[1..] {
        let ns = pos.saturating_sub(context);
        let ne = (pos + context + 1).min(total);
        if merge && ns <= end {
            end = ne;
        } else {
            ranges.push((start, end));
            start = ns;
            end = ne;
        }
    }
    ranges.push((start, end));

    // Build hunks from ranges.
    let mut hunks = Vec::new();
    for (rs, re) in ranges {
        let slice = &indexed[rs..re];
        if slice.is_empty() {
            continue;
        }
        let hunk_old_start = slice[0].old_line;
        let hunk_new_start = slice[0].new_line;
        let mut old_count = 0;
        let mut new_count = 0;
        let mut lines = Vec::new();

        for iop in slice {
            match &iop.op {
                RawOp::Equal(s) => {
                    old_count += 1;
                    new_count += 1;
                    lines.push(DiffLine::Context(s.clone()));
                }
                RawOp::Delete(s) => {
                    old_count += 1;
                    lines.push(DiffLine::Removal(s.clone()));
                }
                RawOp::Insert(s) => {
                    new_count += 1;
                    lines.push(DiffLine::Addition(s.clone()));
                }
            }
        }

        hunks.push(Hunk {
            old_start: hunk_old_start,
            old_count,
            new_start: hunk_new_start,
            new_count,
            section_header: None,
            lines,
        });
    }

    hunks
}

/// Render a FileDiff to a unified diff string.
pub fn render_file_diff(diff: &FileDiff) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "--- {}", diff.header.old_file);
    let _ = writeln!(out, "+++ {}", diff.header.new_file);

    for hunk in &diff.hunks {
        render_hunk(hunk, &mut out);
    }
    out
}

fn render_hunk(hunk: &Hunk, out: &mut String) {
    let section = match &hunk.section_header {
        Some(s) => format!(" {}", s),
        None => String::new(),
    };
    let _ = writeln!(
        out,
        "@@ -{},{} +{},{} @@{}",
        hunk.old_start, hunk.old_count, hunk.new_start, hunk.new_count, section
    );
    for line in &hunk.lines {
        match line {
            DiffLine::Context(s) => {
                let _ = writeln!(out, " {}", s);
            }
            DiffLine::Removal(s) => {
                let _ = writeln!(out, "-{}", s);
            }
            DiffLine::Addition(s) => {
                let _ = writeln!(out, "+{}", s);
            }
        }
    }
}

/// Generate a multi-file unified diff.
pub fn generate_multi_file(diffs: Vec<FileDiff>) -> MultiFileDiff {
    MultiFileDiff { files: diffs }
}

/// Render a multi-file diff to string.
pub fn render_multi_file(multi: &MultiFileDiff) -> String {
    let mut out = String::new();
    for (i, file_diff) in multi.files.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        out.push_str(&render_file_diff(file_diff));
    }
    out
}

/// Generate a color-annotated unified diff.
pub fn generate_colored(old: &str, new: &str, old_name: &str, new_name: &str) -> String {
    let file_diff = generate_file_diff(old, new, old_name, new_name, &DiffConfig::default());
    render_colored(&file_diff)
}

fn render_colored(diff: &FileDiff) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "{}{}--- {}{}",
        AnsiColors::BOLD,
        AnsiColors::RED,
        diff.header.old_file,
        AnsiColors::RESET
    );
    let _ = writeln!(
        out,
        "{}{}+++ {}{}",
        AnsiColors::BOLD,
        AnsiColors::GREEN,
        diff.header.new_file,
        AnsiColors::RESET
    );

    for hunk in &diff.hunks {
        let _ = writeln!(
            out,
            "{}@@ -{},{} +{},{} @@{}",
            AnsiColors::CYAN,
            hunk.old_start,
            hunk.old_count,
            hunk.new_start,
            hunk.new_count,
            AnsiColors::RESET
        );
        for line in &hunk.lines {
            match line {
                DiffLine::Context(s) => {
                    let _ = writeln!(out, " {}", s);
                }
                DiffLine::Removal(s) => {
                    let _ = writeln!(out, "{}-{}{}", AnsiColors::RED, s, AnsiColors::RESET);
                }
                DiffLine::Addition(s) => {
                    let _ = writeln!(out, "{}+{}{}", AnsiColors::GREEN, s, AnsiColors::RESET);
                }
            }
        }
    }
    out
}

// ── Parsing ────────────────────────────────────────────────────────

/// Parse a unified diff string into a FileDiff.
pub fn parse(input: &str) -> Result<FileDiff, UnifiedDiffError> {
    let lines: Vec<&str> = input.lines().collect();
    if lines.len() < 2 {
        return Err(UnifiedDiffError::ParseError(
            "unified diff needs at least --- and +++ lines".into(),
        ));
    }

    // Find the --- and +++ header lines.
    let mut header_start = None;
    for (i, line) in lines.iter().enumerate() {
        if line.starts_with("--- ") && i + 1 < lines.len() && lines[i + 1].starts_with("+++ ") {
            header_start = Some(i);
            break;
        }
    }

    let hs = header_start.ok_or_else(|| {
        UnifiedDiffError::ParseError("no --- / +++ header found".into())
    })?;

    let old_file = lines[hs]
        .strip_prefix("--- ")
        .unwrap_or("")
        .split('\t')
        .next()
        .unwrap_or("")
        .to_string();
    let new_file = lines[hs + 1]
        .strip_prefix("+++ ")
        .unwrap_or("")
        .split('\t')
        .next()
        .unwrap_or("")
        .to_string();

    let header = FileHeader {
        old_file,
        new_file,
        old_timestamp: None,
        new_timestamp: None,
    };

    let mut hunks = Vec::new();
    let mut i = hs + 2;
    while i < lines.len() {
        let line = lines[i];
        if line.starts_with("@@ ") {
            let hunk = parse_hunk_header(line)?;
            let mut hunk_lines = Vec::new();
            i += 1;
            while i < lines.len() && !lines[i].starts_with("@@ ") {
                let l = lines[i];
                if l.starts_with(' ') {
                    hunk_lines.push(DiffLine::Context(l[1..].to_string()));
                } else if l.starts_with('+') {
                    hunk_lines.push(DiffLine::Addition(l[1..].to_string()));
                } else if l.starts_with('-') {
                    hunk_lines.push(DiffLine::Removal(l[1..].to_string()));
                } else if l.starts_with('\\') {
                    // "\ No newline at end of file" — skip.
                } else {
                    // Bare context line (no prefix = context).
                    hunk_lines.push(DiffLine::Context(l.to_string()));
                }
                i += 1;
            }
            hunks.push(Hunk {
                old_start: hunk.0,
                old_count: hunk.1,
                new_start: hunk.2,
                new_count: hunk.3,
                section_header: hunk.4,
                lines: hunk_lines,
            });
        } else {
            i += 1;
        }
    }

    Ok(FileDiff { header, hunks })
}

/// Parse a hunk header: @@ -old_start,old_count +new_start,new_count @@ [section]
fn parse_hunk_header(
    line: &str,
) -> Result<(usize, usize, usize, usize, Option<String>), UnifiedDiffError> {
    let trimmed = line.trim();
    if !trimmed.starts_with("@@ ") {
        return Err(UnifiedDiffError::InvalidHunkHeader(line.into()));
    }

    // Find the closing @@.
    let after_opening = &trimmed[3..];
    let closing = after_opening
        .find("@@")
        .ok_or_else(|| UnifiedDiffError::InvalidHunkHeader(line.into()))?;
    let range_part = after_opening[..closing].trim();
    let section_part = after_opening[closing + 2..].trim();
    let section = if section_part.is_empty() {
        None
    } else {
        Some(section_part.to_string())
    };

    // Parse -old_start,old_count +new_start,new_count
    let parts: Vec<&str> = range_part.split_whitespace().collect();
    if parts.len() < 2 {
        return Err(UnifiedDiffError::InvalidHunkHeader(line.into()));
    }

    let (old_start, old_count) = parse_range(parts[0], '-')?;
    let (new_start, new_count) = parse_range(parts[1], '+')?;

    Ok((old_start, old_count, new_start, new_count, section))
}

fn parse_range(s: &str, prefix: char) -> Result<(usize, usize), UnifiedDiffError> {
    let stripped = s
        .strip_prefix(prefix)
        .ok_or_else(|| UnifiedDiffError::InvalidHunkHeader(s.into()))?;
    if let Some((start_s, count_s)) = stripped.split_once(',') {
        let start: usize = start_s
            .parse()
            .map_err(|_| UnifiedDiffError::InvalidHunkHeader(s.into()))?;
        let count: usize = count_s
            .parse()
            .map_err(|_| UnifiedDiffError::InvalidHunkHeader(s.into()))?;
        Ok((start, count))
    } else {
        let start: usize = stripped
            .parse()
            .map_err(|_| UnifiedDiffError::InvalidHunkHeader(s.into()))?;
        Ok((start, 1))
    }
}

/// Parse a multi-file unified diff.
pub fn parse_multi_file(input: &str) -> Result<MultiFileDiff, UnifiedDiffError> {
    let lines: Vec<&str> = input.lines().collect();
    let mut files = Vec::new();
    let mut segments: Vec<(usize, usize)> = Vec::new();

    // Split on --- lines that are followed by +++ lines.
    let mut i = 0;
    while i < lines.len() {
        if lines[i].starts_with("--- ")
            && i + 1 < lines.len()
            && lines[i + 1].starts_with("+++ ")
        {
            if let Some(last) = segments.last_mut() {
                last.1 = i;
            }
            segments.push((i, lines.len()));
        }
        i += 1;
    }

    for (start, end) in segments {
        let segment: String = lines[start..end].join("\n");
        let file_diff = parse(&segment)?;
        files.push(file_diff);
    }

    Ok(MultiFileDiff { files })
}

/// Merge adjacent hunks within a file diff if they overlap or are close.
pub fn merge_adjacent_hunks(hunks: &[Hunk], gap: usize) -> Vec<Hunk> {
    if hunks.is_empty() {
        return Vec::new();
    }

    let mut merged: Vec<Hunk> = Vec::new();
    merged.push(hunks[0].clone());

    for hunk in &hunks[1..] {
        let last = merged.last().unwrap();
        let last_end = last.old_start + last.old_count;
        if hunk.old_start <= last_end + gap {
            // Merge: extend the last hunk.
            let last = merged.last_mut().unwrap();

            // Add context lines for the gap.
            let gap_lines = if hunk.old_start > last_end {
                hunk.old_start - last_end
            } else {
                0
            };
            for _ in 0..gap_lines {
                last.lines.push(DiffLine::Context(String::new()));
            }
            last.lines.extend(hunk.lines.clone());
            last.old_count = (hunk.old_start + hunk.old_count) - last.old_start;
            last.new_count = (hunk.new_start + hunk.new_count) - last.new_start;
        } else {
            merged.push(hunk.clone());
        }
    }

    merged
}

/// Create a file header with optional timestamps.
pub fn make_header(
    old_file: &str,
    new_file: &str,
    old_ts: Option<&str>,
    new_ts: Option<&str>,
) -> FileHeader {
    FileHeader {
        old_file: old_file.to_string(),
        new_file: new_file.to_string(),
        old_timestamp: old_ts.map(|s| s.to_string()),
        new_timestamp: new_ts.map(|s| s.to_string()),
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_basic_diff() {
        let diff = generate("a\nb\nc", "a\nx\nc", "old.txt", "new.txt");
        assert!(diff.contains("--- old.txt"));
        assert!(diff.contains("+++ new.txt"));
        assert!(diff.contains("@@"));
    }

    #[test]
    fn generate_no_changes() {
        let diff = generate("a\nb\nc", "a\nb\nc", "old.txt", "new.txt");
        // No hunks, just header.
        assert!(diff.contains("--- old.txt"));
        assert!(!diff.contains("@@"));
    }

    #[test]
    fn parse_roundtrip() {
        let original = generate("a\nb\nc\nd\ne", "a\nx\nc\nd\nf", "a.txt", "b.txt");
        let parsed = parse(&original).unwrap();
        assert_eq!(parsed.header.old_file, "a.txt");
        assert_eq!(parsed.header.new_file, "b.txt");
        assert!(!parsed.hunks.is_empty());
    }

    #[test]
    fn parse_hunk_lines() {
        let diff = generate("a\nb\nc", "a\nx\nc", "o.txt", "n.txt");
        let parsed = parse(&diff).unwrap();
        let hunk = &parsed.hunks[0];
        let has_removal = hunk.lines.iter().any(|l| matches!(l, DiffLine::Removal(_)));
        let has_addition = hunk.lines.iter().any(|l| matches!(l, DiffLine::Addition(_)));
        assert!(has_removal);
        assert!(has_addition);
    }

    #[test]
    fn context_lines_configuration() {
        let config = DiffConfig {
            context_lines: 0,
            merge_hunks: false,
        };
        let diff = generate_with_config(
            "a\nb\nc\nd\ne\nf\ng",
            "a\nb\nx\nd\ne\ny\ng",
            "o.txt",
            "n.txt",
            &config,
        );
        // With zero context, hunks should be tighter.
        let parsed = parse(&diff).unwrap();
        assert!(parsed.hunks.len() >= 1);
    }

    #[test]
    fn context_lines_large() {
        let config = DiffConfig {
            context_lines: 10,
            merge_hunks: true,
        };
        let diff = generate_with_config("a\nb\nc", "a\nx\nc", "o", "n", &config);
        assert!(diff.contains("@@"));
    }

    #[test]
    fn colored_output() {
        let colored = generate_colored("a\nb", "a\nc", "old", "new");
        assert!(colored.contains("\x1b[31m")); // red
        assert!(colored.contains("\x1b[32m")); // green
        assert!(colored.contains("\x1b[0m")); // reset
    }

    #[test]
    fn multi_file_diff() {
        let d1 = generate_file_diff("a\nb", "a\nc", "f1.txt", "f1.txt", &DiffConfig::default());
        let d2 = generate_file_diff("x\ny", "x\nz", "f2.txt", "f2.txt", &DiffConfig::default());
        let multi = generate_multi_file(vec![d1, d2]);
        let rendered = render_multi_file(&multi);
        assert!(rendered.contains("f1.txt"));
        assert!(rendered.contains("f2.txt"));
    }

    #[test]
    fn merge_adjacent_hunks_basic() {
        let h1 = Hunk {
            old_start: 1,
            old_count: 3,
            new_start: 1,
            new_count: 3,
            section_header: None,
            lines: vec![DiffLine::Context("a".into())],
        };
        let h2 = Hunk {
            old_start: 5,
            old_count: 3,
            new_start: 5,
            new_count: 3,
            section_header: None,
            lines: vec![DiffLine::Context("b".into())],
        };
        let merged = merge_adjacent_hunks(&[h1, h2], 2);
        assert_eq!(merged.len(), 1);
    }

    #[test]
    fn merge_hunks_far_apart() {
        let h1 = Hunk {
            old_start: 1,
            old_count: 2,
            new_start: 1,
            new_count: 2,
            section_header: None,
            lines: vec![DiffLine::Context("a".into())],
        };
        let h2 = Hunk {
            old_start: 100,
            old_count: 2,
            new_start: 100,
            new_count: 2,
            section_header: None,
            lines: vec![DiffLine::Context("b".into())],
        };
        let merged = merge_adjacent_hunks(&[h1, h2], 1);
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn parse_error_on_empty() {
        let result = parse("");
        assert!(result.is_err());
    }

    #[test]
    fn parse_error_no_header() {
        let result = parse("just some text\nwithout headers");
        assert!(result.is_err());
    }

    #[test]
    fn file_header_construction() {
        let h = make_header("a.rs", "b.rs", Some("2026-01-01"), None);
        assert_eq!(h.old_file, "a.rs");
        assert_eq!(h.new_file, "b.rs");
        assert_eq!(h.old_timestamp, Some("2026-01-01".into()));
        assert!(h.new_timestamp.is_none());
    }

    #[test]
    fn diff_line_types() {
        let ctx = DiffLine::Context("hello".into());
        let add = DiffLine::Addition("world".into());
        let rem = DiffLine::Removal("old".into());
        assert_eq!(ctx, DiffLine::Context("hello".into()));
        assert_eq!(add, DiffLine::Addition("world".into()));
        assert_eq!(rem, DiffLine::Removal("old".into()));
    }

    #[test]
    fn empty_files_diff() {
        let diff = generate("", "hello\nworld", "empty.txt", "new.txt");
        assert!(diff.contains("@@"));
        let parsed = parse(&diff).unwrap();
        assert!(!parsed.hunks.is_empty());
    }

    #[test]
    fn deletion_of_all_content() {
        let diff = generate("hello\nworld", "", "old.txt", "empty.txt");
        assert!(diff.contains("@@"));
    }

    #[test]
    fn multi_file_parse_roundtrip() {
        let d1 = generate_file_diff("a", "b", "f1.txt", "f1.txt", &DiffConfig::default());
        let d2 = generate_file_diff("c", "d", "f2.txt", "f2.txt", &DiffConfig::default());
        let multi = generate_multi_file(vec![d1, d2]);
        let rendered = render_multi_file(&multi);
        let parsed = parse_multi_file(&rendered).unwrap();
        assert_eq!(parsed.files.len(), 2);
    }

    #[test]
    fn error_display() {
        let e = UnifiedDiffError::ParseError("bad".into());
        assert!(e.to_string().contains("bad"));
    }

    #[test]
    fn hunk_with_section_header() {
        let h = Hunk {
            old_start: 1,
            old_count: 1,
            new_start: 1,
            new_count: 1,
            section_header: Some("fn main()".into()),
            lines: vec![DiffLine::Context("code".into())],
        };
        let mut out = String::new();
        render_hunk(&h, &mut out);
        assert!(out.contains("fn main()"));
    }

    #[test]
    fn default_config() {
        let cfg = DiffConfig::default();
        assert_eq!(cfg.context_lines, 3);
        assert!(cfg.merge_hunks);
    }
}
