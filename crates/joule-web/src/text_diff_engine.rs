//! Text diff engine — Myers diff algorithm, unified and side-by-side output,
//! edit scripts, patch generation, and diff statistics.
//!
//! Pure-Rust replacement for jsdiff, diff-match-patch, and Google's diff-match-patch.

use std::fmt;
use std::fmt::Write as FmtWrite;

// ── Types ───────────────────────────────────────────────────────

/// A single edit operation in a diff.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditOp {
    /// Unchanged text.
    Equal(String),
    /// Text inserted in the new version.
    Insert(String),
    /// Text deleted from the old version.
    Delete(String),
}

/// Summary statistics for a diff.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DiffStats {
    pub insertions: usize,
    pub deletions: usize,
    pub unchanged: usize,
    pub total_old_lines: usize,
    pub total_new_lines: usize,
}

/// A hunk in a unified diff patch.
#[derive(Debug, Clone)]
pub struct Hunk {
    pub old_start: usize,
    pub old_count: usize,
    pub new_start: usize,
    pub new_count: usize,
    pub ops: Vec<EditOp>,
}

/// A complete patch with metadata and hunks.
#[derive(Debug, Clone)]
pub struct Patch {
    pub old_name: String,
    pub new_name: String,
    pub hunks: Vec<Hunk>,
}

impl fmt::Display for Patch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "--- {}", self.old_name)?;
        writeln!(f, "+++ {}", self.new_name)?;
        for hunk in &self.hunks {
            writeln!(
                f,
                "@@ -{},{} +{},{} @@",
                hunk.old_start, hunk.old_count, hunk.new_start, hunk.new_count
            )?;
            for op in &hunk.ops {
                match op {
                    EditOp::Equal(s) => writeln!(f, " {s}")?,
                    EditOp::Insert(s) => writeln!(f, "+{s}")?,
                    EditOp::Delete(s) => writeln!(f, "-{s}")?,
                }
            }
        }
        Ok(())
    }
}

/// A single row in a side-by-side diff.
#[derive(Debug, Clone)]
pub struct SideBySideRow {
    pub left_line_num: Option<usize>,
    pub left_content: String,
    pub right_line_num: Option<usize>,
    pub right_content: String,
    pub change_type: ChangeType,
}

/// Type of change for a side-by-side row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeType {
    Equal,
    Modified,
    Added,
    Deleted,
}

/// An edit script — a sequence of commands to transform old into new.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditCommand {
    /// Keep n lines from old.
    Keep(usize),
    /// Delete n lines from old.
    Delete(usize),
    /// Insert the given lines.
    Insert(Vec<String>),
}

// ── Myers diff algorithm ────────────────────────────────────────

/// Compute the diff between two sequences using the Myers algorithm.
/// Returns a minimal edit script of Equal/Insert/Delete operations.
fn myers_diff<T: PartialEq + Clone>(old: &[T], new: &[T]) -> Vec<(DiffTag, usize, usize)> {
    let n = old.len();
    let m = new.len();
    let max_d = n + m;

    if max_d == 0 {
        return Vec::new();
    }

    // For very large inputs, fall back to LCS-based approach.
    if max_d > 10000 {
        return lcs_diff(old, new);
    }

    let offset = max_d;
    let v_size = 2 * max_d + 1;
    let mut v = vec![0i64; v_size];
    let mut trace: Vec<Vec<i64>> = Vec::new();

    let mut found_d = None;
    for d in 0..=max_d {
        let old_v = v.clone();
        trace.push(old_v);

        let d_i = d as i64;
        let mut k = -d_i;
        while k <= d_i {
            let k_idx = (k + offset as i64) as usize;

            let mut x = if k == -d_i
                || (k != d_i
                    && v[(k - 1 + offset as i64) as usize] < v[(k + 1 + offset as i64) as usize])
            {
                v[(k + 1 + offset as i64) as usize]
            } else {
                v[(k - 1 + offset as i64) as usize] + 1
            };

            let mut y = x - k;

            // Follow diagonal (equal elements).
            while (x as usize) < n && (y as usize) < m && old[x as usize] == new[y as usize] {
                x += 1;
                y += 1;
            }

            v[k_idx] = x;

            if (x as usize) >= n && (y as usize) >= m {
                found_d = Some(d);
                break;
            }

            k += 2;
        }

        if found_d.is_some() {
            break;
        }
    }

    let d = found_d.unwrap_or(max_d);

    // Backtrack to recover the edit path.
    let mut result = Vec::new();
    let mut x = n as i64;
    let mut y = m as i64;

    let mut dd = d;
    loop {
        let k = x - y;

        if dd == 0 {
            // At step 0, just emit remaining diagonal (equal) moves.
            while x > 0 && y > 0 {
                x -= 1;
                y -= 1;
                result.push((DiffTag::Equal, x as usize, y as usize));
            }
            break;
        }

        let prev_v = &trace[dd];

        let dd_i = dd as i64;
        let prev_k = if k == -dd_i
            || (k != dd_i
                && prev_v[(k - 1 + offset as i64) as usize]
                    < prev_v[(k + 1 + offset as i64) as usize])
        {
            k + 1
        } else {
            k - 1
        };

        let prev_x = prev_v[(prev_k + offset as i64) as usize];
        let prev_y = prev_x - prev_k;

        // Diagonal (equal).
        while x > prev_x && y > prev_y {
            x -= 1;
            y -= 1;
            result.push((DiffTag::Equal, x as usize, y as usize));
        }

        if x > prev_x {
            x -= 1;
            result.push((DiffTag::Delete, x as usize, 0));
        } else if y > prev_y {
            y -= 1;
            result.push((DiffTag::Insert, 0, y as usize));
        }

        dd -= 1;
    }

    result.reverse();
    result
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DiffTag {
    Equal,
    Insert,
    Delete,
}

/// Fallback LCS-based diff for very large inputs.
fn lcs_diff<T: PartialEq + Clone>(old: &[T], new: &[T]) -> Vec<(DiffTag, usize, usize)> {
    let n = old.len();
    let m = new.len();

    // Build LCS table.
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

    // Backtrack.
    let mut result = Vec::new();
    let mut i = n;
    let mut j = m;

    while i > 0 || j > 0 {
        if i > 0 && j > 0 && old[i - 1] == new[j - 1] {
            result.push((DiffTag::Equal, i - 1, j - 1));
            i -= 1;
            j -= 1;
        } else if j > 0 && (i == 0 || dp[i][j - 1] >= dp[i - 1][j]) {
            result.push((DiffTag::Insert, 0, j - 1));
            j -= 1;
        } else {
            result.push((DiffTag::Delete, i - 1, 0));
            i -= 1;
        }
    }

    result.reverse();
    result
}

// ── Public API ──────────────────────────────────────────────────

/// Compute a line-level diff between two strings.
pub fn diff_lines(old: &str, new: &str) -> Vec<EditOp> {
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

    let tags = myers_diff(&old_lines, &new_lines);
    let mut ops = Vec::new();

    for (tag, old_idx, new_idx) in &tags {
        match tag {
            DiffTag::Equal => ops.push(EditOp::Equal(old_lines[*old_idx].to_string())),
            DiffTag::Insert => ops.push(EditOp::Insert(new_lines[*new_idx].to_string())),
            DiffTag::Delete => ops.push(EditOp::Delete(old_lines[*old_idx].to_string())),
        }
    }

    ops
}

/// Compute a word-level diff between two strings.
pub fn diff_words(old: &str, new: &str) -> Vec<EditOp> {
    let old_words: Vec<&str> = old.split_whitespace().collect();
    let new_words: Vec<&str> = new.split_whitespace().collect();

    let tags = myers_diff(&old_words, &new_words);
    let mut ops = Vec::new();

    for (tag, old_idx, new_idx) in &tags {
        match tag {
            DiffTag::Equal => ops.push(EditOp::Equal(old_words[*old_idx].to_string())),
            DiffTag::Insert => ops.push(EditOp::Insert(new_words[*new_idx].to_string())),
            DiffTag::Delete => ops.push(EditOp::Delete(old_words[*old_idx].to_string())),
        }
    }

    ops
}

/// Compute a character-level diff between two strings.
pub fn diff_chars(old: &str, new: &str) -> Vec<EditOp> {
    let old_chars: Vec<char> = old.chars().collect();
    let new_chars: Vec<char> = new.chars().collect();

    let tags = myers_diff(&old_chars, &new_chars);
    let mut ops = Vec::new();

    for (tag, old_idx, new_idx) in &tags {
        match tag {
            DiffTag::Equal => ops.push(EditOp::Equal(old_chars[*old_idx].to_string())),
            DiffTag::Insert => ops.push(EditOp::Insert(new_chars[*new_idx].to_string())),
            DiffTag::Delete => ops.push(EditOp::Delete(old_chars[*old_idx].to_string())),
        }
    }

    ops
}

/// Calculate statistics from a list of edit operations.
pub fn stats(ops: &[EditOp]) -> DiffStats {
    let mut s = DiffStats::default();
    for op in ops {
        match op {
            EditOp::Equal(_) => {
                s.unchanged += 1;
                s.total_old_lines += 1;
                s.total_new_lines += 1;
            }
            EditOp::Insert(_) => {
                s.insertions += 1;
                s.total_new_lines += 1;
            }
            EditOp::Delete(_) => {
                s.deletions += 1;
                s.total_old_lines += 1;
            }
        }
    }
    s
}

/// Generate a unified diff string.
pub fn unified_diff(old: &str, new: &str, old_name: &str, new_name: &str, context: usize) -> String {
    let patch = create_patch(old, new, old_name, new_name, context);
    patch.to_string()
}

/// Create a patch from two strings.
pub fn create_patch(
    old: &str,
    new: &str,
    old_name: &str,
    new_name: &str,
    context: usize,
) -> Patch {
    let ops = diff_lines(old, new);
    let hunks = build_hunks(&ops, context);

    Patch {
        old_name: old_name.to_string(),
        new_name: new_name.to_string(),
        hunks,
    }
}

/// Build hunks from a flat list of edit operations, grouping changes
/// with `context` lines of surrounding equal content.
fn build_hunks(ops: &[EditOp], context: usize) -> Vec<Hunk> {
    if ops.is_empty() {
        return Vec::new();
    }

    // Find ranges of changes.
    let mut change_ranges: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < ops.len() {
        if matches!(ops[i], EditOp::Insert(_) | EditOp::Delete(_)) {
            let start = i;
            while i < ops.len() && !matches!(ops[i], EditOp::Equal(_)) {
                i += 1;
            }
            change_ranges.push((start, i));
        } else {
            i += 1;
        }
    }

    if change_ranges.is_empty() {
        return Vec::new();
    }

    // Merge close ranges into hunks.
    let mut merged: Vec<(usize, usize)> = Vec::new();
    let mut current = change_ranges[0];

    for range in &change_ranges[1..] {
        if range.0 <= current.1 + 2 * context {
            current.1 = range.1;
        } else {
            merged.push(current);
            current = *range;
        }
    }
    merged.push(current);

    // Build hunks.
    let mut hunks = Vec::new();
    for (start, end) in merged {
        let ctx_start = start.saturating_sub(context);
        let ctx_end = (end + context).min(ops.len());

        let hunk_ops: Vec<EditOp> = ops[ctx_start..ctx_end].to_vec();

        let mut old_count = 0usize;
        let mut new_count = 0usize;
        for op in &hunk_ops {
            match op {
                EditOp::Equal(_) => {
                    old_count += 1;
                    new_count += 1;
                }
                EditOp::Delete(_) => old_count += 1,
                EditOp::Insert(_) => new_count += 1,
            }
        }

        // Calculate old_start, new_start.
        let mut old_start = 1usize;
        let mut new_start = 1usize;
        for op in &ops[..ctx_start] {
            match op {
                EditOp::Equal(_) => {
                    old_start += 1;
                    new_start += 1;
                }
                EditOp::Delete(_) => old_start += 1,
                EditOp::Insert(_) => new_start += 1,
            }
        }

        hunks.push(Hunk {
            old_start,
            old_count,
            new_start,
            new_count,
            ops: hunk_ops,
        });
    }

    hunks
}

/// Generate a side-by-side diff view.
pub fn side_by_side(old: &str, new: &str) -> Vec<SideBySideRow> {
    let ops = diff_lines(old, new);
    let mut rows = Vec::new();
    let mut left_num = 1usize;
    let mut right_num = 1usize;

    let mut i = 0;
    while i < ops.len() {
        match &ops[i] {
            EditOp::Equal(s) => {
                rows.push(SideBySideRow {
                    left_line_num: Some(left_num),
                    left_content: s.clone(),
                    right_line_num: Some(right_num),
                    right_content: s.clone(),
                    change_type: ChangeType::Equal,
                });
                left_num += 1;
                right_num += 1;
                i += 1;
            }
            EditOp::Delete(s) => {
                // Check if next is Insert (modification).
                if i + 1 < ops.len() {
                    if let EditOp::Insert(new_s) = &ops[i + 1] {
                        rows.push(SideBySideRow {
                            left_line_num: Some(left_num),
                            left_content: s.clone(),
                            right_line_num: Some(right_num),
                            right_content: new_s.clone(),
                            change_type: ChangeType::Modified,
                        });
                        left_num += 1;
                        right_num += 1;
                        i += 2;
                        continue;
                    }
                }
                rows.push(SideBySideRow {
                    left_line_num: Some(left_num),
                    left_content: s.clone(),
                    right_line_num: None,
                    right_content: String::new(),
                    change_type: ChangeType::Deleted,
                });
                left_num += 1;
                i += 1;
            }
            EditOp::Insert(s) => {
                rows.push(SideBySideRow {
                    left_line_num: None,
                    left_content: String::new(),
                    right_line_num: Some(right_num),
                    right_content: s.clone(),
                    change_type: ChangeType::Added,
                });
                right_num += 1;
                i += 1;
            }
        }
    }

    rows
}

/// Generate an edit script from the diff.
pub fn edit_script(old: &str, new: &str) -> Vec<EditCommand> {
    let ops = diff_lines(old, new);
    let mut commands = Vec::new();
    let mut i = 0;

    while i < ops.len() {
        match &ops[i] {
            EditOp::Equal(_) => {
                let mut count = 0;
                while i < ops.len() && matches!(ops[i], EditOp::Equal(_)) {
                    count += 1;
                    i += 1;
                }
                commands.push(EditCommand::Keep(count));
            }
            EditOp::Delete(_) => {
                let mut count = 0;
                while i < ops.len() && matches!(ops[i], EditOp::Delete(_)) {
                    count += 1;
                    i += 1;
                }
                commands.push(EditCommand::Delete(count));
            }
            EditOp::Insert(s) => {
                let mut lines = vec![s.clone()];
                i += 1;
                while i < ops.len() {
                    if let EditOp::Insert(s2) = &ops[i] {
                        lines.push(s2.clone());
                        i += 1;
                    } else {
                        break;
                    }
                }
                commands.push(EditCommand::Insert(lines));
            }
        }
    }

    commands
}

/// Apply a patch to the old text, returning the new text.
pub fn apply_patch(old: &str, patch: &Patch) -> Result<String, String> {
    let old_lines: Vec<&str> = if old.is_empty() {
        Vec::new()
    } else {
        old.lines().collect()
    };

    let mut result = Vec::new();
    let mut old_idx = 0;

    for hunk in &patch.hunks {
        // Copy lines before this hunk.
        let hunk_old_start = hunk.old_start.saturating_sub(1);
        while old_idx < hunk_old_start && old_idx < old_lines.len() {
            result.push(old_lines[old_idx].to_string());
            old_idx += 1;
        }

        // Apply hunk operations.
        for op in &hunk.ops {
            match op {
                EditOp::Equal(_s) => {
                    if old_idx < old_lines.len() {
                        result.push(old_lines[old_idx].to_string());
                        old_idx += 1;
                    }
                }
                EditOp::Delete(_) => {
                    old_idx += 1; // Skip the deleted line.
                }
                EditOp::Insert(s) => {
                    result.push(s.clone());
                }
            }
        }
    }

    // Copy remaining lines.
    while old_idx < old_lines.len() {
        result.push(old_lines[old_idx].to_string());
        old_idx += 1;
    }

    Ok(result.join("\n"))
}

/// Format diff operations as a colorized string (ANSI escape codes).
pub fn format_colored(ops: &[EditOp]) -> String {
    let mut out = String::new();
    for op in ops {
        match op {
            EditOp::Equal(s) => {
                let _ = writeln!(out, " {s}");
            }
            EditOp::Insert(s) => {
                let _ = writeln!(out, "\x1b[32m+{s}\x1b[0m");
            }
            EditOp::Delete(s) => {
                let _ = writeln!(out, "\x1b[31m-{s}\x1b[0m");
            }
        }
    }
    out
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_strings() {
        let ops = diff_lines("hello\nworld", "hello\nworld");
        assert!(ops.iter().all(|op| matches!(op, EditOp::Equal(_))));
    }

    #[test]
    fn test_empty_to_something() {
        let ops = diff_lines("", "hello\nworld");
        assert!(ops.iter().all(|op| matches!(op, EditOp::Insert(_))));
        assert_eq!(ops.len(), 2);
    }

    #[test]
    fn test_something_to_empty() {
        let ops = diff_lines("hello\nworld", "");
        assert!(ops.iter().all(|op| matches!(op, EditOp::Delete(_))));
        assert_eq!(ops.len(), 2);
    }

    #[test]
    fn test_single_line_change() {
        let ops = diff_lines("hello", "world");
        let s = stats(&ops);
        assert_eq!(s.deletions, 1);
        assert_eq!(s.insertions, 1);
        assert_eq!(s.unchanged, 0);
    }

    #[test]
    fn test_insertion_in_middle() {
        let old = "line1\nline3";
        let new = "line1\nline2\nline3";
        let ops = diff_lines(old, new);
        let s = stats(&ops);
        assert_eq!(s.insertions, 1);
        assert_eq!(s.unchanged, 2);
    }

    #[test]
    fn test_deletion_in_middle() {
        let old = "line1\nline2\nline3";
        let new = "line1\nline3";
        let ops = diff_lines(old, new);
        let s = stats(&ops);
        assert_eq!(s.deletions, 1);
        assert_eq!(s.unchanged, 2);
    }

    #[test]
    fn test_word_diff() {
        let ops = diff_words("the quick brown fox", "the slow brown cat");
        let s = stats(&ops);
        assert!(s.insertions >= 1);
        assert!(s.deletions >= 1);
    }

    #[test]
    fn test_char_diff() {
        let ops = diff_chars("abc", "adc");
        let s = stats(&ops);
        assert_eq!(s.unchanged, 2); // 'a' and 'c'
    }

    #[test]
    fn test_unified_diff_format() {
        let old = "line1\nline2\nline3";
        let new = "line1\nmodified\nline3";
        let diff = unified_diff(old, new, "a.txt", "b.txt", 3);
        assert!(diff.contains("--- a.txt"));
        assert!(diff.contains("+++ b.txt"));
        assert!(diff.contains("@@"));
    }

    #[test]
    fn test_patch_creation() {
        let old = "a\nb\nc";
        let new = "a\nB\nc";
        let patch = create_patch(old, new, "old.txt", "new.txt", 1);
        assert_eq!(patch.old_name, "old.txt");
        assert_eq!(patch.new_name, "new.txt");
        assert!(!patch.hunks.is_empty());
    }

    #[test]
    fn test_side_by_side() {
        let old = "line1\nline2\nline3";
        let new = "line1\nmodified\nline3";
        let rows = side_by_side(old, new);
        assert!(!rows.is_empty());
        // First and last should be equal.
        assert_eq!(rows[0].change_type, ChangeType::Equal);
        assert_eq!(rows.last().unwrap().change_type, ChangeType::Equal);
    }

    #[test]
    fn test_side_by_side_added() {
        let old = "a\nc";
        let new = "a\nb\nc";
        let rows = side_by_side(old, new);
        assert!(rows.iter().any(|r| r.change_type == ChangeType::Added));
    }

    #[test]
    fn test_edit_script() {
        let old = "a\nb\nc";
        let new = "a\nc";
        let cmds = edit_script(old, new);
        assert!(cmds.contains(&EditCommand::Keep(1)));
        assert!(cmds.contains(&EditCommand::Delete(1)));
    }

    #[test]
    fn test_edit_script_insert() {
        let old = "a\nc";
        let new = "a\nb\nc";
        let cmds = edit_script(old, new);
        assert!(cmds.iter().any(|c| matches!(c, EditCommand::Insert(_))));
    }

    #[test]
    fn test_stats_totals() {
        let ops = diff_lines("a\nb\nc", "a\nB\nc\nd");
        let s = stats(&ops);
        assert_eq!(s.total_old_lines, s.unchanged + s.deletions);
        assert_eq!(s.total_new_lines, s.unchanged + s.insertions);
    }

    #[test]
    fn test_both_empty() {
        let ops = diff_lines("", "");
        assert!(ops.is_empty());
    }

    #[test]
    fn test_colored_output() {
        let ops = vec![
            EditOp::Equal("same".to_string()),
            EditOp::Delete("old".to_string()),
            EditOp::Insert("new".to_string()),
        ];
        let colored = format_colored(&ops);
        assert!(colored.contains(" same"));
        assert!(colored.contains("\x1b[31m-old\x1b[0m"));
        assert!(colored.contains("\x1b[32m+new\x1b[0m"));
    }

    #[test]
    fn test_patch_display() {
        let patch = create_patch("a\nb\nc", "a\nB\nc", "f1", "f2", 1);
        let s = patch.to_string();
        assert!(s.contains("--- f1"));
        assert!(s.contains("+++ f2"));
    }

    #[test]
    fn test_apply_patch() {
        let old = "line1\nline2\nline3";
        let new = "line1\nmodified\nline3";
        let patch = create_patch(old, new, "a", "b", 3);
        let result = apply_patch(old, &patch).unwrap();
        assert_eq!(result, new);
    }

    #[test]
    fn test_large_diff() {
        let old: Vec<String> = (0..100).map(|i| format!("line {i}")).collect();
        let mut new = old.clone();
        new[50] = "CHANGED".to_string();
        let old_str = old.join("\n");
        let new_str = new.join("\n");
        let ops = diff_lines(&old_str, &new_str);
        let s = stats(&ops);
        assert_eq!(s.deletions, 1);
        assert_eq!(s.insertions, 1);
        assert_eq!(s.unchanged, 99);
    }

    #[test]
    fn test_multiline_insert_patch() {
        let old = "a\nc";
        let new = "a\nb1\nb2\nc";
        let patch = create_patch(old, new, "old", "new", 1);
        assert!(!patch.hunks.is_empty());
        let result = apply_patch(old, &patch).unwrap();
        assert_eq!(result, new);
    }

    #[test]
    fn test_word_diff_equal() {
        let ops = diff_words("hello world", "hello world");
        assert!(ops.iter().all(|op| matches!(op, EditOp::Equal(_))));
    }
}
