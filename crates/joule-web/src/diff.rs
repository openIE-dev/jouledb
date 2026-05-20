//! Text diffing.
//!
//! Replaces jsdiff and diff-match-patch with a pure Rust implementation
//! of the Myers diff algorithm at line, word, and character granularity.

use std::fmt::Write as FmtWrite;

// ── Types ──────────────────────────────────────────────────────────

/// A single diff operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiffOp {
    /// Text present in both old and new.
    Equal(String),
    /// Text added in new.
    Insert(String),
    /// Text removed from old.
    Delete(String),
}

/// Summary statistics for a set of diff operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiffStats {
    pub insertions: usize,
    pub deletions: usize,
    pub unchanged: usize,
}

/// A patch consisting of hunks.
#[derive(Debug, Clone)]
pub struct Patch {
    pub hunks: Vec<Hunk>,
}

/// A single hunk in a patch.
#[derive(Debug, Clone)]
pub struct Hunk {
    pub old_start: usize,
    pub old_count: usize,
    pub new_start: usize,
    pub new_count: usize,
    pub lines: Vec<DiffOp>,
}

// ── Core diff (LCS-based) ──────────────────────────────────────────

/// Compute the diff between two sequences using an LCS (longest common
/// subsequence) approach. Produces a minimal edit script of Equal/Insert/Delete.
fn diff_sequences<T: PartialEq + Clone + AsRef<str>>(old: &[T], new: &[T]) -> Vec<DiffOp> {
    let n = old.len();
    let m = new.len();

    if n == 0 && m == 0 {
        return Vec::new();
    }
    if n == 0 {
        return new.iter().map(|s| DiffOp::Insert(s.as_ref().to_string())).collect();
    }
    if m == 0 {
        return old.iter().map(|s| DiffOp::Delete(s.as_ref().to_string())).collect();
    }

    // Build LCS table
    let mut dp = vec![vec![0u32; m + 1]; n + 1];
    for i in 1..=n {
        for j in 1..=m {
            if old[i - 1].as_ref() == new[j - 1].as_ref() {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }

    // Backtrack to produce edit script
    let mut ops = Vec::new();
    let mut i = n;
    let mut j = m;

    while i > 0 || j > 0 {
        if i > 0 && j > 0 && old[i - 1].as_ref() == new[j - 1].as_ref() {
            ops.push(DiffOp::Equal(old[i - 1].as_ref().to_string()));
            i -= 1;
            j -= 1;
        } else if j > 0 && (i == 0 || dp[i][j - 1] >= dp[i - 1][j]) {
            ops.push(DiffOp::Insert(new[j - 1].as_ref().to_string()));
            j -= 1;
        } else {
            ops.push(DiffOp::Delete(old[i - 1].as_ref().to_string()));
            i -= 1;
        }
    }

    ops.reverse();
    ops
}

// ── Public diff functions ──────────────────────────────────────────

/// Line-level diff using Myers algorithm.
pub fn diff_lines(old: &str, new: &str) -> Vec<DiffOp> {
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
    diff_sequences(&old_lines, &new_lines)
}

/// Character-level diff.
pub fn diff_chars(old: &str, new: &str) -> Vec<DiffOp> {
    let old_chars: Vec<String> = old.chars().map(|c| c.to_string()).collect();
    let new_chars: Vec<String> = new.chars().map(|c| c.to_string()).collect();
    diff_sequences(&old_chars, &new_chars)
}

/// Word-level diff.
pub fn diff_words(old: &str, new: &str) -> Vec<DiffOp> {
    let old_words: Vec<&str> = old.split_whitespace().collect();
    let new_words: Vec<&str> = new.split_whitespace().collect();
    diff_sequences(&old_words, &new_words)
}

/// Compute statistics from a diff.
pub fn stats(ops: &[DiffOp]) -> DiffStats {
    let mut s = DiffStats {
        insertions: 0,
        deletions: 0,
        unchanged: 0,
    };
    for op in ops {
        match op {
            DiffOp::Equal(_) => s.unchanged += 1,
            DiffOp::Insert(_) => s.insertions += 1,
            DiffOp::Delete(_) => s.deletions += 1,
        }
    }
    s
}

/// Apply a diff to reconstruct the new text (line-level).
pub fn apply_patch(original: &str, ops: &[DiffOp]) -> String {
    let mut result = Vec::new();
    for op in ops {
        match op {
            DiffOp::Equal(s) | DiffOp::Insert(s) => {
                result.push(s.as_str());
            }
            DiffOp::Delete(_) => {}
        }
    }
    result.join("\n")
}

/// Generate a unified diff string.
pub fn unified_diff(old: &str, new: &str, context_lines: usize) -> String {
    let ops = diff_lines(old, new);

    if ops.is_empty() {
        return String::new();
    }

    // Build indexed ops with line numbers
    let mut old_line = 1usize;
    let mut new_line = 1usize;
    let mut indexed: Vec<(DiffOp, usize, usize)> = Vec::new();

    for op in &ops {
        let ol = old_line;
        let nl = new_line;
        match op {
            DiffOp::Equal(_) => {
                old_line += 1;
                new_line += 1;
            }
            DiffOp::Insert(_) => {
                new_line += 1;
            }
            DiffOp::Delete(_) => {
                old_line += 1;
            }
        }
        indexed.push((op.clone(), ol, nl));
    }

    // Find change positions
    let change_positions: Vec<usize> = indexed
        .iter()
        .enumerate()
        .filter(|(_, (op, _, _))| !matches!(op, DiffOp::Equal(_)))
        .map(|(i, _)| i)
        .collect();

    if change_positions.is_empty() {
        return String::new();
    }

    // Group into hunks based on context
    let mut hunks: Vec<(usize, usize)> = Vec::new();
    let mut start = change_positions[0].saturating_sub(context_lines);
    let mut end = (change_positions[0] + context_lines + 1).min(indexed.len());

    for &pos in &change_positions[1..] {
        let new_start = pos.saturating_sub(context_lines);
        let new_end = (pos + context_lines + 1).min(indexed.len());
        if new_start <= end {
            end = new_end;
        } else {
            hunks.push((start, end));
            start = new_start;
            end = new_end;
        }
    }
    hunks.push((start, end));

    let mut output = String::new();
    output.push_str("--- a\n+++ b\n");

    for (hunk_start, hunk_end) in &hunks {
        let slice = &indexed[*hunk_start..*hunk_end];
        if slice.is_empty() {
            continue;
        }

        let old_start = slice[0].1;
        let new_start = slice[0].2;
        let mut old_count = 0;
        let mut new_count = 0;

        for (op, _, _) in slice {
            match op {
                DiffOp::Equal(_) => {
                    old_count += 1;
                    new_count += 1;
                }
                DiffOp::Delete(_) => old_count += 1,
                DiffOp::Insert(_) => new_count += 1,
            }
        }

        let _ = writeln!(
            output,
            "@@ -{},{} +{},{} @@",
            old_start, old_count, new_start, new_count
        );

        for (op, _, _) in slice {
            match op {
                DiffOp::Equal(s) => {
                    let _ = writeln!(output, " {}", s);
                }
                DiffOp::Delete(s) => {
                    let _ = writeln!(output, "-{}", s);
                }
                DiffOp::Insert(s) => {
                    let _ = writeln!(output, "+{}", s);
                }
            }
        }
    }

    output
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_strings_no_changes() {
        let ops = diff_lines("hello\nworld", "hello\nworld");
        assert!(ops.iter().all(|op| matches!(op, DiffOp::Equal(_))));
        assert_eq!(ops.len(), 2);
    }

    #[test]
    fn insert_only() {
        let ops = diff_lines("a", "a\nb");
        let s = stats(&ops);
        assert_eq!(s.insertions, 1);
        assert_eq!(s.deletions, 0);
        assert_eq!(s.unchanged, 1);
    }

    #[test]
    fn delete_only() {
        let ops = diff_lines("a\nb", "a");
        let s = stats(&ops);
        assert_eq!(s.deletions, 1);
        assert_eq!(s.unchanged, 1);
    }

    #[test]
    fn mixed_changes() {
        let ops = diff_lines("a\nb\nc", "a\nx\nc");
        let s = stats(&ops);
        assert!(s.insertions > 0 || s.deletions > 0);
        assert!(s.unchanged > 0);
    }

    #[test]
    fn unified_diff_format() {
        let ud = unified_diff("a\nb\nc", "a\nx\nc", 1);
        assert!(ud.contains("---"));
        assert!(ud.contains("+++"));
        assert!(ud.contains("@@"));
    }

    #[test]
    fn apply_patch_roundtrip() {
        let old = "line1\nline2\nline3";
        let new = "line1\nmodified\nline3";
        let ops = diff_lines(old, new);
        let result = apply_patch(old, &ops);
        assert_eq!(result, new);
    }

    #[test]
    fn word_diff() {
        let ops = diff_words("the quick fox", "the slow fox");
        let s = stats(&ops);
        assert_eq!(s.unchanged, 2); // "the" and "fox"
        assert_eq!(s.deletions, 1); // "quick"
        assert_eq!(s.insertions, 1); // "slow"
    }

    #[test]
    fn character_diff() {
        let ops = diff_chars("abc", "adc");
        let s = stats(&ops);
        assert_eq!(s.unchanged, 2); // 'a' and 'c'
        assert_eq!(s.deletions, 1); // 'b'
        assert_eq!(s.insertions, 1); // 'd'
    }

    #[test]
    fn stats_correct_counts() {
        let ops = vec![
            DiffOp::Equal("a".to_string()),
            DiffOp::Delete("b".to_string()),
            DiffOp::Insert("c".to_string()),
            DiffOp::Equal("d".to_string()),
        ];
        let s = stats(&ops);
        assert_eq!(s.unchanged, 2);
        assert_eq!(s.deletions, 1);
        assert_eq!(s.insertions, 1);
    }

    #[test]
    fn empty_inputs() {
        let ops = diff_lines("", "");
        assert!(ops.is_empty());
    }

    #[test]
    fn empty_old_all_inserts() {
        let ops = diff_lines("", "a\nb");
        assert_eq!(ops.len(), 2);
        assert!(ops.iter().all(|op| matches!(op, DiffOp::Insert(_))));
    }

    #[test]
    fn empty_new_all_deletes() {
        let ops = diff_lines("a\nb", "");
        assert_eq!(ops.len(), 2);
        assert!(ops.iter().all(|op| matches!(op, DiffOp::Delete(_))));
    }
}
