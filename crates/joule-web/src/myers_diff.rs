//! Myers diff algorithm.
//!
//! O(ND) line-level diff producing minimal edit scripts. Supports insert,
//! delete, and equal operations, longest common subsequence extraction, diff
//! statistics, configurable equality functions, and a patience diff variant
//! that anchors on unique matching lines for more human-readable results.

use std::collections::HashMap;
use std::fmt;

// ── Types ──────────────────────────────────────────────────────────

/// A single edit operation in a diff.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditOp {
    /// A line present in both old and new.
    Equal(String),
    /// A line inserted in new.
    Insert(String),
    /// A line deleted from old.
    Delete(String),
}

/// Summary statistics for a diff result.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DiffStats {
    pub additions: usize,
    pub deletions: usize,
    pub changes: usize,
    pub unchanged: usize,
}

/// The result of a diff operation.
#[derive(Debug, Clone)]
pub struct DiffResult {
    pub ops: Vec<EditOp>,
    pub stats: DiffStats,
}

/// A configurable equality comparator for diff lines.
pub struct EqualityConfig {
    /// If true, ignore leading/trailing whitespace when comparing.
    pub ignore_whitespace: bool,
    /// If true, treat lines case-insensitively.
    pub case_insensitive: bool,
}

impl Default for EqualityConfig {
    fn default() -> Self {
        Self {
            ignore_whitespace: false,
            case_insensitive: false,
        }
    }
}

impl EqualityConfig {
    /// Compare two strings using this configuration.
    pub fn equal(&self, a: &str, b: &str) -> bool {
        let a_norm = self.normalize(a);
        let b_norm = self.normalize(b);
        a_norm == b_norm
    }

    fn normalize(&self, s: &str) -> String {
        let mut result = if self.ignore_whitespace {
            s.trim().to_string()
        } else {
            s.to_string()
        };
        if self.case_insensitive {
            result = result.to_lowercase();
        }
        result
    }
}

impl fmt::Display for EditOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EditOp::Equal(s) => write!(f, " {}", s),
            EditOp::Insert(s) => write!(f, "+{}", s),
            EditOp::Delete(s) => write!(f, "-{}", s),
        }
    }
}

// ── Myers diff core ────────────────────────────────────────────────

/// Compute the shortest edit script between two slices using the
/// Myers O(ND) algorithm. Returns a list of index pairs tracing the
/// path through the edit graph, from which we derive edit ops.
fn myers_shortest_edit(old: &[&str], new: &[&str], eq: &EqualityConfig) -> Vec<EditOp> {
    let n = old.len();
    let m = new.len();

    if n == 0 && m == 0 {
        return Vec::new();
    }
    if n == 0 {
        return new.iter().map(|s| EditOp::Insert(s.to_string())).collect();
    }
    if m == 0 {
        return old.iter().map(|s| EditOp::Delete(s.to_string())).collect();
    }

    let max = n + m;
    // v stores the furthest reaching x for each diagonal k.
    // We offset k by max so indices are always non-negative.
    let size = 2 * max + 1;
    let mut v = vec![0usize; size];
    // trace stores the v snapshot at each step d.
    let mut trace: Vec<Vec<usize>> = Vec::new();

    'outer: for d in 0..=(max as isize) {
        trace.push(v.clone());
        let mut k = -d;
        while k <= d {
            let idx = (k + max as isize) as usize;
            let mut x = if k == -d
                || (k != d && v[idx.wrapping_sub(1)] < v[idx + 1])
            {
                v[idx + 1]
            } else {
                v[idx.wrapping_sub(1)] + 1
            };
            let mut y = (x as isize - k) as usize;

            while x < n && y < m && eq.equal(old[x], new[y]) {
                x += 1;
                y += 1;
            }

            v[idx] = x;

            if x >= n && y >= m {
                break 'outer;
            }
            k += 2;
        }
    }

    // Backtrack through the trace to build the edit script.
    backtrack(&trace, old, new, max, eq)
}

fn backtrack(
    trace: &[Vec<usize>],
    old: &[&str],
    new: &[&str],
    max: usize,
    eq: &EqualityConfig,
) -> Vec<EditOp> {
    let mut x = old.len();
    let mut y = new.len();
    let mut ops: Vec<EditOp> = Vec::new();

    for d in (0..trace.len()).rev() {
        if d == 0 {
            // At step 0, just emit remaining diagonal (equal) moves.
            while x > 0 && y > 0 {
                x -= 1;
                y -= 1;
                ops.push(EditOp::Equal(old[x].to_string()));
            }
            break;
        }

        let v = &trace[d];
        let k = x as isize - y as isize;
        let idx = (k + max as isize) as usize;

        let prev_k = if k == -(d as isize)
            || (k != d as isize && v[idx.wrapping_sub(1)] < v[idx + 1])
        {
            k + 1
        } else {
            k - 1
        };

        let prev_idx = (prev_k + max as isize) as usize;
        let prev_x = v[prev_idx];
        let prev_y = (prev_x as isize - prev_k) as usize;

        // Emit equals for the snake (diagonal moves).
        while x > prev_x && y > prev_y {
            x -= 1;
            y -= 1;
            ops.push(EditOp::Equal(old[x].to_string()));
        }

        if x == prev_x && y > 0 {
            // Vertical move = insert.
            y -= 1;
            ops.push(EditOp::Insert(new[y].to_string()));
        } else if x > 0 {
            // Horizontal move = delete.
            x -= 1;
            ops.push(EditOp::Delete(old[x].to_string()));
        }
    }

    ops.reverse();
    ops
}

// ── Public API ─────────────────────────────────────────────────────

/// Compute a line-level diff between two strings using the Myers algorithm.
pub fn diff_lines(old: &str, new: &str) -> DiffResult {
    diff_lines_with_config(old, new, &EqualityConfig::default())
}

/// Compute a line-level diff with a custom equality configuration.
pub fn diff_lines_with_config(old: &str, new: &str, config: &EqualityConfig) -> DiffResult {
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
    let ops = myers_shortest_edit(&old_lines, &new_lines, config);
    let stats = compute_stats(&ops);
    DiffResult { ops, stats }
}

/// Compute the longest common subsequence of two line sequences.
pub fn longest_common_subsequence(old: &str, new: &str) -> Vec<String> {
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
    lcs_lines(&old_lines, &new_lines)
}

fn lcs_lines(old: &[&str], new: &[&str]) -> Vec<String> {
    let n = old.len();
    let m = new.len();
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
    let mut result = Vec::new();
    let mut i = n;
    let mut j = m;
    while i > 0 && j > 0 {
        if old[i - 1] == new[j - 1] {
            result.push(old[i - 1].to_string());
            i -= 1;
            j -= 1;
        } else if dp[i - 1][j] >= dp[i][j - 1] {
            i -= 1;
        } else {
            j -= 1;
        }
    }
    result.reverse();
    result
}

/// Compute diff statistics from a list of edit operations.
pub fn compute_stats(ops: &[EditOp]) -> DiffStats {
    let mut stats = DiffStats::default();
    let mut i = 0;
    while i < ops.len() {
        match &ops[i] {
            EditOp::Equal(_) => {
                stats.unchanged += 1;
                i += 1;
            }
            EditOp::Delete(_) => {
                // Check if a delete is immediately followed by an insert
                // (which counts as a "change" rather than separate add/delete).
                if i + 1 < ops.len() && matches!(&ops[i + 1], EditOp::Insert(_)) {
                    stats.changes += 1;
                    i += 2;
                } else {
                    stats.deletions += 1;
                    i += 1;
                }
            }
            EditOp::Insert(_) => {
                stats.additions += 1;
                i += 1;
            }
        }
    }
    stats
}

/// Render the diff result as a human-readable string.
pub fn render_diff(result: &DiffResult) -> String {
    let mut out = String::new();
    for op in &result.ops {
        out.push_str(&op.to_string());
        out.push('\n');
    }
    out
}

// ── Patience diff ──────────────────────────────────────────────────

/// Patience diff: anchors on unique matching lines for more human-readable
/// results, then falls back to Myers for the segments between anchors.
pub fn patience_diff(old: &str, new: &str) -> DiffResult {
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

    let ops = patience_diff_lines(&old_lines, &new_lines);
    let stats = compute_stats(&ops);
    DiffResult { ops, stats }
}

fn patience_diff_lines(old: &[&str], new: &[&str]) -> Vec<EditOp> {
    // Find unique lines in both old and new, preserving order.
    let old_unique = unique_lines(old);
    let new_unique = unique_lines(new);

    // Find the common unique lines (anchors) using LCS.
    let anchors = find_anchors(&old_unique, &new_unique, old, new);

    if anchors.is_empty() {
        // No anchors: fall back to Myers.
        return myers_shortest_edit(old, new, &EqualityConfig::default());
    }

    // Diff between anchors using Myers.
    let mut ops = Vec::new();
    let mut old_pos = 0;
    let mut new_pos = 0;

    for (oi, ni) in &anchors {
        // Diff the segment before this anchor.
        if old_pos < *oi || new_pos < *ni {
            let sub = myers_shortest_edit(
                &old[old_pos..*oi],
                &new[new_pos..*ni],
                &EqualityConfig::default(),
            );
            ops.extend(sub);
        }
        ops.push(EditOp::Equal(old[*oi].to_string()));
        old_pos = oi + 1;
        new_pos = ni + 1;
    }

    // Tail segment.
    if old_pos < old.len() || new_pos < new.len() {
        let sub = myers_shortest_edit(
            &old[old_pos..],
            &new[new_pos..],
            &EqualityConfig::default(),
        );
        ops.extend(sub);
    }

    ops
}

/// Returns a map of line content -> list of indices, but only for lines
/// that appear exactly once.
fn unique_lines<'a>(lines: &[&'a str]) -> HashMap<&'a str, usize> {
    let mut counts: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, line) in lines.iter().enumerate() {
        counts.entry(*line).or_default().push(i);
    }
    let mut result = HashMap::new();
    for (line, indices) in counts {
        if indices.len() == 1 {
            result.insert(line, indices[0]);
        }
    }
    result
}

/// Find anchor pairs (old_index, new_index) from unique lines, in order.
fn find_anchors(
    old_unique: &HashMap<&str, usize>,
    new_unique: &HashMap<&str, usize>,
    _old: &[&str],
    _new: &[&str],
) -> Vec<(usize, usize)> {
    // Find common unique lines.
    let mut common: Vec<(usize, usize)> = Vec::new();
    for (line, old_idx) in old_unique {
        if let Some(new_idx) = new_unique.get(line) {
            common.push((*old_idx, *new_idx));
        }
    }
    // Sort by old index.
    common.sort_by_key(|(oi, _)| *oi);

    // Find longest increasing subsequence of new indices to get
    // the anchors in order.
    if common.is_empty() {
        return Vec::new();
    }

    let new_indices: Vec<usize> = common.iter().map(|(_, ni)| *ni).collect();
    let lis_indices = lis(&new_indices);
    lis_indices.iter().map(|i| common[*i]).collect()
}

/// Longest increasing subsequence, returns indices into the input.
fn lis(seq: &[usize]) -> Vec<usize> {
    if seq.is_empty() {
        return Vec::new();
    }
    let n = seq.len();
    let mut tails: Vec<usize> = Vec::new(); // stores indices into seq
    let mut predecessors = vec![usize::MAX; n];
    let mut tail_indices: Vec<usize> = Vec::new();

    for i in 0..n {
        let val = seq[i];
        // Binary search for the position.
        let pos = tails
            .binary_search_by(|idx| seq[*idx].cmp(&val))
            .unwrap_or_else(|p| p);

        if pos == tails.len() {
            tails.push(i);
            tail_indices.push(i);
        } else {
            tails[pos] = i;
            if pos < tail_indices.len() {
                tail_indices[pos] = i;
            }
        }

        if pos > 0 {
            predecessors[i] = tails[pos - 1];
        }
    }

    // Reconstruct the LIS.
    let mut result = Vec::new();
    let mut idx = *tails.last().unwrap();
    loop {
        result.push(idx);
        if predecessors[idx] == usize::MAX {
            break;
        }
        idx = predecessors[idx];
    }
    result.reverse();
    result
}

/// Apply a diff (edit script) to reconstruct the new text.
pub fn apply_edit_script(old: &str, ops: &[EditOp]) -> String {
    let mut result = Vec::new();
    for op in ops {
        match op {
            EditOp::Equal(s) | EditOp::Insert(s) => {
                result.push(s.as_str());
            }
            EditOp::Delete(_) => {}
        }
    }
    result.join("\n")
}

/// Produce the reverse diff: transforms new back to old.
pub fn reverse_diff(ops: &[EditOp]) -> Vec<EditOp> {
    ops.iter()
        .map(|op| match op {
            EditOp::Equal(s) => EditOp::Equal(s.clone()),
            EditOp::Insert(s) => EditOp::Delete(s.clone()),
            EditOp::Delete(s) => EditOp::Insert(s.clone()),
        })
        .collect()
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_strings() {
        let result = diff_lines("hello\nworld", "hello\nworld");
        assert!(result.ops.iter().all(|op| matches!(op, EditOp::Equal(_))));
        assert_eq!(result.stats.unchanged, 2);
        assert_eq!(result.stats.additions, 0);
        assert_eq!(result.stats.deletions, 0);
    }

    #[test]
    fn insert_single_line() {
        let result = diff_lines("a\nc", "a\nb\nc");
        assert_eq!(result.stats.additions, 1);
        assert_eq!(result.stats.unchanged, 2);
        let new_text = apply_edit_script("a\nc", &result.ops);
        assert_eq!(new_text, "a\nb\nc");
    }

    #[test]
    fn delete_single_line() {
        let result = diff_lines("a\nb\nc", "a\nc");
        assert_eq!(result.stats.deletions, 1);
        assert_eq!(result.stats.unchanged, 2);
    }

    #[test]
    fn change_line() {
        let result = diff_lines("a\nb\nc", "a\nx\nc");
        // b->x counts as a change (delete + insert pair).
        assert!(result.stats.changes > 0 || (result.stats.deletions > 0 && result.stats.additions > 0));
    }

    #[test]
    fn empty_old() {
        let result = diff_lines("", "a\nb");
        assert_eq!(result.ops.len(), 2);
        assert!(result.ops.iter().all(|op| matches!(op, EditOp::Insert(_))));
    }

    #[test]
    fn empty_new() {
        let result = diff_lines("a\nb", "");
        assert_eq!(result.ops.len(), 2);
        assert!(result.ops.iter().all(|op| matches!(op, EditOp::Delete(_))));
    }

    #[test]
    fn both_empty() {
        let result = diff_lines("", "");
        assert!(result.ops.is_empty());
    }

    #[test]
    fn lcs_basic() {
        let lcs = longest_common_subsequence("a\nb\nc\nd", "a\nc\nd\ne");
        assert_eq!(lcs, vec!["a", "c", "d"]);
    }

    #[test]
    fn lcs_no_common() {
        let lcs = longest_common_subsequence("a\nb", "c\nd");
        assert!(lcs.is_empty());
    }

    #[test]
    fn lcs_identical() {
        let lcs = longest_common_subsequence("a\nb\nc", "a\nb\nc");
        assert_eq!(lcs, vec!["a", "b", "c"]);
    }

    #[test]
    fn stats_computation() {
        let ops = vec![
            EditOp::Equal("a".into()),
            EditOp::Delete("b".into()),
            EditOp::Insert("x".into()),
            EditOp::Equal("c".into()),
            EditOp::Insert("d".into()),
        ];
        let stats = compute_stats(&ops);
        assert_eq!(stats.unchanged, 2);
        assert_eq!(stats.changes, 1); // delete+insert pair
        assert_eq!(stats.additions, 1); // standalone insert
        assert_eq!(stats.deletions, 0);
    }

    #[test]
    fn whitespace_insensitive_diff() {
        let config = EqualityConfig {
            ignore_whitespace: true,
            case_insensitive: false,
        };
        let result = diff_lines_with_config("  hello  \nworld", "hello\nworld", &config);
        assert_eq!(result.stats.unchanged, 2);
        assert_eq!(result.stats.additions, 0);
    }

    #[test]
    fn case_insensitive_diff() {
        let config = EqualityConfig {
            ignore_whitespace: false,
            case_insensitive: true,
        };
        let result = diff_lines_with_config("Hello\nWorld", "hello\nworld", &config);
        assert_eq!(result.stats.unchanged, 2);
    }

    #[test]
    fn render_diff_output() {
        let result = diff_lines("a\nb", "a\nc");
        let rendered = render_diff(&result);
        assert!(rendered.contains(" a"));
        assert!(rendered.contains("-b") || rendered.contains("+c"));
    }

    #[test]
    fn apply_roundtrip() {
        let old = "line1\nline2\nline3\nline4";
        let new = "line1\nchanged\nline3\nline5\nline6";
        let result = diff_lines(old, new);
        let reconstructed = apply_edit_script(old, &result.ops);
        assert_eq!(reconstructed, new);
    }

    #[test]
    fn reverse_diff_roundtrip() {
        let old = "a\nb\nc";
        let new = "a\nx\nc";
        let result = diff_lines(old, new);
        let reversed = reverse_diff(&result.ops);
        let back = apply_edit_script(new, &reversed);
        assert_eq!(back, old);
    }

    #[test]
    fn patience_diff_basic() {
        let result = patience_diff("a\nb\nc", "a\nx\nc");
        let reconstructed = apply_edit_script("a\nb\nc", &result.ops);
        assert_eq!(reconstructed, "a\nx\nc");
    }

    #[test]
    fn patience_diff_identical() {
        let result = patience_diff("a\nb\nc", "a\nb\nc");
        assert!(result.ops.iter().all(|op| matches!(op, EditOp::Equal(_))));
    }

    #[test]
    fn patience_diff_all_different() {
        let result = patience_diff("a\nb", "c\nd");
        let reconstructed = apply_edit_script("a\nb", &result.ops);
        assert_eq!(reconstructed, "c\nd");
    }

    #[test]
    fn large_diff() {
        let old: String = (0..50).map(|i| format!("line{}", i)).collect::<Vec<_>>().join("\n");
        let new: String = (0..50)
            .map(|i| {
                if i == 25 {
                    "CHANGED".to_string()
                } else {
                    format!("line{}", i)
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        let result = diff_lines(&old, &new);
        let reconstructed = apply_edit_script(&old, &result.ops);
        assert_eq!(reconstructed, new);
        assert_eq!(result.stats.unchanged, 49);
    }

    #[test]
    fn equality_config_default() {
        let config = EqualityConfig::default();
        assert!(!config.ignore_whitespace);
        assert!(!config.case_insensitive);
        assert!(config.equal("hello", "hello"));
        assert!(!config.equal("hello", "Hello"));
    }

    #[test]
    fn equality_config_combined() {
        let config = EqualityConfig {
            ignore_whitespace: true,
            case_insensitive: true,
        };
        assert!(config.equal("  Hello  ", "hello"));
    }

    #[test]
    fn edit_op_display() {
        assert_eq!(format!("{}", EditOp::Equal("x".into())), " x");
        assert_eq!(format!("{}", EditOp::Insert("x".into())), "+x");
        assert_eq!(format!("{}", EditOp::Delete("x".into())), "-x");
    }

    #[test]
    fn diff_single_line_files() {
        let result = diff_lines("only", "only");
        assert_eq!(result.ops.len(), 1);
        assert!(matches!(&result.ops[0], EditOp::Equal(s) if s == "only"));
    }
}
