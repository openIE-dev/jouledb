//! Side-by-side diff view model.
//!
//! Computes line-level and inline diffs, manages line mappings, and supports
//! navigation and context collapsing. Replaces Monaco diff editor model.

// ── Diff line types ─────────────────────────────────────────────

/// A single line in the diff view.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiffLine {
    /// Line is the same in both sides.
    Same(String),
    /// Line was added (only on right side).
    Added(String),
    /// Line was removed (only on left side).
    Removed(String),
    /// Line was modified (old on left, new on right).
    Modified { old: String, new: String },
}

/// An inline diff segment within a line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InlineSegment {
    /// Unchanged text.
    Equal(String),
    /// Inserted text.
    Insert(String),
    /// Deleted text.
    Delete(String),
}

/// Statistics for a diff.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DiffStats {
    pub additions: usize,
    pub deletions: usize,
    pub modifications: usize,
    pub unchanged: usize,
}

/// A mapping between left and right line numbers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineMapping {
    /// Left line number (None if line only exists on right).
    pub left: Option<usize>,
    /// Right line number (None if line only exists on left).
    pub right: Option<usize>,
    /// The diff line.
    pub line: DiffLine,
}

// ── Core diff algorithm (LCS-based) ────────────────────────────

/// Compute the longest common subsequence table.
fn lcs_table(old: &[&str], new: &[&str]) -> Vec<Vec<usize>> {
    let n = old.len();
    let m = new.len();
    let mut table = vec![vec![0usize; m + 1]; n + 1];

    for i in 1..=n {
        for j in 1..=m {
            if old[i - 1] == new[j - 1] {
                table[i][j] = table[i - 1][j - 1] + 1;
            } else {
                table[i][j] = table[i - 1][j].max(table[i][j - 1]);
            }
        }
    }
    table
}

/// Compute a line-level diff between old and new text.
pub fn compute_diff(old_text: &str, new_text: &str) -> Vec<DiffLine> {
    let old_lines: Vec<&str> = old_text.lines().collect();
    let new_lines: Vec<&str> = new_text.lines().collect();

    let table = lcs_table(&old_lines, &new_lines);
    let mut result = Vec::new();

    backtrack_diff(&table, &old_lines, &new_lines, old_lines.len(), new_lines.len(), &mut result);
    result
}

fn backtrack_diff(
    table: &[Vec<usize>],
    old: &[&str],
    new: &[&str],
    i: usize,
    j: usize,
    result: &mut Vec<DiffLine>,
) {
    if i > 0 && j > 0 && old[i - 1] == new[j - 1] {
        backtrack_diff(table, old, new, i - 1, j - 1, result);
        result.push(DiffLine::Same(old[i - 1].to_string()));
    } else if j > 0 && (i == 0 || table[i][j - 1] >= table[i - 1][j]) {
        backtrack_diff(table, old, new, i, j - 1, result);
        result.push(DiffLine::Added(new[j - 1].to_string()));
    } else if i > 0 && (j == 0 || table[i][j - 1] < table[i - 1][j]) {
        backtrack_diff(table, old, new, i - 1, j, result);
        result.push(DiffLine::Removed(old[i - 1].to_string()));
    }
}

/// Post-process: convert adjacent Removed+Added pairs into Modified lines.
pub fn merge_modifications(lines: &[DiffLine]) -> Vec<DiffLine> {
    let mut result = Vec::new();
    let mut i = 0;

    while i < lines.len() {
        if i + 1 < lines.len() {
            if let (DiffLine::Removed(old), DiffLine::Added(new)) = (&lines[i], &lines[i + 1]) {
                result.push(DiffLine::Modified {
                    old: old.clone(),
                    new: new.clone(),
                });
                i += 2;
                continue;
            }
        }
        result.push(lines[i].clone());
        i += 1;
    }
    result
}

// ── Inline diff (word-level or char-level) ──────────────────────

/// Compute an inline (character-level) diff between two strings.
pub fn inline_diff_chars(old: &str, new: &str) -> Vec<InlineSegment> {
    let old_chars: Vec<char> = old.chars().collect();
    let new_chars: Vec<char> = new.chars().collect();

    let n = old_chars.len();
    let m = new_chars.len();

    // LCS for characters
    let mut table = vec![vec![0usize; m + 1]; n + 1];
    for i in 1..=n {
        for j in 1..=m {
            if old_chars[i - 1] == new_chars[j - 1] {
                table[i][j] = table[i - 1][j - 1] + 1;
            } else {
                table[i][j] = table[i - 1][j].max(table[i][j - 1]);
            }
        }
    }

    let mut segments = Vec::new();
    collect_inline_diff(&table, &old_chars, &new_chars, n, m, &mut segments);
    merge_inline_segments(segments)
}

fn collect_inline_diff(
    table: &[Vec<usize>],
    old: &[char],
    new: &[char],
    i: usize,
    j: usize,
    result: &mut Vec<InlineSegment>,
) {
    if i > 0 && j > 0 && old[i - 1] == new[j - 1] {
        collect_inline_diff(table, old, new, i - 1, j - 1, result);
        result.push(InlineSegment::Equal(old[i - 1].to_string()));
    } else if j > 0 && (i == 0 || table[i][j - 1] >= table[i - 1][j]) {
        collect_inline_diff(table, old, new, i, j - 1, result);
        result.push(InlineSegment::Insert(new[j - 1].to_string()));
    } else if i > 0 {
        collect_inline_diff(table, old, new, i - 1, j, result);
        result.push(InlineSegment::Delete(old[i - 1].to_string()));
    }
}

/// Merge consecutive inline segments of the same type.
fn merge_inline_segments(segments: Vec<InlineSegment>) -> Vec<InlineSegment> {
    let mut merged: Vec<InlineSegment> = Vec::new();
    for seg in segments {
        if let Some(last) = merged.last_mut() {
            match (last, &seg) {
                (InlineSegment::Equal(a), InlineSegment::Equal(b)) => {
                    a.push_str(b);
                    continue;
                }
                (InlineSegment::Insert(a), InlineSegment::Insert(b)) => {
                    a.push_str(b);
                    continue;
                }
                (InlineSegment::Delete(a), InlineSegment::Delete(b)) => {
                    a.push_str(b);
                    continue;
                }
                _ => {}
            }
        }
        merged.push(seg);
    }
    merged
}

/// Compute an inline (word-level) diff between two strings.
pub fn inline_diff_words(old: &str, new: &str) -> Vec<InlineSegment> {
    let old_words: Vec<&str> = old.split_inclusive(char::is_whitespace).collect();
    let new_words: Vec<&str> = new.split_inclusive(char::is_whitespace).collect();

    let n = old_words.len();
    let m = new_words.len();

    let mut table = vec![vec![0usize; m + 1]; n + 1];
    for i in 1..=n {
        for j in 1..=m {
            if old_words[i - 1] == new_words[j - 1] {
                table[i][j] = table[i - 1][j - 1] + 1;
            } else {
                table[i][j] = table[i - 1][j].max(table[i][j - 1]);
            }
        }
    }

    let mut raw = Vec::new();
    collect_word_diff(&table, &old_words, &new_words, n, m, &mut raw);
    merge_inline_segments(raw)
}

fn collect_word_diff(
    table: &[Vec<usize>],
    old: &[&str],
    new: &[&str],
    i: usize,
    j: usize,
    result: &mut Vec<InlineSegment>,
) {
    if i > 0 && j > 0 && old[i - 1] == new[j - 1] {
        collect_word_diff(table, old, new, i - 1, j - 1, result);
        result.push(InlineSegment::Equal(old[i - 1].to_string()));
    } else if j > 0 && (i == 0 || table[i][j - 1] >= table[i - 1][j]) {
        collect_word_diff(table, old, new, i, j - 1, result);
        result.push(InlineSegment::Insert(new[j - 1].to_string()));
    } else if i > 0 {
        collect_word_diff(table, old, new, i - 1, j, result);
        result.push(InlineSegment::Delete(old[i - 1].to_string()));
    }
}

// ── Diff model ──────────────────────────────────────────────────

/// A complete side-by-side diff model.
#[derive(Debug, Clone)]
pub struct DiffModel {
    /// The diff lines (with modifications merged).
    pub lines: Vec<DiffLine>,
    /// Line mappings for left/right correspondence.
    pub mappings: Vec<LineMapping>,
    /// Statistics.
    pub stats: DiffStats,
}

impl DiffModel {
    /// Create a diff model from two texts.
    pub fn new(old_text: &str, new_text: &str) -> Self {
        let raw = compute_diff(old_text, new_text);
        let lines = merge_modifications(&raw);

        let mut mappings = Vec::new();
        let mut stats = DiffStats::default();
        let mut left_num = 0usize;
        let mut right_num = 0usize;

        for line in &lines {
            match line {
                DiffLine::Same(_) => {
                    mappings.push(LineMapping {
                        left: Some(left_num),
                        right: Some(right_num),
                        line: line.clone(),
                    });
                    left_num += 1;
                    right_num += 1;
                    stats.unchanged += 1;
                }
                DiffLine::Added(_) => {
                    mappings.push(LineMapping {
                        left: None,
                        right: Some(right_num),
                        line: line.clone(),
                    });
                    right_num += 1;
                    stats.additions += 1;
                }
                DiffLine::Removed(_) => {
                    mappings.push(LineMapping {
                        left: Some(left_num),
                        right: None,
                        line: line.clone(),
                    });
                    left_num += 1;
                    stats.deletions += 1;
                }
                DiffLine::Modified { .. } => {
                    mappings.push(LineMapping {
                        left: Some(left_num),
                        right: Some(right_num),
                        line: line.clone(),
                    });
                    left_num += 1;
                    right_num += 1;
                    stats.modifications += 1;
                }
            }
        }

        Self {
            lines,
            mappings,
            stats,
        }
    }

    /// Get the indices of change regions (Added, Removed, Modified).
    pub fn change_indices(&self) -> Vec<usize> {
        self.lines
            .iter()
            .enumerate()
            .filter(|(_, l)| !matches!(l, DiffLine::Same(_)))
            .map(|(i, _)| i)
            .collect()
    }

    /// Navigate to the next change from the given index.
    pub fn next_change(&self, from: usize) -> Option<usize> {
        self.change_indices().into_iter().find(|i| *i > from)
    }

    /// Navigate to the previous change from the given index.
    pub fn prev_change(&self, from: usize) -> Option<usize> {
        self.change_indices().into_iter().rev().find(|i| *i < from)
    }

    /// Collapse unchanged regions, keeping `context` lines around changes.
    /// Returns the lines with unchanged regions replaced by a separator.
    pub fn collapsed(&self, context: usize) -> Vec<CollapsedLine> {
        let changes = self.change_indices();
        if changes.is_empty() {
            return vec![CollapsedLine::Collapsed {
                count: self.lines.len(),
            }];
        }

        // Mark which lines to show
        let mut visible = vec![false; self.lines.len()];
        for &change_idx in &changes {
            let start = change_idx.saturating_sub(context);
            let end = (change_idx + context + 1).min(self.lines.len());
            for v in &mut visible[start..end] {
                *v = true;
            }
        }

        let mut result = Vec::new();
        let mut hidden_count = 0usize;

        for (i, line) in self.lines.iter().enumerate() {
            if visible[i] {
                if hidden_count > 0 {
                    result.push(CollapsedLine::Collapsed {
                        count: hidden_count,
                    });
                    hidden_count = 0;
                }
                result.push(CollapsedLine::Visible(line.clone()));
            } else {
                hidden_count += 1;
            }
        }

        if hidden_count > 0 {
            result.push(CollapsedLine::Collapsed {
                count: hidden_count,
            });
        }

        result
    }

    /// Total number of lines in the diff.
    pub fn line_count(&self) -> usize {
        self.lines.len()
    }
}

/// A line in a collapsed diff view.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CollapsedLine {
    /// A visible diff line.
    Visible(DiffLine),
    /// A collapsed region of N unchanged lines.
    Collapsed { count: usize },
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_texts() {
        let model = DiffModel::new("hello\nworld", "hello\nworld");
        assert_eq!(model.stats.unchanged, 2);
        assert_eq!(model.stats.additions, 0);
        assert_eq!(model.stats.deletions, 0);
    }

    #[test]
    fn added_line() {
        let model = DiffModel::new("a\nb", "a\nb\nc");
        assert_eq!(model.stats.additions, 1);
        assert!(model.lines.iter().any(|l| matches!(l, DiffLine::Added(s) if s == "c")));
    }

    #[test]
    fn removed_line() {
        let model = DiffModel::new("a\nb\nc", "a\nc");
        assert_eq!(model.stats.deletions, 1);
        assert!(model.lines.iter().any(|l| matches!(l, DiffLine::Removed(s) if s == "b")));
    }

    #[test]
    fn modified_line() {
        let model = DiffModel::new("hello world", "hello rust");
        // Should have a modification
        assert!(
            model.stats.modifications > 0 || model.stats.deletions > 0,
            "expected some change"
        );
    }

    #[test]
    fn line_mappings() {
        let model = DiffModel::new("a\nb\nc", "a\nx\nc");
        // Check that mappings are created
        assert!(!model.mappings.is_empty());
        // First line should map left 0 to right 0
        assert_eq!(model.mappings[0].left, Some(0));
        assert_eq!(model.mappings[0].right, Some(0));
    }

    #[test]
    fn inline_diff_chars_basic() {
        let segments = inline_diff_chars("hello", "hallo");
        // Should have: Equal("h"), Delete("e"), Insert("a"), Equal("llo")
        assert!(segments.iter().any(|s| matches!(s, InlineSegment::Delete(d) if d == "e")));
        assert!(segments.iter().any(|s| matches!(s, InlineSegment::Insert(i) if i == "a")));
    }

    #[test]
    fn inline_diff_words_basic() {
        let segments = inline_diff_words("the quick fox", "the slow fox");
        assert!(segments.iter().any(|s| matches!(s, InlineSegment::Delete(d) if d.contains("quick"))));
        assert!(segments.iter().any(|s| matches!(s, InlineSegment::Insert(i) if i.contains("slow"))));
    }

    #[test]
    fn navigation() {
        let model = DiffModel::new("a\nb\nc\nd\ne", "a\nB\nc\nD\ne");
        let changes = model.change_indices();
        assert!(!changes.is_empty());

        if let Some(first) = changes.first() {
            let next = model.next_change(*first);
            assert!(next.is_some());
            assert!(next.unwrap() > *first);
        }
    }

    #[test]
    fn prev_change() {
        let model = DiffModel::new("a\nb\nc", "a\nB\nC");
        let changes = model.change_indices();
        if changes.len() >= 2 {
            let last = *changes.last().unwrap();
            let prev = model.prev_change(last);
            assert!(prev.is_some());
            assert!(prev.unwrap() < last);
        }
    }

    #[test]
    fn collapsed_view() {
        // Create a diff with many unchanged lines and one change in the middle
        let old = (0..20).map(|i| format!("line {i}")).collect::<Vec<_>>().join("\n");
        let mut new_lines: Vec<String> = (0..20).map(|i| format!("line {i}")).collect();
        new_lines[10] = "CHANGED".to_string();
        let new = new_lines.join("\n");

        let model = DiffModel::new(&old, &new);
        let collapsed = model.collapsed(2);

        // Should have collapsed regions and visible lines around the change
        assert!(collapsed.iter().any(|l| matches!(l, CollapsedLine::Collapsed { .. })));
        let visible_count = collapsed
            .iter()
            .filter(|l| matches!(l, CollapsedLine::Visible(_)))
            .count();
        // Context 2: should show ~5 lines (2 before, 1 change, 2 after)
        assert!(visible_count <= 7);
    }

    #[test]
    fn empty_diff() {
        let model = DiffModel::new("", "");
        assert_eq!(model.line_count(), 0);
    }

    #[test]
    fn diff_stats_complete() {
        let model = DiffModel::new("a\nb\nc\nd", "a\nB\nc\ne\nf");
        let total = model.stats.unchanged
            + model.stats.additions
            + model.stats.deletions
            + model.stats.modifications;
        assert_eq!(total, model.line_count());
    }

    #[test]
    fn merge_modifications_pairs() {
        let lines = vec![
            DiffLine::Removed("old".into()),
            DiffLine::Added("new".into()),
            DiffLine::Same("same".into()),
        ];
        let merged = merge_modifications(&lines);
        assert_eq!(merged.len(), 2);
        assert!(matches!(&merged[0], DiffLine::Modified { old, new } if old == "old" && new == "new"));
    }
}
