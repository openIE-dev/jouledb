//! Three-way merge.
//!
//! Base/ours/theirs merge with conflict detection, auto-resolve for
//! non-overlapping changes, conflict markers, merge result with conflicts
//! list, and line-level and word-level merging.

use std::fmt;

// ── Types ──────────────────────────────────────────────────────────

/// A merge conflict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Conflict {
    /// Zero-based line index in the base where the conflict starts.
    pub base_start: usize,
    /// Number of base lines involved.
    pub base_count: usize,
    /// Our version of the conflicting region.
    pub ours: Vec<String>,
    /// Their version of the conflicting region.
    pub theirs: Vec<String>,
    /// Base version of the conflicting region.
    pub base: Vec<String>,
}

/// The result of a three-way merge.
#[derive(Debug, Clone)]
pub struct MergeResult {
    /// The merged text (may contain conflict markers if unresolved).
    pub text: String,
    /// Whether the merge succeeded without conflicts.
    pub clean: bool,
    /// List of conflicts (empty if clean).
    pub conflicts: Vec<Conflict>,
}

/// Style of conflict markers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkerStyle {
    /// Standard Git-style markers: <<<<<<< ======= >>>>>>>
    Git,
    /// Diff3-style markers including base: <<<<<<< ||||||| ======= >>>>>>>
    Diff3,
}

/// Configuration for three-way merge.
pub struct MergeConfig {
    /// Conflict marker style.
    pub marker_style: MarkerStyle,
    /// Label for "ours" in conflict markers.
    pub ours_label: String,
    /// Label for "theirs" in conflict markers.
    pub theirs_label: String,
    /// Label for "base" in conflict markers (Diff3 only).
    pub base_label: String,
}

impl Default for MergeConfig {
    fn default() -> Self {
        Self {
            marker_style: MarkerStyle::Git,
            ours_label: "ours".into(),
            theirs_label: "theirs".into(),
            base_label: "base".into(),
        }
    }
}

impl fmt::Display for Conflict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "conflict at base line {} ({} lines)",
            self.base_start, self.base_count
        )
    }
}

// ── Diff engine (for merge) ────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
enum DiffOp {
    Equal(usize), // index in old
    Insert(String),
    Delete(usize), // index in old
}

/// Compute a diff between two line sequences, returning edit ops with
/// indices into the old array.
fn line_diff(old: &[&str], new: &[&str]) -> Vec<DiffOp> {
    let n = old.len();
    let m = new.len();

    if n == 0 && m == 0 {
        return Vec::new();
    }
    if n == 0 {
        return new.iter().map(|s| DiffOp::Insert(s.to_string())).collect();
    }
    if m == 0 {
        return (0..n).map(DiffOp::Delete).collect();
    }

    // LCS-based diff.
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
            ops.push(DiffOp::Equal(i - 1));
            i -= 1;
            j -= 1;
        } else if j > 0 && (i == 0 || dp[i][j - 1] >= dp[i - 1][j]) {
            ops.push(DiffOp::Insert(new[j - 1].to_string()));
            j -= 1;
        } else {
            ops.push(DiffOp::Delete(i - 1));
            i -= 1;
        }
    }
    ops.reverse();
    ops
}

/// A change hunk relative to base.
#[derive(Debug, Clone)]
struct ChangeHunk {
    /// Start line in base (inclusive).
    base_start: usize,
    /// End line in base (exclusive).
    base_end: usize,
    /// Replacement lines.
    replacement: Vec<String>,
}

/// Extract change hunks from a diff.
fn extract_hunks(old: &[&str], new: &[&str]) -> Vec<ChangeHunk> {
    let ops = line_diff(old, new);
    let mut hunks = Vec::new();
    let mut i = 0;

    while i < ops.len() {
        match &ops[i] {
            DiffOp::Equal(_) => {
                i += 1;
            }
            DiffOp::Delete(_) | DiffOp::Insert(_) => {
                // Collect a contiguous change region.
                let mut base_start = usize::MAX;
                let mut base_end = 0;
                let mut replacement = Vec::new();

                while i < ops.len() {
                    match &ops[i] {
                        DiffOp::Equal(_) => break,
                        DiffOp::Delete(idx) => {
                            if base_start == usize::MAX {
                                base_start = *idx;
                            }
                            base_end = *idx + 1;
                            i += 1;
                        }
                        DiffOp::Insert(s) => {
                            // If we haven't established a base range,
                            // use the position right after the last equal.
                            if base_start == usize::MAX {
                                // Find position from context.
                                base_start = if i > 0 {
                                    match &ops[i - 1] {
                                        DiffOp::Equal(idx) => *idx + 1,
                                        DiffOp::Delete(idx) => *idx + 1,
                                        _ => 0,
                                    }
                                } else {
                                    0
                                };
                                base_end = base_start;
                            }
                            replacement.push(s.clone());
                            i += 1;
                        }
                    }
                }

                if base_start == usize::MAX {
                    base_start = 0;
                }

                hunks.push(ChangeHunk {
                    base_start,
                    base_end,
                    replacement,
                });
            }
        }
    }

    hunks
}

// ── Core merge ─────────────────────────────────────────────────────

/// Perform a three-way merge.
pub fn merge(base: &str, ours: &str, theirs: &str) -> MergeResult {
    merge_with_config(base, ours, theirs, &MergeConfig::default())
}

/// Perform a three-way merge with custom configuration.
pub fn merge_with_config(
    base: &str,
    ours: &str,
    theirs: &str,
    config: &MergeConfig,
) -> MergeResult {
    let base_lines: Vec<&str> = if base.is_empty() {
        Vec::new()
    } else {
        base.lines().collect()
    };
    let ours_lines: Vec<&str> = if ours.is_empty() {
        Vec::new()
    } else {
        ours.lines().collect()
    };
    let theirs_lines: Vec<&str> = if theirs.is_empty() {
        Vec::new()
    } else {
        theirs.lines().collect()
    };

    let our_hunks = extract_hunks(&base_lines, &ours_lines);
    let their_hunks = extract_hunks(&base_lines, &theirs_lines);

    merge_hunks(&base_lines, &our_hunks, &their_hunks, config)
}

fn merge_hunks(
    base: &[&str],
    our_hunks: &[ChangeHunk],
    their_hunks: &[ChangeHunk],
    config: &MergeConfig,
) -> MergeResult {
    let mut result_lines: Vec<String> = Vec::new();
    let mut conflicts: Vec<Conflict> = Vec::new();
    let mut base_pos = 0;
    let mut oi = 0;
    let mut ti = 0;

    while base_pos <= base.len() || oi < our_hunks.len() || ti < their_hunks.len() {
        // Check if either side has a hunk starting at or before base_pos.
        let our_hunk = our_hunks.get(oi);
        let their_hunk = their_hunks.get(ti);

        let our_start = our_hunk.map_or(usize::MAX, |h| h.base_start);
        let their_start = their_hunk.map_or(usize::MAX, |h| h.base_start);

        // If no more hunks, copy remaining base lines.
        if our_start == usize::MAX && their_start == usize::MAX {
            while base_pos < base.len() {
                result_lines.push(base[base_pos].to_string());
                base_pos += 1;
            }
            break;
        }

        // Copy base lines up to the next hunk.
        let next_start = our_start.min(their_start);
        while base_pos < next_start && base_pos < base.len() {
            result_lines.push(base[base_pos].to_string());
            base_pos += 1;
        }

        // Check for overlap.
        let oh = our_hunk.filter(|h| h.base_start == next_start);
        let th = their_hunk.filter(|h| h.base_start == next_start);

        match (oh, th) {
            (Some(o), Some(t)) => {
                // Both sides have hunks at the same position — check for conflict.
                let o_end = o.base_end;
                let t_end = t.base_end;
                let max_end = o_end.max(t_end);

                if overlaps(o, t) {
                    if o.replacement == t.replacement {
                        // Same change on both sides — no conflict.
                        for line in &o.replacement {
                            result_lines.push(line.clone());
                        }
                    } else {
                        // Genuine conflict.
                        let base_region: Vec<String> = base
                            [o.base_start..max_end.min(base.len())]
                            .iter()
                            .map(|s| s.to_string())
                            .collect();

                        conflicts.push(Conflict {
                            base_start: o.base_start,
                            base_count: max_end - o.base_start,
                            ours: o.replacement.clone(),
                            theirs: t.replacement.clone(),
                            base: base_region,
                        });

                        // Emit conflict markers.
                        emit_conflict_markers(
                            &mut result_lines,
                            o,
                            t,
                            &base[o.base_start..max_end.min(base.len())],
                            config,
                        );
                    }
                    base_pos = max_end;
                    oi += 1;
                    ti += 1;
                } else {
                    // Non-overlapping hunks at the same start — apply the one
                    // that starts earlier (or smaller range first).
                    if o.base_end <= t.base_start {
                        for line in &o.replacement {
                            result_lines.push(line.clone());
                        }
                        base_pos = o.base_end;
                        oi += 1;
                    } else {
                        for line in &t.replacement {
                            result_lines.push(line.clone());
                        }
                        base_pos = t.base_end;
                        ti += 1;
                    }
                }
            }
            (Some(o), None) => {
                for line in &o.replacement {
                    result_lines.push(line.clone());
                }
                base_pos = o.base_end;
                oi += 1;
            }
            (None, Some(t)) => {
                for line in &t.replacement {
                    result_lines.push(line.clone());
                }
                base_pos = t.base_end;
                ti += 1;
            }
            (None, None) => {
                // Advance to next hunk start.
                if base_pos < base.len() {
                    result_lines.push(base[base_pos].to_string());
                    base_pos += 1;
                } else {
                    break;
                }
            }
        }
    }

    let text = result_lines.join("\n");
    let clean = conflicts.is_empty();
    MergeResult {
        text,
        clean,
        conflicts,
    }
}

fn overlaps(a: &ChangeHunk, b: &ChangeHunk) -> bool {
    // Two hunks overlap if their base ranges intersect.
    let a_end = a.base_end.max(a.base_start + 1);
    let b_end = b.base_end.max(b.base_start + 1);
    a.base_start < b_end && b.base_start < a_end
}

fn emit_conflict_markers(
    lines: &mut Vec<String>,
    ours: &ChangeHunk,
    theirs: &ChangeHunk,
    base_lines: &[&str],
    config: &MergeConfig,
) {
    lines.push(format!("<<<<<<< {}", config.ours_label));
    for line in &ours.replacement {
        lines.push(line.clone());
    }
    if config.marker_style == MarkerStyle::Diff3 {
        lines.push(format!("||||||| {}", config.base_label));
        for line in base_lines {
            lines.push(line.to_string());
        }
    }
    lines.push("=======".into());
    for line in &theirs.replacement {
        lines.push(line.clone());
    }
    lines.push(format!(">>>>>>> {}", config.theirs_label));
}

// ── Word-level merge ───────────────────────────────────────────────

/// Perform a word-level three-way merge on single lines.
pub fn merge_words(base: &str, ours: &str, theirs: &str) -> MergeResult {
    let base_words: Vec<&str> = base.split_whitespace().collect();
    let ours_words: Vec<&str> = ours.split_whitespace().collect();
    let theirs_words: Vec<&str> = theirs.split_whitespace().collect();

    let our_changes = word_diff(&base_words, &ours_words);
    let their_changes = word_diff(&base_words, &theirs_words);

    // Simple strategy: if both modified, check if changes overlap.
    if our_changes == their_changes {
        // Same changes.
        let merged = apply_word_changes(&base_words, &our_changes);
        return MergeResult {
            text: merged.join(" "),
            clean: true,
            conflicts: Vec::new(),
        };
    }

    // Check for non-overlapping word changes.
    let has_overlap = word_changes_overlap(&our_changes, &their_changes);
    if !has_overlap {
        // Apply both sets of changes.
        let merged = apply_both_word_changes(&base_words, &our_changes, &their_changes);
        return MergeResult {
            text: merged.join(" "),
            clean: true,
            conflicts: Vec::new(),
        };
    }

    // Conflict.
    MergeResult {
        text: format!(
            "<<<<<<< ours\n{}\n=======\n{}\n>>>>>>> theirs",
            ours, theirs
        ),
        clean: false,
        conflicts: vec![Conflict {
            base_start: 0,
            base_count: 1,
            ours: vec![ours.to_string()],
            theirs: vec![theirs.to_string()],
            base: vec![base.to_string()],
        }],
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct WordChange {
    pos: usize,
    old_count: usize,
    new_words: Vec<String>,
}

fn word_diff(old: &[&str], new: &[&str]) -> Vec<WordChange> {
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

    // Backtrack to find changes.
    let mut changes = Vec::new();
    let mut i = n;
    let mut j = m;
    let mut pending_deletes: Vec<usize> = Vec::new();
    let mut pending_inserts: Vec<String> = Vec::new();

    while i > 0 || j > 0 {
        if i > 0 && j > 0 && old[i - 1] == new[j - 1] {
            flush_word_changes(&mut pending_deletes, &mut pending_inserts, &mut changes);
            i -= 1;
            j -= 1;
        } else if j > 0 && (i == 0 || dp[i][j - 1] >= dp[i - 1][j]) {
            pending_inserts.push(new[j - 1].to_string());
            j -= 1;
        } else {
            pending_deletes.push(i - 1);
            i -= 1;
        }
    }
    flush_word_changes(&mut pending_deletes, &mut pending_inserts, &mut changes);
    changes.reverse();
    changes
}

fn flush_word_changes(
    deletes: &mut Vec<usize>,
    inserts: &mut Vec<String>,
    changes: &mut Vec<WordChange>,
) {
    if deletes.is_empty() && inserts.is_empty() {
        return;
    }
    deletes.reverse();
    inserts.reverse();
    let pos = deletes.first().copied().unwrap_or(0);
    changes.push(WordChange {
        pos,
        old_count: deletes.len(),
        new_words: inserts.clone(),
    });
    deletes.clear();
    inserts.clear();
}

fn apply_word_changes(base: &[&str], changes: &[WordChange]) -> Vec<String> {
    let mut result: Vec<String> = base.iter().map(|s| s.to_string()).collect();
    // Apply in reverse order to preserve indices.
    for change in changes.iter().rev() {
        let end = (change.pos + change.old_count).min(result.len());
        let new_words = change.new_words.clone();
        result.splice(change.pos..end, new_words);
    }
    result
}

fn word_changes_overlap(a: &[WordChange], b: &[WordChange]) -> bool {
    for ac in a {
        for bc in b {
            let a_end = ac.pos + ac.old_count.max(1);
            let b_end = bc.pos + bc.old_count.max(1);
            if ac.pos < b_end && bc.pos < a_end {
                return true;
            }
        }
    }
    false
}

fn apply_both_word_changes(
    base: &[&str],
    a_changes: &[WordChange],
    b_changes: &[WordChange],
) -> Vec<String> {
    // Merge non-overlapping changes. Apply them all, sorted by position descending.
    let mut all_changes: Vec<&WordChange> = a_changes.iter().chain(b_changes.iter()).collect();
    all_changes.sort_by(|a, b| b.pos.cmp(&a.pos));

    let mut result: Vec<String> = base.iter().map(|s| s.to_string()).collect();
    for change in all_changes {
        let end = (change.pos + change.old_count).min(result.len());
        let new_words = change.new_words.clone();
        result.splice(change.pos..end, new_words);
    }
    result
}

/// Check if a merge result has conflicts.
pub fn has_conflicts(result: &MergeResult) -> bool {
    !result.clean
}

/// Count the number of conflicts.
pub fn conflict_count(result: &MergeResult) -> usize {
    result.conflicts.len()
}

/// Strip conflict markers from text, keeping "ours" side.
pub fn resolve_ours(text: &str) -> String {
    let mut result = Vec::new();
    let mut in_theirs = false;
    let mut in_conflict = false;

    for line in text.lines() {
        if line.starts_with("<<<<<<< ") {
            in_conflict = true;
            continue;
        }
        if line.starts_with("||||||| ") {
            in_theirs = true;
            continue;
        }
        if line == "=======" && in_conflict {
            in_theirs = true;
            continue;
        }
        if line.starts_with(">>>>>>> ") {
            in_conflict = false;
            in_theirs = false;
            continue;
        }
        if !in_theirs {
            result.push(line);
        }
    }
    result.join("\n")
}

/// Strip conflict markers from text, keeping "theirs" side.
pub fn resolve_theirs(text: &str) -> String {
    let mut result = Vec::new();
    let mut in_ours = false;
    let mut past_separator = false;

    for line in text.lines() {
        if line.starts_with("<<<<<<< ") {
            in_ours = true;
            past_separator = false;
            continue;
        }
        if line.starts_with("||||||| ") {
            continue;
        }
        if line == "=======" && in_ours {
            past_separator = true;
            continue;
        }
        if line.starts_with(">>>>>>> ") {
            in_ours = false;
            past_separator = false;
            continue;
        }
        if !in_ours || past_separator {
            result.push(line);
        }
    }
    result.join("\n")
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_merge_no_conflicts() {
        let base = "a\nb\nc\nd\ne";
        let ours = "a\nX\nc\nd\ne";   // changed line 2
        let theirs = "a\nb\nc\nY\ne"; // changed line 4
        let result = merge(base, ours, theirs);
        assert!(result.clean);
        assert!(result.conflicts.is_empty());
        assert!(result.text.contains("X"));
        assert!(result.text.contains("Y"));
    }

    #[test]
    fn conflict_on_same_line() {
        let base = "a\nb\nc";
        let ours = "a\nX\nc";
        let theirs = "a\nY\nc";
        let result = merge(base, ours, theirs);
        assert!(!result.clean);
        assert!(!result.conflicts.is_empty());
    }

    #[test]
    fn identical_changes_no_conflict() {
        let base = "a\nb\nc";
        let ours = "a\nX\nc";
        let theirs = "a\nX\nc";
        let result = merge(base, ours, theirs);
        assert!(result.clean);
        assert!(result.text.contains("X"));
    }

    #[test]
    fn no_changes() {
        let base = "a\nb\nc";
        let result = merge(base, base, base);
        assert!(result.clean);
        assert_eq!(result.text, base);
    }

    #[test]
    fn one_side_changed() {
        let base = "a\nb\nc";
        let ours = "a\nX\nc";
        let result = merge(base, ours, base);
        assert!(result.clean);
        assert!(result.text.contains("X"));
    }

    #[test]
    fn conflict_markers_git_style() {
        let base = "a\nb\nc";
        let ours = "a\nX\nc";
        let theirs = "a\nY\nc";
        let result = merge(base, ours, theirs);
        assert!(result.text.contains("<<<<<<<"));
        assert!(result.text.contains("======="));
        assert!(result.text.contains(">>>>>>>"));
    }

    #[test]
    fn conflict_markers_diff3_style() {
        let base = "a\nb\nc";
        let ours = "a\nX\nc";
        let theirs = "a\nY\nc";
        let config = MergeConfig {
            marker_style: MarkerStyle::Diff3,
            ..MergeConfig::default()
        };
        let result = merge_with_config(base, ours, theirs, &config);
        assert!(result.text.contains("|||||||"));
    }

    #[test]
    fn resolve_ours_strips_markers() {
        let text = "a\n<<<<<<< ours\nX\n=======\nY\n>>>>>>> theirs\nc";
        let resolved = resolve_ours(text);
        assert!(resolved.contains("X"));
        assert!(!resolved.contains("Y"));
        assert!(!resolved.contains("<<<<<<<"));
    }

    #[test]
    fn resolve_theirs_strips_markers() {
        let text = "a\n<<<<<<< ours\nX\n=======\nY\n>>>>>>> theirs\nc";
        let resolved = resolve_theirs(text);
        assert!(resolved.contains("Y"));
        assert!(!resolved.contains("X"));
    }

    #[test]
    fn conflict_count_test() {
        let base = "a\nb\nc";
        let ours = "a\nX\nc";
        let theirs = "a\nY\nc";
        let result = merge(base, ours, theirs);
        assert_eq!(conflict_count(&result), 1);
    }

    #[test]
    fn has_conflicts_test() {
        let base = "a\nb\nc";
        let ours = "a\nX\nc";
        let theirs = "a\nY\nc";
        let result = merge(base, ours, theirs);
        assert!(has_conflicts(&result));
    }

    #[test]
    fn empty_base() {
        let result = merge("", "hello", "world");
        // Both added content to an empty base — may conflict.
        assert!(!result.text.is_empty());
    }

    #[test]
    fn conflict_display() {
        let c = Conflict {
            base_start: 5,
            base_count: 3,
            ours: vec!["x".into()],
            theirs: vec!["y".into()],
            base: vec!["z".into()],
        };
        let s = c.to_string();
        assert!(s.contains("5"));
        assert!(s.contains("3"));
    }

    #[test]
    fn word_level_merge_no_conflict() {
        let base = "the quick brown fox";
        let ours = "the fast brown fox";
        let theirs = "the quick brown dog";
        let result = merge_words(base, ours, theirs);
        assert!(result.clean);
    }

    #[test]
    fn word_level_merge_conflict() {
        let base = "the quick brown fox";
        let ours = "the fast brown fox";
        let theirs = "the slow brown fox";
        let result = merge_words(base, ours, theirs);
        // Both changed "quick" to different words.
        assert!(!result.clean);
    }

    #[test]
    fn custom_labels() {
        let config = MergeConfig {
            marker_style: MarkerStyle::Git,
            ours_label: "HEAD".into(),
            theirs_label: "feature-branch".into(),
            base_label: "base".into(),
        };
        let base = "a\nb\nc";
        let ours = "a\nX\nc";
        let theirs = "a\nY\nc";
        let result = merge_with_config(base, ours, theirs, &config);
        assert!(result.text.contains("HEAD"));
        assert!(result.text.contains("feature-branch"));
    }

    #[test]
    fn merge_result_fields() {
        let base = "a\nb";
        let ours = "a\nX";
        let theirs = "a\nY";
        let result = merge(base, ours, theirs);
        assert!(!result.text.is_empty());
        assert_eq!(result.clean, result.conflicts.is_empty());
    }

    #[test]
    fn default_config_values() {
        let cfg = MergeConfig::default();
        assert_eq!(cfg.marker_style, MarkerStyle::Git);
        assert_eq!(cfg.ours_label, "ours");
        assert_eq!(cfg.theirs_label, "theirs");
    }
}
