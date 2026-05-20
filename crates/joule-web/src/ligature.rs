//! Ligature substitution.
//!
//! Replaces sequences of characters/glyphs with ligature glyphs using
//! greedy longest-match lookup, with support for boundary inhibition
//! and language-specific rule sets.

use std::collections::HashMap;

// ── Types ─────────────────────────────────────────────────────────

/// A single ligature rule: a sequence of input code points → a replacement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LigatureRule {
    /// Sequence of characters that form this ligature.
    pub sequence: Vec<char>,
    /// Replacement character (the ligature glyph).
    pub replacement: char,
}

impl LigatureRule {
    pub fn new(sequence: &[char], replacement: char) -> Self {
        Self {
            sequence: sequence.to_vec(),
            replacement,
        }
    }
}

/// A boundary marker that inhibits ligature formation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Boundary {
    /// Zero-width non-joiner (U+200C) — prevents ligature across this point.
    Zwnj,
    /// Morpheme boundary (application-defined).
    Morpheme,
}

// ── Common Latin ligatures ────────────────────────────────────────

/// Unicode code points for common Latin ligatures.
pub const LIGATURE_FI: char = '\u{FB01}';
pub const LIGATURE_FL: char = '\u{FB02}';
pub const LIGATURE_FF: char = '\u{FB00}';
pub const LIGATURE_FFI: char = '\u{FB03}';
pub const LIGATURE_FFL: char = '\u{FB04}';

/// Return the standard Latin ligature rules.
pub fn latin_ligatures() -> Vec<LigatureRule> {
    vec![
        // Longest sequences first for correct greedy matching.
        LigatureRule::new(&['f', 'f', 'i'], LIGATURE_FFI),
        LigatureRule::new(&['f', 'f', 'l'], LIGATURE_FFL),
        LigatureRule::new(&['f', 'f'], LIGATURE_FF),
        LigatureRule::new(&['f', 'i'], LIGATURE_FI),
        LigatureRule::new(&['f', 'l'], LIGATURE_FL),
    ]
}

// ── Ligature Table ────────────────────────────────────────────────

/// A ligature lookup table for efficient substitution.
#[derive(Debug, Clone)]
pub struct LigatureTable {
    /// Rules grouped by first character, sorted longest-first within each group.
    rules_by_first: HashMap<char, Vec<LigatureRule>>,
    /// Maximum sequence length across all rules.
    max_len: usize,
}

impl LigatureTable {
    /// Build a lookup table from a set of rules.
    pub fn new(rules: &[LigatureRule]) -> Self {
        let mut map: HashMap<char, Vec<LigatureRule>> = HashMap::new();
        let mut max_len = 0;

        for rule in rules {
            if rule.sequence.is_empty() {
                continue;
            }
            max_len = max_len.max(rule.sequence.len());
            map.entry(rule.sequence[0])
                .or_default()
                .push(rule.clone());
        }

        // Sort each group longest-first for greedy matching.
        for group in map.values_mut() {
            group.sort_by(|a, b| b.sequence.len().cmp(&a.sequence.len()));
        }

        Self {
            rules_by_first: map,
            max_len,
        }
    }

    /// Create a table with the standard Latin ligatures.
    pub fn latin() -> Self {
        Self::new(&latin_ligatures())
    }

    /// Check if the table is empty.
    pub fn is_empty(&self) -> bool {
        self.rules_by_first.is_empty()
    }

    /// Number of distinct rules.
    pub fn len(&self) -> usize {
        self.rules_by_first.values().map(|v| v.len()).sum()
    }

    /// Try to match a ligature starting at `pos` in `chars`.
    ///
    /// Returns `Some((replacement, consumed_count))` on match.
    fn try_match(&self, chars: &[char], pos: usize, boundaries: &[bool]) -> Option<(char, usize)> {
        let first = chars[pos];
        let candidates = self.rules_by_first.get(&first)?;

        'outer: for rule in candidates {
            let seq_len = rule.sequence.len();
            if pos + seq_len > chars.len() {
                continue;
            }

            // Check for boundary inhibition within the sequence.
            for k in pos..pos + seq_len - 1 {
                if boundaries[k] {
                    continue 'outer;
                }
            }

            // Match characters.
            if chars[pos..pos + seq_len] == rule.sequence[..] {
                return Some((rule.replacement, seq_len));
            }
        }

        None
    }
}

// ── Apply ligatures ───────────────────────────────────────────────

/// Apply ligature substitution to a character sequence.
///
/// `boundaries` marks positions where ligatures are inhibited: if
/// `boundaries[i]` is true, no ligature may span across position `i` and `i+1`.
pub fn apply_ligatures(
    text: &str,
    table: &LigatureTable,
    boundaries: &[bool],
) -> String {
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();

    // Extend boundaries to full length if needed.
    let bounds: Vec<bool> = (0..len)
        .map(|i| boundaries.get(i).copied().unwrap_or(false))
        .collect();

    let mut result = String::new();
    let mut i = 0;

    while i < len {
        if let Some((replacement, consumed)) = table.try_match(&chars, i, &bounds) {
            result.push(replacement);
            i += consumed;
        } else {
            result.push(chars[i]);
            i += 1;
        }
    }

    result
}

/// Apply ligatures with no boundary inhibition.
pub fn apply_ligatures_simple(text: &str, table: &LigatureTable) -> String {
    apply_ligatures(text, table, &[])
}

/// Apply ligatures, inhibiting across ZWNJ (U+200C) characters.
///
/// ZWNJ characters are removed from the output.
pub fn apply_ligatures_zwnj(text: &str, table: &LigatureTable) -> String {
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();

    // Build a filtered char list and boundary list.
    let mut filtered = Vec::new();
    let mut boundaries = Vec::new();

    for (i, &c) in chars.iter().enumerate() {
        if c == '\u{200C}' {
            // Mark boundary at the preceding position.
            if let Some(last) = boundaries.last_mut() {
                *last = true;
            }
        } else {
            filtered.push(c);
            boundaries.push(false);
            // If the next char in source is ZWNJ, the boundary is set above.
            let _ = i; // suppress unused warning
        }
    }

    let text_filtered: String = filtered.iter().collect();
    apply_ligatures(&text_filtered, table, &boundaries)
}

// ── Language-specific rules ───────────────────────────────────────

/// Language identifier for ligature selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Language {
    Latin,
    Turkish,
    Dutch,
}

/// Return ligature rules appropriate for a given language.
///
/// Some languages disable certain ligatures: e.g. Turkish disables fi
/// because 'f' + 'i' should not ligate (the dotless-i distinction matters).
pub fn ligatures_for_language(lang: Language) -> Vec<LigatureRule> {
    match lang {
        Language::Latin => latin_ligatures(),
        Language::Turkish => {
            // Turkish: no fi/ffi ligatures (dotless-i issue).
            latin_ligatures()
                .into_iter()
                .filter(|r| {
                    !r.sequence.contains(&'i')
                })
                .collect()
        }
        Language::Dutch => {
            // Dutch uses standard Latin ligatures plus ij → \u{0133}.
            let mut rules = latin_ligatures();
            rules.push(LigatureRule::new(&['i', 'j'], '\u{0133}'));
            rules
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latin_fi_ligature() {
        let table = LigatureTable::latin();
        let result = apply_ligatures_simple("find", &table);
        assert_eq!(result.chars().next(), Some(LIGATURE_FI));
        assert_eq!(result.chars().count(), 3); // fi→1, n, d
    }

    #[test]
    fn latin_fl_ligature() {
        let table = LigatureTable::latin();
        let result = apply_ligatures_simple("fl", &table);
        assert_eq!(result, LIGATURE_FL.to_string());
    }

    #[test]
    fn latin_ffi_ligature() {
        let table = LigatureTable::latin();
        let result = apply_ligatures_simple("office", &table);
        // o + ffi + c + e → 4 chars
        assert_eq!(result.chars().count(), 4);
        let chars: Vec<char> = result.chars().collect();
        assert_eq!(chars[0], 'o');
        assert_eq!(chars[1], LIGATURE_FFI);
    }

    #[test]
    fn latin_ff_ligature() {
        let table = LigatureTable::latin();
        let result = apply_ligatures_simple("off", &table);
        // o + ff → 2 chars
        assert_eq!(result.chars().count(), 2);
    }

    #[test]
    fn no_ligature_without_match() {
        let table = LigatureTable::latin();
        let result = apply_ligatures_simple("hello", &table);
        assert_eq!(result, "hello");
    }

    #[test]
    fn boundary_inhibits_ligature() {
        let table = LigatureTable::latin();
        // Boundary between f and i prevents fi ligature.
        let result = apply_ligatures("find", &table, &[true, false, false, false]);
        // f should NOT ligate with i.
        let chars: Vec<char> = result.chars().collect();
        assert_eq!(chars[0], 'f');
        assert_eq!(chars[1], 'i');
    }

    #[test]
    fn zwnj_inhibits_ligature() {
        let table = LigatureTable::latin();
        // Insert ZWNJ between f and i.
        let text = "f\u{200C}ind";
        let result = apply_ligatures_zwnj(text, &table);
        let chars: Vec<char> = result.chars().collect();
        assert_eq!(chars[0], 'f');
        assert_eq!(chars[1], 'i');
    }

    #[test]
    fn turkish_no_fi() {
        let rules = ligatures_for_language(Language::Turkish);
        let table = LigatureTable::new(&rules);
        let result = apply_ligatures_simple("find", &table);
        let chars: Vec<char> = result.chars().collect();
        assert_eq!(chars[0], 'f');
        assert_eq!(chars[1], 'i');
    }

    #[test]
    fn dutch_ij_ligature() {
        let rules = ligatures_for_language(Language::Dutch);
        let table = LigatureTable::new(&rules);
        let result = apply_ligatures_simple("ij", &table);
        assert_eq!(result, "\u{0133}");
    }

    #[test]
    fn empty_input() {
        let table = LigatureTable::latin();
        assert_eq!(apply_ligatures_simple("", &table), "");
    }

    #[test]
    fn table_len() {
        let table = LigatureTable::latin();
        assert_eq!(table.len(), 5);
        assert!(!table.is_empty());
    }

    #[test]
    fn multiple_ligatures_in_text() {
        let table = LigatureTable::latin();
        let result = apply_ligatures_simple("find the file", &table);
        // "find" → fi+nd, "the" → the, "file" → fi+le
        // fi(1) n d ' ' t h e ' ' fi(1) l e = 11 chars
        assert_eq!(result.chars().count(), 11);
    }
}
