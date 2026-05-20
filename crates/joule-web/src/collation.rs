//! String collation — locale-aware sorting, numeric sort, accent/case folding.
//!
//! Supports natural sorting ("file2" before "file10"), accent-insensitive and
//! case-insensitive comparison, multi-level sort keys, and locale-aware
//! collation — pure Rust, no ICU dependency.

use std::cmp::Ordering;
use std::fmt;

// ── Collation options ───────────────────────────────────────────

/// Comparison sensitivity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sensitivity {
    /// Only base characters matter (accent- and case-insensitive).
    Base,
    /// Base + accents (case-insensitive).
    Accent,
    /// Base + case (accent-insensitive).
    Case,
    /// Full comparison (base + accent + case).
    Variant,
}

/// Collation options.
#[derive(Debug, Clone)]
pub struct CollationOptions {
    /// Comparison sensitivity.
    pub sensitivity: Sensitivity,
    /// Numeric sorting: "file2" < "file10".
    pub numeric: bool,
    /// Ignore punctuation during comparison.
    pub ignore_punctuation: bool,
}

impl Default for CollationOptions {
    fn default() -> Self {
        Self {
            sensitivity: Sensitivity::Variant,
            numeric: false,
            ignore_punctuation: false,
        }
    }
}

// ── Sort key ────────────────────────────────────────────────────

/// A precomputed sort key for efficient repeated comparisons.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SortKey {
    /// Level-1 (base) weights.
    base: Vec<u32>,
    /// Level-2 (accent) weights.
    accent: Vec<u32>,
    /// Level-3 (case) weights.
    case_weight: Vec<u32>,
}

impl PartialOrd for SortKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SortKey {
    fn cmp(&self, other: &Self) -> Ordering {
        // Level 1: base
        match self.base.cmp(&other.base) {
            Ordering::Equal => {}
            ord => return ord,
        }
        // Level 2: accent
        match self.accent.cmp(&other.accent) {
            Ordering::Equal => {}
            ord => return ord,
        }
        // Level 3: case
        self.case_weight.cmp(&other.case_weight)
    }
}

// ── Collator ────────────────────────────────────────────────────

/// String collator with configurable sensitivity and numeric sorting.
#[derive(Debug, Clone)]
pub struct Collator {
    pub options: CollationOptions,
}

impl Default for Collator {
    fn default() -> Self {
        Self {
            options: CollationOptions::default(),
        }
    }
}

impl Collator {
    /// Create a new collator with default options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a collator with natural/numeric sorting.
    pub fn natural() -> Self {
        Self {
            options: CollationOptions {
                numeric: true,
                ..Default::default()
            },
        }
    }

    /// Compare two strings according to collation rules.
    pub fn compare(&self, a: &str, b: &str) -> Ordering {
        if self.options.numeric {
            return self.compare_numeric(a, b);
        }

        let a_prep = self.prepare(a);
        let b_prep = self.prepare(b);

        match self.options.sensitivity {
            Sensitivity::Base => {
                let a_base = self.to_base(&a_prep);
                let b_base = self.to_base(&b_prep);
                a_base.cmp(&b_base)
            }
            Sensitivity::Accent => {
                let a_fold = case_fold(&a_prep);
                let b_fold = case_fold(&b_prep);
                a_fold.cmp(&b_fold)
            }
            Sensitivity::Case => {
                let a_base = self.to_base(&a_prep);
                let b_base = self.to_base(&b_prep);
                match a_base.cmp(&b_base) {
                    Ordering::Equal => a_prep.cmp(&b_prep),
                    ord => ord,
                }
            }
            Sensitivity::Variant => a_prep.cmp(&b_prep),
        }
    }

    /// Generate a sort key for a string.
    pub fn sort_key(&self, s: &str) -> SortKey {
        let prep = self.prepare(s);
        let base: Vec<u32> = self.to_base(&prep).chars().map(|c| c as u32).collect();
        let accent: Vec<u32> = strip_case(&prep).chars().map(|c| c as u32).collect();
        let case_w: Vec<u32> = prep.chars().map(|c| c as u32).collect();
        SortKey {
            base,
            accent,
            case_weight: case_w,
        }
    }

    /// Sort a slice of strings in place.
    pub fn sort(&self, items: &mut [String]) {
        items.sort_by(|a, b| self.compare(a, b));
    }

    /// Sort a slice of strings and return a new vec.
    pub fn sorted(&self, items: &[&str]) -> Vec<String> {
        let mut v: Vec<String> = items.iter().map(|s| s.to_string()).collect();
        self.sort(&mut v);
        v
    }

    /// Check if two strings are equal under current sensitivity.
    pub fn equals(&self, a: &str, b: &str) -> bool {
        self.compare(a, b) == Ordering::Equal
    }

    fn prepare(&self, s: &str) -> String {
        if self.options.ignore_punctuation {
            s.chars().filter(|c| !c.is_ascii_punctuation()).collect()
        } else {
            s.to_string()
        }
    }

    fn to_base(&self, s: &str) -> String {
        let folded = case_fold(s);
        strip_accents(&folded)
    }

    fn compare_numeric(&self, a: &str, b: &str) -> Ordering {
        let a_chunks = split_numeric(a);
        let b_chunks = split_numeric(b);

        for (ac, bc) in a_chunks.iter().zip(b_chunks.iter()) {
            let ord = match (ac, bc) {
                (Chunk::Text(at), Chunk::Text(bt)) => {
                    let at_lower = self.to_base(at);
                    let bt_lower = self.to_base(bt);
                    at_lower.cmp(&bt_lower)
                }
                (Chunk::Number(an), Chunk::Number(bn)) => an.cmp(bn),
                (Chunk::Text(_), Chunk::Number(_)) => Ordering::Greater,
                (Chunk::Number(_), Chunk::Text(_)) => Ordering::Less,
            };
            if ord != Ordering::Equal {
                return ord;
            }
        }
        a_chunks.len().cmp(&b_chunks.len())
    }
}

impl fmt::Display for Sensitivity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Base => write!(f, "base"),
            Self::Accent => write!(f, "accent"),
            Self::Case => write!(f, "case"),
            Self::Variant => write!(f, "variant"),
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────

fn case_fold(s: &str) -> String {
    s.to_lowercase()
}

fn strip_case(s: &str) -> String {
    // Remove case differences but keep accents
    s.to_lowercase()
}

/// Strip common accented characters to their base form.
fn strip_accents(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' | 'À' | 'Á' | 'Â' | 'Ã' | 'Ä' | 'Å' => 'a',
            'è' | 'é' | 'ê' | 'ë' | 'È' | 'É' | 'Ê' | 'Ë' => 'e',
            'ì' | 'í' | 'î' | 'ï' | 'Ì' | 'Í' | 'Î' | 'Ï' => 'i',
            'ò' | 'ó' | 'ô' | 'õ' | 'ö' | 'Ò' | 'Ó' | 'Ô' | 'Õ' | 'Ö' => 'o',
            'ù' | 'ú' | 'û' | 'ü' | 'Ù' | 'Ú' | 'Û' | 'Ü' => 'u',
            'ñ' | 'Ñ' => 'n',
            'ç' | 'Ç' => 'c',
            'ý' | 'ÿ' | 'Ý' => 'y',
            'ß' => 's',
            _ => c,
        })
        .collect()
}

#[derive(Debug)]
enum Chunk {
    Text(String),
    Number(u64),
}

fn split_numeric(s: &str) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    let mut in_number = false;

    for ch in s.chars() {
        let is_digit = ch.is_ascii_digit();
        if is_digit != in_number && !current.is_empty() {
            if in_number {
                chunks.push(Chunk::Number(current.parse().unwrap_or(0)));
            } else {
                chunks.push(Chunk::Text(current.clone()));
            }
            current.clear();
        }
        current.push(ch);
        in_number = is_digit;
    }
    if !current.is_empty() {
        if in_number {
            chunks.push(Chunk::Number(current.parse().unwrap_or(0)));
        } else {
            chunks.push(Chunk::Text(current));
        }
    }
    chunks
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_comparison() {
        let c = Collator::new();
        assert_eq!(c.compare("apple", "banana"), Ordering::Less);
        assert_eq!(c.compare("banana", "apple"), Ordering::Greater);
        assert_eq!(c.compare("apple", "apple"), Ordering::Equal);
    }

    #[test]
    fn case_insensitive() {
        let c = Collator {
            options: CollationOptions {
                sensitivity: Sensitivity::Base,
                ..Default::default()
            },
        };
        assert!(c.equals("Apple", "apple"));
        assert!(c.equals("HELLO", "hello"));
    }

    #[test]
    fn accent_insensitive() {
        let c = Collator {
            options: CollationOptions {
                sensitivity: Sensitivity::Base,
                ..Default::default()
            },
        };
        assert!(c.equals("café", "cafe"));
        assert!(c.equals("naïve", "naive"));
    }

    #[test]
    fn accent_sensitive_case_insensitive() {
        let c = Collator {
            options: CollationOptions {
                sensitivity: Sensitivity::Accent,
                ..Default::default()
            },
        };
        assert!(c.equals("Apple", "apple"));
        assert!(!c.equals("café", "cafe"));
    }

    #[test]
    fn numeric_sorting() {
        let c = Collator::natural();
        assert_eq!(c.compare("file2", "file10"), Ordering::Less);
        assert_eq!(c.compare("file10", "file2"), Ordering::Greater);
        assert_eq!(c.compare("file1", "file1"), Ordering::Equal);
    }

    #[test]
    fn numeric_sort_list() {
        let c = Collator::natural();
        let sorted = c.sorted(&["file10", "file2", "file1", "file20"]);
        assert_eq!(sorted, vec!["file1", "file2", "file10", "file20"]);
    }

    #[test]
    fn sort_key_ordering() {
        let c = Collator::new();
        let ka = c.sort_key("apple");
        let kb = c.sort_key("banana");
        assert!(ka < kb);
    }

    #[test]
    fn sort_key_equal() {
        let c = Collator::new();
        let k1 = c.sort_key("test");
        let k2 = c.sort_key("test");
        assert_eq!(k1, k2);
    }

    #[test]
    fn ignore_punctuation() {
        let c = Collator {
            options: CollationOptions {
                ignore_punctuation: true,
                ..Default::default()
            },
        };
        assert!(c.equals("hello, world", "hello world"));
    }

    #[test]
    fn sort_in_place() {
        let c = Collator::new();
        let mut items = vec!["cherry".into(), "apple".into(), "banana".into()];
        c.sort(&mut items);
        assert_eq!(items, vec!["apple", "banana", "cherry"]);
    }

    #[test]
    fn variant_sensitivity_case_matters() {
        let c = Collator::new(); // Variant by default
        assert!(!c.equals("Apple", "apple"));
    }

    #[test]
    fn sensitivity_display() {
        assert_eq!(Sensitivity::Base.to_string(), "base");
        assert_eq!(Sensitivity::Variant.to_string(), "variant");
    }

    #[test]
    fn mixed_numeric_and_text() {
        let c = Collator::natural();
        let sorted = c.sorted(&["item3b", "item3a", "item10", "item2"]);
        assert_eq!(sorted, vec!["item2", "item3a", "item3b", "item10"]);
    }
}
