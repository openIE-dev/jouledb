//! Multi-column sorting engine: sort specs, natural sort, case-insensitive,
//! null handling, stable sort, direction cycling.
//!
//! Replaces AG Grid / TanStack Table sort logic with pure Rust.

use std::cmp::Ordering;
use std::collections::HashMap;

// ── SortDirection ───────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortDirection {
    Asc,
    Desc,
}

// ── NullPosition ────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NullPosition {
    First,
    Last,
}

impl Default for NullPosition {
    fn default() -> Self { NullPosition::Last }
}

// ── SortSpec ────────────────────────────────────────────────────

/// A single sort criterion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SortSpec {
    pub field: String,
    pub direction: SortDirection,
}

impl SortSpec {
    pub fn asc(field: impl Into<String>) -> Self {
        Self { field: field.into(), direction: SortDirection::Asc }
    }

    pub fn desc(field: impl Into<String>) -> Self {
        Self { field: field.into(), direction: SortDirection::Desc }
    }
}

// ── SortEngine ──────────────────────────────────────────────────

/// Multi-column sort engine.
#[derive(Debug, Clone)]
pub struct SortEngine {
    /// Primary, secondary, … sort specs.
    pub specs: Vec<SortSpec>,
    /// Case-insensitive comparison.
    pub case_insensitive: bool,
    /// Natural sort (numbers within strings: "item2" < "item10").
    pub natural_sort: bool,
    /// Where nulls / empty strings sort.
    pub null_position: NullPosition,
}

impl SortEngine {
    pub fn new() -> Self {
        Self {
            specs: Vec::new(),
            case_insensitive: true,
            natural_sort: false,
            null_position: NullPosition::Last,
        }
    }

    /// Add a sort spec (appends as next priority).
    pub fn add(&mut self, spec: SortSpec) {
        // Remove existing spec for same field.
        self.specs.retain(|s| s.field != spec.field);
        self.specs.push(spec);
    }

    /// Remove sort on a field.
    pub fn remove(&mut self, field: &str) {
        self.specs.retain(|s| s.field != field);
    }

    /// Clear all sort specs.
    pub fn clear(&mut self) {
        self.specs.clear();
    }

    /// Toggle sort direction cycle: none -> asc -> desc -> none.
    /// Returns the new state for the field (None = removed).
    pub fn toggle(&mut self, field: &str) -> Option<SortDirection> {
        if let Some(pos) = self.specs.iter().position(|s| s.field == field) {
            match self.specs[pos].direction {
                SortDirection::Asc => {
                    self.specs[pos].direction = SortDirection::Desc;
                    Some(SortDirection::Desc)
                }
                SortDirection::Desc => {
                    self.specs.remove(pos);
                    None
                }
            }
        } else {
            self.specs.push(SortSpec::asc(field));
            Some(SortDirection::Asc)
        }
    }

    /// Sort rows in-place.  Each row is a `HashMap<String, String>`.
    pub fn sort(&self, rows: &mut [HashMap<String, String>]) {
        if self.specs.is_empty() {
            return;
        }
        // Stable sort.
        rows.sort_by(|a, b| self.compare_rows(a, b));
    }

    /// Sort and return indices of the original positions.
    pub fn sort_indices(&self, rows: &[HashMap<String, String>]) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..rows.len()).collect();
        if self.specs.is_empty() {
            return indices;
        }
        indices.sort_by(|&ai, &bi| self.compare_rows(&rows[ai], &rows[bi]));
        indices
    }

    fn compare_rows(
        &self,
        a: &HashMap<String, String>,
        b: &HashMap<String, String>,
    ) -> Ordering {
        for spec in &self.specs {
            let va = a.get(&spec.field).map(|s| s.as_str()).unwrap_or("");
            let vb = b.get(&spec.field).map(|s| s.as_str()).unwrap_or("");
            let ord = self.compare_values(va, vb);
            if ord != Ordering::Equal {
                return match spec.direction {
                    SortDirection::Asc => ord,
                    SortDirection::Desc => ord.reverse(),
                };
            }
        }
        Ordering::Equal
    }

    fn compare_values(&self, a: &str, b: &str) -> Ordering {
        let a_empty = a.is_empty();
        let b_empty = b.is_empty();

        // Null handling.
        if a_empty && b_empty {
            return Ordering::Equal;
        }
        if a_empty {
            return match self.null_position {
                NullPosition::First => Ordering::Less,
                NullPosition::Last => Ordering::Greater,
            };
        }
        if b_empty {
            return match self.null_position {
                NullPosition::First => Ordering::Greater,
                NullPosition::Last => Ordering::Less,
            };
        }

        if self.natural_sort {
            return natural_cmp(a, b, self.case_insensitive);
        }

        // Try numeric comparison first.
        if let (Ok(na), Ok(nb)) = (a.parse::<f64>(), b.parse::<f64>()) {
            return na.partial_cmp(&nb).unwrap_or(Ordering::Equal);
        }

        if self.case_insensitive {
            a.to_lowercase().cmp(&b.to_lowercase())
        } else {
            a.cmp(b)
        }
    }
}

// ── Natural sort ────────────────────────────────────────────────

/// Compare two strings with natural (human) ordering.
/// Numbers embedded in strings are compared numerically.
fn natural_cmp(a: &str, b: &str, case_insensitive: bool) -> Ordering {
    let a_chunks = split_natural(a);
    let b_chunks = split_natural(b);

    for (ac, bc) in a_chunks.iter().zip(b_chunks.iter()) {
        let ord = match (ac, bc) {
            (NatChunk::Num(na), NatChunk::Num(nb)) => na.partial_cmp(nb).unwrap_or(Ordering::Equal),
            (NatChunk::Text(ta), NatChunk::Text(tb)) => {
                if case_insensitive {
                    ta.to_lowercase().cmp(&tb.to_lowercase())
                } else {
                    ta.cmp(tb)
                }
            }
            (NatChunk::Num(_), NatChunk::Text(_)) => Ordering::Less,
            (NatChunk::Text(_), NatChunk::Num(_)) => Ordering::Greater,
        };
        if ord != Ordering::Equal {
            return ord;
        }
    }
    a_chunks.len().cmp(&b_chunks.len())
}

#[derive(Debug)]
enum NatChunk {
    Num(f64),
    Text(String),
}

fn split_natural(s: &str) -> Vec<NatChunk> {
    let mut chunks = Vec::new();
    let mut buf = String::new();
    let mut in_num = false;

    for ch in s.chars() {
        let is_digit = ch.is_ascii_digit() || ch == '.';
        if is_digit != in_num && !buf.is_empty() {
            if in_num {
                if let Ok(n) = buf.parse::<f64>() {
                    chunks.push(NatChunk::Num(n));
                } else {
                    chunks.push(NatChunk::Text(buf.clone()));
                }
            } else {
                chunks.push(NatChunk::Text(buf.clone()));
            }
            buf.clear();
        }
        in_num = is_digit;
        buf.push(ch);
    }
    if !buf.is_empty() {
        if in_num {
            if let Ok(n) = buf.parse::<f64>() {
                chunks.push(NatChunk::Num(n));
            } else {
                chunks.push(NatChunk::Text(buf));
            }
        } else {
            chunks.push(NatChunk::Text(buf));
        }
    }
    chunks
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_row(name: &str, age: &str) -> HashMap<String, String> {
        HashMap::from([
            ("name".into(), name.into()),
            ("age".into(), age.into()),
        ])
    }

    #[test]
    fn sort_asc() {
        let mut engine = SortEngine::new();
        engine.add(SortSpec::asc("name"));
        let mut rows = vec![
            make_row("Charlie", "35"),
            make_row("Alice", "30"),
            make_row("Bob", "25"),
        ];
        engine.sort(&mut rows);
        assert_eq!(rows[0]["name"], "Alice");
        assert_eq!(rows[1]["name"], "Bob");
        assert_eq!(rows[2]["name"], "Charlie");
    }

    #[test]
    fn sort_desc() {
        let mut engine = SortEngine::new();
        engine.add(SortSpec::desc("name"));
        let mut rows = vec![
            make_row("Alice", "30"),
            make_row("Charlie", "35"),
            make_row("Bob", "25"),
        ];
        engine.sort(&mut rows);
        assert_eq!(rows[0]["name"], "Charlie");
    }

    #[test]
    fn multi_column_sort() {
        let mut engine = SortEngine::new();
        engine.add(SortSpec::asc("age"));
        engine.add(SortSpec::asc("name"));
        let mut rows = vec![
            make_row("Bob", "30"),
            make_row("Alice", "30"),
            make_row("Charlie", "25"),
        ];
        // Primary: age.  Secondary: name.
        // But specs are ordered: age first, then name.
        engine.sort(&mut rows);
        // age 25 first, then age 30 + Alice, age 30 + Bob.
        assert_eq!(rows[0]["name"], "Charlie");
        assert_eq!(rows[1]["name"], "Alice");
        assert_eq!(rows[2]["name"], "Bob");
    }

    #[test]
    fn natural_sort_numbers_in_strings() {
        let mut engine = SortEngine::new();
        engine.natural_sort = true;
        engine.add(SortSpec::asc("name"));
        let mut rows = vec![
            make_row("item10", ""),
            make_row("item2", ""),
            make_row("item1", ""),
        ];
        engine.sort(&mut rows);
        assert_eq!(rows[0]["name"], "item1");
        assert_eq!(rows[1]["name"], "item2");
        assert_eq!(rows[2]["name"], "item10");
    }

    #[test]
    fn case_insensitive_sort() {
        let mut engine = SortEngine::new();
        engine.case_insensitive = true;
        engine.add(SortSpec::asc("name"));
        let mut rows = vec![
            make_row("banana", ""),
            make_row("Apple", ""),
        ];
        engine.sort(&mut rows);
        assert_eq!(rows[0]["name"], "Apple");
    }

    #[test]
    fn nulls_last() {
        let mut engine = SortEngine::new();
        engine.null_position = NullPosition::Last;
        engine.add(SortSpec::asc("name"));
        let mut rows = vec![
            make_row("", ""),
            make_row("Alice", ""),
            make_row("Bob", ""),
        ];
        engine.sort(&mut rows);
        assert_eq!(rows[0]["name"], "Alice");
        assert_eq!(rows[2]["name"], "");
    }

    #[test]
    fn nulls_first() {
        let mut engine = SortEngine::new();
        engine.null_position = NullPosition::First;
        engine.add(SortSpec::asc("name"));
        let mut rows = vec![
            make_row("Alice", ""),
            make_row("", ""),
            make_row("Bob", ""),
        ];
        engine.sort(&mut rows);
        assert_eq!(rows[0]["name"], "");
        assert_eq!(rows[1]["name"], "Alice");
    }

    #[test]
    fn toggle_cycle() {
        let mut engine = SortEngine::new();
        // none -> asc
        assert_eq!(engine.toggle("name"), Some(SortDirection::Asc));
        assert_eq!(engine.specs.len(), 1);
        // asc -> desc
        assert_eq!(engine.toggle("name"), Some(SortDirection::Desc));
        // desc -> none
        assert_eq!(engine.toggle("name"), None);
        assert!(engine.specs.is_empty());
    }

    #[test]
    fn sort_indices() {
        let mut engine = SortEngine::new();
        engine.add(SortSpec::asc("age"));
        let rows = vec![
            make_row("Charlie", "35"),
            make_row("Alice", "25"),
            make_row("Bob", "30"),
        ];
        let indices = engine.sort_indices(&rows);
        assert_eq!(indices, vec![1, 2, 0]); // 25, 30, 35
    }

    #[test]
    fn stable_sort_preserves_order() {
        let mut engine = SortEngine::new();
        engine.add(SortSpec::asc("age"));
        let mut rows = vec![
            make_row("A", "30"),
            make_row("B", "30"),
            make_row("C", "30"),
        ];
        engine.sort(&mut rows);
        // All same age — stable sort preserves original order.
        assert_eq!(rows[0]["name"], "A");
        assert_eq!(rows[1]["name"], "B");
        assert_eq!(rows[2]["name"], "C");
    }

    #[test]
    fn add_replaces_existing_field() {
        let mut engine = SortEngine::new();
        engine.add(SortSpec::asc("name"));
        engine.add(SortSpec::desc("name"));
        assert_eq!(engine.specs.len(), 1);
        assert_eq!(engine.specs[0].direction, SortDirection::Desc);
    }

    #[test]
    fn remove_and_clear() {
        let mut engine = SortEngine::new();
        engine.add(SortSpec::asc("name"));
        engine.add(SortSpec::asc("age"));
        engine.remove("name");
        assert_eq!(engine.specs.len(), 1);
        engine.clear();
        assert!(engine.specs.is_empty());
    }

    #[test]
    fn numeric_sort() {
        let mut engine = SortEngine::new();
        engine.add(SortSpec::asc("age"));
        let mut rows = vec![
            make_row("", "100"),
            make_row("", "20"),
            make_row("", "3"),
        ];
        engine.sort(&mut rows);
        assert_eq!(rows[0]["age"], "3");
        assert_eq!(rows[1]["age"], "20");
        assert_eq!(rows[2]["age"], "100");
    }
}
