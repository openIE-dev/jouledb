//! List formatter — conjunction, disjunction, and unit lists.
//!
//! Formats lists like "A, B, and C" (conjunction), "A, B, or C" (disjunction),
//! or "A, B, C" (unit), with locale-aware separators and Oxford comma option.

use std::fmt;

// ── List type ───────────────────────────────────────────────────

/// The type of list conjunction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ListType {
    /// "A, B, and C"
    Conjunction,
    /// "A, B, or C"
    Disjunction,
    /// "A, B, C" (no conjunction word)
    Unit,
}

// ── List style ──────────────────────────────────────────────────

/// Formatting style for lists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ListStyle {
    /// Full formatting with conjunction word.
    Long,
    /// Shorter: "A, B, & C"
    Short,
    /// Narrow: "A, B, C" (unit-like even for conjunction).
    Narrow,
}

// ── Locale patterns ─────────────────────────────────────────────

/// Locale-specific list formatting patterns.
#[derive(Debug, Clone)]
pub struct ListLocale {
    /// Separator between items (not the last pair).
    pub separator: &'static str,
    /// Conjunction word for the last pair ("and", "or", etc.).
    pub conjunction: &'static str,
    /// Disjunction word for the last pair.
    pub disjunction: &'static str,
    /// Short conjunction (" & ").
    pub short_conjunction: &'static str,
    /// Short disjunction.
    pub short_disjunction: &'static str,
    /// Two-item conjunction (" and ").
    pub two_conjunction: &'static str,
    /// Two-item disjunction (" or ").
    pub two_disjunction: &'static str,
}

impl ListLocale {
    /// English locale patterns.
    pub fn english() -> Self {
        Self {
            separator: ", ",
            conjunction: " and ",
            disjunction: " or ",
            short_conjunction: " & ",
            short_disjunction: " or ",
            two_conjunction: " and ",
            two_disjunction: " or ",
        }
    }

    /// French locale patterns.
    pub fn french() -> Self {
        Self {
            separator: ", ",
            conjunction: " et ",
            disjunction: " ou ",
            short_conjunction: " et ",
            short_disjunction: " ou ",
            two_conjunction: " et ",
            two_disjunction: " ou ",
        }
    }

    /// Spanish locale patterns.
    pub fn spanish() -> Self {
        Self {
            separator: ", ",
            conjunction: " y ",
            disjunction: " o ",
            short_conjunction: " y ",
            short_disjunction: " o ",
            two_conjunction: " y ",
            two_disjunction: " o ",
        }
    }
}

impl Default for ListLocale {
    fn default() -> Self {
        Self::english()
    }
}

// ── ListFormatter ───────────────────────────────────────────────

/// Formats a list of items into a human-readable string.
#[derive(Debug, Clone)]
pub struct ListFormatter {
    /// List type (conjunction, disjunction, unit).
    pub list_type: ListType,
    /// Formatting style.
    pub style: ListStyle,
    /// Locale patterns.
    pub locale: ListLocale,
    /// Use Oxford comma (serial comma before conjunction).
    pub oxford_comma: bool,
}

impl Default for ListFormatter {
    fn default() -> Self {
        Self {
            list_type: ListType::Conjunction,
            style: ListStyle::Long,
            locale: ListLocale::default(),
            oxford_comma: true,
        }
    }
}

impl ListFormatter {
    /// Create a new conjunction list formatter with Oxford comma.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a conjunction formatter.
    pub fn conjunction() -> Self {
        Self {
            list_type: ListType::Conjunction,
            ..Self::default()
        }
    }

    /// Create a disjunction formatter.
    pub fn disjunction() -> Self {
        Self {
            list_type: ListType::Disjunction,
            ..Self::default()
        }
    }

    /// Create a unit formatter (no conjunction word).
    pub fn unit() -> Self {
        Self {
            list_type: ListType::Unit,
            ..Self::default()
        }
    }

    /// Format a list of items.
    pub fn format<I, S>(&self, items: I) -> String
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let items: Vec<String> = items.into_iter().map(|s| s.as_ref().to_string()).collect();

        match items.len() {
            0 => String::new(),
            1 => items[0].clone(),
            2 => self.format_two(&items[0], &items[1]),
            _ => self.format_many(&items),
        }
    }

    fn format_two(&self, a: &str, b: &str) -> String {
        let connector = match (self.list_type, self.style) {
            (ListType::Unit, _) | (_, ListStyle::Narrow) => self.locale.separator,
            (ListType::Conjunction, ListStyle::Short) => self.locale.short_conjunction,
            (ListType::Disjunction, ListStyle::Short) => self.locale.short_disjunction,
            (ListType::Conjunction, ListStyle::Long) => self.locale.two_conjunction,
            (ListType::Disjunction, ListStyle::Long) => self.locale.two_disjunction,
        };
        format!("{a}{connector}{b}")
    }

    fn format_many(&self, items: &[String]) -> String {
        let last_idx = items.len() - 1;
        let mut result = String::new();

        for (i, item) in items.iter().enumerate() {
            if i == 0 {
                result.push_str(item);
            } else if i == last_idx {
                match (self.list_type, self.style) {
                    (ListType::Unit, _) | (_, ListStyle::Narrow) => {
                        result.push_str(self.locale.separator);
                    }
                    (ListType::Conjunction, ListStyle::Long) => {
                        if self.oxford_comma {
                            result.push(',');
                        }
                        result.push_str(self.locale.conjunction);
                    }
                    (ListType::Conjunction, ListStyle::Short) => {
                        if self.oxford_comma {
                            result.push(',');
                        }
                        result.push_str(self.locale.short_conjunction);
                    }
                    (ListType::Disjunction, ListStyle::Long) => {
                        if self.oxford_comma {
                            result.push(',');
                        }
                        result.push_str(self.locale.disjunction);
                    }
                    (ListType::Disjunction, ListStyle::Short) => {
                        if self.oxford_comma {
                            result.push(',');
                        }
                        result.push_str(self.locale.short_disjunction);
                    }
                }
                result.push_str(item);
            } else {
                result.push_str(self.locale.separator);
                result.push_str(item);
            }
        }
        result
    }

    /// Return the count of items that would be formatted.
    pub fn count<I, S>(&self, items: I) -> usize
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        items.into_iter().count()
    }
}

impl fmt::Display for ListType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Conjunction => write!(f, "conjunction"),
            Self::Disjunction => write!(f, "disjunction"),
            Self::Unit => write!(f, "unit"),
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_list() {
        let f = ListFormatter::conjunction();
        assert_eq!(f.format::<&[&str], _>(&[]), "");
    }

    #[test]
    fn single_item() {
        let f = ListFormatter::conjunction();
        assert_eq!(f.format(&["apple"]), "apple");
    }

    #[test]
    fn two_items_conjunction() {
        let f = ListFormatter::conjunction();
        assert_eq!(f.format(&["apple", "banana"]), "apple and banana");
    }

    #[test]
    fn two_items_disjunction() {
        let f = ListFormatter::disjunction();
        assert_eq!(f.format(&["apple", "banana"]), "apple or banana");
    }

    #[test]
    fn three_items_conjunction_oxford() {
        let f = ListFormatter::conjunction();
        assert_eq!(
            f.format(&["apple", "banana", "cherry"]),
            "apple, banana, and cherry"
        );
    }

    #[test]
    fn three_items_no_oxford() {
        let f = ListFormatter {
            oxford_comma: false,
            ..ListFormatter::conjunction()
        };
        assert_eq!(
            f.format(&["apple", "banana", "cherry"]),
            "apple, banana and cherry"
        );
    }

    #[test]
    fn three_items_disjunction() {
        let f = ListFormatter::disjunction();
        assert_eq!(
            f.format(&["apple", "banana", "cherry"]),
            "apple, banana, or cherry"
        );
    }

    #[test]
    fn unit_list() {
        let f = ListFormatter::unit();
        assert_eq!(
            f.format(&["10 lb", "5 oz", "3 g"]),
            "10 lb, 5 oz, 3 g"
        );
    }

    #[test]
    fn short_style() {
        let f = ListFormatter {
            style: ListStyle::Short,
            ..ListFormatter::conjunction()
        };
        assert_eq!(
            f.format(&["A", "B", "C"]),
            "A, B, & C"
        );
    }

    #[test]
    fn narrow_style() {
        let f = ListFormatter {
            style: ListStyle::Narrow,
            ..ListFormatter::conjunction()
        };
        assert_eq!(f.format(&["A", "B", "C"]), "A, B, C");
    }

    #[test]
    fn french_conjunction() {
        let f = ListFormatter {
            locale: ListLocale::french(),
            oxford_comma: false,
            ..ListFormatter::conjunction()
        };
        assert_eq!(
            f.format(&["pomme", "banane", "cerise"]),
            "pomme, banane et cerise"
        );
    }

    #[test]
    fn many_items() {
        let f = ListFormatter::conjunction();
        let result = f.format(&["A", "B", "C", "D", "E"]);
        assert_eq!(result, "A, B, C, D, and E");
    }

    #[test]
    fn spanish_disjunction() {
        let f = ListFormatter {
            list_type: ListType::Disjunction,
            locale: ListLocale::spanish(),
            oxford_comma: false,
            ..ListFormatter::default()
        };
        assert_eq!(f.format(&["uno", "dos", "tres"]), "uno, dos o tres");
    }

    #[test]
    fn list_type_display() {
        assert_eq!(ListType::Conjunction.to_string(), "conjunction");
        assert_eq!(ListType::Disjunction.to_string(), "disjunction");
        assert_eq!(ListType::Unit.to_string(), "unit");
    }
}
