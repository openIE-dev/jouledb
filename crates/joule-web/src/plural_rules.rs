//! CLDR plural rules engine.
//!
//! Implements plural category selection per UTS #35 for English, French,
//! Arabic, Russian, Polish, and Japanese — pure Rust, no ICU dependency.

use std::fmt;

// ── Plural Category ─────────────────────────────────────────────

/// CLDR plural categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PluralCategory {
    Zero,
    One,
    Two,
    Few,
    Many,
    Other,
}

impl fmt::Display for PluralCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Zero => write!(f, "zero"),
            Self::One => write!(f, "one"),
            Self::Two => write!(f, "two"),
            Self::Few => write!(f, "few"),
            Self::Many => write!(f, "many"),
            Self::Other => write!(f, "other"),
        }
    }
}

// ── Operands ────────────────────────────────────────────────────

/// CLDR plural operands extracted from a numeric value.
///
/// Per UTS #35:
/// - `n` = absolute value of the source number
/// - `i` = integer digits of n
/// - `v` = number of visible fraction digits (with trailing zeros)
/// - `w` = number of visible fraction digits (without trailing zeros)
/// - `f` = visible fraction digits (with trailing zeros), as integer
/// - `t` = visible fraction digits (without trailing zeros), as integer
/// - `e` = exponent (compact notation, always 0 here)
#[derive(Debug, Clone, PartialEq)]
pub struct PluralOperands {
    pub n: f64,
    pub i: u64,
    pub v: u32,
    pub w: u32,
    pub f: u64,
    pub t: u64,
    pub e: u32,
}

impl PluralOperands {
    /// Extract operands from an integer.
    pub fn from_integer(value: i64) -> Self {
        let abs = value.unsigned_abs();
        Self {
            n: abs as f64,
            i: abs,
            v: 0,
            w: 0,
            f: 0,
            t: 0,
            e: 0,
        }
    }

    /// Extract operands from a string representation (preserves trailing zeros).
    pub fn from_str(s: &str) -> Self {
        let s = s.trim().trim_start_matches('-');
        let (int_part, frac_part) = if let Some(dot_pos) = s.find('.') {
            (&s[..dot_pos], &s[dot_pos + 1..])
        } else {
            (s, "")
        };

        let i: u64 = int_part.parse().unwrap_or(0);
        let v = frac_part.len() as u32;
        let f: u64 = if frac_part.is_empty() {
            0
        } else {
            frac_part.parse().unwrap_or(0)
        };
        let trimmed = frac_part.trim_end_matches('0');
        let w = trimmed.len() as u32;
        let t: u64 = if trimmed.is_empty() {
            0
        } else {
            trimmed.parse().unwrap_or(0)
        };
        let n: f64 = s.parse().unwrap_or(i as f64);

        Self {
            n,
            i,
            v,
            w,
            f,
            t,
            e: 0,
        }
    }

    /// Extract operands from an f64 (uses display representation).
    pub fn from_f64(value: f64) -> Self {
        Self::from_str(&format!("{value}"))
    }
}

// ── Locale ──────────────────────────────────────────────────────

/// Supported locales for plural rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PluralLocale {
    /// English: one (i = 1, v = 0), other
    English,
    /// French: one (i = 0,1), other
    French,
    /// Arabic: zero, one, two, few (3..10), many (11..99), other
    Arabic,
    /// Russian: one (mod 10 = 1, mod 100 != 11), few (mod 10 = 2..4, mod 100 != 12..14),
    /// many (mod 10 = 0 or mod 10 = 5..9 or mod 100 = 11..14), other
    Russian,
    /// Polish: one (i = 1, v = 0), few (mod 10 = 2..4, mod 100 != 12..14),
    /// many (i != 1 and mod 10 = 0..1, or mod 10 = 5..9, or mod 100 = 12..14), other
    Polish,
    /// Japanese: other (no plural forms)
    Japanese,
}

impl PluralLocale {
    /// Select the plural category for the given operands.
    pub fn select(&self, ops: &PluralOperands) -> PluralCategory {
        match self {
            Self::English => plural_english(ops),
            Self::French => plural_french(ops),
            Self::Arabic => plural_arabic(ops),
            Self::Russian => plural_russian(ops),
            Self::Polish => plural_polish(ops),
            Self::Japanese => PluralCategory::Other,
        }
    }

    /// Convenience: select for an integer value.
    pub fn select_integer(&self, n: i64) -> PluralCategory {
        self.select(&PluralOperands::from_integer(n))
    }

    /// Return all categories this locale can produce (in CLDR order).
    pub fn categories(&self) -> &'static [PluralCategory] {
        match self {
            Self::English => &[PluralCategory::One, PluralCategory::Other],
            Self::French => &[PluralCategory::One, PluralCategory::Other],
            Self::Arabic => &[
                PluralCategory::Zero,
                PluralCategory::One,
                PluralCategory::Two,
                PluralCategory::Few,
                PluralCategory::Many,
                PluralCategory::Other,
            ],
            Self::Russian => &[
                PluralCategory::One,
                PluralCategory::Few,
                PluralCategory::Many,
                PluralCategory::Other,
            ],
            Self::Polish => &[
                PluralCategory::One,
                PluralCategory::Few,
                PluralCategory::Many,
                PluralCategory::Other,
            ],
            Self::Japanese => &[PluralCategory::Other],
        }
    }
}

// ── Rule implementations ────────────────────────────────────────

fn plural_english(ops: &PluralOperands) -> PluralCategory {
    // one: i = 1 and v = 0
    if ops.i == 1 && ops.v == 0 {
        PluralCategory::One
    } else {
        PluralCategory::Other
    }
}

fn plural_french(ops: &PluralOperands) -> PluralCategory {
    // one: i = 0,1
    if ops.i == 0 || ops.i == 1 {
        PluralCategory::One
    } else {
        PluralCategory::Other
    }
}

fn plural_arabic(ops: &PluralOperands) -> PluralCategory {
    let n = ops.n;
    if n == 0.0 {
        PluralCategory::Zero
    } else if n == 1.0 {
        PluralCategory::One
    } else if n == 2.0 {
        PluralCategory::Two
    } else {
        let mod100 = ops.i % 100;
        if (3..=10).contains(&mod100) {
            PluralCategory::Few
        } else if (11..=99).contains(&mod100) {
            PluralCategory::Many
        } else {
            PluralCategory::Other
        }
    }
}

fn plural_russian(ops: &PluralOperands) -> PluralCategory {
    if ops.v != 0 {
        return PluralCategory::Other;
    }
    let mod10 = ops.i % 10;
    let mod100 = ops.i % 100;
    if mod10 == 1 && mod100 != 11 {
        PluralCategory::One
    } else if (2..=4).contains(&mod10) && !(12..=14).contains(&mod100) {
        PluralCategory::Few
    } else if mod10 == 0 || (5..=9).contains(&mod10) || (11..=14).contains(&mod100) {
        PluralCategory::Many
    } else {
        PluralCategory::Other
    }
}

fn plural_polish(ops: &PluralOperands) -> PluralCategory {
    if ops.i == 1 && ops.v == 0 {
        return PluralCategory::One;
    }
    if ops.v != 0 {
        return PluralCategory::Other;
    }
    let mod10 = ops.i % 10;
    let mod100 = ops.i % 100;
    if (2..=4).contains(&mod10) && !(12..=14).contains(&mod100) {
        PluralCategory::Few
    } else if mod10 == 0
        || mod10 == 1
        || (5..=9).contains(&mod10)
        || (12..=14).contains(&mod100)
    {
        PluralCategory::Many
    } else {
        PluralCategory::Other
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn english_one_other() {
        let en = PluralLocale::English;
        assert_eq!(en.select_integer(1), PluralCategory::One);
        assert_eq!(en.select_integer(0), PluralCategory::Other);
        assert_eq!(en.select_integer(2), PluralCategory::Other);
        assert_eq!(en.select_integer(100), PluralCategory::Other);
    }

    #[test]
    fn english_decimal_not_one() {
        let en = PluralLocale::English;
        let ops = PluralOperands::from_str("1.0");
        // 1.0 has v=1, so not "one" in English
        assert_eq!(en.select(&ops), PluralCategory::Other);
    }

    #[test]
    fn french_zero_is_one() {
        let fr = PluralLocale::French;
        assert_eq!(fr.select_integer(0), PluralCategory::One);
        assert_eq!(fr.select_integer(1), PluralCategory::One);
        assert_eq!(fr.select_integer(2), PluralCategory::Other);
    }

    #[test]
    fn arabic_full_range() {
        let ar = PluralLocale::Arabic;
        assert_eq!(ar.select_integer(0), PluralCategory::Zero);
        assert_eq!(ar.select_integer(1), PluralCategory::One);
        assert_eq!(ar.select_integer(2), PluralCategory::Two);
        assert_eq!(ar.select_integer(5), PluralCategory::Few);
        assert_eq!(ar.select_integer(11), PluralCategory::Many);
        assert_eq!(ar.select_integer(100), PluralCategory::Other);
    }

    #[test]
    fn russian_rules() {
        let ru = PluralLocale::Russian;
        assert_eq!(ru.select_integer(1), PluralCategory::One);
        assert_eq!(ru.select_integer(21), PluralCategory::One);
        assert_eq!(ru.select_integer(2), PluralCategory::Few);
        assert_eq!(ru.select_integer(24), PluralCategory::Few);
        assert_eq!(ru.select_integer(5), PluralCategory::Many);
        assert_eq!(ru.select_integer(11), PluralCategory::Many);
        assert_eq!(ru.select_integer(0), PluralCategory::Many);
    }

    #[test]
    fn polish_rules() {
        let pl = PluralLocale::Polish;
        assert_eq!(pl.select_integer(1), PluralCategory::One);
        assert_eq!(pl.select_integer(2), PluralCategory::Few);
        assert_eq!(pl.select_integer(3), PluralCategory::Few);
        assert_eq!(pl.select_integer(22), PluralCategory::Few);
        assert_eq!(pl.select_integer(5), PluralCategory::Many);
        assert_eq!(pl.select_integer(12), PluralCategory::Many);
        assert_eq!(pl.select_integer(0), PluralCategory::Many);
    }

    #[test]
    fn japanese_always_other() {
        let ja = PluralLocale::Japanese;
        for n in 0..20 {
            assert_eq!(ja.select_integer(n), PluralCategory::Other);
        }
    }

    #[test]
    fn operands_from_string() {
        let ops = PluralOperands::from_str("1.30");
        assert_eq!(ops.i, 1);
        assert_eq!(ops.v, 2); // "30" has 2 visible digits
        assert_eq!(ops.w, 1); // "3" after trimming trailing zeros
        assert_eq!(ops.f, 30);
        assert_eq!(ops.t, 3);
    }

    #[test]
    fn operands_from_integer() {
        let ops = PluralOperands::from_integer(-42);
        assert_eq!(ops.i, 42);
        assert_eq!(ops.n, 42.0);
        assert_eq!(ops.v, 0);
    }

    #[test]
    fn category_display() {
        assert_eq!(PluralCategory::One.to_string(), "one");
        assert_eq!(PluralCategory::Other.to_string(), "other");
        assert_eq!(PluralCategory::Few.to_string(), "few");
    }

    #[test]
    fn locale_categories_count() {
        assert_eq!(PluralLocale::English.categories().len(), 2);
        assert_eq!(PluralLocale::Arabic.categories().len(), 6);
        assert_eq!(PluralLocale::Japanese.categories().len(), 1);
    }

    #[test]
    fn russian_edge_cases() {
        let ru = PluralLocale::Russian;
        // 111 → mod10=1, mod100=11 → many (not one)
        assert_eq!(ru.select_integer(111), PluralCategory::Many);
        // 112 → mod10=2, mod100=12 → many (not few)
        assert_eq!(ru.select_integer(112), PluralCategory::Many);
        // 101 → mod10=1, mod100=1 → one
        assert_eq!(ru.select_integer(101), PluralCategory::One);
    }
}
