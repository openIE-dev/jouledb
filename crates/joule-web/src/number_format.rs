//! Number formatter — decimal, currency, percent, and compact notation.
//!
//! Supports locale-aware grouping separators, min/max fraction digits,
//! significant digits, and compact notation (1K, 1M, 1B) — pure Rust.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Number formatting errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NumberFormatError {
    /// Invalid format options.
    InvalidOptions(String),
}

impl fmt::Display for NumberFormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions(msg) => write!(f, "invalid options: {msg}"),
        }
    }
}

impl std::error::Error for NumberFormatError {}

// ── Locale separators ───────────────────────────────────────────

/// Locale-specific number formatting symbols.
#[derive(Debug, Clone)]
pub struct NumberSymbols {
    /// Decimal separator (e.g., '.' or ',').
    pub decimal: char,
    /// Grouping separator (e.g., ',' or '.').
    pub group: char,
    /// Minus sign.
    pub minus: char,
    /// Percent sign.
    pub percent: char,
}

impl NumberSymbols {
    /// US/English symbols.
    pub fn english() -> Self {
        Self {
            decimal: '.',
            group: ',',
            minus: '-',
            percent: '%',
        }
    }

    /// German/many European locales.
    pub fn german() -> Self {
        Self {
            decimal: ',',
            group: '.',
            minus: '-',
            percent: '%',
        }
    }

    /// French (space as grouping separator).
    pub fn french() -> Self {
        Self {
            decimal: ',',
            group: '\u{202F}', // narrow no-break space
            minus: '-',
            percent: '%',
        }
    }

    /// Indian numbering (uses commas but different grouping).
    pub fn indian() -> Self {
        Self {
            decimal: '.',
            group: ',',
            minus: '-',
            percent: '%',
        }
    }
}

impl Default for NumberSymbols {
    fn default() -> Self {
        Self::english()
    }
}

// ── Currency display mode ───────────────────────────────────────

/// How to display the currency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurrencyDisplay {
    /// Symbol: $, €, £
    Symbol,
    /// ISO 4217 code: USD, EUR, GBP
    Code,
    /// Full name: US Dollar, Euro
    Name,
}

// ── Compact display ─────────────────────────────────────────────

/// Compact notation thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactDisplay {
    /// Short: 1K, 1M, 1B
    Short,
    /// Long: 1 thousand, 1 million
    Long,
}

// ── NumberFormat ─────────────────────────────────────────────────

/// Number formatting options.
#[derive(Debug, Clone)]
pub struct NumberFormat {
    /// Locale symbols.
    pub symbols: NumberSymbols,
    /// Minimum integer digits (pad with zeros).
    pub min_integer_digits: u32,
    /// Minimum fraction digits.
    pub min_fraction_digits: u32,
    /// Maximum fraction digits.
    pub max_fraction_digits: u32,
    /// Use grouping separators.
    pub use_grouping: bool,
    /// Indian-style grouping (first group of 3, then groups of 2).
    pub indian_grouping: bool,
    /// Significant digits constraint (if Some, overrides fraction digits).
    pub significant_digits: Option<(u32, u32)>, // (min, max)
}

impl Default for NumberFormat {
    fn default() -> Self {
        Self {
            symbols: NumberSymbols::default(),
            min_integer_digits: 1,
            min_fraction_digits: 0,
            max_fraction_digits: 3,
            use_grouping: true,
            indian_grouping: false,
            significant_digits: None,
        }
    }
}

impl NumberFormat {
    /// Create a new formatter with default (English) settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Format a number as a decimal string.
    pub fn format_decimal(&self, value: f64) -> String {
        let negative = value < 0.0;
        let abs = value.abs();

        let (int_str, frac_str) = if let Some((min_sig, max_sig)) = self.significant_digits {
            format_significant(abs, min_sig, max_sig)
        } else {
            format_fixed(abs, self.min_fraction_digits, self.max_fraction_digits)
        };

        let grouped = if self.use_grouping {
            if self.indian_grouping {
                group_indian(&int_str, self.symbols.group)
            } else {
                group_western(&int_str, self.symbols.group)
            }
        } else {
            pad_int(&int_str, self.min_integer_digits)
        };

        let padded = pad_int(&grouped, self.min_integer_digits);

        let mut result = String::new();
        if negative {
            result.push(self.symbols.minus);
        }
        result.push_str(&padded);
        if !frac_str.is_empty() {
            result.push(self.symbols.decimal);
            result.push_str(&frac_str);
        }
        result
    }

    /// Format as currency.
    pub fn format_currency(
        &self,
        value: f64,
        code: &str,
        symbol: &str,
        name: &str,
        display: CurrencyDisplay,
    ) -> String {
        let num = self.format_decimal(value);
        match display {
            CurrencyDisplay::Symbol => format!("{symbol}{num}"),
            CurrencyDisplay::Code => format!("{num} {code}"),
            CurrencyDisplay::Name => format!("{num} {name}"),
        }
    }

    /// Format as percentage (value is multiplied by 100).
    pub fn format_percent(&self, value: f64) -> String {
        let pct = value * 100.0;
        let num = self.format_decimal(pct);
        format!("{num}{}", self.symbols.percent)
    }

    /// Format in compact notation.
    pub fn format_compact(&self, value: f64, display: CompactDisplay) -> String {
        let negative = value < 0.0;
        let abs = value.abs();

        let (scaled, suffix) = if abs >= 1_000_000_000.0 {
            (abs / 1_000_000_000.0, match display {
                CompactDisplay::Short => "B",
                CompactDisplay::Long => " billion",
            })
        } else if abs >= 1_000_000.0 {
            (abs / 1_000_000.0, match display {
                CompactDisplay::Short => "M",
                CompactDisplay::Long => " million",
            })
        } else if abs >= 1_000.0 {
            (abs / 1_000.0, match display {
                CompactDisplay::Short => "K",
                CompactDisplay::Long => " thousand",
            })
        } else {
            (abs, "")
        };

        // Format with up to 1 fraction digit for compact
        let (int_str, frac_str) = format_fixed(scaled, 0, 1);
        let mut result = String::new();
        if negative {
            result.push(self.symbols.minus);
        }
        result.push_str(&int_str);
        if !frac_str.is_empty() {
            result.push(self.symbols.decimal);
            result.push_str(&frac_str);
        }
        result.push_str(suffix);
        result
    }
}

// ── Helpers ─────────────────────────────────────────────────────

fn format_fixed(value: f64, min_frac: u32, max_frac: u32) -> (String, String) {
    let rounded = format!("{:.prec$}", value, prec = max_frac as usize);
    let parts: Vec<&str> = rounded.split('.').collect();
    let int_str = parts[0].to_string();

    let frac = if parts.len() > 1 { parts[1] } else { "" };
    // Trim trailing zeros but keep at least min_frac digits
    let mut frac_str = frac.to_string();
    while frac_str.len() > min_frac as usize && frac_str.ends_with('0') {
        frac_str.pop();
    }
    (int_str, frac_str)
}

fn format_significant(value: f64, min_sig: u32, max_sig: u32) -> (String, String) {
    if value == 0.0 {
        let frac = "0".repeat(min_sig.saturating_sub(1) as usize);
        return ("0".to_string(), frac);
    }
    let digits = max_sig as usize;
    let magnitude = value.log10().floor() as i32;
    let frac_digits = (digits as i32 - magnitude - 1).max(0) as usize;
    // Round to the correct number of significant digits
    let rounded_value = if frac_digits == 0 {
        let factor = 10f64.powi(magnitude - digits as i32 + 1);
        (value / factor).round() * factor
    } else {
        value
    };
    let rounded = format!("{:.prec$}", rounded_value, prec = frac_digits);
    let parts: Vec<&str> = rounded.split('.').collect();
    let int_str = parts[0].to_string();
    let frac = if parts.len() > 1 {
        parts[1].to_string()
    } else {
        String::new()
    };
    // Ensure minimum significant digits by padding fraction
    let total_digits = int_str.trim_start_matches('0').len() + frac.len();
    let mut frac = frac;
    if total_digits < min_sig as usize {
        let pad = min_sig as usize - total_digits;
        for _ in 0..pad {
            frac.push('0');
        }
    }
    // Trim trailing zeros down to min_sig total
    let int_digits = int_str.trim_start_matches('0').len().max(1);
    let min_frac = min_sig.saturating_sub(int_digits as u32) as usize;
    while frac.len() > min_frac && frac.ends_with('0') {
        frac.pop();
    }
    (int_str, frac)
}

fn group_western(int_str: &str, sep: char) -> String {
    let bytes: Vec<char> = int_str.chars().collect();
    let mut result = String::new();
    let len = bytes.len();
    for (i, ch) in bytes.iter().enumerate() {
        if i > 0 && (len - i) % 3 == 0 {
            result.push(sep);
        }
        result.push(*ch);
    }
    result
}

fn group_indian(int_str: &str, sep: char) -> String {
    let len = int_str.len();
    if len <= 3 {
        return int_str.to_string();
    }
    let (prefix, last3) = int_str.split_at(len - 3);
    let prefix_chars: Vec<char> = prefix.chars().collect();
    let plen = prefix_chars.len();
    let mut result = String::new();
    for (i, ch) in prefix_chars.iter().enumerate() {
        if i > 0 && (plen - i) % 2 == 0 {
            result.push(sep);
        }
        result.push(*ch);
    }
    result.push(sep);
    result.push_str(last3);
    result
}

fn pad_int(int_str: &str, min_digits: u32) -> String {
    let len = int_str.chars().filter(|c| c.is_ascii_digit()).count();
    if len >= min_digits as usize {
        return int_str.to_string();
    }
    let pad = min_digits as usize - len;
    let mut result = "0".repeat(pad);
    result.push_str(int_str);
    result
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_decimal() {
        let fmt = NumberFormat::new();
        assert_eq!(fmt.format_decimal(1234.5), "1,234.5");
    }

    #[test]
    fn no_grouping() {
        let fmt = NumberFormat {
            use_grouping: false,
            ..Default::default()
        };
        assert_eq!(fmt.format_decimal(1234567.0), "1234567");
    }

    #[test]
    fn min_fraction_digits() {
        let fmt = NumberFormat {
            min_fraction_digits: 2,
            max_fraction_digits: 2,
            ..Default::default()
        };
        assert_eq!(fmt.format_decimal(42.0), "42.00");
        assert_eq!(fmt.format_decimal(42.1), "42.10");
    }

    #[test]
    fn max_fraction_digits() {
        let fmt = NumberFormat {
            max_fraction_digits: 2,
            ..Default::default()
        };
        assert_eq!(fmt.format_decimal(3.14159), "3.14");
    }

    #[test]
    fn negative_number() {
        let fmt = NumberFormat::new();
        assert_eq!(fmt.format_decimal(-1234.5), "-1,234.5");
    }

    #[test]
    fn german_locale() {
        let fmt = NumberFormat {
            symbols: NumberSymbols::german(),
            min_fraction_digits: 2,
            max_fraction_digits: 2,
            ..Default::default()
        };
        assert_eq!(fmt.format_decimal(1234.50), "1.234,50");
    }

    #[test]
    fn currency_symbol() {
        let fmt = NumberFormat {
            min_fraction_digits: 2,
            max_fraction_digits: 2,
            ..Default::default()
        };
        let result = fmt.format_currency(1234.50, "USD", "$", "US dollars", CurrencyDisplay::Symbol);
        assert_eq!(result, "$1,234.50");
    }

    #[test]
    fn currency_code() {
        let fmt = NumberFormat {
            min_fraction_digits: 2,
            max_fraction_digits: 2,
            ..Default::default()
        };
        let result = fmt.format_currency(1234.50, "EUR", "€", "euros", CurrencyDisplay::Code);
        assert_eq!(result, "1,234.50 EUR");
    }

    #[test]
    fn percent_format() {
        let fmt = NumberFormat {
            min_fraction_digits: 1,
            max_fraction_digits: 1,
            use_grouping: false,
            ..Default::default()
        };
        assert_eq!(fmt.format_percent(0.856), "85.6%");
    }

    #[test]
    fn compact_short() {
        let fmt = NumberFormat::new();
        assert_eq!(fmt.format_compact(1_500.0, CompactDisplay::Short), "1.5K");
        assert_eq!(fmt.format_compact(2_500_000.0, CompactDisplay::Short), "2.5M");
        assert_eq!(fmt.format_compact(3_000_000_000.0, CompactDisplay::Short), "3B");
    }

    #[test]
    fn compact_long() {
        let fmt = NumberFormat::new();
        assert_eq!(
            fmt.format_compact(1_200_000.0, CompactDisplay::Long),
            "1.2 million"
        );
    }

    #[test]
    fn indian_grouping() {
        let fmt = NumberFormat {
            indian_grouping: true,
            ..Default::default()
        };
        assert_eq!(fmt.format_decimal(1234567.0), "12,34,567");
    }

    #[test]
    fn significant_digits() {
        let fmt = NumberFormat {
            significant_digits: Some((3, 3)),
            use_grouping: false,
            ..Default::default()
        };
        assert_eq!(fmt.format_decimal(1234.0), "1230");
        assert_eq!(fmt.format_decimal(1.2), "1.20");
    }

    #[test]
    fn zero_value() {
        let fmt = NumberFormat::new();
        assert_eq!(fmt.format_decimal(0.0), "0");
    }

    #[test]
    fn small_number_below_thousand() {
        let fmt = NumberFormat::new();
        assert_eq!(fmt.format_compact(42.0, CompactDisplay::Short), "42");
    }
}
