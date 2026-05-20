//! Currency handling — codes, money arithmetic, formatting, and exchange rates.
//!
//! Replaces currency.js / dinero.js with a pure-Rust money model that
//! prevents currency mismatches at the type level.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;

// ── Errors ──────────────────────────────────────────────────────

/// Currency domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CurrencyError {
    /// Cannot perform arithmetic on different currencies.
    CurrencyMismatch { lhs: CurrencyCode, rhs: CurrencyCode },
    /// Unknown currency code.
    UnknownCurrency(String),
    /// Failed to parse a money string.
    ParseError(String),
    /// No exchange rate available.
    NoExchangeRate { from: CurrencyCode, to: CurrencyCode },
    /// Division by zero.
    DivisionByZero,
}

impl fmt::Display for CurrencyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CurrencyMismatch { lhs, rhs } => {
                write!(f, "currency mismatch: {lhs:?} vs {rhs:?}")
            }
            Self::UnknownCurrency(c) => write!(f, "unknown currency: {c}"),
            Self::ParseError(s) => write!(f, "parse error: {s}"),
            Self::NoExchangeRate { from, to } => {
                write!(f, "no exchange rate from {from:?} to {to:?}")
            }
            Self::DivisionByZero => write!(f, "division by zero"),
        }
    }
}

impl std::error::Error for CurrencyError {}

// ── Currency Code ───────────────────────────────────────────────

/// ISO 4217 currency codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CurrencyCode {
    USD, EUR, GBP, JPY, CNY, CHF, CAD, AUD, NZD, SGD,
    HKD, KRW, INR, BRL, MXN, ZAR, SEK, NOK, DKK, PLN,
    THB, TWD, TRY, AED, SAR,
}

impl CurrencyCode {
    /// The symbol used for display.
    pub fn symbol(self) -> &'static str {
        match self {
            Self::USD => "$",
            Self::EUR => "\u{20ac}",
            Self::GBP => "\u{00a3}",
            Self::JPY => "\u{00a5}",
            Self::CNY => "\u{00a5}",
            Self::CHF => "CHF",
            Self::CAD => "CA$",
            Self::AUD => "A$",
            Self::NZD => "NZ$",
            Self::SGD => "S$",
            Self::HKD => "HK$",
            Self::KRW => "\u{20a9}",
            Self::INR => "\u{20b9}",
            Self::BRL => "R$",
            Self::MXN => "MX$",
            Self::ZAR => "R",
            Self::SEK => "kr",
            Self::NOK => "kr",
            Self::DKK => "kr",
            Self::PLN => "z\u{0142}",
            Self::THB => "\u{0e3f}",
            Self::TWD => "NT$",
            Self::TRY => "\u{20ba}",
            Self::AED => "AED",
            Self::SAR => "SAR",
        }
    }

    /// Number of decimal places (minor unit digits).
    pub fn decimal_places(self) -> u32 {
        match self {
            Self::JPY | Self::KRW | Self::TWD => 0,
            _ => 2,
        }
    }

    /// The minor unit divisor (e.g. 100 for USD, 1 for JPY).
    pub fn minor_unit_divisor(self) -> i64 {
        10i64.pow(self.decimal_places())
    }

    /// ISO code as a string.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::USD => "USD", Self::EUR => "EUR", Self::GBP => "GBP",
            Self::JPY => "JPY", Self::CNY => "CNY", Self::CHF => "CHF",
            Self::CAD => "CAD", Self::AUD => "AUD", Self::NZD => "NZD",
            Self::SGD => "SGD", Self::HKD => "HKD", Self::KRW => "KRW",
            Self::INR => "INR", Self::BRL => "BRL", Self::MXN => "MXN",
            Self::ZAR => "ZAR", Self::SEK => "SEK", Self::NOK => "NOK",
            Self::DKK => "DKK", Self::PLN => "PLN", Self::THB => "THB",
            Self::TWD => "TWD", Self::TRY => "TRY", Self::AED => "AED",
            Self::SAR => "SAR",
        }
    }
}

impl FromStr for CurrencyCode {
    type Err = CurrencyError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "USD" => Ok(Self::USD), "EUR" => Ok(Self::EUR), "GBP" => Ok(Self::GBP),
            "JPY" => Ok(Self::JPY), "CNY" => Ok(Self::CNY), "CHF" => Ok(Self::CHF),
            "CAD" => Ok(Self::CAD), "AUD" => Ok(Self::AUD), "NZD" => Ok(Self::NZD),
            "SGD" => Ok(Self::SGD), "HKD" => Ok(Self::HKD), "KRW" => Ok(Self::KRW),
            "INR" => Ok(Self::INR), "BRL" => Ok(Self::BRL), "MXN" => Ok(Self::MXN),
            "ZAR" => Ok(Self::ZAR), "SEK" => Ok(Self::SEK), "NOK" => Ok(Self::NOK),
            "DKK" => Ok(Self::DKK), "PLN" => Ok(Self::PLN), "THB" => Ok(Self::THB),
            "TWD" => Ok(Self::TWD), "TRY" => Ok(Self::TRY), "AED" => Ok(Self::AED),
            "SAR" => Ok(Self::SAR),
            _ => Err(CurrencyError::UnknownCurrency(s.to_string())),
        }
    }
}

impl fmt::Display for CurrencyCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ── Money ───────────────────────────────────────────────────────

/// An amount of money in a specific currency, stored as minor units (cents).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Money {
    pub amount_cents: i64,
    pub currency: CurrencyCode,
}

impl Money {
    /// Create money from minor units.
    pub fn new(amount_cents: i64, currency: CurrencyCode) -> Self {
        Self { amount_cents, currency }
    }

    /// Create money from a major-unit float (e.g. 12.50 USD).
    pub fn from_major(amount: f64, currency: CurrencyCode) -> Self {
        let cents = (amount * currency.minor_unit_divisor() as f64).round() as i64;
        Self::new(cents, currency)
    }

    /// Convert to major-unit float.
    pub fn to_major(self) -> f64 {
        self.amount_cents as f64 / self.currency.minor_unit_divisor() as f64
    }

    /// Add two Money values (same currency).
    pub fn add(self, other: Money) -> Result<Money, CurrencyError> {
        if self.currency != other.currency {
            return Err(CurrencyError::CurrencyMismatch {
                lhs: self.currency,
                rhs: other.currency,
            });
        }
        Ok(Money::new(self.amount_cents + other.amount_cents, self.currency))
    }

    /// Subtract another Money value (same currency).
    pub fn sub(self, other: Money) -> Result<Money, CurrencyError> {
        if self.currency != other.currency {
            return Err(CurrencyError::CurrencyMismatch {
                lhs: self.currency,
                rhs: other.currency,
            });
        }
        Ok(Money::new(self.amount_cents - other.amount_cents, self.currency))
    }

    /// Multiply by a scalar.
    pub fn multiply(self, factor: f64) -> Money {
        let cents = (self.amount_cents as f64 * factor).round() as i64;
        Money::new(cents, self.currency)
    }

    /// Whether this is zero.
    pub fn is_zero(self) -> bool {
        self.amount_cents == 0
    }

    /// Whether this is negative.
    pub fn is_negative(self) -> bool {
        self.amount_cents < 0
    }

    /// Absolute value.
    pub fn abs(self) -> Money {
        Money::new(self.amount_cents.abs(), self.currency)
    }

    /// Format for display with currency symbol.
    pub fn format_display(self) -> String {
        let sym = self.currency.symbol();
        let dp = self.currency.decimal_places();
        let divisor = self.currency.minor_unit_divisor();

        let sign = if self.amount_cents < 0 { "-" } else { "" };
        let abs_cents = self.amount_cents.abs();

        if dp == 0 {
            format!("{sign}{sym}{abs_cents}")
        } else {
            let major = abs_cents / divisor;
            let minor = abs_cents % divisor;
            format!("{sign}{sym}{major}.{minor:0>width$}", width = dp as usize)
        }
    }

    /// Parse a string like "$12.50", "£100", "¥1000".
    pub fn parse(s: &str) -> Result<Money, CurrencyError> {
        let trimmed = s.trim();
        if trimmed.is_empty() {
            return Err(CurrencyError::ParseError("empty string".into()));
        }

        // Try to match symbol prefix
        let (currency, rest) = parse_symbol_prefix(trimmed)?;
        let cleaned: String = rest.chars().filter(|c| *c != ',' && *c != ' ').collect();

        if cleaned.is_empty() {
            return Err(CurrencyError::ParseError("no amount found".into()));
        }

        let negative = cleaned.starts_with('-');
        let abs_str = cleaned.trim_start_matches('-');

        if currency.decimal_places() == 0 {
            let val: i64 = abs_str
                .parse()
                .map_err(|_| CurrencyError::ParseError(format!("invalid number: {abs_str}")))?;
            let amount = if negative { -val } else { val };
            Ok(Money::new(amount, currency))
        } else {
            let parts: Vec<&str> = abs_str.splitn(2, '.').collect();
            let major: i64 = parts[0]
                .parse()
                .map_err(|_| CurrencyError::ParseError(format!("invalid major: {}", parts[0])))?;
            let minor: i64 = if parts.len() > 1 {
                let frac = parts[1];
                let dp = currency.decimal_places() as usize;
                let padded = if frac.len() < dp {
                    format!("{frac:0<width$}", width = dp)
                } else {
                    frac[..dp].to_string()
                };
                padded.parse().map_err(|_| {
                    CurrencyError::ParseError(format!("invalid minor: {padded}"))
                })?
            } else {
                0
            };
            let divisor = currency.minor_unit_divisor();
            let total = major * divisor + minor;
            let amount = if negative { -total } else { total };
            Ok(Money::new(amount, currency))
        }
    }
}

impl fmt::Display for Money {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.format_display())
    }
}

/// Try to identify a currency from a symbol prefix.
fn parse_symbol_prefix(s: &str) -> Result<(CurrencyCode, &str), CurrencyError> {
    // Multi-char symbols first
    let prefixes: &[(&str, CurrencyCode)] = &[
        ("CA$", CurrencyCode::CAD),
        ("A$", CurrencyCode::AUD),
        ("NZ$", CurrencyCode::NZD),
        ("S$", CurrencyCode::SGD),
        ("HK$", CurrencyCode::HKD),
        ("NT$", CurrencyCode::TWD),
        ("MX$", CurrencyCode::MXN),
        ("R$", CurrencyCode::BRL),
        ("CHF", CurrencyCode::CHF),
        ("AED", CurrencyCode::AED),
        ("SAR", CurrencyCode::SAR),
        ("$", CurrencyCode::USD),
        ("\u{20ac}", CurrencyCode::EUR),
        ("\u{00a3}", CurrencyCode::GBP),
        ("\u{00a5}", CurrencyCode::JPY),
        ("\u{20a9}", CurrencyCode::KRW),
        ("\u{20b9}", CurrencyCode::INR),
        ("R", CurrencyCode::ZAR),
        ("kr", CurrencyCode::SEK),
        ("z\u{0142}", CurrencyCode::PLN),
        ("\u{0e3f}", CurrencyCode::THB),
        ("\u{20ba}", CurrencyCode::TRY),
    ];

    for (prefix, code) in prefixes {
        if s.starts_with(prefix) {
            return Ok((*code, &s[prefix.len()..]));
        }
    }

    // Try 3-letter ISO code prefix
    if s.len() >= 3 {
        if let Ok(code) = s[..3].parse::<CurrencyCode>() {
            let rest = s[3..].trim_start();
            return Ok((code, rest));
        }
    }

    Err(CurrencyError::ParseError(format!(
        "no recognized currency symbol in: {s}"
    )))
}

// ── Exchange Rate Table ─────────────────────────────────────────

/// A table of exchange rates for currency conversion.
#[derive(Debug, Clone, Default)]
pub struct ExchangeRateTable {
    /// Rates stored as (from, to) -> rate.
    rates: HashMap<(CurrencyCode, CurrencyCode), f64>,
}

impl ExchangeRateTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a rate. Setting USD->EUR also implies the inverse EUR->USD.
    pub fn set_rate(&mut self, from: CurrencyCode, to: CurrencyCode, rate: f64) {
        self.rates.insert((from, to), rate);
        if rate != 0.0 {
            self.rates.insert((to, from), 1.0 / rate);
        }
    }

    /// Get the rate from one currency to another.
    pub fn get_rate(&self, from: CurrencyCode, to: CurrencyCode) -> Option<f64> {
        if from == to {
            return Some(1.0);
        }
        self.rates.get(&(from, to)).copied()
    }

    /// Convert money from one currency to another.
    pub fn convert(&self, money: Money, to: CurrencyCode) -> Result<Money, CurrencyError> {
        if money.currency == to {
            return Ok(money);
        }
        let rate = self.get_rate(money.currency, to).ok_or(CurrencyError::NoExchangeRate {
            from: money.currency,
            to,
        })?;
        // Convert through major units to handle different decimal places.
        let from_major = money.to_major();
        let to_major = from_major * rate;
        Ok(Money::from_major(to_major, to))
    }

    /// Number of rate pairs stored.
    pub fn len(&self) -> usize {
        self.rates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rates.is_empty()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn money_creation_and_display() {
        let m = Money::new(1250, CurrencyCode::USD);
        assert_eq!(m.format_display(), "$12.50");
    }

    #[test]
    fn money_from_major() {
        let m = Money::from_major(99.99, CurrencyCode::EUR);
        assert_eq!(m.amount_cents, 9999);
    }

    #[test]
    fn jpy_zero_decimals() {
        let m = Money::new(1000, CurrencyCode::JPY);
        assert_eq!(m.format_display(), "\u{00a5}1000");
        assert_eq!(m.to_major(), 1000.0);
    }

    #[test]
    fn add_same_currency() {
        let a = Money::new(500, CurrencyCode::USD);
        let b = Money::new(300, CurrencyCode::USD);
        let sum = a.add(b).unwrap();
        assert_eq!(sum.amount_cents, 800);
    }

    #[test]
    fn add_different_currency_fails() {
        let a = Money::new(500, CurrencyCode::USD);
        let b = Money::new(300, CurrencyCode::EUR);
        assert!(a.add(b).is_err());
    }

    #[test]
    fn subtract() {
        let a = Money::new(1000, CurrencyCode::GBP);
        let b = Money::new(400, CurrencyCode::GBP);
        let diff = a.sub(b).unwrap();
        assert_eq!(diff.amount_cents, 600);
    }

    #[test]
    fn multiply() {
        let m = Money::new(1000, CurrencyCode::USD);
        let doubled = m.multiply(2.5);
        assert_eq!(doubled.amount_cents, 2500);
    }

    #[test]
    fn parse_usd() {
        let m = Money::parse("$12.50").unwrap();
        assert_eq!(m.currency, CurrencyCode::USD);
        assert_eq!(m.amount_cents, 1250);
    }

    #[test]
    fn parse_gbp() {
        let m = Money::parse("\u{00a3}100").unwrap();
        assert_eq!(m.currency, CurrencyCode::GBP);
        assert_eq!(m.amount_cents, 10000);
    }

    #[test]
    fn parse_jpy() {
        let m = Money::parse("\u{00a5}5000").unwrap();
        assert_eq!(m.currency, CurrencyCode::JPY);
        assert_eq!(m.amount_cents, 5000);
    }

    #[test]
    fn parse_eur_symbol() {
        let m = Money::parse("\u{20ac}42.00").unwrap();
        assert_eq!(m.currency, CurrencyCode::EUR);
        assert_eq!(m.amount_cents, 4200);
    }

    #[test]
    fn exchange_rate_conversion() {
        let mut table = ExchangeRateTable::new();
        table.set_rate(CurrencyCode::USD, CurrencyCode::EUR, 0.92);

        let usd = Money::new(10000, CurrencyCode::USD); // $100.00
        let eur = table.convert(usd, CurrencyCode::EUR).unwrap();
        assert_eq!(eur.currency, CurrencyCode::EUR);
        assert_eq!(eur.amount_cents, 9200); // €92.00
    }

    #[test]
    fn exchange_rate_inverse() {
        let mut table = ExchangeRateTable::new();
        table.set_rate(CurrencyCode::USD, CurrencyCode::GBP, 0.79);

        let gbp = Money::new(7900, CurrencyCode::GBP); // £79.00
        let usd = table.convert(gbp, CurrencyCode::USD).unwrap();
        assert_eq!(usd.currency, CurrencyCode::USD);
        assert_eq!(usd.amount_cents, 10000); // $100.00
    }

    #[test]
    fn same_currency_conversion_noop() {
        let table = ExchangeRateTable::new();
        let m = Money::new(500, CurrencyCode::USD);
        let result = table.convert(m, CurrencyCode::USD).unwrap();
        assert_eq!(result, m);
    }

    #[test]
    fn no_rate_returns_error() {
        let table = ExchangeRateTable::new();
        let m = Money::new(500, CurrencyCode::USD);
        assert!(table.convert(m, CurrencyCode::EUR).is_err());
    }

    #[test]
    fn currency_code_from_str() {
        assert_eq!("usd".parse::<CurrencyCode>().unwrap(), CurrencyCode::USD);
        assert_eq!("JPY".parse::<CurrencyCode>().unwrap(), CurrencyCode::JPY);
        assert!("XYZ".parse::<CurrencyCode>().is_err());
    }

    #[test]
    fn negative_money() {
        let m = Money::new(-500, CurrencyCode::USD);
        assert!(m.is_negative());
        assert_eq!(m.format_display(), "-$5.00");
        assert_eq!(m.abs().amount_cents, 500);
    }

    #[test]
    fn at_least_25_currencies() {
        // Verify the enum has at least 25 variants by parsing all codes.
        let codes = [
            "USD", "EUR", "GBP", "JPY", "CNY", "CHF", "CAD", "AUD", "NZD", "SGD",
            "HKD", "KRW", "INR", "BRL", "MXN", "ZAR", "SEK", "NOK", "DKK", "PLN",
            "THB", "TWD", "TRY", "AED", "SAR",
        ];
        for code in &codes {
            assert!(code.parse::<CurrencyCode>().is_ok(), "failed to parse {code}");
        }
    }
}
