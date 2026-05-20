//! Localization — number formatting by locale, date formatting, currency,
//! relative time, list formatting, collation-aware sorting.
//!
//! Pure-Rust replacement for Intl.NumberFormat, Intl.DateTimeFormat,
//! Intl.RelativeTimeFormat, and Intl.ListFormat.

// ── Locale Data ─────────────────────────────────────────────────

/// Locale-specific formatting symbols.
#[derive(Debug, Clone)]
pub struct LocaleData {
    pub decimal_sep: char,
    pub grouping_sep: char,
    pub grouping_size: usize,
    pub currency_symbol: &'static str,
    pub currency_code: &'static str,
    pub currency_prefix: bool,
    pub percent_suffix: &'static str,
    pub months_short: [&'static str; 12],
    pub months_long: [&'static str; 12],
    pub weekdays_short: [&'static str; 7],
    pub weekdays_long: [&'static str; 7],
    pub date_short: &'static str,
    pub date_medium: &'static str,
    pub date_long: &'static str,
    pub list_conjunction: &'static str,
    pub list_disjunction: &'static str,
}

const EN_MONTHS_SHORT: [&str; 12] = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
const EN_MONTHS_LONG: [&str; 12] = ["January","February","March","April","May","June","July","August","September","October","November","December"];
const EN_WEEKDAYS_SHORT: [&str; 7] = ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"];
const EN_WEEKDAYS_LONG: [&str; 7] = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"];

const DE_MONTHS_SHORT: [&str; 12] = ["Jan","Feb","Mär","Apr","Mai","Jun","Jul","Aug","Sep","Okt","Nov","Dez"];
const DE_MONTHS_LONG: [&str; 12] = ["Januar","Februar","März","April","Mai","Juni","Juli","August","September","Oktober","November","Dezember"];
const DE_WEEKDAYS_SHORT: [&str; 7] = ["So","Mo","Di","Mi","Do","Fr","Sa"];
const DE_WEEKDAYS_LONG: [&str; 7] = ["Sonntag","Montag","Dienstag","Mittwoch","Donnerstag","Freitag","Samstag"];

const FR_MONTHS_SHORT: [&str; 12] = ["janv.","févr.","mars","avr.","mai","juin","juil.","août","sept.","oct.","nov.","déc."];
const FR_MONTHS_LONG: [&str; 12] = ["janvier","février","mars","avril","mai","juin","juillet","août","septembre","octobre","novembre","décembre"];
const FR_WEEKDAYS_SHORT: [&str; 7] = ["dim.","lun.","mar.","mer.","jeu.","ven.","sam."];
const FR_WEEKDAYS_LONG: [&str; 7] = ["dimanche","lundi","mardi","mercredi","jeudi","vendredi","samedi"];

const JA_MONTHS_SHORT: [&str; 12] = ["1月","2月","3月","4月","5月","6月","7月","8月","9月","10月","11月","12月"];
const JA_MONTHS_LONG: [&str; 12] = ["1月","2月","3月","4月","5月","6月","7月","8月","9月","10月","11月","12月"];
const JA_WEEKDAYS_SHORT: [&str; 7] = ["日","月","火","水","木","金","土"];
const JA_WEEKDAYS_LONG: [&str; 7] = ["日曜日","月曜日","火曜日","水曜日","木曜日","金曜日","土曜日"];

fn locale_data_en() -> LocaleData {
    LocaleData {
        decimal_sep: '.', grouping_sep: ',', grouping_size: 3,
        currency_symbol: "$", currency_code: "USD", currency_prefix: true,
        percent_suffix: "%",
        months_short: EN_MONTHS_SHORT, months_long: EN_MONTHS_LONG,
        weekdays_short: EN_WEEKDAYS_SHORT, weekdays_long: EN_WEEKDAYS_LONG,
        date_short: "M/d/yyyy", date_medium: "MMM d, yyyy", date_long: "MMMM d, yyyy",
        list_conjunction: "and", list_disjunction: "or",
    }
}

fn locale_data_de() -> LocaleData {
    LocaleData {
        decimal_sep: ',', grouping_sep: '.', grouping_size: 3,
        currency_symbol: "€", currency_code: "EUR", currency_prefix: false,
        percent_suffix: " %",
        months_short: DE_MONTHS_SHORT, months_long: DE_MONTHS_LONG,
        weekdays_short: DE_WEEKDAYS_SHORT, weekdays_long: DE_WEEKDAYS_LONG,
        date_short: "dd.MM.yyyy", date_medium: "dd. MMM yyyy", date_long: "d. MMMM yyyy",
        list_conjunction: "und", list_disjunction: "oder",
    }
}

fn locale_data_fr() -> LocaleData {
    LocaleData {
        decimal_sep: ',', grouping_sep: '\u{202f}', grouping_size: 3,
        currency_symbol: "€", currency_code: "EUR", currency_prefix: false,
        percent_suffix: " %",
        months_short: FR_MONTHS_SHORT, months_long: FR_MONTHS_LONG,
        weekdays_short: FR_WEEKDAYS_SHORT, weekdays_long: FR_WEEKDAYS_LONG,
        date_short: "dd/MM/yyyy", date_medium: "d MMM yyyy", date_long: "d MMMM yyyy",
        list_conjunction: "et", list_disjunction: "ou",
    }
}

fn locale_data_ja() -> LocaleData {
    LocaleData {
        decimal_sep: '.', grouping_sep: ',', grouping_size: 3,
        currency_symbol: "¥", currency_code: "JPY", currency_prefix: true,
        percent_suffix: "%",
        months_short: JA_MONTHS_SHORT, months_long: JA_MONTHS_LONG,
        weekdays_short: JA_WEEKDAYS_SHORT, weekdays_long: JA_WEEKDAYS_LONG,
        date_short: "yyyy/MM/dd", date_medium: "yyyy年M月d日", date_long: "yyyy年M月d日",
        list_conjunction: "、", list_disjunction: "または",
    }
}

/// Get locale data for a language code.
pub fn get_locale_data(lang: &str) -> LocaleData {
    let primary = lang.split(['-', '_']).next().unwrap_or("en");
    match primary {
        "de" => locale_data_de(),
        "fr" => locale_data_fr(),
        "ja" => locale_data_ja(),
        _ => locale_data_en(),
    }
}

// ── Number Formatting ───────────────────────────────────────────

/// Format a number with locale-specific grouping and decimal separators.
pub fn format_number(n: f64, decimals: Option<usize>, data: &LocaleData) -> String {
    let neg = n < 0.0;
    let abs = n.abs();
    let dec = decimals.unwrap_or_else(|| {
        let frac = abs - abs.trunc();
        if frac.abs() < 1e-10 { 0 } else { 2 }
    });
    let int_part = abs.trunc() as u64;
    let int_str = format_int_grouped(int_part, data.grouping_sep, data.grouping_size);
    let result = if dec == 0 {
        int_str
    } else {
        let frac = abs - (int_part as f64);
        let frac_scaled = (frac * 10_f64.powi(dec as i32)).round() as u64;
        let frac_str = format!("{:0>width$}", frac_scaled, width = dec);
        format!("{int_str}{}{frac_str}", data.decimal_sep)
    };
    if neg { format!("-{result}") } else { result }
}

fn format_int_grouped(n: u64, sep: char, group_size: usize) -> String {
    let s = n.to_string();
    if s.len() <= group_size { return s; }
    let bytes = s.as_bytes();
    let len = bytes.len();
    let mut result = String::with_capacity(len + len / group_size);
    for (i, &b) in bytes.iter().enumerate() {
        if i > 0 && (len - i) % group_size == 0 { result.push(sep); }
        result.push(b as char);
    }
    result
}

/// Format a number as a percentage.
pub fn format_percent(n: f64, decimals: usize, data: &LocaleData) -> String {
    let pct = n * 100.0;
    let formatted = format_number(pct, Some(decimals), data);
    format!("{formatted}{}", data.percent_suffix)
}

/// Format a currency value.
pub fn format_currency(n: f64, data: &LocaleData) -> String {
    let formatted = format_number(n.abs(), Some(2), data);
    let sign = if n < 0.0 { "-" } else { "" };
    if data.currency_prefix {
        format!("{sign}{}{formatted}", data.currency_symbol)
    } else {
        format!("{sign}{formatted} {}", data.currency_symbol)
    }
}

/// Format a currency value with explicit symbol.
pub fn format_currency_with_symbol(n: f64, symbol: &str, prefix: bool, data: &LocaleData) -> String {
    let formatted = format_number(n.abs(), Some(2), data);
    let sign = if n < 0.0 { "-" } else { "" };
    if prefix {
        format!("{sign}{symbol}{formatted}")
    } else {
        format!("{sign}{formatted} {symbol}")
    }
}

// ── Date Formatting ─────────────────────────────────────────────

/// A simple date struct (no chrono dependency required by callers).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SimpleDate {
    pub year: i32,
    pub month: u32,
    pub day: u32,
    pub weekday: u32,
    pub hour: u32,
    pub minute: u32,
    pub second: u32,
}

impl SimpleDate {
    pub fn new(year: i32, month: u32, day: u32) -> Self {
        let weekday = zellers_weekday(year, month, day);
        Self { year, month, day, weekday, hour: 0, minute: 0, second: 0 }
    }

    pub fn with_time(mut self, hour: u32, minute: u32, second: u32) -> Self {
        self.hour = hour;
        self.minute = minute;
        self.second = second;
        self
    }
}

/// Zeller's congruence -> 0=Sun, 1=Mon, ..., 6=Sat.
fn zellers_weekday(year: i32, month: u32, day: u32) -> u32 {
    let (y, m) = if month <= 2 { (year - 1, month + 12) } else { (year, month) };
    let q = day as i32;
    let k = y % 100;
    let j = y / 100;
    let m_i = m as i32;
    let h = (q + (13 * (m_i + 1)) / 5 + k + k / 4 + j / 4 + 5 * j) % 7;
    ((h + 6) % 7) as u32
}

/// Date formatting style.
#[derive(Debug, Clone, PartialEq)]
pub enum DateStyle { Short, Medium, Long }

/// Format a date with locale data.
pub fn format_date(date: &SimpleDate, style: &DateStyle, data: &LocaleData) -> String {
    match style {
        DateStyle::Short => {
            let pattern = data.date_short;
            apply_date_pattern(date, pattern, data)
        }
        DateStyle::Medium => {
            let pattern = data.date_medium;
            apply_date_pattern(date, pattern, data)
        }
        DateStyle::Long => {
            let pattern = data.date_long;
            apply_date_pattern(date, pattern, data)
        }
    }
}

fn apply_date_pattern(date: &SimpleDate, pattern: &str, data: &LocaleData) -> String {
    let mut result = pattern.to_string();
    // Use placeholders without letters M/d/y to prevent cascading replacements
    // (e.g., the 'M' in "March" being replaced by the single-M month-number rule).
    result = result.replace("yyyy", "\x01\x02\x03\x04");
    result = result.replace("MMMM", "\x01\x02\x03\x05");
    result = result.replace("MMM", "\x01\x02\x03\x06");
    result = result.replace("MM", &format!("{:02}", date.month));
    result = result.replace('M', &format!("{}", date.month));
    result = result.replace("dd", &format!("{:02}", date.day));
    result = result.replace('d', &format!("{}", date.day));
    // Resolve placeholders last.
    result = result.replace("\x01\x02\x03\x04", &format!("{}", date.year));
    result = result.replace("\x01\x02\x03\x05", data.months_long[date.month as usize - 1]);
    result = result.replace("\x01\x02\x03\x06", data.months_short[date.month as usize - 1]);
    result
}

// ── Relative Time ───────────────────────────────────────────────

/// Relative time unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelativeTimeUnit { Second, Minute, Hour, Day, Week, Month, Year }

/// Format a relative time value (e.g., "3 days ago", "in 2 hours").
pub fn format_relative_time(value: i64, unit: RelativeTimeUnit, lang: &str) -> String {
    let primary = lang.split(['-', '_']).next().unwrap_or("en");
    let unit_str = match primary {
        "de" => relative_time_unit_de(unit, value.unsigned_abs()),
        "fr" => relative_time_unit_fr(unit, value.unsigned_abs()),
        _ => relative_time_unit_en(unit, value.unsigned_abs()),
    };
    let abs_val = value.unsigned_abs();
    match primary {
        "de" => {
            if value > 0 { format!("in {abs_val} {unit_str}") }
            else if value < 0 { format!("vor {abs_val} {unit_str}") }
            else { format!("jetzt") }
        }
        "fr" => {
            if value > 0 { format!("dans {abs_val} {unit_str}") }
            else if value < 0 { format!("il y a {abs_val} {unit_str}") }
            else { format!("maintenant") }
        }
        _ => {
            if value > 0 { format!("in {abs_val} {unit_str}") }
            else if value < 0 { format!("{abs_val} {unit_str} ago") }
            else { format!("now") }
        }
    }
}

fn relative_time_unit_en(unit: RelativeTimeUnit, n: u64) -> &'static str {
    match (unit, n == 1) {
        (RelativeTimeUnit::Second, true) => "second", (RelativeTimeUnit::Second, false) => "seconds",
        (RelativeTimeUnit::Minute, true) => "minute", (RelativeTimeUnit::Minute, false) => "minutes",
        (RelativeTimeUnit::Hour, true) => "hour", (RelativeTimeUnit::Hour, false) => "hours",
        (RelativeTimeUnit::Day, true) => "day", (RelativeTimeUnit::Day, false) => "days",
        (RelativeTimeUnit::Week, true) => "week", (RelativeTimeUnit::Week, false) => "weeks",
        (RelativeTimeUnit::Month, true) => "month", (RelativeTimeUnit::Month, false) => "months",
        (RelativeTimeUnit::Year, true) => "year", (RelativeTimeUnit::Year, false) => "years",
    }
}

fn relative_time_unit_de(unit: RelativeTimeUnit, n: u64) -> &'static str {
    match (unit, n == 1) {
        (RelativeTimeUnit::Second, true) => "Sekunde", (RelativeTimeUnit::Second, false) => "Sekunden",
        (RelativeTimeUnit::Minute, true) => "Minute", (RelativeTimeUnit::Minute, false) => "Minuten",
        (RelativeTimeUnit::Hour, true) => "Stunde", (RelativeTimeUnit::Hour, false) => "Stunden",
        (RelativeTimeUnit::Day, true) => "Tag", (RelativeTimeUnit::Day, false) => "Tagen",
        (RelativeTimeUnit::Week, true) => "Woche", (RelativeTimeUnit::Week, false) => "Wochen",
        (RelativeTimeUnit::Month, true) => "Monat", (RelativeTimeUnit::Month, false) => "Monaten",
        (RelativeTimeUnit::Year, true) => "Jahr", (RelativeTimeUnit::Year, false) => "Jahren",
    }
}

fn relative_time_unit_fr(unit: RelativeTimeUnit, n: u64) -> &'static str {
    match (unit, n <= 1) {
        (RelativeTimeUnit::Second, true) => "seconde", (RelativeTimeUnit::Second, false) => "secondes",
        (RelativeTimeUnit::Minute, true) => "minute", (RelativeTimeUnit::Minute, false) => "minutes",
        (RelativeTimeUnit::Hour, true) => "heure", (RelativeTimeUnit::Hour, false) => "heures",
        (RelativeTimeUnit::Day, true) => "jour", (RelativeTimeUnit::Day, false) => "jours",
        (RelativeTimeUnit::Week, true) => "semaine", (RelativeTimeUnit::Week, false) => "semaines",
        (RelativeTimeUnit::Month, true) => "mois", (RelativeTimeUnit::Month, false) => "mois",
        (RelativeTimeUnit::Year, true) => "an", (RelativeTimeUnit::Year, false) => "ans",
    }
}

// ── List Formatting ─────────────────────────────────────────────

/// List format type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ListType { Conjunction, Disjunction }

/// Format a list of items with locale-appropriate conjunctions.
pub fn format_list(items: &[&str], list_type: ListType, data: &LocaleData) -> String {
    let conj = match list_type {
        ListType::Conjunction => data.list_conjunction,
        ListType::Disjunction => data.list_disjunction,
    };
    match items.len() {
        0 => String::new(),
        1 => items[0].to_string(),
        2 => format!("{} {} {}", items[0], conj, items[1]),
        _ => {
            let last = items.len() - 1;
            let head: Vec<&str> = items[..last].to_vec();
            format!("{}, {} {}", head.join(", "), conj, items[last])
        }
    }
}

// ── Collation-Aware Sorting ─────────────────────────────────────

/// Sort strings with basic collation awareness (case-insensitive, accent folding).
pub fn collation_sort(items: &mut [String], _lang: &str) {
    items.sort_by(|a, b| {
        let a_key = collation_key(a);
        let b_key = collation_key(b);
        a_key.cmp(&b_key)
    });
}

/// Generate a collation key for a string (lowercase + basic accent folding).
pub fn collation_key(s: &str) -> String {
    s.chars().map(|c| fold_accent(c).to_ascii_lowercase()).collect()
}

fn fold_accent(c: char) -> char {
    match c {
        '\u{00e0}'..='\u{00e5}' | '\u{00c0}'..='\u{00c5}' => 'a',
        '\u{00e8}'..='\u{00eb}' | '\u{00c8}'..='\u{00cb}' => 'e',
        '\u{00ec}'..='\u{00ef}' | '\u{00cc}'..='\u{00cf}' => 'i',
        '\u{00f2}'..='\u{00f6}' | '\u{00d2}'..='\u{00d6}' => 'o',
        '\u{00f9}'..='\u{00fc}' | '\u{00d9}'..='\u{00dc}' => 'u',
        '\u{00f1}' | '\u{00d1}' => 'n',
        '\u{00e7}' | '\u{00c7}' => 'c',
        '\u{00df}' => 's',
        _ => c,
    }
}

// ── Compact Number Formatting ───────────────────────────────────

/// Format a number in compact form (1K, 2.3M, etc.)
pub fn format_compact(n: f64, lang: &str) -> String {
    let primary = lang.split(['-', '_']).next().unwrap_or("en");
    let (divisor, suffix) = if n.abs() >= 1_000_000_000.0 {
        (1_000_000_000.0, compact_suffix(primary, "billion"))
    } else if n.abs() >= 1_000_000.0 {
        (1_000_000.0, compact_suffix(primary, "million"))
    } else if n.abs() >= 1_000.0 {
        (1_000.0, compact_suffix(primary, "thousand"))
    } else {
        return format_number_simple(n);
    };
    let val = n / divisor;
    if (val - val.round()).abs() < 0.05 {
        format!("{}{}", val.round() as i64, suffix)
    } else {
        format!("{:.1}{}", val, suffix)
    }
}

fn compact_suffix(lang: &str, magnitude: &str) -> &'static str {
    match (lang, magnitude) {
        (_, "thousand") => "K",
        ("de", "million") => " Mio.",
        ("de", "billion") => " Mrd.",
        (_, "million") => "M",
        (_, "billion") => "B",
        _ => "",
    }
}

fn format_number_simple(n: f64) -> String {
    if n == n.trunc() { format!("{}", n as i64) } else { format!("{:.1}", n) }
}

// ── Unit Formatting ─────────────────────────────────────────────

/// Common measurement units.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeasureUnit { Meter, Kilometer, Kilogram, Gram, Liter, Celsius, Fahrenheit }

/// Format a value with a unit.
pub fn format_unit(value: f64, unit: MeasureUnit, lang: &str) -> String {
    let primary = lang.split(['-', '_']).next().unwrap_or("en");
    let suffix = match (primary, unit) {
        (_, MeasureUnit::Meter) => "m",
        (_, MeasureUnit::Kilometer) => "km",
        (_, MeasureUnit::Kilogram) => "kg",
        (_, MeasureUnit::Gram) => "g",
        ("en", MeasureUnit::Liter) => "L",
        (_, MeasureUnit::Liter) => "l",
        (_, MeasureUnit::Celsius) => "\u{00b0}C",
        (_, MeasureUnit::Fahrenheit) => "\u{00b0}F",
    };
    let data = get_locale_data(lang);
    let formatted = format_number(value, None, &data);
    format!("{formatted} {suffix}")
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn number_en_grouping() {
        let data = get_locale_data("en");
        assert_eq!(format_number(1234567.89, Some(2), &data), "1,234,567.89");
    }

    #[test]
    fn number_de_grouping() {
        let data = get_locale_data("de");
        assert_eq!(format_number(1234567.89, Some(2), &data), "1.234.567,89");
    }

    #[test]
    fn number_fr_grouping() {
        let data = get_locale_data("fr");
        let result = format_number(1234567.89, Some(2), &data);
        assert!(result.contains("1\u{202f}234\u{202f}567,89"));
    }

    #[test]
    fn number_no_decimals() {
        let data = get_locale_data("en");
        assert_eq!(format_number(1000.0, Some(0), &data), "1,000");
    }

    #[test]
    fn number_negative() {
        let data = get_locale_data("en");
        assert_eq!(format_number(-42.5, Some(2), &data), "-42.50");
    }

    #[test]
    fn percent_en() {
        let data = get_locale_data("en");
        assert_eq!(format_percent(0.456, 1, &data), "45.6%");
    }

    #[test]
    fn percent_de() {
        let data = get_locale_data("de");
        assert_eq!(format_percent(0.456, 1, &data), "45,6 %");
    }

    #[test]
    fn currency_en() {
        let data = get_locale_data("en");
        assert_eq!(format_currency(1234.56, &data), "$1,234.56");
    }

    #[test]
    fn currency_de() {
        let data = get_locale_data("de");
        assert_eq!(format_currency(1234.56, &data), "1.234,56 €");
    }

    #[test]
    fn currency_negative() {
        let data = get_locale_data("en");
        assert_eq!(format_currency(-99.99, &data), "-$99.99");
    }

    #[test]
    fn currency_custom_symbol() {
        let data = get_locale_data("en");
        assert_eq!(format_currency_with_symbol(50.0, "£", true, &data), "£50.00");
    }

    #[test]
    fn date_short_en() {
        let d = SimpleDate::new(2026, 3, 9);
        let data = get_locale_data("en");
        assert_eq!(format_date(&d, &DateStyle::Short, &data), "3/9/2026");
    }

    #[test]
    fn date_medium_en() {
        let d = SimpleDate::new(2026, 3, 9);
        let data = get_locale_data("en");
        assert_eq!(format_date(&d, &DateStyle::Medium, &data), "Mar 9, 2026");
    }

    #[test]
    fn date_long_en() {
        let d = SimpleDate::new(2026, 3, 9);
        let data = get_locale_data("en");
        assert_eq!(format_date(&d, &DateStyle::Long, &data), "March 9, 2026");
    }

    #[test]
    fn date_short_de() {
        let d = SimpleDate::new(2026, 3, 9);
        let data = get_locale_data("de");
        assert_eq!(format_date(&d, &DateStyle::Short, &data), "09.03.2026");
    }

    #[test]
    fn weekday_calculation() {
        // 2026-03-09 is a Monday
        let d = SimpleDate::new(2026, 3, 9);
        assert_eq!(d.weekday, 1);
    }

    #[test]
    fn relative_time_en_past() {
        assert_eq!(format_relative_time(-3, RelativeTimeUnit::Day, "en"), "3 days ago");
    }

    #[test]
    fn relative_time_en_future() {
        assert_eq!(format_relative_time(2, RelativeTimeUnit::Hour, "en"), "in 2 hours");
    }

    #[test]
    fn relative_time_en_now() {
        assert_eq!(format_relative_time(0, RelativeTimeUnit::Second, "en"), "now");
    }

    #[test]
    fn relative_time_en_singular() {
        assert_eq!(format_relative_time(-1, RelativeTimeUnit::Day, "en"), "1 day ago");
    }

    #[test]
    fn relative_time_de() {
        assert_eq!(format_relative_time(-5, RelativeTimeUnit::Minute, "de"), "vor 5 Minuten");
        assert_eq!(format_relative_time(1, RelativeTimeUnit::Hour, "de"), "in 1 Stunde");
    }

    #[test]
    fn relative_time_fr() {
        assert_eq!(format_relative_time(-2, RelativeTimeUnit::Year, "fr"), "il y a 2 ans");
    }

    #[test]
    fn list_conjunction_en() {
        let data = get_locale_data("en");
        assert_eq!(format_list(&["a", "b", "c"], ListType::Conjunction, &data), "a, b, and c");
    }

    #[test]
    fn list_disjunction_en() {
        let data = get_locale_data("en");
        assert_eq!(format_list(&["a", "b"], ListType::Disjunction, &data), "a or b");
    }

    #[test]
    fn list_single_item() {
        let data = get_locale_data("en");
        assert_eq!(format_list(&["only"], ListType::Conjunction, &data), "only");
    }

    #[test]
    fn list_empty() {
        let data = get_locale_data("en");
        assert_eq!(format_list(&[], ListType::Conjunction, &data), "");
    }

    #[test]
    fn list_de() {
        let data = get_locale_data("de");
        assert_eq!(format_list(&["Apfel", "Birne", "Kirsche"], ListType::Conjunction, &data), "Apfel, Birne, und Kirsche");
    }

    #[test]
    fn collation_sort_case_insensitive() {
        let mut items = vec!["Banana".into(), "apple".into(), "cherry".into()];
        collation_sort(&mut items, "en");
        assert_eq!(items, vec!["apple", "Banana", "cherry"]);
    }

    #[test]
    fn collation_sort_accents() {
        let mut items = vec!["étude".into(), "ecole".into(), "ephemeral".into()];
        collation_sort(&mut items, "fr");
        assert_eq!(items, vec!["ecole", "ephemeral", "étude"]);
    }

    #[test]
    fn collation_key_accent_fold() {
        assert_eq!(collation_key("café"), "cafe");
        assert_eq!(collation_key("über"), "uber");
        assert_eq!(collation_key("naïve"), "naive");
    }

    #[test]
    fn compact_thousands() {
        assert_eq!(format_compact(1500.0, "en"), "1.5K");
        assert_eq!(format_compact(1000.0, "en"), "1K");
    }

    #[test]
    fn compact_millions() {
        assert_eq!(format_compact(2_300_000.0, "en"), "2.3M");
    }

    #[test]
    fn compact_billions() {
        assert_eq!(format_compact(1_000_000_000.0, "en"), "1B");
    }

    #[test]
    fn compact_small_number() {
        assert_eq!(format_compact(42.0, "en"), "42");
    }

    #[test]
    fn compact_de() {
        assert_eq!(format_compact(5_000_000.0, "de"), "5 Mio.");
    }

    #[test]
    fn unit_formatting() {
        assert_eq!(format_unit(100.0, MeasureUnit::Kilometer, "en"), "100 km");
        assert_eq!(format_unit(36.6, MeasureUnit::Celsius, "en"), "36.60 °C");
    }

    #[test]
    fn date_with_time() {
        let d = SimpleDate::new(2026, 1, 15).with_time(14, 30, 0);
        assert_eq!(d.hour, 14);
        assert_eq!(d.minute, 30);
    }

    #[test]
    fn locale_data_ja_currency() {
        let data = get_locale_data("ja");
        assert_eq!(data.currency_symbol, "¥");
        assert!(data.currency_prefix);
    }

    #[test]
    fn number_small() {
        let data = get_locale_data("en");
        assert_eq!(format_number(42.0, Some(0), &data), "42");
    }

    #[test]
    fn number_zero() {
        let data = get_locale_data("en");
        assert_eq!(format_number(0.0, Some(2), &data), "0.00");
    }
}
