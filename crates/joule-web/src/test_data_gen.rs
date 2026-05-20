//! Test data generation.
//!
//! Replaces `faker.js`, `fake-rs`, `chance.js`, and similar fake data libraries
//! with a pure-Rust deterministic generator. Supports fake data for names, emails,
//! addresses, dates, UUIDs, IP addresses, lorem ipsum, and more. All generation
//! is deterministic given a seed, supports builder patterns for complex objects,
//! batch generation, and unique value guarantees.

use std::collections::HashSet;
use std::fmt;

// ── PRNG ─────────────────────────────────────────────────────────

/// Xoshiro256** for deterministic generation.
#[derive(Debug, Clone)]
struct Rng {
    state: [u64; 4],
}

impl Rng {
    fn new(seed: u64) -> Self {
        let mut sm = seed;
        let mut state = [0u64; 4];
        for s in &mut state {
            sm = sm.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = sm;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            *s = z ^ (z >> 31);
        }
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        let result = self.state[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.state[1] << 17;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);
        result
    }

    fn next_usize(&mut self, max: usize) -> usize {
        if max == 0 {
            return 0;
        }
        (self.next_u64() as usize) % max
    }

    fn next_u32_range(&mut self, lo: u32, hi: u32) -> u32 {
        if lo >= hi {
            return lo;
        }
        let range = (hi - lo) as u64 + 1;
        lo + (self.next_u64() % range) as u32
    }

    fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.next_usize(items.len())]
    }
}

// ── Word Lists ───────────────────────────────────────────────────

const FIRST_NAMES: &[&str] = &[
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael",
    "Linda", "David", "Elizabeth", "William", "Barbara", "Richard", "Susan",
    "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen", "Alice",
    "Bob", "Carol", "Daniel", "Emma", "Frank", "Grace", "Henry", "Iris",
    "Julia",
];

const LAST_NAMES: &[&str] = &[
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
    "Ramirez", "Lewis", "Robinson",
];

const STREET_NAMES: &[&str] = &[
    "Main", "Oak", "Elm", "Maple", "Cedar", "Pine", "Walnut", "Park",
    "Lake", "Hill", "Washington", "Lincoln", "Forest", "Spring", "River",
    "Valley", "Bridge", "Harbor", "Sunset", "Highland",
];

const STREET_SUFFIXES: &[&str] = &[
    "St", "Ave", "Blvd", "Dr", "Ln", "Ct", "Way", "Rd", "Pl", "Cir",
];

const CITIES: &[&str] = &[
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia",
    "San Antonio", "San Diego", "Dallas", "San Jose", "Austin", "Jacksonville",
    "Fort Worth", "Columbus", "Charlotte", "Indianapolis", "Denver",
    "Seattle", "Nashville", "Portland",
];

const STATES: &[&str] = &[
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
];

const DOMAINS: &[&str] = &[
    "example.com", "test.org", "demo.net", "sample.io", "mock.dev",
    "fake.co", "testmail.com", "mailtest.org",
];

const TLDS: &[&str] = &[
    "com", "org", "net", "io", "dev", "co", "app", "tech",
];

const LOREM_WORDS: &[&str] = &[
    "lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing",
    "elit", "sed", "do", "eiusmod", "tempor", "incididunt", "ut", "labore",
    "et", "dolore", "magna", "aliqua", "enim", "ad", "minim", "veniam",
    "quis", "nostrud", "exercitation", "ullamco", "laboris", "nisi",
    "aliquip", "ex", "ea", "commodo", "consequat", "duis", "aute", "irure",
    "in", "reprehenderit", "voluptate", "velit", "esse", "cillum", "fugiat",
    "nulla", "pariatur", "excepteur", "sint", "occaecat", "cupidatat",
];

const COMPANY_SUFFIXES: &[&str] = &[
    "Inc", "LLC", "Corp", "Ltd", "Group", "Solutions", "Technologies",
    "Systems", "Partners", "Associates",
];

// ── DataGen ──────────────────────────────────────────────────────

/// Deterministic fake data generator.
#[derive(Debug, Clone)]
pub struct DataGen {
    rng: Rng,
    /// Track unique values to guarantee no duplicates.
    unique_strings: HashSet<String>,
}

impl DataGen {
    /// Create a new generator with the given seed.
    pub fn new(seed: u64) -> Self {
        Self {
            rng: Rng::new(seed),
            unique_strings: HashSet::new(),
        }
    }

    /// Reset the generator to a new seed.
    pub fn reseed(&mut self, seed: u64) {
        self.rng = Rng::new(seed);
        self.unique_strings.clear();
    }

    // ── Names ────────────────────────────────────────────────────

    /// Generate a random first name.
    pub fn first_name(&mut self) -> String {
        self.rng.pick(FIRST_NAMES).to_string()
    }

    /// Generate a random last name.
    pub fn last_name(&mut self) -> String {
        self.rng.pick(LAST_NAMES).to_string()
    }

    /// Generate a full name (first + last).
    pub fn full_name(&mut self) -> String {
        let first = self.first_name();
        let last = self.last_name();
        format!("{first} {last}")
    }

    // ── Email ────────────────────────────────────────────────────

    /// Generate a random email address.
    pub fn email(&mut self) -> String {
        let first = self.rng.pick(FIRST_NAMES).to_lowercase();
        let last = self.rng.pick(LAST_NAMES).to_lowercase();
        let num = self.rng.next_u32_range(1, 999);
        let domain = self.rng.pick(DOMAINS);
        format!("{first}.{last}{num}@{domain}")
    }

    /// Generate an email for a specific name.
    pub fn email_for(&mut self, first: &str, last: &str) -> String {
        let domain = self.rng.pick(DOMAINS);
        let num = self.rng.next_u32_range(1, 99);
        format!("{}.{}{num}@{domain}", first.to_lowercase(), last.to_lowercase())
    }

    // ── Address ──────────────────────────────────────────────────

    /// Generate a street address.
    pub fn street_address(&mut self) -> String {
        let number = self.rng.next_u32_range(100, 9999);
        let street = self.rng.pick(STREET_NAMES);
        let suffix = self.rng.pick(STREET_SUFFIXES);
        format!("{number} {street} {suffix}")
    }

    /// Generate a city name.
    pub fn city(&mut self) -> String {
        self.rng.pick(CITIES).to_string()
    }

    /// Generate a state abbreviation.
    pub fn state(&mut self) -> String {
        self.rng.pick(STATES).to_string()
    }

    /// Generate a ZIP code.
    pub fn zip_code(&mut self) -> String {
        let code = self.rng.next_u32_range(10000, 99999);
        format!("{code:05}")
    }

    /// Generate a full address.
    pub fn full_address(&mut self) -> String {
        let street = self.street_address();
        let city = self.city();
        let state = self.state();
        let zip = self.zip_code();
        format!("{street}, {city}, {state} {zip}")
    }

    // ── Phone ────────────────────────────────────────────────────

    /// Generate a US phone number.
    pub fn phone(&mut self) -> String {
        let area = self.rng.next_u32_range(200, 999);
        let prefix = self.rng.next_u32_range(200, 999);
        let line = self.rng.next_u32_range(1000, 9999);
        format!("({area}) {prefix}-{line}")
    }

    // ── Date/Time ────────────────────────────────────────────────

    /// Generate a random date string (YYYY-MM-DD) between 1970 and 2030.
    pub fn date(&mut self) -> String {
        let year = self.rng.next_u32_range(1970, 2030);
        let month = self.rng.next_u32_range(1, 12);
        let max_day = match month {
            2 => 28,
            4 | 6 | 9 | 11 => 30,
            _ => 31,
        };
        let day = self.rng.next_u32_range(1, max_day);
        format!("{year:04}-{month:02}-{day:02}")
    }

    /// Generate a random date in a specific year.
    pub fn date_in_year(&mut self, year: u32) -> String {
        let month = self.rng.next_u32_range(1, 12);
        let max_day = match month {
            2 => 28,
            4 | 6 | 9 | 11 => 30,
            _ => 31,
        };
        let day = self.rng.next_u32_range(1, max_day);
        format!("{year:04}-{month:02}-{day:02}")
    }

    /// Generate a random ISO 8601 datetime.
    pub fn datetime(&mut self) -> String {
        let date = self.date();
        let hour = self.rng.next_u32_range(0, 23);
        let minute = self.rng.next_u32_range(0, 59);
        let second = self.rng.next_u32_range(0, 59);
        format!("{date}T{hour:02}:{minute:02}:{second:02}Z")
    }

    // ── UUID ─────────────────────────────────────────────────────

    /// Generate a random UUID v4 string.
    pub fn uuid(&mut self) -> String {
        let a = self.rng.next_u64();
        let b = self.rng.next_u64();
        // Set version 4 and variant bits
        let time_hi = ((a >> 48) & 0x0FFF) | 0x4000;
        let clock_seq = ((b >> 48) & 0x3FFF) | 0x8000;
        format!(
            "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
            (a >> 32) as u32,
            (a >> 16) as u16 & 0xFFFF,
            time_hi as u16,
            clock_seq as u16,
            b & 0xFFFFFFFFFFFF,
        )
    }

    // ── IP Addresses ─────────────────────────────────────────────

    /// Generate a random IPv4 address.
    pub fn ipv4(&mut self) -> String {
        let a = self.rng.next_u32_range(1, 254);
        let b = self.rng.next_u32_range(0, 255);
        let c = self.rng.next_u32_range(0, 255);
        let d = self.rng.next_u32_range(1, 254);
        format!("{a}.{b}.{c}.{d}")
    }

    /// Generate a random IPv6 address.
    pub fn ipv6(&mut self) -> String {
        let parts: Vec<String> = (0..8)
            .map(|_| format!("{:04x}", self.rng.next_u64() as u16))
            .collect();
        parts.join(":")
    }

    /// Generate a private IPv4 address (10.x.x.x).
    pub fn private_ipv4(&mut self) -> String {
        let b = self.rng.next_u32_range(0, 255);
        let c = self.rng.next_u32_range(0, 255);
        let d = self.rng.next_u32_range(1, 254);
        format!("10.{b}.{c}.{d}")
    }

    // ── Lorem Ipsum ──────────────────────────────────────────────

    /// Generate N random words.
    pub fn words(&mut self, count: usize) -> String {
        (0..count)
            .map(|_| *self.rng.pick(LOREM_WORDS))
            .collect::<Vec<&str>>()
            .join(" ")
    }

    /// Generate a sentence (5-15 words, capitalized, with period).
    pub fn sentence(&mut self) -> String {
        let word_count = self.rng.next_u32_range(5, 15) as usize;
        let mut text = self.words(word_count);
        if let Some(first) = text.get_mut(0..1) {
            first.make_ascii_uppercase();
        }
        text.push('.');
        text
    }

    /// Generate a paragraph (3-7 sentences).
    pub fn paragraph(&mut self) -> String {
        let sentence_count = self.rng.next_u32_range(3, 7) as usize;
        (0..sentence_count)
            .map(|_| self.sentence())
            .collect::<Vec<String>>()
            .join(" ")
    }

    // ── Company ──────────────────────────────────────────────────

    /// Generate a company name.
    pub fn company_name(&mut self) -> String {
        let last = self.rng.pick(LAST_NAMES);
        let suffix = self.rng.pick(COMPANY_SUFFIXES);
        format!("{last} {suffix}")
    }

    // ── Domain/URL ───────────────────────────────────────────────

    /// Generate a domain name.
    pub fn domain(&mut self) -> String {
        let word = self.rng.pick(LOREM_WORDS);
        let tld = self.rng.pick(TLDS);
        format!("{word}.{tld}")
    }

    /// Generate a URL.
    pub fn url(&mut self) -> String {
        let domain = self.domain();
        let path = self.rng.pick(LOREM_WORDS);
        format!("https://{domain}/{path}")
    }

    // ── Numbers ──────────────────────────────────────────────────

    /// Generate a random integer in a range.
    pub fn int(&mut self, lo: i64, hi: i64) -> i64 {
        if lo >= hi {
            return lo;
        }
        let range = (hi - lo) as u64 + 1;
        lo + (self.rng.next_u64() % range) as i64
    }

    /// Generate a random float in [lo, hi).
    pub fn float(&mut self, lo: f64, hi: f64) -> f64 {
        let f = (self.rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        lo + f * (hi - lo)
    }

    /// Generate a boolean.
    pub fn boolean(&mut self) -> bool {
        self.rng.next_u64() & 1 == 1
    }

    // ── Unique Values ────────────────────────────────────────────

    /// Generate a unique string value. Retries up to 1000 times.
    pub fn unique<F>(&mut self, max_retries: usize, generator: F) -> Option<String>
    where
        F: Fn(&mut Self) -> String,
    {
        for _ in 0..max_retries {
            let value = generator(self);
            if !self.unique_strings.contains(&value) {
                self.unique_strings.insert(value.clone());
                return Some(value);
            }
        }
        None
    }

    /// Number of tracked unique values.
    pub fn unique_count(&self) -> usize {
        self.unique_strings.len()
    }

    /// Clear unique value tracking.
    pub fn clear_unique(&mut self) {
        self.unique_strings.clear();
    }

    // ── Batch ────────────────────────────────────────────────────

    /// Generate a batch of values.
    pub fn batch<F, T>(&mut self, count: usize, generator: F) -> Vec<T>
    where
        F: Fn(&mut Self) -> T,
    {
        (0..count).map(|_| generator(self)).collect()
    }

    // ── Hex / Alphanumeric strings ───────────────────────────────

    /// Generate a random hex string of given length.
    pub fn hex_string(&mut self, len: usize) -> String {
        (0..len)
            .map(|_| {
                let nibble = (self.rng.next_u64() % 16) as u8;
                if nibble < 10 {
                    (b'0' + nibble) as char
                } else {
                    (b'a' + nibble - 10) as char
                }
            })
            .collect()
    }

    /// Generate a random alphanumeric string.
    pub fn alphanumeric(&mut self, len: usize) -> String {
        const CHARS: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        (0..len)
            .map(|_| CHARS[self.rng.next_usize(CHARS.len())] as char)
            .collect()
    }

    // ── Username / Password ──────────────────────────────────────

    /// Generate a username.
    pub fn username(&mut self) -> String {
        let first = self.rng.pick(FIRST_NAMES).to_lowercase();
        let num = self.rng.next_u32_range(1, 999);
        format!("{first}{num}")
    }

    /// Generate a password-like string.
    pub fn password(&mut self, len: usize) -> String {
        const CHARS: &[u8] =
            b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@$%^&*";
        (0..len)
            .map(|_| CHARS[self.rng.next_usize(CHARS.len())] as char)
            .collect()
    }

    // ── Color ────────────────────────────────────────────────────

    /// Generate a random hex color (e.g., "ff3a2b").
    pub fn hex_color(&mut self) -> String {
        let r = self.rng.next_u64() as u8;
        let g = self.rng.next_u64() as u8;
        let b = self.rng.next_u64() as u8;
        format!("{r:02x}{g:02x}{b:02x}")
    }
}

impl fmt::Display for DataGen {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DataGen(unique_tracked={})", self.unique_strings.len())
    }
}

// ── Person Builder ───────────────────────────────────────────────

/// Builder for generating a complete person record.
#[derive(Debug, Clone, Default)]
pub struct PersonBuilder {
    first_name: Option<String>,
    last_name: Option<String>,
    email: Option<String>,
    phone: Option<String>,
    address: Option<String>,
}

impl PersonBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_first_name(mut self, name: &str) -> Self {
        self.first_name = Some(name.to_string());
        self
    }

    pub fn with_last_name(mut self, name: &str) -> Self {
        self.last_name = Some(name.to_string());
        self
    }

    pub fn with_email(mut self, email: &str) -> Self {
        self.email = Some(email.to_string());
        self
    }

    /// Build the person, filling in missing fields from the generator.
    pub fn build(self, dg: &mut DataGen) -> Person {
        let first = self.first_name.unwrap_or_else(|| dg.first_name());
        let last = self.last_name.unwrap_or_else(|| dg.last_name());
        let email = self.email.unwrap_or_else(|| {
            dg.email_for(&first, &last)
        });
        let phone = self.phone.unwrap_or_else(|| dg.phone());
        let address = self.address.unwrap_or_else(|| dg.full_address());

        Person {
            first_name: first,
            last_name: last,
            email,
            phone,
            address,
        }
    }
}

/// A generated person record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Person {
    pub first_name: String,
    pub last_name: String,
    pub email: String,
    pub phone: String,
    pub address: String,
}

impl Person {
    pub fn full_name(&self) -> String {
        format!("{} {}", self.first_name, self.last_name)
    }
}

impl fmt::Display for Person {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {} <{}> {}",
            self.first_name, self.last_name, self.email, self.phone
        )
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_with_seed() {
        let mut g1 = DataGen::new(42);
        let mut g2 = DataGen::new(42);
        assert_eq!(g1.first_name(), g2.first_name());
        assert_eq!(g1.email(), g2.email());
        assert_eq!(g1.uuid(), g2.uuid());
    }

    #[test]
    fn different_seeds_different_values() {
        let mut g1 = DataGen::new(1);
        let mut g2 = DataGen::new(2);
        // Different seeds should produce at least some different values
        let names1: Vec<String> = (0..10).map(|_| g1.first_name()).collect();
        let names2: Vec<String> = (0..10).map(|_| g2.first_name()).collect();
        assert_ne!(names1, names2);
    }

    #[test]
    fn full_name_format() {
        let mut dg = DataGen::new(42);
        let name = dg.full_name();
        assert!(name.contains(' '));
        let parts: Vec<&str> = name.split(' ').collect();
        assert_eq!(parts.len(), 2);
    }

    #[test]
    fn email_format() {
        let mut dg = DataGen::new(42);
        let email = dg.email();
        assert!(email.contains('@'));
        assert!(email.contains('.'));
    }

    #[test]
    fn street_address_has_number() {
        let mut dg = DataGen::new(42);
        let addr = dg.street_address();
        // Should start with a number
        assert!(addr.chars().next().unwrap().is_ascii_digit());
    }

    #[test]
    fn zip_code_format() {
        let mut dg = DataGen::new(42);
        let zip = dg.zip_code();
        assert_eq!(zip.len(), 5);
        assert!(zip.chars().all(|c| c.is_ascii_digit()));
    }

    #[test]
    fn date_format() {
        let mut dg = DataGen::new(42);
        let date = dg.date();
        assert_eq!(date.len(), 10);
        assert_eq!(date.chars().nth(4), Some('-'));
        assert_eq!(date.chars().nth(7), Some('-'));
    }

    #[test]
    fn datetime_format() {
        let mut dg = DataGen::new(42);
        let dt = dg.datetime();
        assert!(dt.contains('T'));
        assert!(dt.ends_with('Z'));
    }

    #[test]
    fn uuid_format() {
        let mut dg = DataGen::new(42);
        let uuid = dg.uuid();
        let parts: Vec<&str> = uuid.split('-').collect();
        assert_eq!(parts.len(), 5);
        assert_eq!(parts[0].len(), 8);
        assert_eq!(parts[1].len(), 4);
        assert_eq!(parts[2].len(), 4);
        assert_eq!(parts[3].len(), 4);
        assert_eq!(parts[4].len(), 12);
        // Version 4 indicator
        assert!(parts[2].starts_with('4'));
    }

    #[test]
    fn ipv4_format() {
        let mut dg = DataGen::new(42);
        let ip = dg.ipv4();
        let parts: Vec<&str> = ip.split('.').collect();
        assert_eq!(parts.len(), 4);
        for part in parts {
            let n: u32 = part.parse().unwrap();
            assert!(n <= 255);
        }
    }

    #[test]
    fn ipv6_format() {
        let mut dg = DataGen::new(42);
        let ip = dg.ipv6();
        let parts: Vec<&str> = ip.split(':').collect();
        assert_eq!(parts.len(), 8);
    }

    #[test]
    fn private_ipv4_starts_with_10() {
        let mut dg = DataGen::new(42);
        let ip = dg.private_ipv4();
        assert!(ip.starts_with("10."));
    }

    #[test]
    fn lorem_words() {
        let mut dg = DataGen::new(42);
        let text = dg.words(5);
        let words: Vec<&str> = text.split(' ').collect();
        assert_eq!(words.len(), 5);
    }

    #[test]
    fn sentence_format() {
        let mut dg = DataGen::new(42);
        let s = dg.sentence();
        assert!(s.ends_with('.'));
        assert!(s.chars().next().unwrap().is_uppercase());
    }

    #[test]
    fn paragraph_multiple_sentences() {
        let mut dg = DataGen::new(42);
        let p = dg.paragraph();
        let sentence_count = p.matches('.').count();
        assert!(sentence_count >= 3);
    }

    #[test]
    fn batch_generation() {
        let mut dg = DataGen::new(42);
        let emails = dg.batch(10, |g| g.email());
        assert_eq!(emails.len(), 10);
        for email in &emails {
            assert!(email.contains('@'));
        }
    }

    #[test]
    fn unique_values() {
        let mut dg = DataGen::new(42);
        let e1 = dg.unique(100, |g| g.email());
        let e2 = dg.unique(100, |g| g.email());
        assert!(e1.is_some());
        assert!(e2.is_some());
        assert_ne!(e1, e2);
        assert_eq!(dg.unique_count(), 2);
    }

    #[test]
    fn hex_string_format() {
        let mut dg = DataGen::new(42);
        let hex = dg.hex_string(16);
        assert_eq!(hex.len(), 16);
        assert!(hex.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn alphanumeric_string() {
        let mut dg = DataGen::new(42);
        let s = dg.alphanumeric(20);
        assert_eq!(s.len(), 20);
        assert!(s.chars().all(|c| c.is_alphanumeric()));
    }

    #[test]
    fn username_format() {
        let mut dg = DataGen::new(42);
        let u = dg.username();
        assert!(!u.is_empty());
        assert!(u.chars().last().unwrap().is_ascii_digit());
    }

    #[test]
    fn password_length() {
        let mut dg = DataGen::new(42);
        let p = dg.password(16);
        assert_eq!(p.len(), 16);
    }

    #[test]
    fn person_builder() {
        let mut dg = DataGen::new(42);
        let person = PersonBuilder::new()
            .with_first_name("Alice")
            .build(&mut dg);
        assert_eq!(person.first_name, "Alice");
        assert!(!person.last_name.is_empty());
        assert!(person.email.contains("alice"));
    }

    #[test]
    fn person_display() {
        let mut dg = DataGen::new(42);
        let person = PersonBuilder::new().build(&mut dg);
        let s = format!("{person}");
        assert!(s.contains('@'));
        assert!(s.contains('<'));
    }

    #[test]
    fn reseed_resets_state() {
        let mut dg = DataGen::new(42);
        let name1 = dg.first_name();
        dg.reseed(42);
        let name2 = dg.first_name();
        assert_eq!(name1, name2);
    }

    #[test]
    fn company_name_format() {
        let mut dg = DataGen::new(42);
        let name = dg.company_name();
        assert!(name.contains(' '));
    }

    #[test]
    fn url_format() {
        let mut dg = DataGen::new(42);
        let url = dg.url();
        assert!(url.starts_with("https://"));
    }

    #[test]
    fn int_range() {
        let mut dg = DataGen::new(42);
        for _ in 0..100 {
            let v = dg.int(10, 20);
            assert!(v >= 10 && v <= 20);
        }
    }

    #[test]
    fn float_range() {
        let mut dg = DataGen::new(42);
        for _ in 0..100 {
            let v = dg.float(0.0, 1.0);
            assert!(v >= 0.0 && v < 1.0);
        }
    }

    #[test]
    fn hex_color_format() {
        let mut dg = DataGen::new(42);
        let color = dg.hex_color();
        assert_eq!(color.len(), 6);
        assert!(color.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn phone_format() {
        let mut dg = DataGen::new(42);
        let phone = dg.phone();
        assert!(phone.starts_with('('));
        assert!(phone.contains(')'));
        assert!(phone.contains('-'));
    }

    #[test]
    fn date_in_year_correct_year() {
        let mut dg = DataGen::new(42);
        let date = dg.date_in_year(2025);
        assert!(date.starts_with("2025-"));
    }
}
