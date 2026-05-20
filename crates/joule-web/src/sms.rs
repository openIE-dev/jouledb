//! SMS message model — GSM-7/UCS-2 encoding, segmentation, E.164 validation.
//!
//! Replaces Twilio SDK / Vonage / Plivo with a pure-Rust SMS domain model.
//! No HTTP calls — only message construction, validation, and segment math.

use chrono::{DateTime, Utc};
use std::collections::HashMap;
use uuid::Uuid;

// ── Constants ───────────────────────────────────────────────────

/// GSM-7 basic character set (standard table).
const GSM7_BASIC: &str =
    "@£$¥èéùìòÇ\nØø\rÅåΔ_ΦΓΛΩΠΨΣΘΞ ÆæßÉ !\"#¤%&'()*+,-./0123456789:;<=>?\
     ¡ABCDEFGHIJKLMNOPQRSTUVWXYZ\
     ÄÖÑÜabcdefghijklmnopqrstuvwxyz\
     äöñüà§";

/// GSM-7 extended characters (require escape, count as 2).
const GSM7_EXTENDED: &str = "^{}\\[~]|€";

// ── Delivery Status ─────────────────────────────────────────────

/// SMS delivery status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeliveryStatus {
    Pending,
    Sent,
    Delivered,
    Failed,
}

// ── Encoding ────────────────────────────────────────────────────

/// Detected encoding for the message body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmsEncoding {
    Gsm7,
    Ucs2,
}

// ── SmsMessage ──────────────────────────────────────────────────

/// An SMS message.
#[derive(Debug, Clone)]
pub struct SmsMessage {
    pub id: Uuid,
    pub from: String,
    pub to: String,
    pub body: String,
    pub timestamp: DateTime<Utc>,
    pub status: DeliveryStatus,
}

impl SmsMessage {
    /// Create a new SMS message.
    pub fn new(from: &str, to: &str, body: &str) -> Self {
        Self {
            id: Uuid::new_v4(),
            from: from.to_string(),
            to: to.to_string(),
            body: body.to_string(),
            timestamp: Utc::now(),
            status: DeliveryStatus::Pending,
        }
    }

    /// Detect the encoding required for this message.
    pub fn encoding(&self) -> SmsEncoding {
        detect_encoding(&self.body)
    }

    /// Count the number of GSM-7 septets (or UCS-2 code units) in the body.
    pub fn char_count(&self) -> usize {
        match self.encoding() {
            SmsEncoding::Gsm7 => gsm7_char_count(&self.body),
            SmsEncoding::Ucs2 => self.body.chars().count(),
        }
    }

    /// Number of SMS segments required.
    pub fn segment_count(&self) -> usize {
        let count = self.char_count();
        match self.encoding() {
            SmsEncoding::Gsm7 => {
                if count <= 160 {
                    1
                } else {
                    (count + 152) / 153 // ceil division
                }
            }
            SmsEncoding::Ucs2 => {
                if count <= 70 {
                    1
                } else {
                    (count + 66) / 67
                }
            }
        }
    }

    /// Validate the "from" phone number (E.164).
    pub fn validate_from(&self) -> bool {
        validate_e164(&self.from)
    }

    /// Validate the "to" phone number (E.164).
    pub fn validate_to(&self) -> bool {
        validate_e164(&self.to)
    }

    /// Update delivery status.
    pub fn set_status(&mut self, status: DeliveryStatus) {
        self.status = status;
    }
}

// ── Phone number validation ─────────────────────────────────────

/// Validate an E.164 phone number: +[country code][number], 1-15 digits total.
pub fn validate_e164(number: &str) -> bool {
    if !number.starts_with('+') {
        return false;
    }
    let digits = &number[1..];
    if digits.is_empty() || digits.len() > 15 {
        return false;
    }
    digits.chars().all(|c| c.is_ascii_digit())
}

// ── GSM-7 helpers ───────────────────────────────────────────────

/// Detect whether all characters fit in GSM-7.
fn detect_encoding(text: &str) -> SmsEncoding {
    for ch in text.chars() {
        if !GSM7_BASIC.contains(ch) && !GSM7_EXTENDED.contains(ch) {
            return SmsEncoding::Ucs2;
        }
    }
    SmsEncoding::Gsm7
}

/// Count GSM-7 septets (extended chars count as 2).
fn gsm7_char_count(text: &str) -> usize {
    let mut count = 0;
    for ch in text.chars() {
        if GSM7_EXTENDED.contains(ch) {
            count += 2;
        } else {
            count += 1;
        }
    }
    count
}

/// Check if a single character is in the GSM-7 charset.
pub fn is_gsm7_char(ch: char) -> bool {
    GSM7_BASIC.contains(ch) || GSM7_EXTENDED.contains(ch)
}

// ── Message Threading ───────────────────────────────────────────

/// Thread messages by phone number pair.
#[derive(Debug, Clone, Default)]
pub struct MessageThread {
    threads: HashMap<String, Vec<SmsMessage>>,
}

impl MessageThread {
    pub fn new() -> Self {
        Self::default()
    }

    /// Generate a thread key from two phone numbers (sorted for consistency).
    fn thread_key(a: &str, b: &str) -> String {
        let mut pair = [a, b];
        pair.sort();
        format!("{}:{}", pair[0], pair[1])
    }

    /// Add a message to its thread.
    pub fn add(&mut self, msg: SmsMessage) {
        let key = Self::thread_key(&msg.from, &msg.to);
        self.threads.entry(key).or_default().push(msg);
    }

    /// Get messages in a thread between two numbers.
    pub fn get_thread(&self, a: &str, b: &str) -> Option<&[SmsMessage]> {
        let key = Self::thread_key(a, b);
        self.threads.get(&key).map(|v| v.as_slice())
    }

    /// Number of distinct threads.
    pub fn thread_count(&self) -> usize {
        self.threads.len()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gsm7_detection() {
        let msg = SmsMessage::new("+1234567890", "+0987654321", "Hello world");
        assert_eq!(msg.encoding(), SmsEncoding::Gsm7);
    }

    #[test]
    fn test_ucs2_detection() {
        let msg = SmsMessage::new("+1234567890", "+0987654321", "Hello 你好");
        assert_eq!(msg.encoding(), SmsEncoding::Ucs2);
    }

    #[test]
    fn test_gsm7_extended_counts_double() {
        // '{' is extended GSM-7, counts as 2 septets
        let msg = SmsMessage::new("+1234567890", "+0987654321", "{");
        assert_eq!(msg.encoding(), SmsEncoding::Gsm7);
        assert_eq!(msg.char_count(), 2);
    }

    #[test]
    fn test_single_segment_gsm7() {
        let body = "A".repeat(160);
        let msg = SmsMessage::new("+1234567890", "+0987654321", &body);
        assert_eq!(msg.segment_count(), 1);
    }

    #[test]
    fn test_multi_segment_gsm7() {
        let body = "A".repeat(161);
        let msg = SmsMessage::new("+1234567890", "+0987654321", &body);
        assert_eq!(msg.segment_count(), 2); // 161 chars -> 2 segments at 153 each
    }

    #[test]
    fn test_single_segment_ucs2() {
        let body = "你".repeat(70);
        let msg = SmsMessage::new("+1234567890", "+0987654321", &body);
        assert_eq!(msg.encoding(), SmsEncoding::Ucs2);
        assert_eq!(msg.segment_count(), 1);
    }

    #[test]
    fn test_multi_segment_ucs2() {
        let body = "你".repeat(71);
        let msg = SmsMessage::new("+1234567890", "+0987654321", &body);
        assert_eq!(msg.segment_count(), 2);
    }

    #[test]
    fn test_e164_valid() {
        assert!(validate_e164("+1234567890"));
        assert!(validate_e164("+44207123456"));
    }

    #[test]
    fn test_e164_invalid() {
        assert!(!validate_e164("1234567890")); // no +
        assert!(!validate_e164("+")); // no digits
        assert!(!validate_e164("+1234567890123456")); // too long (16 digits)
        assert!(!validate_e164("+123-456")); // non-digit
    }

    #[test]
    fn test_delivery_status() {
        let mut msg = SmsMessage::new("+1234567890", "+0987654321", "Test");
        assert_eq!(msg.status, DeliveryStatus::Pending);
        msg.set_status(DeliveryStatus::Sent);
        assert_eq!(msg.status, DeliveryStatus::Sent);
        msg.set_status(DeliveryStatus::Delivered);
        assert_eq!(msg.status, DeliveryStatus::Delivered);
    }

    #[test]
    fn test_message_threading() {
        let mut threads = MessageThread::new();
        threads.add(SmsMessage::new("+1111", "+2222", "Hello"));
        threads.add(SmsMessage::new("+2222", "+1111", "Hi back"));
        threads.add(SmsMessage::new("+1111", "+3333", "Other thread"));

        assert_eq!(threads.thread_count(), 2);
        let thread = threads.get_thread("+1111", "+2222").unwrap();
        assert_eq!(thread.len(), 2);
        assert_eq!(thread[0].body, "Hello");
        assert_eq!(thread[1].body, "Hi back");
    }

    #[test]
    fn test_validate_from_to() {
        let msg = SmsMessage::new("+12025551234", "+447911123456", "Test");
        assert!(msg.validate_from());
        assert!(msg.validate_to());

        let bad = SmsMessage::new("not-a-number", "+1234", "Test");
        assert!(!bad.validate_from());
    }

    #[test]
    fn test_three_segment_gsm7() {
        let body = "A".repeat(307); // 307 / 153 = 2.006 -> 3 segments
        let msg = SmsMessage::new("+1234567890", "+0987654321", &body);
        assert_eq!(msg.segment_count(), 3);
    }

    #[test]
    fn test_gsm7_char_validation() {
        assert!(is_gsm7_char('A'));
        assert!(is_gsm7_char('0'));
        assert!(is_gsm7_char('€'));
        assert!(!is_gsm7_char('你'));
    }
}
