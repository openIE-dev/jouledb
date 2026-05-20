//! Input masking engine — mask patterns, cursor tracking, partial validation.
//!
//! Replaces inputmask.js, cleave.js, and imask.js with a pure-Rust masking
//! engine that enforces input patterns with placeholder display and cursor
//! position tracking.

use serde::{Deserialize, Serialize};

// ── Errors ──────────────────────────────────────────────────────

/// Masking errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MaskError {
    /// Character does not match the mask slot.
    InvalidCharacter { position: usize, expected: SlotKind },
    /// Input exceeds mask length.
    InputTooLong { max: usize },
    /// Mask pattern is empty or invalid.
    InvalidMask(String),
}

impl std::fmt::Display for MaskError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidCharacter { position, expected } => {
                write!(f, "invalid char at {position}, expected {expected:?}")
            }
            Self::InputTooLong { max } => write!(f, "input exceeds max length {max}"),
            Self::InvalidMask(s) => write!(f, "invalid mask: {s}"),
        }
    }
}

impl std::error::Error for MaskError {}

// ── Mask Slots ──────────────────────────────────────────────────

/// Kind of mask slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SlotKind {
    /// `#` — digit only (0-9).
    Digit,
    /// `A` — letter only (a-z, A-Z).
    Letter,
    /// `*` — any character.
    Any,
    /// A literal character that is auto-inserted.
    Literal(char),
}

impl SlotKind {
    /// Check if a character matches this slot.
    pub fn matches(&self, ch: char) -> bool {
        match self {
            SlotKind::Digit => ch.is_ascii_digit(),
            SlotKind::Letter => ch.is_ascii_alphabetic(),
            SlotKind::Any => true,
            SlotKind::Literal(lit) => ch == *lit,
        }
    }

    /// The placeholder character for display.
    pub fn placeholder(&self) -> char {
        match self {
            SlotKind::Digit => '_',
            SlotKind::Letter => '_',
            SlotKind::Any => '_',
            SlotKind::Literal(c) => *c,
        }
    }
}

// ── Mask Definition ─────────────────────────────────────────────

/// A parsed mask pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaskPattern {
    pub slots: Vec<SlotKind>,
    /// Original pattern string.
    pub pattern: String,
}

impl MaskPattern {
    /// Parse a mask pattern string.
    ///
    /// - `#` → digit
    /// - `A` → letter
    /// - `*` → any
    /// - `\\` → escape next char as literal
    /// - anything else → literal
    pub fn parse(pattern: &str) -> Result<Self, MaskError> {
        if pattern.is_empty() {
            return Err(MaskError::InvalidMask("empty pattern".into()));
        }

        let mut slots = Vec::new();
        let mut chars = pattern.chars().peekable();

        while let Some(ch) = chars.next() {
            match ch {
                '\\' => {
                    if let Some(next) = chars.next() {
                        slots.push(SlotKind::Literal(next));
                    } else {
                        return Err(MaskError::InvalidMask("trailing backslash".into()));
                    }
                }
                '#' => slots.push(SlotKind::Digit),
                'A' => slots.push(SlotKind::Letter),
                '*' => slots.push(SlotKind::Any),
                other => slots.push(SlotKind::Literal(other)),
            }
        }

        Ok(Self {
            slots,
            pattern: pattern.to_string(),
        })
    }

    /// Number of input slots (non-literal).
    pub fn input_slot_count(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| !matches!(s, SlotKind::Literal(_)))
            .count()
    }

    /// Total length of the masked output.
    pub fn total_len(&self) -> usize {
        self.slots.len()
    }

    /// Generate the placeholder string.
    pub fn placeholder(&self) -> String {
        self.slots.iter().map(|s| s.placeholder()).collect()
    }
}

// ── Masked Input ────────────────────────────────────────────────

/// State of a masked input field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaskedInput {
    pub mask: MaskPattern,
    /// Raw input characters (only user-typed, no literals).
    pub raw: Vec<char>,
    /// Current cursor position in the display string.
    pub cursor: usize,
}

impl MaskedInput {
    pub fn new(mask: MaskPattern) -> Self {
        let cursor = first_input_slot(&mask.slots);
        Self {
            mask,
            raw: Vec::new(),
            cursor,
        }
    }

    /// Insert a character at the current cursor position.
    pub fn insert(&mut self, ch: char) -> Result<(), MaskError> {
        // Find the mask slot at the current cursor position.
        let slot_idx = self.cursor;
        if slot_idx >= self.mask.slots.len() {
            return Err(MaskError::InputTooLong {
                max: self.mask.slots.len(),
            });
        }

        // Skip literal slots.
        let mut idx = slot_idx;
        while idx < self.mask.slots.len() {
            if let SlotKind::Literal(_) = &self.mask.slots[idx] {
                idx += 1;
            } else {
                break;
            }
        }

        if idx >= self.mask.slots.len() {
            return Err(MaskError::InputTooLong {
                max: self.mask.slots.len(),
            });
        }

        let slot = &self.mask.slots[idx];
        if !slot.matches(ch) {
            return Err(MaskError::InvalidCharacter {
                position: idx,
                expected: *slot,
            });
        }

        // Calculate raw index from display index.
        let raw_idx = self.display_to_raw_index(idx);
        if raw_idx <= self.raw.len() {
            self.raw.insert(raw_idx, ch);
        } else {
            self.raw.push(ch);
        }

        // Advance cursor past this slot and any following literals.
        self.cursor = idx + 1;
        while self.cursor < self.mask.slots.len() {
            if let SlotKind::Literal(_) = &self.mask.slots[self.cursor] {
                self.cursor += 1;
            } else {
                break;
            }
        }

        Ok(())
    }

    /// Remove the last raw character.
    pub fn backspace(&mut self) {
        if self.raw.pop().is_some() {
            // Recalculate cursor: find display position of last input slot filled.
            self.cursor = self.raw_to_display_index(self.raw.len());
        }
    }

    /// Get the display string with placeholders.
    pub fn display(&self) -> String {
        let mut out = String::new();
        let mut raw_idx = 0;

        for slot in &self.mask.slots {
            match slot {
                SlotKind::Literal(c) => out.push(*c),
                _ => {
                    if raw_idx < self.raw.len() {
                        out.push(self.raw[raw_idx]);
                        raw_idx += 1;
                    } else {
                        out.push(slot.placeholder());
                    }
                }
            }
        }

        out
    }

    /// Get the raw value (user input only, no literals).
    pub fn raw_value(&self) -> String {
        self.raw.iter().collect()
    }

    /// Get the unmasked value (raw input with no formatting).
    pub fn unmasked(&self) -> String {
        self.raw.iter().collect()
    }

    /// Get the masked value (only filled portion, including literals).
    pub fn masked_value(&self) -> String {
        let mut out = String::new();
        let mut raw_idx = 0;

        for slot in &self.mask.slots {
            if raw_idx >= self.raw.len() && !matches!(slot, SlotKind::Literal(_)) {
                break;
            }
            match slot {
                SlotKind::Literal(c) => out.push(*c),
                _ => {
                    if raw_idx < self.raw.len() {
                        out.push(self.raw[raw_idx]);
                        raw_idx += 1;
                    }
                }
            }
        }

        out
    }

    /// Check if the input is complete (all slots filled).
    pub fn is_complete(&self) -> bool {
        self.raw.len() == self.mask.input_slot_count()
    }

    /// Check if the current input is valid (all entered chars match their slots).
    pub fn is_valid_partial(&self) -> bool {
        let mut raw_idx = 0;
        for slot in &self.mask.slots {
            if raw_idx >= self.raw.len() {
                break;
            }
            match slot {
                SlotKind::Literal(_) => {}
                _ => {
                    if !slot.matches(self.raw[raw_idx]) {
                        return false;
                    }
                    raw_idx += 1;
                }
            }
        }
        true
    }

    /// Clear all input.
    pub fn clear(&mut self) {
        self.raw.clear();
        self.cursor = first_input_slot(&self.mask.slots);
    }

    /// Set the input from a raw string (only input characters, no literals).
    pub fn set_raw(&mut self, input: &str) -> Result<(), MaskError> {
        self.clear();
        for ch in input.chars() {
            self.insert(ch)?;
        }
        Ok(())
    }

    fn display_to_raw_index(&self, display_idx: usize) -> usize {
        let mut count = 0;
        for (i, slot) in self.mask.slots.iter().enumerate() {
            if i >= display_idx {
                break;
            }
            if !matches!(slot, SlotKind::Literal(_)) {
                count += 1;
            }
        }
        count
    }

    fn raw_to_display_index(&self, raw_idx: usize) -> usize {
        let mut count = 0;
        for (i, slot) in self.mask.slots.iter().enumerate() {
            if !matches!(slot, SlotKind::Literal(_)) {
                if count == raw_idx {
                    return i;
                }
                count += 1;
            }
        }
        self.mask.slots.len()
    }
}

fn first_input_slot(slots: &[SlotKind]) -> usize {
    slots
        .iter()
        .position(|s| !matches!(s, SlotKind::Literal(_)))
        .unwrap_or(0)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_phone_mask() {
        let mask = MaskPattern::parse("(###) ###-####").unwrap();
        assert_eq!(mask.total_len(), 14);
        assert_eq!(mask.input_slot_count(), 10);
    }

    #[test]
    fn placeholder_display() {
        let mask = MaskPattern::parse("##/##/####").unwrap();
        assert_eq!(mask.placeholder(), "__/__/____");
    }

    #[test]
    fn insert_digits() {
        let mask = MaskPattern::parse("###-####").unwrap();
        let mut input = MaskedInput::new(mask);
        input.insert('5').unwrap();
        input.insert('5').unwrap();
        input.insert('5').unwrap();
        assert_eq!(input.display(), "555-____");
        input.insert('1').unwrap();
        input.insert('2').unwrap();
        input.insert('3').unwrap();
        input.insert('4').unwrap();
        assert_eq!(input.display(), "555-1234");
        assert!(input.is_complete());
    }

    #[test]
    fn reject_invalid_char() {
        let mask = MaskPattern::parse("###").unwrap();
        let mut input = MaskedInput::new(mask);
        assert!(input.insert('a').is_err());
    }

    #[test]
    fn letter_mask() {
        let mask = MaskPattern::parse("AA-###").unwrap();
        let mut input = MaskedInput::new(mask);
        input.insert('A').unwrap();
        input.insert('B').unwrap();
        input.insert('1').unwrap();
        input.insert('2').unwrap();
        input.insert('3').unwrap();
        assert_eq!(input.display(), "AB-123");
        assert!(input.is_complete());
    }

    #[test]
    fn any_mask() {
        let mask = MaskPattern::parse("**-**").unwrap();
        let mut input = MaskedInput::new(mask);
        input.insert('A').unwrap();
        input.insert('1').unwrap();
        input.insert('B').unwrap();
        input.insert('2').unwrap();
        assert_eq!(input.display(), "A1-B2");
    }

    #[test]
    fn backspace() {
        let mask = MaskPattern::parse("###").unwrap();
        let mut input = MaskedInput::new(mask);
        input.insert('1').unwrap();
        input.insert('2').unwrap();
        input.backspace();
        assert_eq!(input.display(), "1__");
        assert_eq!(input.raw_value(), "1");
    }

    #[test]
    fn raw_and_masked_value() {
        let mask = MaskPattern::parse("(##) ##").unwrap();
        let mut input = MaskedInput::new(mask);
        input.insert('1').unwrap();
        input.insert('2').unwrap();
        input.insert('3').unwrap();
        assert_eq!(input.raw_value(), "123");
        assert_eq!(input.masked_value(), "(12) 3");
    }

    #[test]
    fn set_raw() {
        let mask = MaskPattern::parse("##-##").unwrap();
        let mut input = MaskedInput::new(mask);
        input.set_raw("1234").unwrap();
        assert_eq!(input.display(), "12-34");
        assert!(input.is_complete());
    }

    #[test]
    fn clear() {
        let mask = MaskPattern::parse("###").unwrap();
        let mut input = MaskedInput::new(mask);
        input.insert('1').unwrap();
        input.clear();
        assert_eq!(input.display(), "___");
        assert!(input.raw.is_empty());
    }

    #[test]
    fn input_too_long() {
        let mask = MaskPattern::parse("##").unwrap();
        let mut input = MaskedInput::new(mask);
        input.insert('1').unwrap();
        input.insert('2').unwrap();
        assert!(input.insert('3').is_err());
    }

    #[test]
    fn escaped_literal() {
        let mask = MaskPattern::parse("\\##-#").unwrap();
        // \# means literal '#', then digit slot, '-', digit slot
        assert_eq!(mask.slots.len(), 4);
        assert_eq!(mask.slots[0], SlotKind::Literal('#'));
        assert_eq!(mask.input_slot_count(), 2);
    }

    #[test]
    fn partial_validity() {
        let mask = MaskPattern::parse("###-AAA").unwrap();
        let mut input = MaskedInput::new(mask);
        input.insert('1').unwrap();
        input.insert('2').unwrap();
        assert!(input.is_valid_partial());
        assert!(!input.is_complete());
    }

    #[test]
    fn empty_mask_error() {
        assert!(MaskPattern::parse("").is_err());
    }

    #[test]
    fn serialization_roundtrip() {
        let mask = MaskPattern::parse("##-##").unwrap();
        let json = serde_json::to_string(&mask).unwrap();
        let restored: MaskPattern = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.pattern, "##-##");
        assert_eq!(restored.slots.len(), mask.slots.len());
    }
}
