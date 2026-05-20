//! Input masking: format user input according to pattern rules.
//!
//! Replaces react-input-mask / imask.js. Pattern characters:
//! `9` = digit, `A` = letter, `a` = letter or digit, `*` = any,
//! all other characters are literals that pass through.

// ── MaskPattern ────────────────────────────────────────────────

/// A mask pattern that describes the expected input format.
#[derive(Debug, Clone)]
pub struct MaskPattern {
    pub pattern: String,
}

impl MaskPattern {
    pub fn new(pattern: &str) -> Self {
        Self { pattern: pattern.to_string() }
    }

    /// Format raw input according to the mask, inserting literals.
    pub fn apply(&self, input: &str) -> String {
        let chars: Vec<char> = self.pattern.chars().collect();
        let mut input_iter = input.chars().filter(|c| !c.is_ascii_whitespace() || *c == ' ');
        let mut out = String::with_capacity(chars.len());

        for &pc in &chars {
            match pc {
                '9' | 'A' | 'a' | '*' => {
                    loop {
                        match input_iter.next() {
                            Some(ic) if char_matches_slot(pc, ic) => {
                                out.push(ic);
                                break;
                            }
                            Some(_) => continue,
                            None => return out,
                        }
                    }
                }
                literal => {
                    out.push(literal);
                }
            }
        }
        out
    }

    /// Check whether `input` completely fills every slot in the mask.
    pub fn validate(&self, input: &str) -> bool {
        let formatted = self.apply(input);
        if formatted.len() != self.pattern.len() {
            return false;
        }
        let pchars: Vec<char> = self.pattern.chars().collect();
        let fchars: Vec<char> = formatted.chars().collect();
        for (i, &pc) in pchars.iter().enumerate() {
            match pc {
                '9' | 'A' | 'a' | '*' => {
                    if !char_matches_slot(pc, fchars[i]) {
                        return false;
                    }
                }
                literal => {
                    if fchars[i] != literal {
                        return false;
                    }
                }
            }
        }
        true
    }

    // ── Common masks ──────────────────────────────────────────

    pub fn phone_us() -> Self { Self::new("(999) 999-9999") }
    pub fn date() -> Self { Self::new("99/99/9999") }
    pub fn time() -> Self { Self::new("99:99") }
    pub fn ssn() -> Self { Self::new("999-99-9999") }
    pub fn zip_code() -> Self { Self::new("99999") }
    pub fn credit_card() -> Self { Self::new("9999 9999 9999 9999") }
    pub fn ip_address() -> Self { Self::new("999.999.999.999") }
}

fn char_matches_slot(slot: char, c: char) -> bool {
    match slot {
        '9' => c.is_ascii_digit(),
        'A' => c.is_ascii_alphabetic(),
        'a' => c.is_ascii_alphanumeric(),
        '*' => true,
        _ => false,
    }
}

fn is_literal(pc: char) -> bool {
    !matches!(pc, '9' | 'A' | 'a' | '*')
}

// ── MaskState ──────────────────────────────────────────────────

/// Stateful mask that tracks cursor position for interactive editing.
#[derive(Debug, Clone)]
pub struct MaskState {
    pub pattern: MaskPattern,
    pub value: String,
    pub cursor: usize,
}

impl MaskState {
    pub fn new(pattern: MaskPattern) -> Self {
        Self { pattern, value: String::new(), cursor: 0 }
    }

    /// Insert a character at the current cursor, respecting mask rules.
    /// Returns `true` if the character was accepted.
    pub fn insert_char(&mut self, c: char) -> bool {
        let pchars: Vec<char> = self.pattern.pattern.chars().collect();
        if self.cursor >= pchars.len() {
            return false;
        }

        // Skip over literal positions — auto-insert them.
        while self.cursor < pchars.len() && is_literal(pchars[self.cursor]) {
            self.value.push(pchars[self.cursor]);
            self.cursor += 1;
        }

        if self.cursor >= pchars.len() {
            return false;
        }

        let slot = pchars[self.cursor];
        if char_matches_slot(slot, c) {
            self.value.push(c);
            self.cursor += 1;
            // Auto-insert trailing literals.
            while self.cursor < pchars.len() && is_literal(pchars[self.cursor]) {
                self.value.push(pchars[self.cursor]);
                self.cursor += 1;
            }
            true
        } else {
            false
        }
    }

    /// Delete the last character (and any trailing literals before it).
    pub fn delete_back(&mut self) -> bool {
        if self.value.is_empty() { return false; }
        let pchars: Vec<char> = self.pattern.pattern.chars().collect();

        // Pop trailing literals first.
        while self.cursor > 0 && is_literal(pchars[self.cursor - 1]) {
            self.value.pop();
            self.cursor -= 1;
        }
        if self.cursor > 0 {
            self.value.pop();
            self.cursor -= 1;
            true
        } else {
            false
        }
    }

    /// Return the raw value with mask literal characters stripped.
    pub fn raw_value(&self) -> String {
        let pchars: Vec<char> = self.pattern.pattern.chars().collect();
        let vchars: Vec<char> = self.value.chars().collect();
        let mut raw = String::new();
        for (i, &vc) in vchars.iter().enumerate() {
            if i < pchars.len() && !is_literal(pchars[i]) {
                raw.push(vc);
            }
        }
        raw
    }

    /// Return the current formatted value.
    pub fn formatted_value(&self) -> String {
        self.value.clone()
    }

    /// Whether all mask slots have been filled.
    pub fn is_complete(&self) -> bool {
        self.cursor >= self.pattern.pattern.len()
    }

    /// Current cursor position in the formatted string.
    pub fn cursor_position(&self) -> usize {
        self.cursor
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phone_mask_formats_digits() {
        let mask = MaskPattern::phone_us();
        let out = mask.apply("5551234567");
        assert_eq!(out, "(555) 123-4567");
    }

    #[test]
    fn date_mask_formats() {
        let mask = MaskPattern::date();
        let out = mask.apply("12252025");
        assert_eq!(out, "12/25/2025");
    }

    #[test]
    fn validate_complete() {
        let mask = MaskPattern::phone_us();
        assert!(mask.validate("5551234567"));
    }

    #[test]
    fn validate_incomplete() {
        let mask = MaskPattern::phone_us();
        assert!(!mask.validate("555123"));
    }

    #[test]
    fn raw_value_strips_literals() {
        let mut state = MaskState::new(MaskPattern::phone_us());
        for c in "5551234567".chars() {
            state.insert_char(c);
        }
        assert_eq!(state.raw_value(), "5551234567");
        assert_eq!(state.formatted_value(), "(555) 123-4567");
    }

    #[test]
    fn credit_card_spacing() {
        let mask = MaskPattern::credit_card();
        let out = mask.apply("4111111111111111");
        assert_eq!(out, "4111 1111 1111 1111");
    }

    #[test]
    fn insert_non_matching_rejected() {
        let mut state = MaskState::new(MaskPattern::zip_code());
        assert!(state.insert_char('1'));
        assert!(!state.insert_char('x')); // only digits
        assert_eq!(state.raw_value(), "1");
    }

    #[test]
    fn delete_back() {
        let mut state = MaskState::new(MaskPattern::phone_us());
        for c in "555".chars() {
            state.insert_char(c);
        }
        assert!(state.delete_back());
        assert!(state.formatted_value().len() < "(555) ".len());
    }

    #[test]
    fn cursor_position_advances() {
        let mut state = MaskState::new(MaskPattern::time());
        assert_eq!(state.cursor_position(), 0);
        state.insert_char('1');
        state.insert_char('2');
        // After "12" the ':' literal is auto-inserted, cursor at 3
        assert_eq!(state.cursor_position(), 3);
        state.insert_char('3');
        state.insert_char('0');
        assert!(state.is_complete());
        assert_eq!(state.formatted_value(), "12:30");
    }

    #[test]
    fn ssn_mask() {
        let mask = MaskPattern::ssn();
        let out = mask.apply("123456789");
        assert_eq!(out, "123-45-6789");
    }

    #[test]
    fn ip_address_mask() {
        let mask = MaskPattern::ip_address();
        let out = mask.apply("192168001001");
        assert_eq!(out, "192.168.001.001");
    }

    #[test]
    fn letter_mask_slot() {
        let mask = MaskPattern::new("AA-9999");
        let out = mask.apply("CA1234");
        assert_eq!(out, "CA-1234");
    }
}
