//! Unicode Bidirectional Algorithm (UAX #9).
//!
//! Resolves paragraph embedding levels and reorders mixed LTR/RTL text
//! for correct visual display — pure Rust, no ICU dependency.

// ── Bidi Class ────────────────────────────────────────────────────

/// Unicode bidirectional character classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BidiClass {
    /// Strong left-to-right.
    L,
    /// Strong right-to-left.
    R,
    /// Arabic letter (strong RTL).
    AL,
    /// European number.
    EN,
    /// European separator.
    ES,
    /// European terminator.
    ET,
    /// Arabic number.
    AN,
    /// Common separator.
    CS,
    /// Non-spacing mark.
    NSM,
    /// Boundary neutral.
    BN,
    /// Paragraph separator.
    B,
    /// Segment separator.
    S,
    /// Whitespace.
    WS,
    /// Other neutral.
    ON,
    // Explicit embeddings
    /// Left-to-right embedding.
    LRE,
    /// Left-to-right override.
    LRO,
    /// Right-to-left embedding.
    RLE,
    /// Right-to-left override.
    RLO,
    /// Pop directional format.
    PDF,
    // Isolates (UAX #9 rev 39+)
    /// Left-to-right isolate.
    LRI,
    /// Right-to-left isolate.
    RLI,
    /// First-strong isolate.
    FSI,
    /// Pop directional isolate.
    PDI,
}

// ── Character classification ──────────────────────────────────────

/// Classify a character into its bidi class (simplified).
pub fn classify(c: char) -> BidiClass {
    match c {
        // Explicit formatting characters
        '\u{202A}' => BidiClass::LRE,
        '\u{202D}' => BidiClass::LRO,
        '\u{202B}' => BidiClass::RLE,
        '\u{202E}' => BidiClass::RLO,
        '\u{202C}' => BidiClass::PDF,
        '\u{2066}' => BidiClass::LRI,
        '\u{2067}' => BidiClass::RLI,
        '\u{2068}' => BidiClass::FSI,
        '\u{2069}' => BidiClass::PDI,

        // Paragraph/segment separators
        '\n' | '\r' | '\u{0085}' | '\u{2029}' => BidiClass::B,
        '\t' | '\u{001F}' => BidiClass::S,

        // Whitespace
        ' ' | '\u{00A0}' | '\u{2000}'..='\u{200A}' | '\u{3000}' => BidiClass::WS,

        // BN
        '\u{200B}'..='\u{200D}' | '\u{FEFF}' => BidiClass::BN,

        // Arabic block
        '\u{0600}'..='\u{0605}' | '\u{0608}' | '\u{060B}' | '\u{060D}' => BidiClass::AN,
        '\u{0660}'..='\u{0669}' => BidiClass::AN,
        '\u{0621}'..='\u{064A}' | '\u{066E}'..='\u{06D3}' | '\u{06D5}'
        | '\u{06FA}'..='\u{06FF}' => BidiClass::AL,
        '\u{064B}'..='\u{065F}' | '\u{0670}' => BidiClass::NSM,

        // Hebrew block
        '\u{0590}'..='\u{05FF}' => BidiClass::R,

        // European numbers
        '0'..='9' => BidiClass::EN,
        '+' | '-' => BidiClass::ES,
        '#' | '$' | '%' | '\u{00A2}'..='\u{00A5}' => BidiClass::ET,
        ',' | '.' | ':' => BidiClass::CS,

        // Latin, Greek, Cyrillic, CJK — LTR
        'A'..='Z' | 'a'..='z' | '\u{00C0}'..='\u{024F}' => BidiClass::L,
        '\u{0370}'..='\u{03FF}' => BidiClass::L,
        '\u{0400}'..='\u{04FF}' => BidiClass::L,
        '\u{4E00}'..='\u{9FFF}' => BidiClass::L,

        // Default: other neutral
        _ => BidiClass::ON,
    }
}

// ── Paragraph level ───────────────────────────────────────────────

/// Resolve the paragraph embedding level (P2/P3 of UAX #9).
///
/// Returns 0 for LTR paragraphs, 1 for RTL paragraphs.
pub fn resolve_paragraph_level(text: &str) -> u8 {
    for c in text.chars() {
        match classify(c) {
            BidiClass::L => return 0,
            BidiClass::R | BidiClass::AL => return 1,
            _ => continue,
        }
    }
    0 // default LTR
}

// ── Embedding levels ──────────────────────────────────────────────

/// An entry in the directional-status stack used during explicit level resolution.
#[derive(Debug, Clone, Copy)]
struct DirectionalStatus {
    level: u8,
    override_status: Option<BidiClass>,
    isolate: bool,
}

/// Resolve embedding levels for a paragraph.
///
/// Implements a simplified version of rules X1–X8 (explicit embeddings)
/// followed by W1–W7 (weak types) and N1–N2 (neutral types).
pub fn resolve_levels(text: &str) -> Vec<u8> {
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    if len == 0 {
        return vec![];
    }

    let paragraph_level = resolve_paragraph_level(text);
    let mut classes: Vec<BidiClass> = chars.iter().map(|c| classify(*c)).collect();
    let mut levels: Vec<u8> = vec![paragraph_level; len];

    // ── X1–X8: Explicit embeddings ───────────────────────────
    let max_depth: u8 = 125;
    let mut stack: Vec<DirectionalStatus> = vec![DirectionalStatus {
        level: paragraph_level,
        override_status: None,
        isolate: false,
    }];
    let mut overflow_isolate: u32 = 0;
    let mut overflow_embedding: u32 = 0;
    let mut valid_isolate: u32 = 0;

    for i in 0..len {
        let current_level = stack.last().unwrap().level;

        match classes[i] {
            BidiClass::RLE | BidiClass::RLO | BidiClass::LRE | BidiClass::LRO => {
                let is_rtl =
                    classes[i] == BidiClass::RLE || classes[i] == BidiClass::RLO;
                let is_override =
                    classes[i] == BidiClass::RLO || classes[i] == BidiClass::LRO;
                let new_level = if is_rtl {
                    (current_level + 1) | 1 // next odd
                } else {
                    (current_level + 2) & !1 // next even
                };
                if new_level <= max_depth && overflow_isolate == 0 && overflow_embedding == 0
                {
                    let override_status = if is_override {
                        Some(if is_rtl { BidiClass::R } else { BidiClass::L })
                    } else {
                        None
                    };
                    stack.push(DirectionalStatus {
                        level: new_level,
                        override_status,
                        isolate: false,
                    });
                } else if overflow_isolate == 0 {
                    overflow_embedding += 1;
                }
                levels[i] = current_level;
                classes[i] = BidiClass::BN;
            }
            BidiClass::RLI | BidiClass::LRI | BidiClass::FSI => {
                levels[i] = current_level;
                let is_rtl = classes[i] == BidiClass::RLI;
                let new_level = if is_rtl {
                    (current_level + 1) | 1
                } else {
                    (current_level + 2) & !1
                };
                if new_level <= max_depth && overflow_isolate == 0 && overflow_embedding == 0
                {
                    valid_isolate += 1;
                    stack.push(DirectionalStatus {
                        level: new_level,
                        override_status: None,
                        isolate: true,
                    });
                } else {
                    overflow_isolate += 1;
                }
            }
            BidiClass::PDI => {
                if overflow_isolate > 0 {
                    overflow_isolate -= 1;
                } else if valid_isolate > 0 {
                    overflow_embedding = 0;
                    while let Some(entry) = stack.pop() {
                        if entry.isolate {
                            break;
                        }
                    }
                    valid_isolate -= 1;
                }
                levels[i] = stack.last().unwrap().level;
            }
            BidiClass::PDF => {
                if overflow_isolate == 0 {
                    if overflow_embedding > 0 {
                        overflow_embedding -= 1;
                    } else if stack.len() >= 2 && !stack.last().unwrap().isolate {
                        stack.pop();
                    }
                }
                levels[i] = stack.last().unwrap().level;
                classes[i] = BidiClass::BN;
            }
            BidiClass::B => {
                levels[i] = paragraph_level;
            }
            _ => {
                levels[i] = stack.last().unwrap().level;
                if let Some(ovr) = stack.last().unwrap().override_status {
                    classes[i] = ovr;
                }
            }
        }
    }

    // ── W1–W7: Weak type resolution ─────────────────────────
    resolve_weak_types(&mut classes, &levels, paragraph_level);

    // ── N1–N2: Neutral type resolution ──────────────────────
    resolve_neutral_types(&mut classes, &levels, paragraph_level);

    // ── I1–I2: Implicit level adjustment ────────────────────
    for i in 0..len {
        match classes[i] {
            BidiClass::R => {
                if levels[i] % 2 == 0 {
                    levels[i] += 1;
                }
            }
            BidiClass::AN | BidiClass::EN => {
                if levels[i] % 2 == 0 {
                    levels[i] += 2;
                } else {
                    levels[i] += 1;
                }
            }
            BidiClass::L => {
                if levels[i] % 2 == 1 {
                    levels[i] += 1;
                }
            }
            _ => {}
        }
    }

    levels
}

fn resolve_weak_types(classes: &mut [BidiClass], levels: &[u8], _para_level: u8) {
    let len = classes.len();
    if len == 0 {
        return;
    }

    // W1: NSM
    let mut prev = BidiClass::ON;
    for i in 0..len {
        if classes[i] == BidiClass::NSM {
            classes[i] = prev;
        }
        prev = classes[i];
    }

    // W2: EN after AL → AN
    let mut last_strong = BidiClass::ON;
    for i in 0..len {
        match classes[i] {
            BidiClass::L | BidiClass::R | BidiClass::AL => last_strong = classes[i],
            BidiClass::EN => {
                if last_strong == BidiClass::AL {
                    classes[i] = BidiClass::AN;
                }
            }
            _ => {}
        }
    }

    // W3: AL → R
    for cls in classes.iter_mut() {
        if *cls == BidiClass::AL {
            *cls = BidiClass::R;
        }
    }

    // W4: EN ES EN → EN EN EN; EN CS EN → EN EN EN; AN CS AN → AN AN AN
    for i in 1..len.saturating_sub(1) {
        if classes[i] == BidiClass::ES
            && classes[i - 1] == BidiClass::EN
            && classes[i + 1] == BidiClass::EN
            && levels[i - 1] == levels[i]
            && levels[i] == levels[i + 1]
        {
            classes[i] = BidiClass::EN;
        }
        if classes[i] == BidiClass::CS && levels[i - 1] == levels[i] && levels[i] == levels[i + 1]
        {
            if classes[i - 1] == BidiClass::EN && classes[i + 1] == BidiClass::EN {
                classes[i] = BidiClass::EN;
            } else if classes[i - 1] == BidiClass::AN && classes[i + 1] == BidiClass::AN {
                classes[i] = BidiClass::AN;
            }
        }
    }

    // W5: ET adjacent to EN → EN
    for i in 0..len {
        if classes[i] == BidiClass::ET {
            let mut found_en = false;
            // look left
            if i > 0 && classes[i - 1] == BidiClass::EN && levels[i - 1] == levels[i] {
                found_en = true;
            }
            // look right
            if i + 1 < len && classes[i + 1] == BidiClass::EN && levels[i] == levels[i + 1] {
                found_en = true;
            }
            if found_en {
                classes[i] = BidiClass::EN;
            }
        }
    }

    // W6: ES, ET, CS → ON
    for cls in classes.iter_mut() {
        if matches!(*cls, BidiClass::ES | BidiClass::ET | BidiClass::CS) {
            *cls = BidiClass::ON;
        }
    }

    // W7: EN with context L → L
    let mut last_strong = BidiClass::L; // sos type defaults
    for i in 0..len {
        match classes[i] {
            BidiClass::L | BidiClass::R => last_strong = classes[i],
            BidiClass::EN => {
                if last_strong == BidiClass::L {
                    classes[i] = BidiClass::L;
                }
            }
            _ => {}
        }
    }
}

fn resolve_neutral_types(classes: &mut [BidiClass], levels: &[u8], para_level: u8) {
    let len = classes.len();
    if len == 0 {
        return;
    }

    // N1/N2: Neutrals take the direction of the surrounding strong types.
    let mut i = 0;
    while i < len {
        if is_neutral(classes[i]) {
            let start = i;
            while i < len && is_neutral(classes[i]) {
                i += 1;
            }
            let end = i;

            // Find preceding strong type
            let prev_strong = if start > 0 {
                embedding_direction(classes[start - 1], levels[start - 1])
            } else {
                if para_level % 2 == 0 {
                    BidiClass::L
                } else {
                    BidiClass::R
                }
            };

            // Find following strong type
            let next_strong = if end < len {
                embedding_direction(classes[end], levels[end])
            } else {
                if para_level % 2 == 0 {
                    BidiClass::L
                } else {
                    BidiClass::R
                }
            };

            let resolved = if prev_strong == next_strong {
                prev_strong // N1
            } else {
                // N2: use embedding direction
                if levels[start] % 2 == 0 {
                    BidiClass::L
                } else {
                    BidiClass::R
                }
            };

            for cls in &mut classes[start..end] {
                *cls = resolved;
            }
        } else {
            i += 1;
        }
    }
}

fn is_neutral(cls: BidiClass) -> bool {
    matches!(
        cls,
        BidiClass::ON | BidiClass::WS | BidiClass::S | BidiClass::B | BidiClass::BN
    )
}

fn embedding_direction(cls: BidiClass, level: u8) -> BidiClass {
    match cls {
        BidiClass::L => BidiClass::L,
        BidiClass::R | BidiClass::EN | BidiClass::AN => BidiClass::R,
        _ => {
            if level % 2 == 0 {
                BidiClass::L
            } else {
                BidiClass::R
            }
        }
    }
}

// ── Reordering ────────────────────────────────────────────────────

/// Reorder characters for visual display based on resolved embedding levels.
///
/// Returns indices into the original string in visual order.
pub fn reorder(levels: &[u8]) -> Vec<usize> {
    let len = levels.len();
    if len == 0 {
        return vec![];
    }

    let mut indices: Vec<usize> = (0..len).collect();
    let max_level = *levels.iter().max().unwrap();
    let min_odd_level = *levels
        .iter()
        .filter(|&&l| l % 2 == 1)
        .min()
        .unwrap_or(&max_level);

    // L2: Reverse runs at each level from max_level down to min_odd_level.
    let mut level = max_level;
    while level >= min_odd_level && level > 0 {
        let mut i = 0;
        while i < len {
            if levels[indices[i]] >= level {
                let start = i;
                while i < len && levels[indices[i]] >= level {
                    i += 1;
                }
                indices[start..i].reverse();
            } else {
                i += 1;
            }
        }
        level -= 1;
    }

    indices
}

/// Convenience: reorder a string for visual display.
pub fn reorder_text(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let levels = resolve_levels(text);
    let order = reorder(&levels);
    order.iter().map(|i| chars[*i]).collect()
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_latin() {
        assert_eq!(classify('A'), BidiClass::L);
        assert_eq!(classify('z'), BidiClass::L);
    }

    #[test]
    fn classify_arabic() {
        assert_eq!(classify('\u{0628}'), BidiClass::AL); // Baa
    }

    #[test]
    fn classify_hebrew() {
        assert_eq!(classify('\u{05D0}'), BidiClass::R); // Alef
    }

    #[test]
    fn classify_digits() {
        assert_eq!(classify('0'), BidiClass::EN);
        assert_eq!(classify('9'), BidiClass::EN);
    }

    #[test]
    fn paragraph_level_ltr() {
        assert_eq!(resolve_paragraph_level("Hello world"), 0);
    }

    #[test]
    fn paragraph_level_rtl() {
        assert_eq!(resolve_paragraph_level("\u{05D0}\u{05D1}\u{05D2}"), 1);
    }

    #[test]
    fn paragraph_level_default() {
        assert_eq!(resolve_paragraph_level("  123  "), 0);
    }

    #[test]
    fn levels_pure_ltr() {
        let levels = resolve_levels("Hello");
        assert_eq!(levels, vec![0, 0, 0, 0, 0]);
    }

    #[test]
    fn levels_pure_rtl() {
        let levels = resolve_levels("\u{05D0}\u{05D1}\u{05D2}");
        // Hebrew chars at para level 1 → level 1
        for l in &levels {
            assert_eq!(*l, 1);
        }
    }

    #[test]
    fn reorder_pure_ltr() {
        let order = reorder(&[0, 0, 0]);
        assert_eq!(order, vec![0, 1, 2]);
    }

    #[test]
    fn reorder_pure_rtl() {
        let order = reorder(&[1, 1, 1]);
        assert_eq!(order, vec![2, 1, 0]);
    }

    #[test]
    fn reorder_mixed() {
        // Simulating: LTR(0) RTL(1) RTL(1) LTR(0)
        let order = reorder(&[0, 1, 1, 0]);
        assert_eq!(order, vec![0, 2, 1, 3]);
    }

    #[test]
    fn reorder_text_pure_ltr() {
        assert_eq!(reorder_text("Hello"), "Hello");
    }

    #[test]
    fn empty_text() {
        assert_eq!(resolve_levels(""), Vec::<u8>::new());
        assert_eq!(reorder(&[]), Vec::<usize>::new());
    }

    #[test]
    fn classify_whitespace() {
        assert_eq!(classify(' '), BidiClass::WS);
        assert_eq!(classify('\n'), BidiClass::B);
    }
}
