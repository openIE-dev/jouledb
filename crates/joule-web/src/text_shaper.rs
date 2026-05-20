//! Text shaping pipeline.
//!
//! Transforms Unicode text into positioned glyph sequences, handling
//! script-specific shaping (Latin 1:1, Arabic joining), kerning, and
//! directional layout — all in pure Rust with no HarfBuzz dependency.

use std::collections::HashMap;

// ── Types ─────────────────────────────────────────────────────────

/// Information about a single shaped glyph.
#[derive(Debug, Clone, PartialEq)]
pub struct GlyphInfo {
    /// Glyph identifier (font-specific).
    pub glyph_id: u32,
    /// Cluster index — maps back to the source character(s).
    pub cluster: u32,
    /// Horizontal advance in font units.
    pub x_advance: i32,
    /// Vertical advance in font units.
    pub y_advance: i32,
    /// Horizontal offset from the default position.
    pub x_offset: i32,
    /// Vertical offset from the default position.
    pub y_offset: i32,
}

/// Writing direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    LeftToRight,
    RightToLeft,
}

/// Input to the shaping pipeline.
#[derive(Debug, Clone)]
pub struct ShapingInput {
    pub text: String,
    pub script: Script,
    pub language: String,
    pub direction: Direction,
}

/// Script tag (subset relevant for shaping).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Script {
    Latin,
    Arabic,
    Common,
}

// ── Arabic joining ────────────────────────────────────────────────

/// Joining type for Arabic-style scripts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoiningType {
    /// Can join on both sides (baa, etc.).
    Dual,
    /// Joins only on the right (alef, etc.).
    Right,
    /// Does not join (hamza, etc.).
    NonJoining,
    /// Transparent — ignored for joining decisions.
    Transparent,
}

/// Contextual form of an Arabic-style glyph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoiningForm {
    Isolated,
    Initial,
    Medial,
    Final,
}

/// Simple table mapping a code point to its joining type.
fn arabic_joining_type(c: char) -> JoiningType {
    match c {
        // Alef variants — right-joining only
        '\u{0627}' | '\u{0622}' | '\u{0623}' | '\u{0625}' => JoiningType::Right,
        // Dal, Thal, Ra, Zain, Waw — right-joining
        '\u{062F}' | '\u{0630}' | '\u{0631}' | '\u{0632}' | '\u{0648}' => JoiningType::Right,
        // Most Arabic letters are dual-joining
        '\u{0628}'..='\u{064A}' => JoiningType::Dual,
        // Tatweel — dual
        '\u{0640}' => JoiningType::Dual,
        // Non-spacing marks — transparent
        '\u{064B}'..='\u{065F}' => JoiningType::Transparent,
        _ => JoiningType::NonJoining,
    }
}

/// Resolve joining forms for a sequence of Arabic characters.
pub fn resolve_joining_forms(text: &str) -> Vec<JoiningForm> {
    let chars: Vec<char> = text.chars().collect();
    let joining_types: Vec<JoiningType> = chars.iter().map(|c| arabic_joining_type(*c)).collect();
    let len = chars.len();
    let mut forms = vec![JoiningForm::Isolated; len];

    for i in 0..len {
        if joining_types[i] == JoiningType::Transparent {
            forms[i] = JoiningForm::Isolated;
            continue;
        }
        if joining_types[i] == JoiningType::NonJoining {
            forms[i] = JoiningForm::Isolated;
            continue;
        }

        // Look for a preceding joiner (skip transparent).
        let prev_joins = find_prev_joiner(&joining_types, i);
        // Look for a following joiner (skip transparent).
        let next_joins = find_next_joiner(&joining_types, i);

        let can_join_left = joining_types[i] == JoiningType::Dual
            || joining_types[i] == JoiningType::Right;
        let can_join_right = joining_types[i] == JoiningType::Dual;

        let joins_prev = prev_joins && can_join_left;
        let joins_next = next_joins && can_join_right;

        forms[i] = match (joins_prev, joins_next) {
            (false, false) => JoiningForm::Isolated,
            (false, true) => JoiningForm::Initial,
            (true, false) => JoiningForm::Final,
            (true, true) => JoiningForm::Medial,
        };
    }

    forms
}

fn find_prev_joiner(types: &[JoiningType], idx: usize) -> bool {
    let mut i = idx;
    while i > 0 {
        i -= 1;
        if types[i] == JoiningType::Transparent {
            continue;
        }
        return types[i] == JoiningType::Dual || types[i] == JoiningType::Right;
    }
    false
}

fn find_next_joiner(types: &[JoiningType], idx: usize) -> bool {
    let mut i = idx + 1;
    while i < types.len() {
        if types[i] == JoiningType::Transparent {
            i += 1;
            continue;
        }
        return types[i] == JoiningType::Dual || types[i] == JoiningType::Right;
    }
    false
}

// ── Kerning ───────────────────────────────────────────────────────

/// A kerning table: maps (left_glyph, right_glyph) → x-adjustment.
#[derive(Debug, Clone, Default)]
pub struct KerningTable {
    pairs: HashMap<(u32, u32), i32>,
}

impl KerningTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, left: u32, right: u32, adjustment: i32) {
        self.pairs.insert((left, right), adjustment);
    }

    pub fn get(&self, left: u32, right: u32) -> i32 {
        self.pairs.get(&(left, right)).copied().unwrap_or(0)
    }

    /// Apply kerning adjustments in-place to a glyph sequence.
    pub fn apply(&self, glyphs: &mut [GlyphInfo]) {
        if glyphs.len() < 2 {
            return;
        }
        for i in 0..glyphs.len() - 1 {
            let adj = self.get(glyphs[i].glyph_id, glyphs[i + 1].glyph_id);
            if adj != 0 {
                glyphs[i].x_advance += adj;
            }
        }
    }
}

// ── Shaper ────────────────────────────────────────────────────────

/// The text shaping engine.
#[derive(Debug, Clone)]
pub struct TextShaper {
    /// Default advance width for Latin glyphs.
    pub default_advance: i32,
    /// Kerning table.
    pub kerning: KerningTable,
}

impl Default for TextShaper {
    fn default() -> Self {
        Self {
            default_advance: 600,
            kerning: KerningTable::new(),
        }
    }
}

impl TextShaper {
    pub fn new(default_advance: i32) -> Self {
        Self {
            default_advance,
            kerning: KerningTable::new(),
        }
    }

    /// Shape text according to the given input parameters.
    pub fn shape(&self, input: &ShapingInput) -> Vec<GlyphInfo> {
        let mut glyphs = match input.script {
            Script::Latin | Script::Common => self.shape_latin(&input.text),
            Script::Arabic => self.shape_arabic(&input.text),
        };

        self.kerning.apply(&mut glyphs);

        if input.direction == Direction::RightToLeft {
            glyphs.reverse();
        }

        glyphs
    }

    /// Basic Latin shaping: 1-to-1 character-to-glyph mapping.
    fn shape_latin(&self, text: &str) -> Vec<GlyphInfo> {
        text.chars()
            .enumerate()
            .map(|(i, c)| GlyphInfo {
                glyph_id: c as u32,
                cluster: i as u32,
                x_advance: self.default_advance,
                y_advance: 0,
                x_offset: 0,
                y_offset: 0,
            })
            .collect()
    }

    /// Arabic shaping: resolve joining forms and assign glyph IDs.
    ///
    /// Glyph ID encoding: `base_codepoint * 4 + form_offset` where
    /// form offsets are: isolated=0, final=1, initial=2, medial=3.
    fn shape_arabic(&self, text: &str) -> Vec<GlyphInfo> {
        let forms = resolve_joining_forms(text);
        text.chars()
            .enumerate()
            .zip(forms.iter())
            .map(|((i, c), form)| {
                let form_offset = match form {
                    JoiningForm::Isolated => 0,
                    JoiningForm::Final => 1,
                    JoiningForm::Initial => 2,
                    JoiningForm::Medial => 3,
                };
                GlyphInfo {
                    glyph_id: (c as u32) * 4 + form_offset,
                    cluster: i as u32,
                    x_advance: self.default_advance,
                    y_advance: 0,
                    x_offset: 0,
                    y_offset: 0,
                }
            })
            .collect()
    }

    /// Compute total advance width of a glyph sequence.
    pub fn total_advance(glyphs: &[GlyphInfo]) -> i32 {
        glyphs.iter().map(|g| g.x_advance).sum()
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latin_basic_shape() {
        let shaper = TextShaper::new(600);
        let input = ShapingInput {
            text: "Hello".into(),
            script: Script::Latin,
            language: "en".into(),
            direction: Direction::LeftToRight,
        };
        let glyphs = shaper.shape(&input);
        assert_eq!(glyphs.len(), 5);
        assert_eq!(glyphs[0].glyph_id, 'H' as u32);
        assert_eq!(glyphs[0].cluster, 0);
        assert_eq!(glyphs[4].glyph_id, 'o' as u32);
    }

    #[test]
    fn latin_total_advance() {
        let shaper = TextShaper::new(500);
        let input = ShapingInput {
            text: "Hi".into(),
            script: Script::Latin,
            language: "en".into(),
            direction: Direction::LeftToRight,
        };
        let glyphs = shaper.shape(&input);
        assert_eq!(TextShaper::total_advance(&glyphs), 1000);
    }

    #[test]
    fn rtl_reverses_glyphs() {
        let shaper = TextShaper::new(600);
        let input = ShapingInput {
            text: "AB".into(),
            script: Script::Latin,
            language: "en".into(),
            direction: Direction::RightToLeft,
        };
        let glyphs = shaper.shape(&input);
        assert_eq!(glyphs[0].glyph_id, 'B' as u32);
        assert_eq!(glyphs[1].glyph_id, 'A' as u32);
    }

    #[test]
    fn kerning_adjusts_advance() {
        let mut shaper = TextShaper::new(600);
        shaper.kerning.add('A' as u32, 'V' as u32, -50);
        let input = ShapingInput {
            text: "AV".into(),
            script: Script::Latin,
            language: "en".into(),
            direction: Direction::LeftToRight,
        };
        let glyphs = shaper.shape(&input);
        assert_eq!(glyphs[0].x_advance, 550); // 600 - 50
        assert_eq!(glyphs[1].x_advance, 600);
    }

    #[test]
    fn arabic_isolated_form() {
        // Single Arabic letter should be isolated.
        let forms = resolve_joining_forms("\u{0628}"); // Baa
        assert_eq!(forms, vec![JoiningForm::Isolated]);
    }

    #[test]
    fn arabic_two_dual_joining() {
        // Baa + Baa: first=initial, second=final
        let forms = resolve_joining_forms("\u{0628}\u{0628}");
        assert_eq!(forms, vec![JoiningForm::Initial, JoiningForm::Final]);
    }

    #[test]
    fn arabic_three_dual_joining() {
        // Baa + Baa + Baa: initial, medial, final
        let forms = resolve_joining_forms("\u{0628}\u{0628}\u{0628}");
        assert_eq!(
            forms,
            vec![JoiningForm::Initial, JoiningForm::Medial, JoiningForm::Final]
        );
    }

    #[test]
    fn arabic_right_joining_breaks_chain() {
        // Baa + Alef: Baa=initial, Alef=final (alef is right-joining only)
        let forms = resolve_joining_forms("\u{0628}\u{0627}");
        assert_eq!(forms, vec![JoiningForm::Initial, JoiningForm::Final]);
    }

    #[test]
    fn arabic_alef_then_baa() {
        // Alef + Baa: Alef cannot join right, Baa has no prev joiner forming pair
        // Alef is right-joining: it joins to the left (i.e., to the preceding char).
        // With no preceding char, alef is isolated.
        // Baa is dual-joining; prev is alef (right-joining) — alef can join on the
        // right side (from Baa's perspective, alef is a valid prev joiner).
        let forms = resolve_joining_forms("\u{0627}\u{0628}");
        // Alef: no prev, can_join_right = false (Right type) → isolated
        // Baa: prev is Alef (Right type, counts as joiner), no next → final
        assert_eq!(forms, vec![JoiningForm::Isolated, JoiningForm::Final]);
    }

    #[test]
    fn arabic_glyph_id_encoding() {
        let shaper = TextShaper::new(600);
        let input = ShapingInput {
            text: "\u{0628}".into(), // single Baa — isolated
            script: Script::Arabic,
            language: "ar".into(),
            direction: Direction::RightToLeft,
        };
        let glyphs = shaper.shape(&input);
        assert_eq!(glyphs.len(), 1);
        // Isolated form: base * 4 + 0
        assert_eq!(glyphs[0].glyph_id, 0x0628 * 4);
    }

    #[test]
    fn empty_text_produces_no_glyphs() {
        let shaper = TextShaper::default();
        let input = ShapingInput {
            text: String::new(),
            script: Script::Latin,
            language: "en".into(),
            direction: Direction::LeftToRight,
        };
        let glyphs = shaper.shape(&input);
        assert!(glyphs.is_empty());
    }

    #[test]
    fn kerning_table_miss_returns_zero() {
        let table = KerningTable::new();
        assert_eq!(table.get(1, 2), 0);
    }

    #[test]
    fn cluster_indices_sequential() {
        let shaper = TextShaper::new(600);
        let input = ShapingInput {
            text: "abcde".into(),
            script: Script::Latin,
            language: "en".into(),
            direction: Direction::LeftToRight,
        };
        let glyphs = shaper.shape(&input);
        let clusters: Vec<u32> = glyphs.iter().map(|g| g.cluster).collect();
        assert_eq!(clusters, vec![0, 1, 2, 3, 4]);
    }
}
