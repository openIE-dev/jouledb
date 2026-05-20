//! Flyweight pattern — shared intrinsic state with per-instance extrinsic state.
//!
//! Provides a `FlyweightFactory` that deduplicates intrinsic state objects,
//! `FlyweightRef` handles that pair intrinsic with extrinsic data, memory
//! savings tracking, and a character-rendering example.

use std::collections::HashMap;
use std::fmt;

// ── Intrinsic state ────────────────────────────────────────────────

/// Shared, immutable intrinsic state stored once in the pool.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IntrinsicState {
    /// A key that identifies this flyweight.
    pub key: String,
    /// The shared data (e.g. font family, glyph bitmap, texture, ...).
    pub data: Vec<u8>,
    /// Arbitrary properties.
    pub properties: Vec<(String, String)>,
}

impl IntrinsicState {
    pub fn new(key: impl Into<String>, data: Vec<u8>) -> Self {
        Self {
            key: key.into(),
            data,
            properties: Vec::new(),
        }
    }

    pub fn with_property(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.properties.push((name.into(), value.into()));
        self
    }

    /// Estimated memory footprint in bytes.
    pub fn size_bytes(&self) -> usize {
        self.key.len()
            + self.data.len()
            + self
                .properties
                .iter()
                .map(|(k, v)| k.len() + v.len())
                .sum::<usize>()
    }
}

// ── Extrinsic state ────────────────────────────────────────────────

/// Per-instance data that varies between usages of the same flyweight.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtrinsicState {
    pub x: f64,
    pub y: f64,
    pub scale: f64,
    pub extra: HashMap<String, String>,
}

impl ExtrinsicState {
    pub fn new(x: f64, y: f64) -> Self {
        Self {
            x,
            y,
            scale: 1.0,
            extra: HashMap::new(),
        }
    }

    pub fn with_scale(mut self, scale: f64) -> Self {
        self.scale = scale;
        self
    }

    pub fn with_extra(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra.insert(key.into(), value.into());
        self
    }
}

// ── Flyweight reference ────────────────────────────────────────────

/// A reference to a pooled flyweight combined with extrinsic state.
#[derive(Debug, Clone)]
pub struct FlyweightRef {
    /// Index into the factory's pool.
    pool_index: usize,
    /// Per-instance extrinsic state.
    pub extrinsic: ExtrinsicState,
}

impl FlyweightRef {
    /// Return the pool index (for factory lookups).
    pub fn pool_index(&self) -> usize {
        self.pool_index
    }
}

// ── Factory / Pool ─────────────────────────────────────────────────

/// Pool statistics.
#[derive(Debug, Clone)]
pub struct PoolStats {
    /// Number of unique intrinsic objects in the pool.
    pub unique_count: usize,
    /// Total number of flyweight references created.
    pub total_refs: u64,
    /// Total bytes of intrinsic data stored (shared).
    pub shared_bytes: usize,
    /// Hypothetical bytes if every ref had its own copy.
    pub unshared_bytes: usize,
    /// Bytes saved by sharing.
    pub savings_bytes: usize,
}

/// Flyweight factory that manages the pool of intrinsic states.
pub struct FlyweightFactory {
    /// Intrinsic state pool (append-only).
    pool: Vec<IntrinsicState>,
    /// Map from key to pool index for dedup.
    index: HashMap<String, usize>,
    /// Counts how many refs have been created per pool entry.
    ref_counts: Vec<u64>,
    /// Total refs created.
    total_refs: u64,
}

impl FlyweightFactory {
    /// Create an empty factory.
    pub fn new() -> Self {
        Self {
            pool: Vec::new(),
            index: HashMap::new(),
            ref_counts: Vec::new(),
            total_refs: 0,
        }
    }

    /// Get or create a flyweight for the given intrinsic state, paired with
    /// the provided extrinsic state.
    pub fn get_flyweight(
        &mut self,
        intrinsic: IntrinsicState,
        extrinsic: ExtrinsicState,
    ) -> FlyweightRef {
        let key = intrinsic.key.clone();
        let pool_index = if let Some(&idx) = self.index.get(&key) {
            idx
        } else {
            let idx = self.pool.len();
            self.pool.push(intrinsic);
            self.index.insert(key, idx);
            self.ref_counts.push(0);
            idx
        };

        self.ref_counts[pool_index] += 1;
        self.total_refs += 1;

        FlyweightRef {
            pool_index,
            extrinsic,
        }
    }

    /// Look up the intrinsic state for a given flyweight ref.
    pub fn intrinsic(&self, fw: &FlyweightRef) -> Option<&IntrinsicState> {
        self.pool.get(fw.pool_index)
    }

    /// Look up intrinsic state by key.
    pub fn get_by_key(&self, key: &str) -> Option<&IntrinsicState> {
        self.index.get(key).and_then(|idx| self.pool.get(*idx))
    }

    /// Number of unique intrinsic objects in the pool.
    pub fn unique_count(&self) -> usize {
        self.pool.len()
    }

    /// Total flyweight references created.
    pub fn total_refs(&self) -> u64 {
        self.total_refs
    }

    /// Reference count for a given key.
    pub fn ref_count(&self, key: &str) -> u64 {
        self.index
            .get(key)
            .and_then(|idx| self.ref_counts.get(*idx))
            .copied()
            .unwrap_or(0)
    }

    /// Whether a key is already in the pool.
    pub fn contains_key(&self, key: &str) -> bool {
        self.index.contains_key(key)
    }

    /// All keys in the pool (sorted for determinism).
    pub fn keys(&self) -> Vec<String> {
        let mut keys: Vec<String> = self.index.keys().cloned().collect();
        keys.sort();
        keys
    }

    /// Compute pool statistics.
    pub fn stats(&self) -> PoolStats {
        let shared_bytes: usize = self.pool.iter().map(|s| s.size_bytes()).sum();
        let unshared_bytes: usize = self
            .pool
            .iter()
            .enumerate()
            .map(|(i, s)| s.size_bytes() * (self.ref_counts[i] as usize).max(1))
            .sum();
        let savings = unshared_bytes.saturating_sub(shared_bytes);

        PoolStats {
            unique_count: self.pool.len(),
            total_refs: self.total_refs,
            shared_bytes,
            unshared_bytes,
            savings_bytes: savings,
        }
    }

    /// Memory savings as a fraction (0.0 to 1.0).
    pub fn savings_ratio(&self) -> f64 {
        let stats = self.stats();
        if stats.unshared_bytes == 0 {
            return 0.0;
        }
        stats.savings_bytes as f64 / stats.unshared_bytes as f64
    }
}

impl Default for FlyweightFactory {
    fn default() -> Self {
        Self::new()
    }
}

// ── Character rendering example ────────────────────────────────────

/// A character glyph flyweight for text rendering.
#[derive(Debug, Clone)]
pub struct CharGlyph {
    pub character: char,
    pub font_family: String,
    pub font_size: u32,
}

impl CharGlyph {
    pub fn to_intrinsic(&self) -> IntrinsicState {
        let key = format!("{}:{}:{}", self.character, self.font_family, self.font_size);
        // Simulate glyph bitmap data — size proportional to font_size squared.
        let data_size = (self.font_size as usize) * (self.font_size as usize);
        let data = vec![0u8; data_size];
        IntrinsicState::new(key, data)
            .with_property("char", &self.character.to_string())
            .with_property("font", &self.font_family)
            .with_property("size", &self.font_size.to_string())
    }
}

/// A positioned character in a text layout.
#[derive(Debug, Clone)]
pub struct PositionedChar {
    pub flyweight: FlyweightRef,
    pub color: [u8; 4],
}

impl PositionedChar {
    pub fn new(flyweight: FlyweightRef, color: [u8; 4]) -> Self {
        Self { flyweight, color }
    }
}

/// Render a string of text using the flyweight factory.
/// Returns a list of positioned characters.
pub fn render_text(
    factory: &mut FlyweightFactory,
    text: &str,
    font_family: &str,
    font_size: u32,
    color: [u8; 4],
    start_x: f64,
    start_y: f64,
) -> Vec<PositionedChar> {
    let advance = font_size as f64 * 0.6;
    text.chars()
        .enumerate()
        .map(|(i, ch)| {
            let glyph = CharGlyph {
                character: ch,
                font_family: font_family.to_string(),
                font_size,
            };
            let intrinsic = glyph.to_intrinsic();
            let extrinsic = ExtrinsicState::new(start_x + (i as f64) * advance, start_y);
            let fw = factory.get_flyweight(intrinsic, extrinsic);
            PositionedChar::new(fw, color)
        })
        .collect()
}

impl fmt::Display for FlyweightFactory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = self.stats();
        write!(
            f,
            "FlyweightFactory(unique={}, refs={}, saved={} bytes)",
            s.unique_count, s.total_refs, s.savings_bytes
        )
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_intrinsic(key: &str, size: usize) -> IntrinsicState {
        IntrinsicState::new(key, vec![0u8; size])
    }

    #[test]
    fn factory_new_is_empty() {
        let f = FlyweightFactory::new();
        assert_eq!(f.unique_count(), 0);
        assert_eq!(f.total_refs(), 0);
    }

    #[test]
    fn get_flyweight_creates_entry() {
        let mut f = FlyweightFactory::new();
        let fw = f.get_flyweight(make_intrinsic("a", 100), ExtrinsicState::new(0.0, 0.0));
        assert_eq!(f.unique_count(), 1);
        assert_eq!(f.total_refs(), 1);
        assert_eq!(fw.pool_index(), 0);
    }

    #[test]
    fn dedup_same_key() {
        let mut f = FlyweightFactory::new();
        let fw1 = f.get_flyweight(make_intrinsic("x", 50), ExtrinsicState::new(1.0, 1.0));
        let fw2 = f.get_flyweight(make_intrinsic("x", 50), ExtrinsicState::new(2.0, 2.0));
        assert_eq!(f.unique_count(), 1);
        assert_eq!(f.total_refs(), 2);
        assert_eq!(fw1.pool_index(), fw2.pool_index());
    }

    #[test]
    fn different_keys_separate_entries() {
        let mut f = FlyweightFactory::new();
        f.get_flyweight(make_intrinsic("a", 10), ExtrinsicState::new(0.0, 0.0));
        f.get_flyweight(make_intrinsic("b", 20), ExtrinsicState::new(0.0, 0.0));
        assert_eq!(f.unique_count(), 2);
    }

    #[test]
    fn intrinsic_lookup() {
        let mut f = FlyweightFactory::new();
        let fw = f.get_flyweight(make_intrinsic("key1", 64), ExtrinsicState::new(0.0, 0.0));
        let intr = f.intrinsic(&fw).unwrap();
        assert_eq!(intr.key, "key1");
        assert_eq!(intr.data.len(), 64);
    }

    #[test]
    fn get_by_key() {
        let mut f = FlyweightFactory::new();
        f.get_flyweight(make_intrinsic("abc", 10), ExtrinsicState::new(0.0, 0.0));
        assert!(f.get_by_key("abc").is_some());
        assert!(f.get_by_key("xyz").is_none());
    }

    #[test]
    fn contains_key() {
        let mut f = FlyweightFactory::new();
        assert!(!f.contains_key("k"));
        f.get_flyweight(make_intrinsic("k", 1), ExtrinsicState::new(0.0, 0.0));
        assert!(f.contains_key("k"));
    }

    #[test]
    fn ref_count_tracking() {
        let mut f = FlyweightFactory::new();
        f.get_flyweight(make_intrinsic("r", 10), ExtrinsicState::new(0.0, 0.0));
        f.get_flyweight(make_intrinsic("r", 10), ExtrinsicState::new(1.0, 1.0));
        f.get_flyweight(make_intrinsic("r", 10), ExtrinsicState::new(2.0, 2.0));
        assert_eq!(f.ref_count("r"), 3);
        assert_eq!(f.ref_count("nonexistent"), 0);
    }

    #[test]
    fn keys_sorted() {
        let mut f = FlyweightFactory::new();
        f.get_flyweight(make_intrinsic("z", 1), ExtrinsicState::new(0.0, 0.0));
        f.get_flyweight(make_intrinsic("a", 1), ExtrinsicState::new(0.0, 0.0));
        f.get_flyweight(make_intrinsic("m", 1), ExtrinsicState::new(0.0, 0.0));
        assert_eq!(f.keys(), vec!["a", "m", "z"]);
    }

    #[test]
    fn stats_calculation() {
        let mut f = FlyweightFactory::new();
        // One intrinsic (100 bytes of data + key "k" = 101 bytes).
        // Two refs.
        f.get_flyweight(make_intrinsic("k", 100), ExtrinsicState::new(0.0, 0.0));
        f.get_flyweight(make_intrinsic("k", 100), ExtrinsicState::new(1.0, 1.0));

        let stats = f.stats();
        assert_eq!(stats.unique_count, 1);
        assert_eq!(stats.total_refs, 2);
        assert_eq!(stats.shared_bytes, 101); // "k" + 100 data bytes
        assert_eq!(stats.unshared_bytes, 202); // 101 * 2 refs
        assert_eq!(stats.savings_bytes, 101);
    }

    #[test]
    fn savings_ratio() {
        let mut f = FlyweightFactory::new();
        f.get_flyweight(make_intrinsic("x", 100), ExtrinsicState::new(0.0, 0.0));
        f.get_flyweight(make_intrinsic("x", 100), ExtrinsicState::new(0.0, 0.0));
        let ratio = f.savings_ratio();
        assert!(ratio > 0.0);
        assert!(ratio < 1.0);
    }

    #[test]
    fn savings_ratio_empty() {
        let f = FlyweightFactory::new();
        assert_eq!(f.savings_ratio(), 0.0);
    }

    #[test]
    fn extrinsic_state_builder() {
        let ext = ExtrinsicState::new(10.0, 20.0)
            .with_scale(2.0)
            .with_extra("color", "red");
        assert_eq!(ext.x, 10.0);
        assert_eq!(ext.y, 20.0);
        assert_eq!(ext.scale, 2.0);
        assert_eq!(ext.extra.get("color").unwrap(), "red");
    }

    #[test]
    fn intrinsic_with_properties() {
        let intr = IntrinsicState::new("glyph", vec![1, 2, 3])
            .with_property("font", "Arial")
            .with_property("style", "bold");
        assert_eq!(intr.properties.len(), 2);
        assert!(intr.size_bytes() > 0);
    }

    #[test]
    fn char_glyph_render() {
        let mut factory = FlyweightFactory::new();
        let chars = render_text(&mut factory, "hello", "Arial", 12, [0, 0, 0, 255], 0.0, 0.0);

        assert_eq!(chars.len(), 5);
        // 'l' appears twice — only 4 unique glyphs.
        assert_eq!(factory.unique_count(), 4);
        assert_eq!(factory.total_refs(), 5);
    }

    #[test]
    fn char_glyph_repeat_text() {
        let mut factory = FlyweightFactory::new();
        render_text(&mut factory, "aaa", "Mono", 10, [0, 0, 0, 255], 0.0, 0.0);
        // Only one unique glyph for 'a'.
        assert_eq!(factory.unique_count(), 1);
        assert_eq!(factory.total_refs(), 3);
        assert!(factory.savings_ratio() > 0.5);
    }

    #[test]
    fn display_impl() {
        let mut f = FlyweightFactory::new();
        f.get_flyweight(make_intrinsic("d", 50), ExtrinsicState::new(0.0, 0.0));
        let s = format!("{f}");
        assert!(s.contains("unique=1"));
        assert!(s.contains("refs=1"));
    }
}
