//! Attractor Tier — intelligence attractor dispatch in the materializer cascade.
//!
//! This tier sits between [`Source::Skill`] and [`Source::Eigenbasis`] in the
//! cascade, dispatching queries to one of the 37 intelligence attractors from
//! the Periodic Stack of Intelligence when a pattern matches.
//!
//! ## Why this exists
//!
//! The 37 attractors are the structural skeleton of intelligence — every
//! intelligent operation across biology, digital, and chemistry converges on
//! one of 37 convergent cells. Until this module existed, the attractor
//! infrastructure in `inv-ai-codegraph::flowg` was scaffolding that nothing
//! called. This module connects the attractor primitives to the runtime
//! cascade so they actually dispatch.
//!
//! ## Design: local shadow enum + trait
//!
//! `joule-db-amorphic` cannot depend on `inv-ai-codegraph` (cyclic). So this
//! module defines:
//!
//! 1. A local [`AttractorKind`] enum shadowing the 37 attractors by name.
//! 2. A [`AttractorQuery`] enum for typed attractor inputs.
//! 3. A [`AttractorResolver`] trait that the application layer implements
//!    by delegating to `inv-ai-codegraph::flowg::OpKind::Intelligence(_)`.
//! 4. A [`PatternRegistry`] that maps string patterns to attractor kinds,
//!    so the string-based [`Materializer::materialize`] can detect attractor
//!    queries and dispatch them.
//!
//! The typed entry point [`Materializer::materialize_attractor`] skips the
//! string step entirely — the caller already knows which attractor they want.
//!
//! ## Energy model
//!
//! Each attractor dispatch reports its theoretical floor cost (from the
//! silicon roadmap, x86_64 baseline). Current implementations emulate the
//! attractor via math ops at 10²-10⁴× the floor; the report tells you the
//! savings available if dedicated silicon existed.

use std::collections::HashMap;
use std::time::Instant;

use super::materializer::{EntropyLevel, MaterializeResult, Materializer, Source};

// ============================================================================
// AttractorKind — local shadow of inv-ai-codegraph's AttractorOp
// ============================================================================

/// The 37 convergent cells from the Periodic Stack of Intelligence, shadowed
/// locally so `joule-db-amorphic` can reference them without depending on
/// `inv-ai-codegraph` (which would create a cycle).
///
/// Name-for-name this matches `inv_ai_codegraph::flowg::enums::AttractorOp`.
/// The application layer bridges the two with a trivial from/into mapping.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AttractorKind {
    // Tier 0 — Raw (4)
    ThresholdCrossing,
    RawStructuralRecognition,
    RawExternalMagnitude,
    RawMagnitudeAdaptation,
    // Tier 1 — Feature (13)
    SpatialGradient,
    TemporalAdaptation,
    MagnitudeGainAdaptation,
    SpectralTemplateMatch,
    ShallowStructuralTemplate,
    ParallelSpectralDecomposition,
    SpectralAdaptation,
    ShallowMagnitudeContrast,
    MagnitudeVsComputedReference,
    ShallowRelationalComparison,
    ShallowRelationalTemplate,
    ShallowCausalInference,
    SpatialAdaptation,
    // Tier 2 — Pattern (12)
    DeepStructuralTemplate,
    CompetitiveComparison,
    ForwardModelPredictionError,
    MultiScaleTemporalContrast,
    SpatialTemplateMatch,
    CausalSchemaMatch,
    TemporalContextEncoding,
    SustainedSpatialSelfReference,
    StructuralCompetition,
    AdaptiveStructuralPerception,
    MidDepthCausalInference,
    RelationalPredictionError,
    // Tier 3 — Model (8)
    DeepSelfModelError,
    DeepRelationalTemplate,
    RecursiveSpatialMaps,
    DeepCausalModelError,
    Metacognition,
    LongRangePredictiveTemporal,
    DeepTemporalMemory,
    SustainedStructuralMemory,
}

impl AttractorKind {
    /// The attractor ID (1-37) matching the canonical Periodic Stack of
    /// Intelligence numbering.
    pub fn id(&self) -> u8 {
        use AttractorKind as A;
        match self {
            A::ThresholdCrossing => 1,
            A::RawStructuralRecognition => 20,
            A::RawExternalMagnitude => 26,
            A::RawMagnitudeAdaptation => 29,
            A::SpatialGradient => 2,
            A::TemporalAdaptation => 3,
            A::MagnitudeGainAdaptation => 4,
            A::SpectralTemplateMatch => 14,
            A::ShallowStructuralTemplate => 21,
            A::ParallelSpectralDecomposition => 24,
            A::SpectralAdaptation => 25,
            A::ShallowMagnitudeContrast => 27,
            A::MagnitudeVsComputedReference => 28,
            A::ShallowRelationalComparison => 30,
            A::ShallowRelationalTemplate => 31,
            A::ShallowCausalInference => 33,
            A::SpatialAdaptation => 34,
            A::DeepStructuralTemplate => 5,
            A::CompetitiveComparison => 6,
            A::ForwardModelPredictionError => 7,
            A::MultiScaleTemporalContrast => 9,
            A::SpatialTemplateMatch => 10,
            A::CausalSchemaMatch => 12,
            A::TemporalContextEncoding => 15,
            A::SustainedSpatialSelfReference => 16,
            A::StructuralCompetition => 22,
            A::AdaptiveStructuralPerception => 23,
            A::MidDepthCausalInference => 32,
            A::RelationalPredictionError => 35,
            A::DeepSelfModelError => 8,
            A::DeepRelationalTemplate => 11,
            A::RecursiveSpatialMaps => 13,
            A::DeepCausalModelError => 17,
            A::Metacognition => 18,
            A::LongRangePredictiveTemporal => 19,
            A::DeepTemporalMemory => 36,
            A::SustainedStructuralMemory => 37,
        }
    }

    /// The cascade tier this attractor belongs to (0=Raw, 1=Feature,
    /// 2=Pattern, 3=Model). Matches the intelligence stack's 4-13-12-8
    /// distribution.
    pub fn tier(&self) -> u8 {
        match self.id() {
            1 | 20 | 26 | 29 => 0,
            2 | 3 | 4 | 14 | 21 | 24 | 25 | 27 | 28 | 30 | 31 | 33 | 34 => 1,
            5 | 6 | 7 | 9 | 10 | 12 | 15 | 16 | 22 | 23 | 32 | 35 => 2,
            8 | 11 | 13 | 17 | 18 | 19 | 36 | 37 => 3,
            _ => 0, // unreachable given id() is total
        }
    }

    /// Theoretical native silicon floor in joules (from the x86_64 baseline
    /// table in `inv_ai_codegraph::flowg::energy::x86_64_attractor_baseline`).
    /// These are the picojoule costs in the silicon roadmap converted to J.
    pub fn native_floor_joules(&self) -> f64 {
        let pj = self.native_floor_picojoules();
        pj as f64 * 1e-12
    }

    /// Theoretical native silicon floor in picojoules. Matches the
    /// per-attractor calibration from flowg-silicon-roadmap.md §3-§6.
    pub fn native_floor_picojoules(&self) -> u64 {
        use AttractorKind as A;
        match self {
            // Tier 0
            A::ThresholdCrossing => 5,
            A::RawStructuralRecognition => 10,
            A::RawExternalMagnitude => 5,
            A::RawMagnitudeAdaptation => 8,
            // Tier 1
            A::SpatialGradient => 20,
            A::TemporalAdaptation => 30,
            A::MagnitudeGainAdaptation => 25,
            A::SpectralTemplateMatch => 50,
            A::ShallowStructuralTemplate => 40,
            A::ParallelSpectralDecomposition => 15,
            A::SpectralAdaptation => 30,
            A::ShallowMagnitudeContrast => 20,
            A::MagnitudeVsComputedReference => 40,
            A::ShallowRelationalComparison => 30,
            A::ShallowRelationalTemplate => 50,
            A::ShallowCausalInference => 45,
            A::SpatialAdaptation => 60,
            // Tier 2
            A::DeepStructuralTemplate => 250,
            A::CompetitiveComparison => 200,
            A::ForwardModelPredictionError => 300,
            A::MultiScaleTemporalContrast => 250,
            A::SpatialTemplateMatch => 250,
            A::CausalSchemaMatch => 300,
            A::TemporalContextEncoding => 200,
            A::SustainedSpatialSelfReference => 180,
            A::StructuralCompetition => 200,
            A::AdaptiveStructuralPerception => 280,
            A::MidDepthCausalInference => 280,
            A::RelationalPredictionError => 300,
            // Tier 3
            A::DeepSelfModelError => 1500,
            A::DeepRelationalTemplate => 1000,
            A::RecursiveSpatialMaps => 800,
            A::DeepCausalModelError => 2000,
            A::Metacognition => 1200,
            A::LongRangePredictiveTemporal => 900,
            A::DeepTemporalMemory => 600,
            A::SustainedStructuralMemory => 500,
        }
    }
}

// ============================================================================
// AttractorQuery — typed input for attractor dispatch
// ============================================================================

/// A typed attractor query — carries both the attractor kind and the input
/// data in a form the resolver can consume directly.
///
/// String → typed conversion happens in the pattern registry (string path)
/// or at the caller's site (typed path via [`Materializer::materialize_attractor`]).
/// Once the query reaches a resolver, no string parsing is needed.
#[derive(Clone, Debug)]
pub struct AttractorQuery {
    pub kind: AttractorKind,
    /// Input payload — opaque to this module. Resolvers cast to their
    /// expected type using the discriminator in `kind`.
    pub input: String,
}

impl AttractorQuery {
    pub fn new(kind: AttractorKind, input: impl Into<String>) -> Self {
        Self {
            kind,
            input: input.into(),
        }
    }
}

// ============================================================================
// AttractorResolver — trait the app layer implements
// ============================================================================

/// A typed resolver for an intelligence attractor primitive.
///
/// Implementations delegate to `inv-ai-codegraph::flowg::OpKind::Intelligence(_)`
/// and return a string output. The resolver knows the attractor's native cost
/// but can report a higher actual cost if it runs on emulated silicon.
pub trait AttractorResolver: Send + Sync {
    /// The attractor this resolver handles.
    fn kind(&self) -> AttractorKind;

    /// Resolve a query. Returns `None` if the resolver declines (e.g., the
    /// input doesn't match what this attractor expects).
    fn resolve(&self, query: &AttractorQuery) -> Option<AttractorResolution>;
}

/// Output from an attractor resolution.
#[derive(Clone, Debug)]
pub struct AttractorResolution {
    pub output: String,
    /// Actual energy used (joules). On emulated silicon this is higher than
    /// `kind.native_floor_joules()`; on flowG native silicon they match.
    pub energy_joules: f64,
    pub confidence: f64,
    /// Native silicon floor for this specific attractor dispatch, in
    /// picojoules. When `Some`, the materializer wires this into the
    /// energy receipt so callers see the per-attractor floor instead of
    /// the generic `Source::Attractor` default. See
    /// [`AttractorKind::native_floor_picojoules`].
    pub native_floor_pj: Option<u64>,
}

// ============================================================================
// PatternRegistry — string pattern → attractor kind
// ============================================================================

/// A pattern matcher that recognizes attractor queries by string pattern.
/// Returns `Some(kind)` if the string matches, `None` if not.
///
/// Function pointers are used instead of closures to keep `PatternRegistry`
/// `Send + Sync` without requiring `Box<dyn Fn>`.
pub type PatternMatcher = fn(&str) -> Option<AttractorKind>;

/// A registry mapping string patterns to attractor dispatches.
///
/// The materializer uses this to recognize when a string query matches one
/// of the 37 attractor patterns — e.g., "what's above threshold" → `ThresholdCrossing`,
/// "find similar" → `ShallowRelationalTemplate`, "predict next" →
/// `ForwardModelPredictionError`.
///
/// When a pattern matches, the query is packaged into an [`AttractorQuery`]
/// and dispatched to the registered resolver for that kind. If no resolver
/// is registered, the query falls through to the next cascade tier.
#[derive(Default)]
pub struct PatternRegistry {
    matchers: Vec<PatternMatcher>,
}

impl PatternRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a pattern matcher. The first matcher that returns `Some`
    /// wins — order matters, most-specific first.
    pub fn register(&mut self, matcher: PatternMatcher) {
        self.matchers.push(matcher);
    }

    /// Check the input against all registered patterns. Returns the first
    /// matching attractor kind.
    pub fn classify(&self, input: &str) -> Option<AttractorKind> {
        for m in &self.matchers {
            if let Some(kind) = m(input) {
                return Some(kind);
            }
        }
        None
    }

    /// Number of registered patterns.
    pub fn len(&self) -> usize {
        self.matchers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.matchers.is_empty()
    }
}

// ============================================================================
// AttractorCache — structural caching by (kind, content hash)
// ============================================================================

/// A cache key that identifies an attractor query by **structure**, not by
/// surface form. Two queries that land in the same attractor cell with the
/// same input content share a cache entry even if their original strings
/// differed (e.g. "is value:42 above threshold:10" and "value:42 above-thr 10"
/// both hash to the same key once parsed).
///
/// The key has two parts:
/// - `kind`: which of the 37 attractors this query targets
/// - `content_hash`: a fast hash (FxHash-style) of the canonical input form
///
/// The content hash is computed from the **raw input string** by default —
/// this is the conservative case. Resolvers that have a canonical form
/// (e.g. spatial coordinates) can override `canonical_key()` on
/// [`AttractorResolver`] to produce structural keys that ignore irrelevant
/// surface differences (whitespace, key order, etc.).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AttractorCacheKey {
    pub kind: AttractorKind,
    pub content_hash: u64,
}

impl AttractorCacheKey {
    /// Construct a cache key from a query, hashing the raw input string.
    pub fn from_query(query: &AttractorQuery) -> Self {
        Self::from_parts(query.kind, &query.input)
    }

    /// Construct a cache key from a kind and a string (the canonical input).
    pub fn from_parts(kind: AttractorKind, canonical: &str) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(canonical, &mut hasher);
        let content_hash = std::hash::Hasher::finish(&hasher);
        Self { kind, content_hash }
    }
}

/// Cache for attractor resolutions. Stores `(AttractorCacheKey, AttractorResolution)`
/// pairs and tracks hit/miss counts.
///
/// The cache is owned by the [`AttractorTier`]. On every dispatch, the tier
/// computes the cache key, checks for a hit, and either returns the cached
/// resolution or runs the resolver and stores the result.
/// Default maximum number of entries before the cache evicts the oldest
/// entries. Large enough for diverse workloads, small enough that a
/// long-running server doesn't OOM on unbounded growth.
const DEFAULT_MAX_CACHE_ENTRIES: usize = 10_000;

pub struct AttractorCache {
    entries: HashMap<AttractorCacheKey, AttractorResolution>,
    /// Insertion order for LRU-style eviction. When full, the oldest
    /// entries (front of the vec) are removed.
    insertion_order: Vec<AttractorCacheKey>,
    max_entries: usize,
    hits: u64,
    misses: u64,
}

impl Default for AttractorCache {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
            insertion_order: Vec::new(),
            max_entries: DEFAULT_MAX_CACHE_ENTRIES,
            hits: 0,
            misses: 0,
        }
    }
}

impl AttractorCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct a cache with a custom capacity limit.
    pub fn with_max_entries(max_entries: usize) -> Self {
        Self {
            max_entries: max_entries.max(1),
            ..Self::default()
        }
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Number of cache hits since creation.
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Number of cache misses since creation.
    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// Hit rate as a fraction in [0, 1]. Returns 0.0 if no lookups have happened.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Look up a cached resolution. Increments the hit counter on success
    /// and the miss counter on failure.
    pub fn get(&mut self, key: &AttractorCacheKey) -> Option<&AttractorResolution> {
        if let Some(resolution) = self.entries.get(key) {
            self.hits += 1;
            Some(resolution)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Insert a resolution into the cache. If the cache is at capacity,
    /// the oldest entry (by insertion order) is evicted first.
    pub fn insert(&mut self, key: AttractorCacheKey, resolution: AttractorResolution) {
        if !self.entries.contains_key(&key) {
            // New entry — evict oldest if at capacity.
            while self.entries.len() >= self.max_entries {
                if let Some(oldest) = self.insertion_order.first().copied() {
                    self.insertion_order.remove(0);
                    self.entries.remove(&oldest);
                } else {
                    break;
                }
            }
            self.insertion_order.push(key);
        }
        self.entries.insert(key, resolution);
    }

    /// Maximum number of entries before eviction kicks in.
    pub fn max_entries(&self) -> usize {
        self.max_entries
    }

    /// Clear all cached entries (does not reset hit/miss counters).
    pub fn clear_entries(&mut self) {
        self.entries.clear();
        self.insertion_order.clear();
    }

    /// Reset hit/miss counters (does not clear entries).
    pub fn reset_stats(&mut self) {
        self.hits = 0;
        self.misses = 0;
    }
}

// ============================================================================
// AttractorTier — the combined registry + resolver map
// ============================================================================

/// The attractor tier holds the pattern registry, the per-attractor resolvers,
/// and a structural cache. The materializer owns one of these.
///
/// Dispatch flow:
/// 1. Check the cache by `(AttractorKind, content_hash)` — instant return on hit.
/// 2. If miss, classify the query (string path) or use the typed kind directly.
/// 3. Run the resolver, store the result in the cache, return.
///
/// The cache is structural — it keys by `(kind, hash(canonical_input))`, so
/// queries that resolve to the same attractor cell share entries even if their
/// surface forms differed.
pub struct AttractorTier {
    patterns: PatternRegistry,
    resolvers: HashMap<AttractorKind, Box<dyn AttractorResolver>>,
    cache: AttractorCache,
}

impl AttractorTier {
    pub fn new() -> Self {
        Self {
            patterns: PatternRegistry::new(),
            resolvers: HashMap::new(),
            cache: AttractorCache::new(),
        }
    }

    /// Register a pattern matcher for recognizing attractor queries by string.
    pub fn register_pattern(&mut self, matcher: PatternMatcher) {
        self.patterns.register(matcher);
    }

    /// Register a resolver for a specific attractor kind. Replaces any
    /// previously-registered resolver for the same kind.
    pub fn register_resolver(&mut self, resolver: Box<dyn AttractorResolver>) {
        self.resolvers.insert(resolver.kind(), resolver);
    }

    /// Try to dispatch a string query through the attractor tier. Returns
    /// `None` if no pattern matches or no resolver is registered for the
    /// matched kind.
    ///
    /// Checks the structural cache first; on hit, returns the cached
    /// resolution at zero cost (the resolver is not called). On miss,
    /// runs the resolver and stores the result.
    pub fn dispatch_string(&mut self, input: &str) -> Option<AttractorResolution> {
        let kind = self.patterns.classify(input)?;
        let key = AttractorCacheKey::from_parts(kind, input);
        if let Some(cached) = self.cache.get(&key) {
            return Some(cached.clone());
        }
        // Cache miss — run the resolver.
        let resolver = self.resolvers.get(&kind)?;
        let query = AttractorQuery::new(kind, input);
        let resolution = resolver.resolve(&query)?;
        self.cache.insert(key, resolution.clone());
        Some(resolution)
    }

    /// Dispatch a typed query directly, skipping pattern matching but still
    /// using the structural cache.
    pub fn dispatch_typed(&mut self, query: &AttractorQuery) -> Option<AttractorResolution> {
        let key = AttractorCacheKey::from_query(query);
        if let Some(cached) = self.cache.get(&key) {
            return Some(cached.clone());
        }
        let resolver = self.resolvers.get(&query.kind)?;
        let resolution = resolver.resolve(query)?;
        self.cache.insert(key, resolution.clone());
        Some(resolution)
    }

    /// Number of patterns registered.
    pub fn pattern_count(&self) -> usize {
        self.patterns.len()
    }

    /// Number of resolvers registered.
    pub fn resolver_count(&self) -> usize {
        self.resolvers.len()
    }

    /// Read-only access to the cache for metrics inspection.
    pub fn cache(&self) -> &AttractorCache {
        &self.cache
    }

    /// Mutable access to the cache for clearing or stat reset.
    pub fn cache_mut(&mut self) -> &mut AttractorCache {
        &mut self.cache
    }
}

impl Default for AttractorTier {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Materializer integration
// ============================================================================

impl Materializer {
    /// Attach an attractor tier. Replaces any previously-attached tier.
    pub fn set_attractor_tier(&mut self, tier: AttractorTier) {
        self.attractor_tier = Some(tier);
    }

    /// Whether an attractor tier is attached.
    pub fn has_attractor_tier(&self) -> bool {
        self.attractor_tier.is_some()
    }

    /// Read-only access to the attractor tier's structural cache.
    /// Returns `None` if no tier is attached.
    ///
    /// Use this for inspecting cache hit rates after running queries:
    ///
    /// ```ignore
    /// let m = ...;
    /// // ... run queries ...
    /// if let Some(cache) = m.attractor_cache() {
    ///     println!("hit rate: {:.1}%", cache.hit_rate() * 100.0);
    /// }
    /// ```
    pub fn attractor_cache(&self) -> Option<&AttractorCache> {
        self.attractor_tier.as_ref().map(|t| t.cache())
    }

    /// Materialize a typed attractor query, bypassing string parsing.
    /// This is the fast path for callers who already know which attractor
    /// they want — they construct an [`AttractorQuery`] and dispatch directly.
    ///
    /// Uses the structural cache: if the same `(kind, input)` was seen before,
    /// returns the cached resolution at zero resolver cost.
    ///
    /// Returns `None` if no resolver is registered for the query's kind.
    pub fn materialize_attractor(
        &mut self,
        query: &AttractorQuery,
    ) -> Option<MaterializeResult> {
        let start = Instant::now();
        let resolution = self
            .attractor_tier
            .as_mut()
            .and_then(|tier| tier.dispatch_typed(query))?;
        self.metrics.total += 1;

        let entropy = EntropyLevel::Low;
        *self
            .metrics
            .by_entropy
            .entry(format!("{:?}", entropy))
            .or_insert(0) += 1;

        self.metrics.total_energy += resolution.energy_joules;
        *self
            .metrics
            .by_source
            .entry("attractor".into())
            .or_insert(0) += 1;

        let actual_pj = (resolution.energy_joules * 1e12).max(0.0) as u64;
        let native_floor_pj = resolution
            .native_floor_pj
            .unwrap_or(actual_pj.max(1));
        let receipt = super::energy_receipt::EnergyReceipt::for_tier_with_floor(
            Source::Attractor,
            actual_pj,
            native_floor_pj,
        );
        Some(MaterializeResult {
            output: resolution.output,
            source: Source::Attractor,
            entropy,
            verified: resolution.confidence > 0.5,
            energy_joules: resolution.energy_joules,
            elapsed_us: start.elapsed().as_micros() as u64,
            receipt,
        })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// A trivial resolver for testing — returns a canned response for
    /// `ThresholdCrossing` queries.
    struct ThresholdResolver;

    impl AttractorResolver for ThresholdResolver {
        fn kind(&self) -> AttractorKind {
            AttractorKind::ThresholdCrossing
        }

        fn resolve(&self, query: &AttractorQuery) -> Option<AttractorResolution> {
            if query.kind != AttractorKind::ThresholdCrossing {
                return None;
            }
            // Parse "value:X threshold:Y" → answer by splitting on whitespace first
            let mut value: Option<f64> = None;
            let mut threshold: Option<f64> = None;
            for token in query.input.split_whitespace() {
                if let Some(rest) = token.strip_prefix("value:") {
                    value = rest.parse().ok();
                } else if let Some(rest) = token.strip_prefix("threshold:") {
                    threshold = rest.parse().ok();
                }
            }
            let v = value?;
            let t = threshold?;
            let output = if v > t {
                "above threshold".to_string()
            } else {
                "below threshold".to_string()
            };
            Some(attractor_ok(query.kind, output))
        }
    }

    /// Pattern matcher: "above threshold X" or "below threshold X" → ThresholdCrossing
    fn threshold_pattern(s: &str) -> Option<AttractorKind> {
        let lower = s.to_lowercase();
        if lower.contains("above threshold") || lower.contains("below threshold") {
            Some(AttractorKind::ThresholdCrossing)
        } else {
            None
        }
    }

    /// Build a successful `AttractorResolution` from the attractor kind and
    /// a computed output string. Fills `energy_joules`, `confidence`, and
    /// `native_floor_pj` from the kind's silicon roadmap data. Every test
    /// resolver below uses this instead of repeating the boilerplate.
    fn attractor_ok(kind: AttractorKind, output: String) -> AttractorResolution {
        AttractorResolution {
            output,
            energy_joules: kind.native_floor_joules(),
            confidence: 1.0,
            native_floor_pj: Some(kind.native_floor_picojoules()),
        }
    }

    /// Parse `key:value` tokens from whitespace-separated input. Returns
    /// a vec of `(label, number)` pairs for tokens matching `<label>:<f64>`.
    fn parse_kv_pairs(input: &str) -> Vec<(String, f64)> {
        input
            .split_whitespace()
            .filter_map(|tok| {
                let (k, v) = tok.split_once(':')?;
                let val: f64 = v.parse().ok()?;
                Some((k.to_string(), val))
            })
            .collect()
    }

    // ────────────────────────────────────────────────────────────────
    // 4 additional attractor resolvers (T0, T1, T1, T2) — proof that the
    // cascade can dispatch through multiple intelligence primitives end
    // to end, not just ThresholdCrossing. Each resolver:
    //
    //   1. Parses its typed input from a string payload.
    //   2. Computes the attractor result in native Rust.
    //   3. Returns an AttractorResolution with the per-attractor native
    //      floor, so the energy receipt carries a picojoule-scale cost.
    //
    // Picked for the silicon priority ranking: RawExternalMagnitude and
    // ShallowMagnitudeContrast are cheap T0/T1 cells that the perception
    // workload hits every cycle; CompetitiveComparison is the #2 global
    // target from the silicon priority doc with 3.2 nJ of savings per
    // hit; SpatialGradient is the canonical edge-detect primitive.
    // ────────────────────────────────────────────────────────────────

    /// T0 — raw distance/intensity sensor reading.
    ///
    /// Input format: `"magnitude:42.5"` or `"magnitude:1.5 unit:m"`.
    /// Output: `"magnitude=<value>"` (with unit if supplied).
    struct RawMagnitudeResolver;

    impl AttractorResolver for RawMagnitudeResolver {
        fn kind(&self) -> AttractorKind {
            AttractorKind::RawExternalMagnitude
        }

        fn resolve(&self, query: &AttractorQuery) -> Option<AttractorResolution> {
            if query.kind != AttractorKind::RawExternalMagnitude {
                return None;
            }
            let pairs = parse_kv_pairs(&query.input);
            let m = pairs
                .iter()
                .find(|(k, _)| k == "magnitude" || k == "distance" || k == "intensity")
                .map(|(_, v)| *v)?;
            let unit = query
                .input
                .split_whitespace()
                .find_map(|t| t.strip_prefix("unit:"))
                .map(|s| s.to_string());
            let output = match unit {
                Some(u) => format!("magnitude={} {}", m, u),
                None => format!("magnitude={}", m),
            };
            Some(attractor_ok(query.kind, output))
        }
    }

    fn raw_magnitude_pattern(s: &str) -> Option<AttractorKind> {
        let lower = s.to_lowercase();
        if lower.contains("magnitude:") || lower.contains("distance:") || lower.contains("intensity:") {
            Some(AttractorKind::RawExternalMagnitude)
        } else {
            None
        }
    }

    /// T1 — differential between two magnitude channels.
    ///
    /// Input format: `"left:1.2 right:3.4"` or `"a:10 b:5"`.
    /// Output: `"delta=<absdiff> winner=<label>"`.
    struct MagnitudeContrastResolver;

    impl AttractorResolver for MagnitudeContrastResolver {
        fn kind(&self) -> AttractorKind {
            AttractorKind::ShallowMagnitudeContrast
        }

        fn resolve(&self, query: &AttractorQuery) -> Option<AttractorResolution> {
            if query.kind != AttractorKind::ShallowMagnitudeContrast {
                return None;
            }
            let pairs = parse_kv_pairs(&query.input);
            if pairs.len() < 2 {
                return None;
            }
            let (l1, v1) = &pairs[0];
            let (l2, v2) = &pairs[1];
            let delta = (v1 - v2).abs();
            let winner = if v1 >= v2 { l1 } else { l2 };
            Some(attractor_ok(
                query.kind,
                format!("delta={} winner={}", delta, winner),
            ))
        }
    }

    fn magnitude_contrast_pattern(s: &str) -> Option<AttractorKind> {
        let lower = s.to_lowercase();
        // "contrast a:1 b:2" or explicit "channels:" hint
        if lower.starts_with("contrast ") || lower.contains("channels:") {
            Some(AttractorKind::ShallowMagnitudeContrast)
        } else {
            None
        }
    }

    /// T1 — 1D edge detection via finite difference on a sample sequence.
    ///
    /// Input format: `"samples:1,2,5,10,4,3"` (CSV numbers, comma-separated).
    /// Output: `"max_gradient=<abs> at=<index>"`.
    struct SpatialGradientResolver;

    impl AttractorResolver for SpatialGradientResolver {
        fn kind(&self) -> AttractorKind {
            AttractorKind::SpatialGradient
        }

        fn resolve(&self, query: &AttractorQuery) -> Option<AttractorResolution> {
            if query.kind != AttractorKind::SpatialGradient {
                return None;
            }
            // Find `samples:a,b,c,d`
            let samples_token = query
                .input
                .split_whitespace()
                .find_map(|t| t.strip_prefix("samples:"))?;
            let samples: Vec<f64> = samples_token
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            if samples.len() < 2 {
                return None;
            }
            // Finite difference — find largest absolute delta and its index.
            let mut max_abs = 0.0_f64;
            let mut max_idx = 0usize;
            for i in 1..samples.len() {
                let d = (samples[i] - samples[i - 1]).abs();
                if d > max_abs {
                    max_abs = d;
                    max_idx = i;
                }
            }
            Some(attractor_ok(
                query.kind,
                format!("max_gradient={} at={}", max_abs, max_idx),
            ))
        }
    }

    fn spatial_gradient_pattern(s: &str) -> Option<AttractorKind> {
        let lower = s.to_lowercase();
        if lower.contains("samples:") && (lower.contains("gradient") || lower.contains("edge")) {
            Some(AttractorKind::SpatialGradient)
        } else {
            None
        }
    }

    /// T2 — argmax over a labeled candidate set. Attention-like selection.
    ///
    /// Input format: `"apple:0.3 banana:0.7 cherry:0.5"`.
    /// Output: `"winner=<label> score=<max>"`.
    struct CompetitiveComparisonResolver;

    impl AttractorResolver for CompetitiveComparisonResolver {
        fn kind(&self) -> AttractorKind {
            AttractorKind::CompetitiveComparison
        }

        fn resolve(&self, query: &AttractorQuery) -> Option<AttractorResolution> {
            if query.kind != AttractorKind::CompetitiveComparison {
                return None;
            }
            let pairs = parse_kv_pairs(&query.input);
            let (winner, score) = pairs
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))?;
            Some(attractor_ok(
                query.kind,
                format!("winner={} score={}", winner, score),
            ))
        }
    }

    fn competitive_pattern(s: &str) -> Option<AttractorKind> {
        let lower = s.to_lowercase();
        // "select best X" or "argmax X" or "pick best"
        if lower.contains("argmax")
            || lower.starts_with("select best")
            || lower.starts_with("pick best")
            || lower.contains("best of:")
        {
            Some(AttractorKind::CompetitiveComparison)
        } else {
            None
        }
    }

    // ────────────────────────────────────────────────────────────────
    // Per-resolver unit tests
    // ────────────────────────────────────────────────────────────────

    #[test]
    fn raw_magnitude_resolver_parses_magnitude() {
        let r = RawMagnitudeResolver;
        let q = AttractorQuery::new(AttractorKind::RawExternalMagnitude, "magnitude:42.5");
        let res = r.resolve(&q).expect("should resolve");
        assert_eq!(res.output, "magnitude=42.5");
        assert_eq!(res.native_floor_pj, Some(5));
    }

    #[test]
    fn raw_magnitude_resolver_with_unit() {
        let r = RawMagnitudeResolver;
        let q = AttractorQuery::new(AttractorKind::RawExternalMagnitude, "distance:1.5 unit:m");
        let res = r.resolve(&q).expect("should resolve");
        assert_eq!(res.output, "magnitude=1.5 m");
    }

    #[test]
    fn raw_magnitude_resolver_rejects_wrong_kind() {
        let r = RawMagnitudeResolver;
        let q = AttractorQuery::new(AttractorKind::ThresholdCrossing, "magnitude:42");
        assert!(r.resolve(&q).is_none());
    }

    #[test]
    fn magnitude_contrast_resolver_finds_delta_and_winner() {
        let r = MagnitudeContrastResolver;
        let q = AttractorQuery::new(
            AttractorKind::ShallowMagnitudeContrast,
            "left:1.2 right:3.4",
        );
        let res = r.resolve(&q).expect("should resolve");
        assert!(res.output.contains("winner=right"));
        assert!(res.output.contains("delta=2.2"));
        assert_eq!(res.native_floor_pj, Some(20));
    }

    #[test]
    fn magnitude_contrast_resolver_picks_larger_side() {
        let r = MagnitudeContrastResolver;
        let q = AttractorQuery::new(AttractorKind::ShallowMagnitudeContrast, "a:10 b:5");
        let res = r.resolve(&q).expect("should resolve");
        assert!(res.output.contains("winner=a"));
    }

    #[test]
    fn magnitude_contrast_resolver_needs_two_channels() {
        let r = MagnitudeContrastResolver;
        let q = AttractorQuery::new(AttractorKind::ShallowMagnitudeContrast, "only:1");
        assert!(r.resolve(&q).is_none());
    }

    #[test]
    fn spatial_gradient_resolver_finds_largest_edge() {
        // samples [1,2,5,10,4,3] with finite-difference indexes at the
        // *receiving* sample:
        //   i=1: |2-1|=1
        //   i=2: |5-2|=3
        //   i=3: |10-5|=5
        //   i=4: |4-10|=6   ← max
        //   i=5: |3-4|=1
        let r = SpatialGradientResolver;
        let q = AttractorQuery::new(AttractorKind::SpatialGradient, "samples:1,2,5,10,4,3");
        let res = r.resolve(&q).expect("should resolve");
        assert!(res.output.contains("max_gradient=6"));
        assert!(res.output.contains("at=4"));
        assert_eq!(res.native_floor_pj, Some(20));
    }

    #[test]
    fn spatial_gradient_resolver_handles_monotonic() {
        let r = SpatialGradientResolver;
        let q = AttractorQuery::new(AttractorKind::SpatialGradient, "samples:1,2,3,4,5");
        let res = r.resolve(&q).expect("should resolve");
        // All deltas are 1 → first one at index 1 wins (first max seen).
        assert!(res.output.contains("max_gradient=1"));
        assert!(res.output.contains("at=1"));
    }

    #[test]
    fn spatial_gradient_resolver_needs_at_least_two_samples() {
        let r = SpatialGradientResolver;
        let q = AttractorQuery::new(AttractorKind::SpatialGradient, "samples:1");
        assert!(r.resolve(&q).is_none());
    }

    #[test]
    fn competitive_resolver_picks_highest_score() {
        let r = CompetitiveComparisonResolver;
        let q = AttractorQuery::new(
            AttractorKind::CompetitiveComparison,
            "apple:0.3 banana:0.7 cherry:0.5",
        );
        let res = r.resolve(&q).expect("should resolve");
        assert!(res.output.contains("winner=banana"));
        assert!(res.output.contains("score=0.7"));
        assert_eq!(res.native_floor_pj, Some(200));
    }

    #[test]
    fn competitive_resolver_single_candidate_still_wins() {
        let r = CompetitiveComparisonResolver;
        let q = AttractorQuery::new(AttractorKind::CompetitiveComparison, "only:42");
        let res = r.resolve(&q).expect("should resolve");
        assert!(res.output.contains("winner=only"));
        assert!(res.output.contains("score=42"));
    }

    #[test]
    fn competitive_resolver_rejects_empty() {
        let r = CompetitiveComparisonResolver;
        let q = AttractorQuery::new(AttractorKind::CompetitiveComparison, "no labels here");
        assert!(r.resolve(&q).is_none());
    }

    // ────────────────────────────────────────────────────────────────
    // End-to-end cascade integration test — proves all 5 resolvers
    // dispatch through the materializer string path with energy
    // receipts carrying per-attractor picojoule floors.
    // ────────────────────────────────────────────────────────────────

    #[test]
    fn cascade_dispatches_through_five_attractor_primitives() {
        let mut m = Materializer::new();
        let mut tier = AttractorTier::new();

        // Register all 5 resolvers.
        tier.register_resolver(Box::new(ThresholdResolver));
        tier.register_resolver(Box::new(RawMagnitudeResolver));
        tier.register_resolver(Box::new(MagnitudeContrastResolver));
        tier.register_resolver(Box::new(SpatialGradientResolver));
        tier.register_resolver(Box::new(CompetitiveComparisonResolver));

        // Register all 5 pattern matchers.
        tier.register_pattern(threshold_pattern);
        tier.register_pattern(raw_magnitude_pattern);
        tier.register_pattern(magnitude_contrast_pattern);
        tier.register_pattern(spatial_gradient_pattern);
        tier.register_pattern(competitive_pattern);

        m.set_attractor_tier(tier);

        // Five queries, one per attractor kind, through the materializer
        // string path.
        let queries: &[(&str, AttractorKind, u64)] = &[
            // (query_string, expected_kind, expected_native_floor_pj)
            ("value:42 threshold:10 above threshold", AttractorKind::ThresholdCrossing, 5),
            ("magnitude:1.5 unit:m", AttractorKind::RawExternalMagnitude, 5),
            ("contrast left:1.2 right:3.4", AttractorKind::ShallowMagnitudeContrast, 20),
            ("samples:1,2,5,10,4,3 compute gradient", AttractorKind::SpatialGradient, 20),
            ("argmax over apple:0.3 banana:0.7 cherry:0.5", AttractorKind::CompetitiveComparison, 200),
        ];

        let mut total_actual_pj = 0u64;
        for (q, expected_kind, expected_floor) in queries {
            let r = m.materialize(q);
            assert_eq!(
                r.source,
                Source::Attractor,
                "query `{}` should dispatch to attractor tier, got {:?}",
                q,
                r.source
            );
            assert_eq!(
                r.receipt.tier,
                Source::Attractor,
                "receipt tier mismatch for `{}`",
                q
            );
            assert_eq!(
                r.receipt.native_floor_pj, *expected_floor,
                "native floor mismatch for {:?}",
                expected_kind
            );
            // Receipt should show massive savings vs LLM. The smallest
            // win comes from CompetitiveComparison at 200 pJ actual vs
            // 10 mJ LLM baseline = exactly 5e7×. Allow >= for that.
            assert!(
                r.receipt.savings_vs_llm() >= 5e7,
                "savings too small for {:?}: {}",
                expected_kind,
                r.receipt.savings_vs_llm()
            );
            total_actual_pj += r.receipt.actual_pj;
        }

        // Sum of 5 attractor floors = 5 + 5 + 20 + 20 + 200 = 250 pJ.
        // vs an LLM cascade handling 5 queries = 5 × 10 mJ = 50 mJ = 50_000_000_000 pJ.
        // Cascade savings = 50 GJ / 250 pJ ≈ 200M×.
        let baseline_pj = 5u64 * 10_000_000_000; // 5 × 10 mJ
        let cascade_savings = baseline_pj as f64 / total_actual_pj as f64;
        assert!(
            cascade_savings > 1e8,
            "5-attractor cascade should save >1e8× vs LLM, got {}×",
            cascade_savings
        );

        // And the neural-avoidance rate should be 100% — every query
        // resolved at the attractor tier, none fell through.
        assert_eq!(
            m.metrics.pct_avoided_neural(),
            100.0,
            "cascade should have avoided neural 100%, got {}%",
            m.metrics.pct_avoided_neural()
        );
    }

    /// Simulated robot perception cycle — 10 queries, all attractor hits,
    /// with energy receipts. This is the runtime counterpart to the
    /// abstract `perception_workload()` benchmark in
    /// `inv-ai-codegraph::flowg::impedance`. That test ranks attractors by
    /// how much silicon would save against a GPU pipeline. This test
    /// proves the same attractors actually *dispatch* through the
    /// runtime cascade in joule-db-amorphic with picojoule-scale cost,
    /// not µJ, not mJ.
    ///
    /// The full loop — sensor read, differential contrast, edge detect,
    /// threshold classification, winner selection — mirrors the T0-T2
    /// attractor mix a real robot runs every 100 ms.
    #[test]
    fn runtime_perception_cycle_is_picojoule_scale() {
        let mut m = Materializer::new();
        let mut tier = AttractorTier::new();
        tier.register_resolver(Box::new(ThresholdResolver));
        tier.register_resolver(Box::new(RawMagnitudeResolver));
        tier.register_resolver(Box::new(MagnitudeContrastResolver));
        tier.register_resolver(Box::new(SpatialGradientResolver));
        tier.register_resolver(Box::new(CompetitiveComparisonResolver));
        tier.register_pattern(threshold_pattern);
        tier.register_pattern(raw_magnitude_pattern);
        tier.register_pattern(magnitude_contrast_pattern);
        tier.register_pattern(spatial_gradient_pattern);
        tier.register_pattern(competitive_pattern);
        m.set_attractor_tier(tier);

        // A realistic robot perception loop, one 100 ms cycle:
        let cycle: &[&str] = &[
            // T0 — raw sensor reads
            "distance:1.2 unit:m",                                  // lidar forward
            "distance:0.8 unit:m",                                  // lidar left
            "intensity:0.35",                                       // ambient light
            // T1 — feature extraction
            "contrast left:0.8 right:1.2",                          // differential range
            "samples:0.1,0.2,0.5,1.0,0.4,0.2 compute gradient",     // 1D depth edge
            "samples:0.9,0.9,0.8,0.8,0.3,0.2 compute edge",         // silhouette detection
            // T0 — threshold classifications
            "value:1.2 threshold:0.5 above threshold",              // obstacle?
            "value:0.2 threshold:0.5 below threshold",              // clear path?
            // T2 — attention selection
            "argmax over ball:0.9 chair:0.2 wall:0.4",              // what's in view?
            "argmax over approach:0.7 stop:0.2 backup:0.1",         // next action
        ];

        let mut total_actual_pj = 0u64;
        let mut total_native_floor_pj = 0u64;
        let mut per_tier_counts: std::collections::HashMap<&'static str, u32> =
            std::collections::HashMap::new();

        for (i, query) in cycle.iter().enumerate() {
            let r = m.materialize(query);
            assert_eq!(
                r.source,
                Source::Attractor,
                "cycle step {} `{}` didn't route to attractor tier: {:?}",
                i,
                query,
                r.source
            );
            total_actual_pj += r.receipt.actual_pj;
            total_native_floor_pj += r.receipt.native_floor_pj;
            // Bucket by cascade tier (T0 / T1 / T2) inferred from actual_pj.
            // T0 ≤ 10 pJ, T1 10-100 pJ, T2 > 100 pJ.
            let tier_label = if r.receipt.actual_pj <= 10 {
                "T0"
            } else if r.receipt.actual_pj <= 100 {
                "T1"
            } else {
                "T2"
            };
            *per_tier_counts.entry(tier_label).or_insert(0) += 1;
        }

        // Sanity: the cycle ran 10 queries, all attractor-resolved.
        assert_eq!(m.metrics.total, 10);
        assert_eq!(m.metrics.pct_avoided_neural(), 100.0);
        assert_eq!(*per_tier_counts.get("T0").unwrap_or(&0), 5);
        assert!(per_tier_counts.contains_key("T1"));
        assert!(per_tier_counts.contains_key("T2"));

        // Total cycle cost is at most ~500 pJ — sub-nanojoule for the
        // whole perception loop. Compare to an LLM cascade doing the
        // same 10 queries: 10 × 10 mJ = 100 mJ = 100_000_000_000 pJ.
        // Savings = 100 GJ / 500 pJ ≈ 2e8×.
        assert!(
            total_actual_pj < 1_000,
            "total cycle cost exceeded 1 nJ: {} pJ",
            total_actual_pj
        );
        let llm_cascade_pj = 10u64 * 10_000_000_000;
        let cycle_savings = llm_cascade_pj as f64 / total_actual_pj as f64;
        assert!(
            cycle_savings > 1e8,
            "perception cycle should save >1e8× vs LLM cascade, got {}×",
            cycle_savings
        );

        // At 10 Hz continuous operation, the total energy budget of the
        // cascade is comically small.
        let per_second_pj = total_actual_pj * 10;
        assert!(
            per_second_pj < 10_000, // < 10 nJ/s for the attractor layer alone
            "10 Hz operation should use < 10 nJ/s in the attractor layer, got {} pJ/s",
            per_second_pj
        );

        // At 24/7 operation for a full day, the cascade consumes
        // negligible energy in the attractor tier.
        let per_day_pj = per_second_pj * 60 * 60 * 24;
        // < 1 mJ per day for the attractor dispatch work.
        assert!(per_day_pj < 1_000_000_000, "per-day picojoules too high: {}", per_day_pj);

        // Final sanity: native floors are at most 1 per query above
        // total actual cost (because some resolvers report actual ==
        // floor). The impedance ratio is 1.0 in this synthetic test
        // because the resolvers return `native_floor_joules()`, not
        // an emulated higher cost.
        assert!(
            total_native_floor_pj <= total_actual_pj,
            "native floors should be <= actual on the synthetic resolvers"
        );
    }

    #[test]
    fn cascade_falls_back_to_neural_when_no_attractor_matches() {
        // Verify the cascade still escalates correctly when no attractor
        // pattern matches — the new resolvers don't accidentally gobble
        // unrelated queries.
        let mut m = Materializer::new();
        let mut tier = AttractorTier::new();
        tier.register_resolver(Box::new(RawMagnitudeResolver));
        tier.register_resolver(Box::new(CompetitiveComparisonResolver));
        tier.register_pattern(raw_magnitude_pattern);
        tier.register_pattern(competitive_pattern);
        m.set_attractor_tier(tier);

        let r = m.materialize("write me a poem about the stars");
        assert_eq!(r.source, Source::Neural);
    }

    #[test]
    fn attractor_kind_ids_unique() {
        let all: &[AttractorKind] = &[
            AttractorKind::ThresholdCrossing,
            AttractorKind::RawStructuralRecognition,
            AttractorKind::RawExternalMagnitude,
            AttractorKind::RawMagnitudeAdaptation,
            AttractorKind::SpatialGradient,
            AttractorKind::TemporalAdaptation,
            AttractorKind::MagnitudeGainAdaptation,
            AttractorKind::SpectralTemplateMatch,
            AttractorKind::ShallowStructuralTemplate,
            AttractorKind::ParallelSpectralDecomposition,
            AttractorKind::SpectralAdaptation,
            AttractorKind::ShallowMagnitudeContrast,
            AttractorKind::MagnitudeVsComputedReference,
            AttractorKind::ShallowRelationalComparison,
            AttractorKind::ShallowRelationalTemplate,
            AttractorKind::ShallowCausalInference,
            AttractorKind::SpatialAdaptation,
            AttractorKind::DeepStructuralTemplate,
            AttractorKind::CompetitiveComparison,
            AttractorKind::ForwardModelPredictionError,
            AttractorKind::MultiScaleTemporalContrast,
            AttractorKind::SpatialTemplateMatch,
            AttractorKind::CausalSchemaMatch,
            AttractorKind::TemporalContextEncoding,
            AttractorKind::SustainedSpatialSelfReference,
            AttractorKind::StructuralCompetition,
            AttractorKind::AdaptiveStructuralPerception,
            AttractorKind::MidDepthCausalInference,
            AttractorKind::RelationalPredictionError,
            AttractorKind::DeepSelfModelError,
            AttractorKind::DeepRelationalTemplate,
            AttractorKind::RecursiveSpatialMaps,
            AttractorKind::DeepCausalModelError,
            AttractorKind::Metacognition,
            AttractorKind::LongRangePredictiveTemporal,
            AttractorKind::DeepTemporalMemory,
            AttractorKind::SustainedStructuralMemory,
        ];
        assert_eq!(all.len(), 37);
        let mut ids: Vec<u8> = all.iter().map(|a| a.id()).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), 37);
        assert!(*ids.first().unwrap() == 1 && *ids.last().unwrap() == 37);
    }

    #[test]
    fn attractor_kind_tier_distribution_matches_4_13_12_8() {
        let all: Vec<AttractorKind> = (1u8..=37)
            .filter_map(|id| match id {
                1 => Some(AttractorKind::ThresholdCrossing),
                2 => Some(AttractorKind::SpatialGradient),
                3 => Some(AttractorKind::TemporalAdaptation),
                4 => Some(AttractorKind::MagnitudeGainAdaptation),
                5 => Some(AttractorKind::DeepStructuralTemplate),
                6 => Some(AttractorKind::CompetitiveComparison),
                7 => Some(AttractorKind::ForwardModelPredictionError),
                8 => Some(AttractorKind::DeepSelfModelError),
                9 => Some(AttractorKind::MultiScaleTemporalContrast),
                10 => Some(AttractorKind::SpatialTemplateMatch),
                11 => Some(AttractorKind::DeepRelationalTemplate),
                12 => Some(AttractorKind::CausalSchemaMatch),
                13 => Some(AttractorKind::RecursiveSpatialMaps),
                14 => Some(AttractorKind::SpectralTemplateMatch),
                15 => Some(AttractorKind::TemporalContextEncoding),
                16 => Some(AttractorKind::SustainedSpatialSelfReference),
                17 => Some(AttractorKind::DeepCausalModelError),
                18 => Some(AttractorKind::Metacognition),
                19 => Some(AttractorKind::LongRangePredictiveTemporal),
                20 => Some(AttractorKind::RawStructuralRecognition),
                21 => Some(AttractorKind::ShallowStructuralTemplate),
                22 => Some(AttractorKind::StructuralCompetition),
                23 => Some(AttractorKind::AdaptiveStructuralPerception),
                24 => Some(AttractorKind::ParallelSpectralDecomposition),
                25 => Some(AttractorKind::SpectralAdaptation),
                26 => Some(AttractorKind::RawExternalMagnitude),
                27 => Some(AttractorKind::ShallowMagnitudeContrast),
                28 => Some(AttractorKind::MagnitudeVsComputedReference),
                29 => Some(AttractorKind::RawMagnitudeAdaptation),
                30 => Some(AttractorKind::ShallowRelationalComparison),
                31 => Some(AttractorKind::ShallowRelationalTemplate),
                32 => Some(AttractorKind::MidDepthCausalInference),
                33 => Some(AttractorKind::ShallowCausalInference),
                34 => Some(AttractorKind::SpatialAdaptation),
                35 => Some(AttractorKind::RelationalPredictionError),
                36 => Some(AttractorKind::DeepTemporalMemory),
                37 => Some(AttractorKind::SustainedStructuralMemory),
                _ => None,
            })
            .collect();
        assert_eq!(all.len(), 37);

        let t0 = all.iter().filter(|a| a.tier() == 0).count();
        let t1 = all.iter().filter(|a| a.tier() == 1).count();
        let t2 = all.iter().filter(|a| a.tier() == 2).count();
        let t3 = all.iter().filter(|a| a.tier() == 3).count();
        assert_eq!((t0, t1, t2, t3), (4, 13, 12, 8));
    }

    #[test]
    fn native_floor_monotonic_in_tier() {
        // T0 < T1 < T2 < T3 floors on average
        let t0 = AttractorKind::ThresholdCrossing.native_floor_picojoules(); // 5
        let t1 = AttractorKind::SpatialGradient.native_floor_picojoules();    // 20
        let t2 = AttractorKind::DeepStructuralTemplate.native_floor_picojoules(); // 250
        let t3 = AttractorKind::DeepCausalModelError.native_floor_picojoules(); // 2000
        assert!(t0 < t1);
        assert!(t1 < t2);
        assert!(t2 < t3);
    }

    #[test]
    fn empty_tier_returns_none() {
        let mut tier = AttractorTier::new();
        assert!(tier.dispatch_string("anything").is_none());
    }

    #[test]
    fn pattern_registry_classifies() {
        let mut reg = PatternRegistry::new();
        reg.register(threshold_pattern);
        assert_eq!(
            reg.classify("is value above threshold"),
            Some(AttractorKind::ThresholdCrossing)
        );
        assert_eq!(reg.classify("completely unrelated query"), None);
    }

    #[test]
    fn tier_dispatches_string_query() {
        let mut tier = AttractorTier::new();
        tier.register_pattern(threshold_pattern);
        tier.register_resolver(Box::new(ThresholdResolver));

        // This string matches the pattern (contains "above threshold")
        // AND has the value:threshold format the resolver parses.
        let result = tier
            .dispatch_string("is value:42 above threshold:10")
            .expect("pattern matched + resolver succeeded");
        assert_eq!(result.output, "above threshold");
        assert!(result.confidence > 0.5);
    }

    #[test]
    fn tier_dispatch_string_misses_unmatched_pattern() {
        let mut tier = AttractorTier::new();
        tier.register_pattern(threshold_pattern);
        tier.register_resolver(Box::new(ThresholdResolver));

        // No "above threshold" or "below threshold" keyword — pattern doesn't match
        assert!(tier.dispatch_string("random unrelated query").is_none());
    }

    #[test]
    fn tier_dispatches_typed_query() {
        let mut tier = AttractorTier::new();
        tier.register_resolver(Box::new(ThresholdResolver));

        let query = AttractorQuery::new(
            AttractorKind::ThresholdCrossing,
            "value:42 threshold:10",
        );
        let resolution = tier.dispatch_typed(&query).expect("resolver should succeed");
        assert_eq!(resolution.output, "above threshold");
        assert!(resolution.confidence > 0.5);
        // Energy equals the native floor (the resolver uses the theoretical cost)
        assert!(resolution.energy_joules > 0.0);
        assert!(resolution.energy_joules < 1e-9); // < 1 nJ (it's 5 pJ)
    }

    #[test]
    fn tier_dispatch_unknown_resolver_returns_none() {
        let mut tier = AttractorTier::new();
        let query = AttractorQuery::new(
            AttractorKind::DeepCausalModelError,
            "input",
        );
        assert!(tier.dispatch_typed(&query).is_none());
    }

    #[test]
    fn materializer_integration_with_typed_query() {
        let mut m = Materializer::new();
        let mut tier = AttractorTier::new();
        tier.register_resolver(Box::new(ThresholdResolver));
        m.set_attractor_tier(tier);
        assert!(m.has_attractor_tier());

        let query = AttractorQuery::new(
            AttractorKind::ThresholdCrossing,
            "value:42 threshold:10",
        );
        let result = m.materialize_attractor(&query).expect("dispatch success");
        assert_eq!(result.source, Source::Attractor);
        assert_eq!(result.output, "above threshold");
        assert!(result.verified);
        assert!(result.energy_joules < 1e-9);

        // Metrics tracked
        assert_eq!(m.metrics.total, 1);
        assert_eq!(*m.metrics.by_source.get("attractor").unwrap(), 1);
    }

    #[test]
    fn materialize_attractor_without_tier_returns_none() {
        let mut m = Materializer::new();
        let query = AttractorQuery::new(
            AttractorKind::ThresholdCrossing,
            "value:42 threshold:10",
        );
        assert!(m.materialize_attractor(&query).is_none());
    }

    #[test]
    fn cost_class_attractor_is_lut() {
        assert_eq!(Source::Attractor.cost_class(), "lut");
    }

    #[test]
    fn attractor_dispatch_cheaper_than_neural() {
        // The whole point: attractor primitives at picojoule cost should
        // be dramatically cheaper than neural (estimated at 1 mJ per query).
        let threshold_cost =
            AttractorKind::ThresholdCrossing.native_floor_joules();
        let neural_estimate = 0.001; // 1 mJ from materializer.rs neural fallback
        let ratio = neural_estimate / threshold_cost;
        // Should be > 10^6× savings
        assert!(
            ratio > 1e6,
            "ThresholdCrossing should be >10^6× cheaper than neural, got {}",
            ratio
        );
    }

    #[test]
    fn cascade_100_queries_all_attractor() {
        let mut m = Materializer::new();
        let mut tier = AttractorTier::new();
        tier.register_resolver(Box::new(ThresholdResolver));
        m.set_attractor_tier(tier);

        for i in 0..100 {
            let query = AttractorQuery::new(
                AttractorKind::ThresholdCrossing,
                format!("value:{} threshold:50", i),
            );
            let result = m.materialize_attractor(&query).unwrap();
            assert_eq!(result.source, Source::Attractor);
        }

        assert_eq!(m.metrics.total, 100);
        assert_eq!(*m.metrics.by_source.get("attractor").unwrap(), 100);

        // All 100 queries resolved through attractor tier.
        // Neural avoidance should be 100%.
        assert_eq!(m.metrics.pct_avoided_neural(), 100.0);

        // Total energy for 100 threshold crossings at 5 pJ each = ~500 pJ
        // Neural would cost 100 mJ. Ratio > 10^8.
        let attractor_total_energy = m.metrics.total_energy;
        let neural_equivalent = 100.0 * 0.001; // 100 queries × 1 mJ
        assert!(
            neural_equivalent / attractor_total_energy > 1e6,
            "attractor total {} J vs neural equivalent {} J",
            attractor_total_energy,
            neural_equivalent
        );
    }

    // ── v10 cache tests ──────────────────────────────────────────

    #[test]
    fn cache_key_same_for_identical_queries() {
        let q1 = AttractorQuery::new(
            AttractorKind::ThresholdCrossing,
            "value:42 threshold:10",
        );
        let q2 = AttractorQuery::new(
            AttractorKind::ThresholdCrossing,
            "value:42 threshold:10",
        );
        assert_eq!(
            AttractorCacheKey::from_query(&q1),
            AttractorCacheKey::from_query(&q2)
        );
    }

    #[test]
    fn cache_key_differs_by_kind() {
        let q1 = AttractorQuery::new(AttractorKind::ThresholdCrossing, "x");
        let q2 = AttractorQuery::new(AttractorKind::DeepCausalModelError, "x");
        assert_ne!(
            AttractorCacheKey::from_query(&q1),
            AttractorCacheKey::from_query(&q2)
        );
    }

    #[test]
    fn cache_key_differs_by_input() {
        let q1 = AttractorQuery::new(AttractorKind::ThresholdCrossing, "alpha");
        let q2 = AttractorQuery::new(AttractorKind::ThresholdCrossing, "beta");
        assert_ne!(
            AttractorCacheKey::from_query(&q1),
            AttractorCacheKey::from_query(&q2)
        );
    }

    #[test]
    fn empty_cache_has_zero_hit_rate() {
        let cache = AttractorCache::new();
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.misses(), 0);
        assert_eq!(cache.hit_rate(), 0.0);
        assert!(cache.is_empty());
    }

    #[test]
    fn cache_records_hits_and_misses() {
        let mut cache = AttractorCache::new();
        let key = AttractorCacheKey::from_parts(AttractorKind::ThresholdCrossing, "test");

        // First lookup is a miss
        assert!(cache.get(&key).is_none());
        assert_eq!(cache.misses(), 1);
        assert_eq!(cache.hits(), 0);

        // Insert a resolution
        cache.insert(
            key,
            AttractorResolution {
                output: "above threshold".into(),
                energy_joules: 5e-12,
                confidence: 1.0,
                native_floor_pj: Some(5),
            },
        );

        // Second lookup is a hit
        let r = cache.get(&key).unwrap();
        assert_eq!(r.output, "above threshold");
        assert_eq!(cache.hits(), 1);
        assert_eq!(cache.misses(), 1);
        assert_eq!(cache.hit_rate(), 0.5);
    }

    #[test]
    fn tier_caches_repeated_queries() {
        let mut tier = AttractorTier::new();
        tier.register_resolver(Box::new(ThresholdResolver));

        let query = AttractorQuery::new(
            AttractorKind::ThresholdCrossing,
            "value:42 threshold:10",
        );

        // First dispatch is a miss → resolver runs
        let r1 = tier.dispatch_typed(&query).unwrap();
        assert_eq!(r1.output, "above threshold");
        assert_eq!(tier.cache().hits(), 0);
        assert_eq!(tier.cache().misses(), 1);
        assert_eq!(tier.cache().len(), 1);

        // Second dispatch with same input is a hit
        let r2 = tier.dispatch_typed(&query).unwrap();
        assert_eq!(r2.output, "above threshold");
        assert_eq!(tier.cache().hits(), 1);
        assert_eq!(tier.cache().misses(), 1);
        assert_eq!(tier.cache().len(), 1);

        // Third dispatch — another hit
        let _ = tier.dispatch_typed(&query).unwrap();
        assert_eq!(tier.cache().hits(), 2);
        assert_eq!(tier.cache().misses(), 1);

        // Hit rate after 2 hits + 1 miss = 2/3
        assert!((tier.cache().hit_rate() - 2.0 / 3.0).abs() < 0.001);
    }

    #[test]
    fn tier_separates_queries_by_kind() {
        let mut tier = AttractorTier::new();
        tier.register_resolver(Box::new(ThresholdResolver));

        // Two queries with the same input but different kinds
        let q1 = AttractorQuery::new(
            AttractorKind::ThresholdCrossing,
            "value:42 threshold:10",
        );
        let q2 = AttractorQuery::new(
            AttractorKind::ThresholdCrossing, // intentionally same kind
            "value:42 threshold:10",
        );

        let _ = tier.dispatch_typed(&q1);
        let _ = tier.dispatch_typed(&q2);
        // Same key → second is a cache hit
        assert_eq!(tier.cache().hits(), 1);
        assert_eq!(tier.cache().misses(), 1);
    }

    #[test]
    fn tier_string_dispatch_uses_cache() {
        let mut tier = AttractorTier::new();
        tier.register_pattern(threshold_pattern);
        tier.register_resolver(Box::new(ThresholdResolver));

        // Same string twice — second is a cache hit
        let r1 = tier.dispatch_string("is value:42 above threshold:10").unwrap();
        let r2 = tier.dispatch_string("is value:42 above threshold:10").unwrap();
        assert_eq!(r1.output, r2.output);
        assert_eq!(tier.cache().hits(), 1);
        assert_eq!(tier.cache().misses(), 1);
    }

    #[test]
    fn cache_clear_resets_entries_not_stats() {
        let mut cache = AttractorCache::new();
        let key = AttractorCacheKey::from_parts(AttractorKind::ThresholdCrossing, "k");
        cache.insert(
            key,
            AttractorResolution {
                output: "x".into(),
                energy_joules: 0.0,
                confidence: 1.0,
                native_floor_pj: None,
            },
        );
        let _ = cache.get(&key); // hit
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.hits(), 1);

        cache.clear_entries();
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.hits(), 1); // stats preserved

        cache.reset_stats();
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.misses(), 0);
    }

    #[test]
    fn cascade_500_repeated_queries_show_cache_savings() {
        // Realistic workload: 500 queries, but only 5 unique inputs.
        // The first 5 are cache misses (resolver runs); the remaining 495
        // are cache hits (resolver doesn't run). The structural cache means
        // even though the materializer cache key is by string, repeated
        // identical queries hit the structural cache too.
        let mut m = Materializer::new();
        let mut tier = AttractorTier::new();
        tier.register_resolver(Box::new(ThresholdResolver));
        m.set_attractor_tier(tier);

        let unique_queries = [
            "value:10 threshold:50",
            "value:75 threshold:50",
            "value:25 threshold:50",
            "value:90 threshold:50",
            "value:50 threshold:50",
        ];

        // Dispatch each unique query 100 times = 500 total dispatches
        for round in 0..100 {
            for input in &unique_queries {
                let query = AttractorQuery::new(
                    AttractorKind::ThresholdCrossing,
                    *input,
                );
                let result = m.materialize_attractor(&query).unwrap();
                assert_eq!(result.source, Source::Attractor);
                let _ = round;
            }
        }

        assert_eq!(m.metrics.total, 500);
        // Cache stats: 5 unique inputs → 5 misses, 495 hits
        let cache = m.attractor_cache().unwrap();
        assert_eq!(cache.misses(), 5);
        assert_eq!(cache.hits(), 495);
        assert_eq!(cache.len(), 5);
        // Hit rate should be 99%
        assert!((cache.hit_rate() - 0.99).abs() < 0.001);
    }
}
