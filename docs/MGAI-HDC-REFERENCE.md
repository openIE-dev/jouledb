# MGAI HDC Reference

**Version 1.0 — 2026-05-18**
**Crate:** [`joule-db-hdc`](../crates/joule-db-hdc/)
**Primitive APIs:** [`crates/joule-db-hdc/src/primitives/`](../crates/joule-db-hdc/src/primitives/)
**Sister doc:** [`crates/joule-db-hdc/PRIMITIVES.md`](../crates/joule-db-hdc/PRIMITIVES.md) — primitive-level design notes

---

## 1. What HDC is in the OpenIE stack

Hyperdimensional Computing (HDC) is the energy-efficient substrate of the OpenIE stack. Where transformers run gradient descent over learned embeddings, HDC operates on **10,000-dimensional binary or bipolar hypervectors** with four canonical operations — bind, bundle, permute, similarity — extended in the JouleDB implementation to **15 irreducible primitives**.

The 15-primitive framework is the HDC equivalent of PyTorch's primitive set: an irreducible vocabulary that everything else decomposes into. PyTorch has 6 (`tensor`, `autograd`, `nn.Module`, `optim`, `DataLoader`, `loss`). HDC has 15 because contrast-recognition needs primitives that gradient descent doesn't have (Route, Forget, Reflect, Test, Spawn, Inhibit, Coarsen, Synchronize).

**Headline benchmark (WN18RR knowledge-graph link prediction):**
- **83.8% Hits@10** (28 pts above SOTA)
- **MRR 0.468** (matches PathHD)
- **9 µJ per prediction** (≈10⁶× more efficient than GPU inference)
- **Zero training** (starts empty, reads live)

See [`WHITEPAPER-JOULEDB-2026-05.md`](WHITEPAPER-JOULEDB-2026-05.md) §3 for full results.

---

## 2. The 15 primitives

### 2.1 Quick reference

| # | Primitive | Implementation | Energy class | What transformers lack (limitation # from whitepaper §1.2) |
|---|---|---|---|---|
| 1 | **Encode** | `BinaryHV::from_data()`, trigram, hash, structural | ~µJ | — (transformers encode via learned embeddings) |
| 2 | **Bind** | XOR, MAP, Fourier, GHRR block-circular | ~ns / op | #6 No compositionality |
| 3 | **Permute** | Circular bit shift | ~ns / op | #9 No causal model (symmetry) |
| 4 | **Compare** | Hamming distance, cosine (18 ns) | ~ns / op | — |
| 5 | **Route** | `MetabolicController`: 4 states, 6 orders of magnitude | varies by route | #3 Fixed compute per token |
| 6 | **Remember** | `TurboHolographic`, SDM write | ~µJ / write | #1 No real memory |
| 7 | **Forget** | Half-life decay, decoherence | passive | #7 Catastrophic forgetting |
| 8 | **Reflect** | Subject XOR context → self-model | ~µJ | #8 No self-model |
| 9 | **Merge / Un-merge** | `BundleAccumulator`, resonator factorize | ~µJ | #6 No compositionality |
| 10 | **Update** | In-place bind, bundle | ~ns / op | #4 No learning at inference |
| 11 | **Test** | Verify orthogonality, `HologramHealth` | ~µJ | #16 No error correction |
| 12 | **Spawn** | Autonomous entity + clonal expansion | ~µJ + lifetime | #15 No multi-agent |
| 13 | **Inhibit** | Winner-take-all via resonator cleanup | ~µJ | #13 No negative knowledge |
| 14 | **Coarsen** | XOR fold, block majority, pyramid | ~µJ | #14 No abstraction hierarchy |
| 15 | **Synchronize** | Barrier + `SyncGroup` | ~ns | #15 No multi-agent |

### 2.2 What each primitive does

#### 1. Encode — `joule-db-hdc::binary_hd`, `ternary_hv`, `hyper`

Maps reality into the algebra. Entry point — nothing exists in the algebra until encoded.

```rust,ignore
use joule_db_hdc::BinaryHV;
let hv = BinaryHV::random(10_000, seed);
let hv_from_text = BinaryHV::from_data(b"apple");
```

Irreducible: every other primitive operates on encoded hypervectors. There's no algebra-internal way to create one from nothing.

#### 2. Bind — XOR / MAP / Fourier / GHRR block-circular

Creates structured relationships. In binary HDC, `a ⊕ b` ("a bound to b").

```rust,ignore
let red_apple = apple.bind(&red);
```

Irreducible: many VSAs make bind symmetric (`A·B = B·A`); to break symmetry you need a separate primitive (Permute).

#### 3. Permute — Circular bit shift

Imposes order / sequence. Breaks the symmetry of Bind.

```rust,ignore
let position_1 = apple.permute(1);  // "apple in position 1"
let sequence = position_1.bind(&banana.permute(2));  // "apple then banana"
```

Irreducible: without Permute, "dog bites man" = "man bites dog".

#### 4. Compare — Hamming distance, cosine (18 ns)

Detects contrast (similarity / distance). Measurement without modification.

```rust,ignore
let similarity = a.similarity(&b);   // 0.5 = random; 1.0 = identical; 0.0 = opposite
let distance = a.hamming_distance(&b);
```

Irreducible: every other primitive either creates, modifies, or routes; only Compare measures.

#### 5. Route — `MetabolicController` (4 states, 6 orders of magnitude)

Allocates compute proportional to novelty. The system's metabolism — decides how much energy to spend on a given input.

States: Resting (~1 µJ), Walking, Running, Surge (~5 J). Selection is contrast-driven: trivial / repeated inputs stay in Resting; novel inputs escalate.

Irreducible: every other primitive performs work; only Route decides *how much* work.

#### 6. Remember — `TurboHolographic`, SDM write

Stores without retraining. Associative write — one-shot, no gradient update.

```rust,ignore
store.put(key_hv, value_hv);  // single write, no training loop
let recovered = store.get(key_hv);
```

Irreducible: gradient-trained networks rebuild associations through backprop. HDC associative write is structurally different — one shot, no loss function.

#### 7. Forget — Half-life decay, decoherence

Natural decoherence / decay. Passive — information fades without active maintenance, matching biological memory.

Implemented in [`primitives/forget.rs`](../crates/joule-db-hdc/src/primitives/forget.rs) — `Forgettable`, `DecayRate`, `Decay` types. 5 tests.

Irreducible: without Forget, the system either keeps everything (catastrophic accumulation) or actively deletes (which is a different operation requiring a deletion policy).

#### 8. Reflect — Subject XOR context → self-model

Applies the system to its own output. Metacognition — contrast on contrast.

Implemented in [`primitives/reflect.rs`](../crates/joule-db-hdc/src/primitives/reflect.rs) — `Reflectable`, `Reflection` types. 5 tests.

Irreducible: Compare is outward-facing (compare two external things). Reflect is inward-facing (compare the system's state to itself). Self-model emerges from repeated reflection.

#### 9. Merge / Un-merge — `BundleAccumulator`, resonator factorize

Combines representations, decomposes back into parts. Bundle algebra — superposition and factorization.

```rust,ignore
let combined = BundleAccumulator::new(10_000);
combined.add(&apple);
combined.add(&banana);
combined.add(&cherry);
let bundle = combined.threshold();  // "apple OR banana OR cherry"

// Un-merge: given the bundle and one component, recover the others
let resonator = Resonator::new(codebook);
let factors = resonator.factorize(&compound_hv);
```

Irreducible: gradient-trained models entangle features; HDC bundles preserve constituent identity (high similarity to each member, ~0 similarity to non-members).

#### 10. Update — In-place bind, bundle

Modifies existing representation in-place. Mutation without re-encoding from scratch.

Irreducible: reconstructing from Encode would lose accumulated state. In-place Update preserves history.

#### 11. Test — `verify_orthogonality`, `HologramHealth`

Verifies whether a representation still holds. Inward-facing contrast (Compare is outward-facing).

Irreducible: Compare answers "are these similar?"; Test answers "is this still valid?" The second question requires knowing what valid means, which is system-internal.

#### 12. Spawn — Autonomous entity + clonal expansion

Creates a new independent computational entity. Used for multi-agent setups, clonal expansion in immune-system models, parallel decomposition.

Implemented in [`primitives/spawn.rs`](../crates/joule-db-hdc/src/primitives/spawn.rs) — `Spawner`, `SpawnedEntity` types. 5 tests.

Irreducible: none of primitives 1–11 produce an autonomous process. They all run inside one execution context.

#### 13. Inhibit — Winner-take-all via resonator cleanup

Active competitive suppression. Operationalizes negative knowledge ("NOT X" as suppression, not just absence).

Irreducible: Compare measures distance but doesn't suppress. Route allocates compute but doesn't kill losers. Inhibit is the only primitive that says "this representation is actively wrong."

#### 14. Coarsen — XOR fold, block majority, pyramid

Scale-changing lossy projection. Abstraction.

Implemented in [`primitives/coarsen.rs`](../crates/joule-db-hdc/src/primitives/coarsen.rs) — `Coarsenable`, `CoarsenedView`, `CoarsenStrategy` types. 6 tests.

Irreducible: Merge combines at the same scale; Coarsen *changes* the scale. The output lives in a smaller-dimensional space than the input.

#### 15. Synchronize — Barrier + `SyncGroup`

Temporal alignment / coordination barrier. Coordinates *when*, not *what*.

Implemented in [`primitives/sync.rs`](../crates/joule-db-hdc/src/primitives/sync.rs) — `Synchronizable`, `Barrier`, `SyncGroup` types. 6 tests.

Irreducible in concurrent systems: every other primitive can run independently; Synchronize is the only one that requires multiple agents to be at the same point.

---

## 3. What is NOT a primitive

A common challenge: "you missed X." For each candidate, X decomposes into compositions of existing primitives:

| Candidate | Decomposition |
|---|---|
| Predict | Reflect + Compare |
| Attention | Compare + Route + Merge |
| Communicate | Encode + Route + Merge |
| Mutate / Explore | Update + Encode(noise) |
| Lambda abstraction | Encode + Bind |
| Oscillate | Inhibit + Update (cyclic circuit) |
| Gate | Route (binary) or Inhibit (graded) |
| Restrict / Scope | Spawn + Bind |

After ~15, new candidates consistently decompose. The framework is self-improving in this sense: Reflect + Test + Update applied to the primitive set itself discovers redundancies and gaps. The set stabilises around 15.

---

## 4. Implementation status

All 15 are implemented. Test counts per primitive:

| # | Primitive | Module path | Tests |
|---|---|---|---|
| 1 | Encode | `binary_hd.rs`, `ternary_hv.rs`, `hyper/` | (covered by core hypervector tests) |
| 2 | Bind | `binary_hd.rs`, `map_bind.rs`, `fourier_bind.rs` | (covered) |
| 3 | Permute | `binary_hd.rs`, `map_bind.rs` | (covered) |
| 4 | Compare | `binary_hd.rs`, `ternary_hv.rs`, `hyper/` | (covered) |
| 5 | Route | `primitives/route.rs`, `metabolic.rs`, `selector.rs` | (covered) |
| 6 | Remember | `turbo_holographic.rs`, `sdm.rs` | (covered) |
| 7 | Forget | [`primitives/forget.rs`](../crates/joule-db-hdc/src/primitives/forget.rs) | 5 |
| 8 | Reflect | [`primitives/reflect.rs`](../crates/joule-db-hdc/src/primitives/reflect.rs) | 5 |
| 9 | Merge / Un-merge | `binary_hd.rs` (`bundle`), `resonator.rs` | (covered) |
| 10 | Update | `binary_hd.rs`, `turbo_holographic.rs` | (covered) |
| 11 | Test | `binary_hd.rs::verify_orthogonality`, `HologramHealth` | (covered) |
| 12 | Spawn | [`primitives/spawn.rs`](../crates/joule-db-hdc/src/primitives/spawn.rs) | 5 |
| 13 | Inhibit | `resonator.rs` (cleanup), `spiking/` | (covered) |
| 14 | Coarsen | [`primitives/coarsen.rs`](../crates/joule-db-hdc/src/primitives/coarsen.rs) | 6 |
| 15 | Synchronize | [`primitives/sync.rs`](../crates/joule-db-hdc/src/primitives/sync.rs) | 6 |

The five primitives with their own dedicated `primitives/*.rs` file (Forget, Reflect, Spawn, Coarsen, Synchronize) were extracted as first-class APIs because they weren't naturally expressed by the existing HDC primitive surface.

---

## 5. Design principles

1. **Orthogonality is a guide, not a law.** Reality has domain overlap. Mathematicians demand orthogonal; engineers don't. Resilience over elegance.
2. **Self-improving.** The 15 aren't a closed set — extensible. But diminishing returns after ~15 — new candidates decompose into compositions.
3. **Energy-proportional.** Every primitive has a known energy cost at the Landauer floor. The framework tracks joules, not just FLOPs.
4. **One-shot.** No primitive requires a training loop. Encode once, bind once, remember once. Contrast is detected, not learned.

---

## 6. Related concepts in academic literature

| System | What it has | What it lacks vs. JouleDB's 15 |
|---|---|---|
| **Academic HDC libraries** (torchhd, hdlib, HDTorch) | Encode, Bind, Bundle, Compare | All of: Route, Forget, Reflect, Test, Spawn, Inhibit, Coarsen, Synchronize |
| **HPVM-HDC** (ISCA 2025) | 27 parameterized variants of the same 4 | Same as above — variants, not new primitives |
| **PathHD** | Knowledge-graph reasoning via HDC paths | Implements 4 primitives + path encoding. No knowledge system, no metabolic Route. |
| **HDReason** (FPGA HDC) | Encode, Bind, Compare in hardware | Hardware specialization, not a primitive-level alternative |
| **GHRR** (block-circular HRR) | Non-commutative Bind | Subset — one new variant of Bind |
| **Spiking neural networks** | Temporal coding + winner-take-all | Provides hardware models for Inhibit and Synchronize, doesn't define a primitive vocabulary |

JouleDB's 15 primitives subsume all of the above. No equivalent framework exists at this primitive count.

---

## 7. Usage example — end-to-end

```rust,ignore
use joule_db_hdc::{BinaryHV, BundleAccumulator};
use joule_db_hdc::primitives::{forget::DecayRate, reflect::Reflectable};

// 1. Encode
let apple = BinaryHV::random(10_000, 42);
let red = BinaryHV::random(10_000, 43);
let yellow = BinaryHV::random(10_000, 44);

// 2. Bind — create relationships
let red_apple = apple.bind(&red);

// 9. Merge — combine multiple
let mut acc = BundleAccumulator::new(10_000);
acc.add(&red_apple);
acc.add(&apple.bind(&yellow));
let apples = acc.threshold();

// 4. Compare — measure
let sim_to_red_apple = apples.similarity(&red_apple);  // ~0.7
let sim_to_unrelated = apples.similarity(&BinaryHV::random(10_000, 99));  // ~0.5

// 7. Forget — let it decay
let decayed = apples.forget(DecayRate::halflife(1.0));

// 8. Reflect — apply to own output
let self_model = apples.reflect();
```

---

## 8. See also

- [`joule-db-hdc/README.md`](../crates/joule-db-hdc/README.md) — crate-level overview
- [`joule-db-hdc/PRIMITIVES.md`](../crates/joule-db-hdc/PRIMITIVES.md) — primitive-level design notes
- [`WHITEPAPER-JOULEDB-2026-05.md`](WHITEPAPER-JOULEDB-2026-05.md) §2.3 — primitives in the architectural narrative
- [`jouledb-ai-runtime`](../crates/jouledb-ai-runtime/) — the Awareness Engine cascade built on these primitives
- [`joule-db-domains`](../crates/joule-db-domains/) — 22 domain-specific encoders that target these primitives
- [`joule-db-ternary`](../crates/joule-db-ternary/) — packed ternary encoding for some hypervector representations

---

*Drafted 2026-05-18 as wave 2 of the JouleDB documentation parity pass. Closes Open Item §10.4 in [`MGAI-SPEC-DOMAIN-JOULEDB.md`](MGAI-SPEC-DOMAIN-JOULEDB.md).*
