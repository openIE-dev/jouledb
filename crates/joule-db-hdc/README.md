# joule-db-hdc

Hyperdimensional Computing (HDC) library for high-performance similarity search and the substrate for JouleDB's energy-efficient knowledge layer.

## What it is

`joule-db-hdc` is the HDC / Vector Symbolic Architecture engine for JouleDB. It treats data as 10,000-dimensional binary or bipolar hypervectors and provides the four canonical operations — **bind**, **bundle**, **permute**, **similarity** — that make HDC a one-shot, content-addressable, distributed alternative to gradient-trained embeddings.

Used by [`jouledb-ai-runtime`](../jouledb-ai-runtime/), [`joule-db-amorphic`](../joule-db-amorphic/), [`joule-db-domains`](../joule-db-domains/), and [`joule-db-features`](../joule-db-features/).

## Core operations

| Operation | What it does |
|---|---|
| **Bind (XOR)** | Creates an association between two concepts. `red ⊕ apple = "red apple"` |
| **Bundle (majority vote)** | Combines multiple vectors into one. Sum + threshold. |
| **Permute (rotate)** | Encodes sequence / position information |
| **Similarity (Hamming)** | Measures how related two vectors are |

## Quickstart

```rust,ignore
use joule_db_hdc::{BinaryHV, BundleAccumulator};

let apple = BinaryHV::random(10000, 42);
let red = BinaryHV::random(10000, 43);

// Bind creates the "red apple" association
let red_apple = apple.bind(&red);

// Bundle multiple items with majority vote
let mut acc = BundleAccumulator::new(10000);
acc.add(&apple);
acc.add(&red_apple);
let combined = acc.threshold();

// High similarity = related concepts
println!("Similarity: {}", combined.similarity(&apple));
```

## Architecture

A unified VSA combining:

- **Binary hypervectors** — 10,000-D vectors with XOR binding
- **Holographic memory** — distributed pattern storage with superposition
- **Sparse distributed memory (SDM)** — Kanerva-style content-addressable memory
- **Predictive prefetching** — Markov / n-gram query prediction

## Feature flags — extended HDC components

| Feature | Default | Purpose |
|---|---|---|
| `holographic` | — | VSA / holographic storage |
| `sdm` | — | Sparse distributed memory |
| `spiking` | off | Spiking neural networks — temporal data processing |
| `neurosymbolic` | off | Neural + symbolic reasoning hybrid |
| `invertible` | off | Invertible encodings — data embedded in visualizations |
| `pqc` | off | Post-quantum crypto — NIST FIPS 203/204/205, HQC KEM |
| `thermodynamic` | off | Thermodynamic optimizer — SA-based query optimization |
| `manifold` | off | Information manifold — geodesic similarity |
| `learned` | off | Learned indexes — ML-optimized data access, Neural LSH |
| `hdc-research` | off | Full suite — Amorphic Engine + all research modules |

## Holographic key-value store

A probabilistic KV store where `put` / `get` are vector binding / unbinding rather than direct addressing.

```rust,ignore
use joule_db_hdc::holographic_kv::{HolographicKV, HolographicKVConfig};

let config = HolographicKVConfig::default();
let mut store: HolographicKV<4096> = HolographicKV::new(config);

store.put(b"user:123", b"Alice").unwrap();
let val = store.get(b"user:123").unwrap(); // fuzzy retrieval supported
```

## Design principles

1. **Pure Rust** — no WASM dependencies in core
2. **Feature-gated** — enable only what you need
3. **Platform-agnostic** — native and WASM both work

## See also

- [PRIMITIVES.md](PRIMITIVES.md) — primitive-level reference
- `docs/MGAI-HDC-REFERENCE.md` *(in progress)* — the 15 irreducible primitives, user-facing
- [joule-db-ternary](../joule-db-ternary/) — packed ternary encoding used by some hypervector representations
- [joule-db-domains](../joule-db-domains/) — domain-specific encoders that target this substrate
