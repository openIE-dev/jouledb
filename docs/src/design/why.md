# Why JouleDB: Prior Art, Architecture, Evaluation & Licensing

Data movement across disconnected infrastructure silos—OLTP database, columnar warehouse,
vector index, property-graph engine, container daemon, and external LLM weight servers—consumes
orders of magnitude more joules and tail latency than arithmetic inside an ALU. JouleDB
consolidates transactional storage, hybrid retrieval, and sub-1-bit `.jsyn` neural execution
onto a single Copy-on-Write Serializable Snapshot Isolation (`SSI`) substrate written in
Rust (`41` workspace member crates / `737,265` LOC in [`crates/`](https://github.com/openIE-dev/jouledb/tree/main/crates), `v0.10.0` pre-1.0 preview).

This chapter states the prior art JouleDB builds on, where established systems outperform
JouleDB, the exact `1..15` hyperdimensional computing (`HDC`) primitives, benchmark protocols
(including where `Hits@1` drops sharply relative to `Hits@10`), and licensing/packaging history.

---

## 1. The Core Problem & Prior Art: Data Movement vs. Arithmetic Energy

Energy-aware data management and low-bitwidth inference build on a deep systems literature:

- **Whole-System Sorting & I/O Energy — JouleSort (Rivoire, Shah, Ranganathan & Kozyrakis,
  *SIGMOD* 2007):** Introduced whole-system joules-per-record sorting benchmarks and showed
  that storage I/O, DRAM traffic, and system base power dominate raw CPU arithmetic.
- **Energy-Proportional Data Systems (Harizopoulos, Shah, Meza & Ranganathan, *CIDR* 2009):**
  Framed software-level query energy optimization and hardware energy proportionality as
  first-class database architecture objectives.
- **Why "Fastest = Most Energy-Efficient" on Conventional Servers (Tsirogiannis, Harizopoulos
  & Shah, *SIGMOD* 2010):** Demonstrated experimentally across DBMS operators that on
  conventional servers with high static/idle socket power, **the fastest execution plan is
  almost always the most energy-efficient plan**, because keeping the package awake longer
  outweighs small dynamic-power differences between join or scan operators.
- **Ternary Weight Compression — BitNet b1.58 (Ma et al., 2024):** Showed that ternary
  $\{-1, 0, +1\}$ weight matrices replace floating-point multiply-accumulate (`FP16 MAC`) with
  integer addition while drastically reducing weight-fetch memory bandwidth.

### Where Tsirogiannis et al. (2010) Governs — and Where JouleDB Operates Differently

Inside a single relational scan, join, or aggregation on an active server, **Tsirogiannis et
al.'s result holds directly for JouleDB**: morsel-driven `2,048`-row vectorization, `ZoneMap`
cold-granule skipping (`97.96%` skip on `ClickBench`-style filters), and `SIMD POPCNT` save
energy primarily by minimizing wall-clock time.

JouleDB targets three cross-layer regimes outside single-engine SQL plan selection:

1. **Eliminating Cross-Service Serialization & IPC:** Collapsing a multi-service stack
   (`PostgreSQL -> JSON/Arrow -> Vector Store -> Graph Engine -> HTTP LLM Server`) into one
   shared-memory `SSI` snapshot removes repeated UTF-8 JSON marshaling, socket round-trips,
   and duplicated OS page caches.
2. **Multi-Tier Algorithmic Complexity Cascade:** Rather than running a fixed-cost neural
   pass on every request, JouleDB routes queries through an escalating cost cascade:
   `LSN LangCache (~95 nJ est) -> Engram O(1) exact lookup (~220 nJ est) -> TriModal RRF
   (BM25 ⋈ RaBitQ 1-bit POPCNT ⋈ HDC, ~1–10 µJ) -> .jsyn Living Bitplanes (mJ–J)`.
   Short-circuiting repeated or exact factual lookups at `O(1)` avoids waking GPU/NPU rails
   at all—reducing work by $10^3\times$–$10^6\times$ rather than tuning operator watts by 10%.
3. **Sub-1-bit `.jsyn` (`LittleBit-2 ⊕ Sakana SVF`) Memory-Traffic Reduction (Within Rank Limits):**
   Compressing compact bidirectional encoders (`all-MiniLM-L6-v2-LittleBit` at `0.44 BPW`, `22.2×` smaller)
   preserves `0.9441` cosine similarity, whereas ultra-low `0.02–0.17 BPW` settings severely degrade
   autoregressive LLMs (`cos = 0.27–0.45`) and are not drop-in LLM replacements.

---

## 2. Baseline Positioning & When *Not* to Use JouleDB

| System | Core Architecture | Primary Sweet Spot | Where It Beats JouleDB | Where JouleDB Fits Better |
|---|---|---|---|---|
| **DuckDB** (Raasveldt & Mühleisen, *SIGMOD* 2019) | Embedded vectorized columnar OLAP (`C++`) | In-process analytics over Parquet, CSV, and Arrow | Richer analytical SQL optimizer, mature extension ecosystem, fastest pure single-node Parquet OLAP | Multi-client `SSI` server + `pgwire` + CoW branching + integrated `GraphRAG` and `.jsyn` weights |
| **ClickHouse** | Shared-nothing distributed columnar OLAP | Petabyte-scale append-heavy logs, traces, and BI | Horizontal multi-node merge-tree ingestion and petabyte aggregation throughput | Transactional `SSI` point updates, `1.13 µs` CoW forks, and zero-ETL hybrid retrieval |
| **PostgreSQL + `pgvector`** | Extensible MVCC relational OLTP + HNSW / IVFFlat | General-purpose production OLTP with moderate vector search | Decades of production hardening, logical replication, extensions (`PostGIS`, FDWs), managed cloud HA | `32x` smaller `RaBitQ` 1-bit vectors + `BM25` + `HDC` + in-engine `.jsyn` inference without sidecars |
| **Standalone `llama.cpp` / `vLLM`** | Dedicated `GGUF` (`C++`) or PagedAttention (`CUDA`) inference servers | High-throughput dense/quantized LLM token generation | Higher tokens/sec on `FP8`/`Q4_K_M` long-form creative generation and multi-GPU tensor parallelism | Sub-1-bit `.jsyn` circuits coupled to exact SQL/`Engram` facts with per-request energy receipts |

### When to Choose DuckDB, PostgreSQL, ClickHouse, or `vLLM` Instead of JouleDB

- **Use DuckDB** for embedded analytical SQL over local Parquet/Arrow datasets in Python, R, or CLI scripts.
- **Use ClickHouse** for multi-terabyte or petabyte distributed event/observability warehousing.
- **Use PostgreSQL + `pgvector`** when you need battle-tested relational OLTP, full PostgreSQL extension compatibility, mature logical replication, or a managed cloud SLA.
- **Use `vLLM` or `llama.cpp`** when your primary workload is high-concurrency open-ended text generation over multi-GPU `BF16`/`FP8`/`Q4` models without database state coupling.

---

## 3. The 15 Core HDC Primitives (Explicitly Enumerated `1..15`)

Earlier summary prose referred to *"15 irreducible primitives"* while naming only 14 operations
(omitting `Synchronize` or counting `Merge / Un-merge` inconsistently). In
[`crates/joule-db-hdc`](https://github.com/openIE-dev/jouledb/tree/main/crates/joule-db-hdc)
(see [`PRIMITIVES.md`](https://github.com/openIE-dev/jouledb/blob/main/crates/joule-db-hdc/PRIMITIVES.md)
and [`MGAI-HDC-REFERENCE.md`](https://github.com/openIE-dev/jouledb/blob/main/docs/MGAI-HDC-REFERENCE.md)),
the four classical Vector Symbolic Architecture (`VSA`) operations (`Encode`, `Bind`, `Permute`,
`Compare`) are extended to the following **1-to-1 numbered list of 15 primitives**:

| # | Primitive | Algebraic Role | Rust Module (`joule-db-hdc` / `joule-db-amorphic`) | Coverage |
|---:|---|---|---|---|
| 1 | **Encode** | Project scalars, tokens, or structs into `10,000`-bit hypervectors | `binary_hd`, `ternary_hv`, `holo.rs` | Core |
| 2 | **Bind** | Associative/relational binding (`XOR`, `MAP`, Fourier, `GHRR` block-circular) | `binary_hd`, `map_bind`, `fourier_bind` | Core |
| 3 | **Permute** | Cyclic bit-shift imposing sequence and causal asymmetry (`A -> B != B -> A`) | `binary_hd`, `map_bind` | Core |
| 4 | **Compare** | Non-destructive similarity measurement (`SIMD` Hamming / cosine distance) | `binary_hd`, `ternary_hv`, `contrast.rs` | Core |
| 5 | **Route** | Novelty-proportional compute allocation across metabolic tiers | `amorphic::metabolic`, `selector.rs` | Core |
| 6 | **Remember** | One-shot associative write (`SDM` / holographic memory, `0` gradient steps) | `turbo_holographic`, `sdm`, `memory.rs` | Core |
| 7 | **Forget** | Passive half-life decay / temporal decoherence without full table scans | `primitives/forget.rs` (`Forgettable`, `Decay`) | Domain-scoped |
| 8 | **Reflect** | Second-order residual self-comparison (`state XOR context`) | `primitives/reflect.rs`, `variational.rs` | Domain-scoped |
| 9 | **Merge / Un-merge** | Majority-vote bundle superposition paired with resonator factorization | `binary_hd` (`BundleAccumulator`), `resonator` | Core (dual pair) |
| 10 | **Update** | In-place incremental hypervector delta mutation without full re-encoding | `binary_hd`, `turbo_holographic`, `tx.rs` | Core |
| 11 | **Test** | Structural orthogonality and capacity health check (`HologramHealth`) | `binary_hd`, `amorphic::partition` | Core |
| 12 | **Spawn** | Fork an isolated autonomous entity / sub-workspace | `primitives/spawn.rs` (`Spawner`, `SpawnedEntity`) | Domain-scoped |
| 13 | **Inhibit** | Active winner-take-all competitive cleanup suppression | `resonator`, `spiking` | Core |
| 14 | **Coarsen** | Scale-changing lossy projection (`XOR` fold / block majority hierarchy) | `primitives/coarsen.rs`, `quantization.rs` | Domain-scoped |
| 15 | **Synchronize** | Multi-entity causal/temporal barrier coordination (`Barrier`, `SyncGroup`) | `primitives/sync.rs`, `spiking/temporal.rs` | Domain-scoped |

*Engineering note:* As noted in `PRIMITIVES.md`, "irreducible" denotes the lowest-level API
vocabulary in `joule-db-hdc` rather than a strict mathematical independence proof; composite
operators (`Attention = Compare + Route + Merge`, `Predict = Reflect + Compare`) are built on
top of `1..15` without backpropagation.

---

## 4. Evaluation Protocols, Metrics & Orthogonal Evidence Classes

### 4.1 Energy Evidence Taxonomy (`measured_j ⊥ reported_j ⊥ est_j ⊥ unmetered`)

JouleDB enforces four mutually orthogonal energy classes—values from distinct classes must
never be summed, averaged, or relabeled:

- **`measured_j` (External Inline Meter):** Physical DC rail shunt (`INA238` / `INA3221`) or
  calibrated AC wall analyzer. **JouleDB software alone never emits `measured_j`** unless an
  external hardware meter is attached.
- **`reported_j` (On-Chip / OS Hardware Counter):** Kernel/PMU counters on bare-metal hosts:
  Apple Silicon `IOReport` / `proc_pid_rusage(RUSAGE_INFO_V6).ri_energy_nj` on macOS and
  `/sys/class/powercap/intel-rapl*` (Intel & AMD RAPL) on Linux. Socket/package counters
  reflect co-tenant load unless isolated via per-PID counters or idle-drift checks.
- **`est_j` (Analytical Silicon Ledger):** Deterministic instruction + cache/DRAM byte-movement
  cost model (`crates/joule-energy`, e.g., `~95 nJ` `LangCache`, `~220 nJ` `Engram`) used for
  sub-microsecond single-operator attribution or inside `WASM` / unprivileged containers.
- **`unmetered` (`joules: null`):** Exact wall-clock nanoseconds, CPU time, and byte counts
  when no hardware energy counter is readable without root.

### 4.2 `WN18RR` / `PathHD` Filtered Link-Prediction Protocol & `Hits@1` vs. `Hits@10`

Knowledge-graph link prediction is evaluated on **`WN18RR`** (`40,943` entities, `11` lexical
relations, `86,835` train / `3,034` validation / `3,134` test triples) under the **standard
filtered ranking protocol (Bordes et al., *NeurIPS* 2013)**:

1. For each test triple $(h, r, t)$, evaluate both tail corruption $(h, r, t')$ and head
   corruption $(h', r, t)$ across all $40,943$ entities (`6,268` ranked queries total) with
   zero gradient updates.
2. Remove every corrupted candidate $(h, r, t') \neq (h, r, t)$ that appears in
   `train ∪ valid ∪ test` (*filtered* setting) so known true facts do not artificially demote
   the ground-truth target.
3. Break ties deterministically (strict/mean rank, never optimistic rank-1 ties) and report
   **both `Hits@1` and `Hits@10`** alongside `MRR`:

| Engine Generation | Structural Representation | `WN18RR` `Hits@1` | `WN18RR` `Hits@10` | `MRR` | Energy / Pred (`reported_j`) | Relation-Level Behavior & Limitation |
|---|---|---:|---:|---:|---:|---|
| **1-Hop `PathStore` (`v0.1`–`v0.2`)** | 1-hop bundle superposition (`WHITEPAPER-2026-04/05`) | **`26.4%`** | `83.8%` | `0.4680` | `~9 µJ` | Places right neighborhood in top-10, **but `Hits@1` drops to `26.4%` overall** (`11%` on `_hypernym` `IsA`; `0%` on `_has_part` `PartOf`) due to `1-to-N` superposition ambiguity |
| **3-Hop `PathHD` `GHRR` (`v0.3`–`v0.4`)** | Non-commutative 3-hop `GHRR` block-circular path composition + degree/type constraints | **`70.0%`** | **`88.1%`** | `0.7628` | `~10 µJ` | Resolves multi-hop asymmetric composition; high-fanout `1-to-N` relations without relational SQL verification still exhibit a `Hits@1`-vs-`Hits@10` gap |

**Reporting rule:** `Hits@10` must never be cited without `Hits@1`. Workloads requiring exact
rank-1 factual recall should query MVCC SQL / `Engram` keys directly or apply relational
constraints over `PathHD` candidates.

---

## 5. Licensing, `crates.io` Packaging & Repository Availability Disclosure

1. **Complete Open-Source Workspace in `github.com/openIE-dev/jouledb` (`Apache-2.0 OR MIT`):**
   All **41 Rust workspace member crates** (`crates/*`, `737,265` LOC across `986` `.rs` files, with `29 / 41` crates enforcing `#![cfg_attr(not(test), deny(clippy::unwrap_used))]`), Python bindings (`python/`), examples, and docs in this
   repository are dual-licensed under **`Apache-2.0 OR MIT`** ([`LICENSE-APACHE`](https://github.com/openIE-dev/jouledb/blob/main/LICENSE-APACHE),
   [`LICENSE-MIT`](https://github.com/openIE-dev/jouledb/blob/main/LICENSE-MIT),
   [`COPYRIGHT`](https://github.com/openIE-dev/jouledb/blob/main/COPYRIGHT)) with zero
   proprietary dependencies. Build and verify from source directly:
   ```bash
   cargo build --release --locked -p joule-db-server --bin jouledb
   cargo test --workspace --locked
   ```
2. **What the `crates.io` `jouledb` Crate (`stubs/jouledb`) Does:**
   Because the 41 internal workspace crates are not yet published individually on `crates.io`,
   the top-level `jouledb` package on `crates.io` ([`stubs/jouledb/`](https://github.com/openIE-dev/jouledb/tree/main/stubs/jouledb),
   `v0.10.0`, `MIT OR Apache-2.0`; `1.0.0` yanked) is a **`cargo-binstall` binary-fetcher stub**:
   - `cargo binstall jouledb` downloads the signed release archive built from this repository
     by [`.github/workflows/release.yml`](https://github.com/openIE-dev/jouledb/blob/main/.github/workflows/release.yml).
   - Plain `cargo install jouledb` compiles `stubs/jouledb/src/main.rs`, which prints install
     instructions and exits with status `2`. For a pure-source build, clone this repository and
     run `cargo build --release --locked -p joule-db-server --bin jouledb`.
3. **Historical `v0.1` `BSL-1.1` & Private-Mirror Transition:**
   Prior to `v0.2.0` (November 23, 2026), `openIE-dev/jouledb` was an initial `v0.1` release-only
   surface synced from an internal mirror (`joulesperbit`), prebuilt binaries carried `BSL-1.1`
   notices, an early workflow built only `jouledb-cli`, and this page (`why.md`) was a 10-line
   placeholder. In `v0.2.0`, OpenIE relicensed the entire codebase to **`Apache-2.0 OR MIT`**
   and committed all 40 workspace crates to `crates/`. Stale `v0.1` mirror language in
   `CONTRIBUTING.md`, `install.md`, and `introduction.md` has been reconciled accordingly.

---

## References

1. Rivoire, S., Shah, M. A., Ranganathan, P., & Kozyrakis, C. (2007). *JouleSort: A Balanced Energy-Efficiency Benchmark*. ACM SIGMOD, 365–376.
2. Harizopoulos, S., Shah, M. A., Meza, J., & Ranganathan, P. (2009). *Energy Efficiency: The New Holy Grail of Data Management Systems*. CIDR.
3. Tsirogiannis, D., Harizopoulos, S., & Shah, M. A. (2010). *Analyzing the Energy Efficiency of a Database Server*. ACM SIGMOD, 231–242.
4. Bordes, A., Usunier, N., Garcia-Duran, A., Weston, J., & Yakhnenko, O. (2013). *Translating Embeddings for Modeling Multi-relational Data*. NeurIPS.
5. Raasveldt, M., & Mühleisen, H. (2019). *DuckDB: An Embeddable Analytical Database*. ACM SIGMOD, 1981–1984.
6. Ma, S., Wang, H., Ma, L., Wang, L., Wang, W., Huang, S., Dong, L., Wang, R., Xue, J., & Wei, F. (2024). *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits* (`BitNet b1.58`). arXiv:2402.17764.
