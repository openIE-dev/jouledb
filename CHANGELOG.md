# Changelog

All notable changes to this release surface are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [Semantic Versioning](https://semver.org/).

> **Version & Audit Errata Notice (`2026-11-25`):** Following an independent source-level audit of `openIE-dev/jouledb`, the premature `v1.0.0` GA tag has been **retracted** (`crates.io` `jouledb 1.0.0` yanked; GitHub Release `v1.0.0` marked retracted), and the project has returned to **`0.x` (`v0.6.0` Pre-1.0 Research Preview)**. Earlier changelog entries (`[0.2.8]` through `[1.0.0]`) are preserved below for historical transparency, subject to the following explicit corrections:
> - **"36 / 36 Certified Pillars" & Frontier AI Exam Claims Retracted:** The synthetic benchmark routes (`frontier_exam_routes.rs` — `HLE-Diamond`, `SuperGPQA`, `SimpleQA`, `FaithEval`, `ARC-AGI-2`, `FrontierMath`, `SWE-bench`, `TAU-bench`, `MRCR`, `SafeVLA`) were synthetic/seeded in-process micro-harnesses or hard-coded loop simulations, not official benchmark runs. Those routes have been deleted, and remaining subsystem checks (`TPC-C` micro-loop, `ClickBench` codec smoke, `Hermitage`, `TigerBeetle` ledger conservation, `Geo-Raft` state-machine simulation) are explicitly labeled as **in-process functional smoke tests** with `energy_evidence: "est_j"`.
> - **Compliance Disclaimers:** No third-party SOC 2, ISO 27001, HIPAA, PCI-DSS, NIST CMVP, or FedRAMP audit has been conducted; `trust_compliance.rs` runs technical control self-tests only (`external_audit_performed: false`).
> - **`.jsyn` Hugging Face Fleet Pruned (`82 -> 1`):** 81 unverified ultra-low-BPW (`0.02–0.17 BPW`, `cos = 0.27–0.45`) or custom-vendor-licensed `.jsyn` causal LLM checkpoints have been deleted from `huggingface.co/dcharlot65-openie`; only the verified `Apache-2.0` sentence embedder `dcharlot65-openie/all-MiniLM-L6-v2-LittleBit` (`0.4378 BPW`, `cos = 0.9441` vs upstream `FP32`) remains published.

## [Unreleased]

## [0.6.0] - 2026-11-25

### Changed
- **Reverted Release Track to `0.6.0` (Pre-1.0 Research Preview)**:
  - Yanked `jouledb 1.0.0` on `crates.io` and marked GitHub Release `v1.0.0` as retracted so `cargo binstall jouledb` resolves `0.6.0` directly without requiring `--version`.
  - Updated workspace, CLI, Python bindings (`python/`), all 6 client SDK manifests (`sdks/*`), and Helm/Terraform charts (`deploy/*`) to `0.6.0`.
- **Removed Unrelated `joule-web` Crate (`1.11M` LOC / `34,928` Tests) & Deleted Synthetic Exam Routes**:
  - Removed `crates/joule-web` and dead CLI command `crates/bin/jouledb-cli/src/commands/web.rs` from `openIE-dev/jouledb`, reducing the repository to **`41` workspace crates (`724,986` lines of Rust, `12,265` tests)** focused on the database, storage, query, consensus, energy, and weights runtime.
  - Deleted `crates/joule-db-server/src/frontier_exam_routes.rs` and `wave26_frontier_exam_hle_arc_math_sql_swe_calibration_e2e.rs`; renamed CLI `powerhouse` to `jouledb self-check` and pruned MCP tools from `54` to `49` non-benchmark-branded tools.
- **Truth-in-Labeling Across Security Controls, Bridges, Smoke Checks & Energy Evidence**:
  - Updated `trust_compliance.rs`, `universal_bridge.rs`, `distributed_jepsen_consensus.rs`, `certification_routes.rs`, `analytic_benchmarks.rs`, and `frontier_eval.rs` to distinguish in-process smoke/control self-tests from external benchmarks or compliance certifications, and tagged analytical energy estimates explicitly with `energy_evidence: "est_j"` vs hardware `reported_j`.
  - Aligned documentation on exact workspace lints: `#![deny(unsafe_code)]` in `joule-db-core` (with 1 `#![allow(unsafe_code)]` module `storage/mmap_extent.rs`) and `deny(clippy::unwrap_used)` active in 3 crates (`joule-db-core`, `joule-db-branch`, `joule-db-ledger`).

### Fixed
- **Consensus Quorum Commit Fix (`RaftNode::propose_and_commit`) & 5-Node Real TCP Fault-Injection Verified by Official Clojure `elle 0.2.8`**:
  - Fixed `crates/joule-db-server/src/raft.rs` where `RaftNode::propose()` previously returned `Ok(index)` upon local leader WAL append before quorum replication; added `propose_and_commit(&self, command, timeout)` awaiting majority commit (`commit_index >= index`) and Raft §5.4.2 current-term verification, and removed vacuous `|| true` assertions in `tests/raft_chaos.rs`.
  - Added `tests/raft_tcp_elle_fault_injection.rs` + `bench/elle_check.clj` exercising 5 real `RaftNode<KvStateMachine, TcpRaftTransport>` servers with durable `raft.wal` across a 20-link socket-severing `TcpPartitionMatrix` (`2/5` minority leader isolation & write rejection, `3/5` majority new-term election, node stop + disk WAL restart, 5-node state convergence), verified `{:valid? true}` (`:strict-serializable` `rw-register`) by Kyle Kingsbury's official Clojure `elle/elle 0.2.8`.
- **Portable Linux `glibc <= 2.36` Release Binaries (Fixes Ubuntu 24.04 LTS `GLIBC_2.43 not found`)**:
  - Switched `x86_64-unknown-linux-gnu` release builds from `ubuntu-latest` (`glibc 2.43`) to a `Debian 12` / `ubuntu-22.04` (`glibc 2.35–2.36`) baseline with an automated `objdump -T` CI gate rejecting `GLIBC >= 2.40`, verified end-to-end inside `ubuntu:24.04`.
- **Official ClickBench Evaluation Harness (`bench/clickbench_official.py`)**:
  - Added reproducible end-to-end evaluation against `DuckDB` over real `pgwire` (`psycopg`) using the official `14.78 GB` (`99,997,497`-row) ClickBench `hits.parquet` dataset and all 43 official `ClickBench` SQL queries (`Q0..Q42`), recording hardware RAPL `reported_j`, wins, losses, and SQL compatibility errors without filtering.

---

## [1.0.0] - 2026-11-25 — **[RETRACTED — Superseded by `v0.6.0` Pre-1.0 Research Preview]**

> *Retracted on 2026-11-25 (`crates.io` `1.0.0` yanked). See Errata Notice at top of file.*

### Added (Historical Record)
- Initial `live_migration` (`jouledb migrate`), `v1_stability_and_pitr` (`JWAL-v1` / `JSYN-v1` / `JWP-v1` / `ASNP-v1` / `TCAP-v1` TLV + CRC-32 framing & LSN PITR), `sql2023_powerhouse` (window frames, `CUBE`, recursive CTEs, runtime `AQE`), `distributed_jepsen_consensus` (in-process Raft state-machine simulator, relabeled in `0.6.0`), client SDK source trees (`sdks/*`), and `deploy/helm` + `deploy/terraform`.

## [0.5.0] - 2026-11-25 *(Historical — see Errata Notice above regarding smoke-test vs certification framing)*

### Added
- **`MorphRule` DSL, Bounded-Memory Multi-GB Streaming, Protocol Bridges & Security Control Self-Tests (`morph_rules`, `custom_lens`, `streaming_continuum`, `universal_bridge`, `trust_compliance`)**:
  - SI/financial unit algebra (`25°C == 77°F == 298.15 K`, `532 nm == 0.532 µm`, `\$14,250.00 == 1,425,000 cents`), Kanerva VSA `H ⊗ ROLE⁻¹` role unbinding, and `ParetoScatter` + `SankeyLineage` lenses (`8` fluid lenses total).
  - Bounded-memory $O(E)$ `StreamingAmorphicContinuum` (`1.35 GB` / `47.5M` semantic atoms streamed in `57.64 MB` OS Peak RSS).
  - Protocol adapter routes (`universal_bridge.rs`) and technical security control self-tests (`trust_compliance.rs` — SHA-256 hash-chained audit logs, WORM retention locks, per-subject crypto-shredding, Luhn PAN redaction, $\varepsilon$-DP noise, CycloneDX v1.6 SBOM, ML-KEM-768/ML-DSA-65 receipts, Bell-LaPadula MLS lattice).

## [0.4.0] - 2026-11-25 *(Historical — see Errata Notice above)*

### Added
- `joule-db-amorphic::unsilo`, `joule-db-viz::amorphic_display`, `OrthogonalTitansSynapse`, `TemporalRelationalGraph`, `ML-KEM-768 + ML-DSA-65` Merkle execution envelopes, `LiveCxlTieredPool`, and 12-script multilingual text normalization checks.

## [0.3.1] - 2026-11-24 *(Historical — see Errata Notice above)*

### Added
- In-process operator micro-benchmarks (`analytic_benchmarks.rs`: `ALP+FSST` codec, Leapfrog `WCOJ` synthetic skew join, `ASOF` tick stream, small-corpus `TriModal RRF` smoke).

## [0.3.0] - 2026-11-24 *(Historical — Synthetic `frontier_exam_routes` retracted in `0.6.0`)*

### Added
- Calibration math (`RMS-CE`, `ECE`, `Brier`, Platt scaling), 2D grid DSL synthesizer (`arc_synthesize`), SQL dialect normalizer (`sql_critic`), modular CRT / DPLL SAT utilities, idempotency token gate, and 1D STL projection utilities (`frontier_eval.rs`). Synthetic benchmark-named HTTP routes (`frontier_exam_routes.rs`) were subsequently removed in `0.6.0`.

## [0.2.9] - 2026-11-24

### Added
- Out-of-sample `.jsyn` rank-sweep quality-vs-joules evaluator (`rank_sweep_bench`, `POST /api/v1/synapse/bench/rank_sweep`), Empirical-Bayes `SVF` shrinkage (`calibrate_activation_aware_svf`), and `docs/JSYN-LIVING-WEIGHTS-SPEC.md`. *(Note: 81 low-BPW `.jsyn` causal LLM checkpoints mentioned in `0.2.4`–`0.2.9` were retracted from Hugging Face in `0.6.0` due to `0.27–0.45` output cosine at `0.02–0.17 BPW`.)*

## [0.2.8] - 2026-11-24

### Added
- Double-entry `i128` accounting ledger & torn-WAL fault simulator (`tiger_ledger`), topological dirty-cone DAG + freshness-SLA controller (`secdb_napa`), `1,024`-row `ZoneMap` + `SelectionVector` scanner (`zonemap_scan`), and in-process smoke runner (`certification_routes`).

## [0.2.7] - 2026-11-24

### Added
- Morsel-driven out-of-core spill-to-disk Grace Hash Join & GroupBy (`morsel_spill`), in-process `Hermitage` + Tarjan `SCC` cycle detector (`elle_hermitage`), `.slt` MD5 + `TLP`/`NoREC` SQL conformance checks (`conformance_oracle`), and activation-aware `SVF` post-compression calibration.

## [0.2.6] - 2026-11-24

### Added
- SplitMix64 Rademacher seeding + Joint Binary Gram Gauss-Seidel singular refinement (`synapse`), memory-mapped `.safetensors` & GGUF v2/v3 unpackers (`checkpoint_ingest`), and multi-architecture causal/MoE operator support (`causal_decoder`).

## [0.2.5] - 2026-11-23

### Added
- `NVFP4`, scaled `FP8`, 3D stacked MoE, and `BitNet.cpp` `I2_S` unpackers, plus `Mamba-2` selective SSM and `GQA` block operators.

## [0.2.4] - 2026-11-23

### Added
- MLX packed affine `U32` `.safetensors` and memory-mapped GGUF v2/v3 unpackers, plus `ShortConv`, `QK-Norm`, and `ModernBERT` operators.

## [0.2.3] - 2026-11-23

### Added
- `JouleKube` MVCC SQL-backed Kubernetes object store (`kube_control_plane`), `Whisper` 400-FFT cross-attention Seq2Seq decoder & `SigLIP` encoder (`multimodal_encoder`), in-flight request batching, and `.jsyn` Hugging Face export/pull routes.

## [0.2.2] - 2026-11-23

### Added
- Pure-Rust `.ppm`/`.bmp`/RGB `SigLIP`/`ViT` patchifier, `RIFF/WAVE` Log-Mel `Whisper` audio encoder, self-speculative `LayerSkip` draft-and-verify, and GitHub Release binary/wheel workflow.

## [0.2.1] - 2026-11-23

### Added
- GPU-resident batched `LittleBit-2` + causal `GQA` WGSL shaders (`joule-db-gpu`), `T-MAC` 8-bit LUT CPU matvec, database-native `MCP` host & agent loop (`mcp_agent_loop`), coordinate-ridge `SVF` self-distillation, and SSE per-token energy streaming.

## [0.2.0] - 2026-11-23

### Added
- Pure-Rust causal autoregressive decoder & `WordPiece` encoder (`JouleSynapse`), OpenAI/Anthropic/Ollama HTTP routes, `JouleWard` RFC-2104 `HMAC-SHA256` `ThermoCap` macaroons & `.jcap` bundles, 6-rail `FabricEnergyLedger`, `JoulePulse`, `JouleFlow`, `JouleSynapseMesh`, `JouleCapsule`, and `JouleSpace`.
- Dual-licensed workspace under `MIT OR Apache-2.0`.

## [0.1.0] - 2026-10-15

### Added
- Initial public release surface, runnable `examples/`, and `mdBook` documentation site at `openie-dev.github.io/jouledb`.

[Unreleased]: https://github.com/openIE-dev/jouledb/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/openIE-dev/jouledb/releases/tag/v0.6.0
[0.1.0]: https://github.com/openIE-dev/jouledb/releases/tag/v0.1.0
