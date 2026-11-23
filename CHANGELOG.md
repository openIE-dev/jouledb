# Changelog

All notable changes to this release surface are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Self-Distilled `SVF` Calibration (`JouleSynapse`)**: Dataset-free Coordinate-Ridge singular-value self-distillation (`distill_svf_against_dense`) + exact per-row Cauchy-Schwarz gain projection during `.safetensors` ingestion (`distill_svf: true` by default).
- **SSE Per-Token Nanojoule Streaming & Zero-Hallucination Grammar/Tool Engine**: Native `stream: true` `text/event-stream` streaming with per-token `x_jouledb_token_nj` deltas plus `XGrammar`-class jump-forward JSON Schema (`response_format`) and function/tool-call (`tools` -> `tool_calls` / `tool_use`) constraining across OpenAI `/v1/chat/completions`, Anthropic `/v1/messages`, Python SDK (`chat_stream`, `chat_structured`, `chat_tools`), and Rust SDK (`HttpFabricClient`).

## [0.2.0] - 2026-11-23

### Added
- **Zero-PyTorch Causal Autoregressive Decoder & Multi-Layer Encoder (`JouleSynapse`)**:
  - Full `Llama-3` / `SmolLM2` / `Qwen2.5` / `Mistral` causal autoregressive decoder (`RoPE` + Grouped-Query Attention `GQA` + `RMSNorm` + `SwiGLU` + weight-tied/untied `LMHead` + pure-Rust byte-level `BpeTokenizer` + materialized `Mooncake` `(K, V)` prefix block reuse) executing directly over sub-1-bit `LittleBit-2` bitplanes (`POST /api/v1/synapse/decoder/generate`, `/v1/chat/completions`, `/v1/messages`).
  - Pure-Rust `WordPieceTokenizer` (`vocab.txt` & `tokenizer.json`) + full $L$-layer Multi-Head Self-Attention + `GELU` FFN + `LayerNorm` + Mean-Pooling encoder (`all-MiniLM-L6-v2` / BERT / BGE) with zero external ML runtime (`ort` / `candle` / PyTorch / `llama.cpp`).
- **Drop-In AI Model Server Disintermediation**: OpenAI (`GET /v1/models`, `POST /v1/embeddings`, `POST /v1/chat/completions`), Anthropic (`POST /v1/messages`), and Ollama (`GET /api/tags`, `POST /api/generate`) wire compatibility backed by a 4-tier thermodynamic cascade (`LangCache` ~`95 nJ` -> `Engram` $O(1)$ ~`220 nJ` -> `TriModal` RRF -> `LittleBit-2` hardware `wgpu` Metal/Vulkan `POPCNT` + `Titans`).
- **Cryptographic Capability Confinement & Single-File `.jcap` Appliances (`JouleWard`)**: RFC-2104 `HMAC-SHA256` `ThermoCap` macaroons with offline caveat attenuation, native OS kernel confinement (`macOS Seatbelt SBPL`, `Linux Landlock + bwrap + seccomp-bpf`, `WASI 0.2`), `85 nJ` Merkle CAS action memoization, and SHA-256 verified `JCAP_v1` single-file bundles.
- **High-Resolution (`1 nJ` / `1 ns`) Heterogeneous 6-Rail Fabric Energy Ledger + Live Hardware Sensor Fusion**: Tracks `cpu_nj`, `gpu_nj`, `npu_ane_nj`, `dram_uma_nj`, `storage_nvme_nj`, and `network_wire_nj` across 7 fabric compute tiers (`MCU` -> `WASM` -> `Mobile NPU` -> `Edge Robotic` -> `Apple UMA` -> `Linux Accelerator` -> `Cloud CXL`), fusing live OS/hardware power telemetry (`EnergySnapshot` / `HardwarePowerSample`) directly into per-hop receipts.
- **Database-Native Observability, Guardrails, Durable Workflows & Federated Skill Sync**: `JoulePulse` (`45 nJ` inline prompt-injection/SSRF/secret guardrail firewall, `POST /v1/traces` OTLP span ring, Pareto CI regression gate), `JouleFlow` (crash-idempotent journaled DAGs + carbon deferral + Saga compensations), and `JouleSynapseMesh` (`< 4 KB` `HMAC-SHA256` signed skill deltas with Byzantine Multi-Krum aggregation).
- **Container & Post-Git Disintermediation (`JouleCapsule` & `JouleSpace`)**: Pure-Rust OCI `.tar.gz` whiteout extractor, `O(1)` CoW branch VFS capsules, `Joulefile` DAG builder, `jj`-style conflict-free delta oplog with automatic secret scrubbing, and `LiveKit` multimodal room barge-in.
- **Python & Rust SDK Upgrades**: Added `synapse_status`, `ward_ledger`, `ward_mint_token`, `ward_exec`, `embed`, `chat`, `messages`, and `flow_execute` to both the `jouledb` Python SDK (`0.2.0`) and `joule-db-client::HttpFabricClient`.

### Changed
- License: MIT OR Apache-2.0 dual-licensing across the workspace, binaries, documentation, and examples.

## [0.1.0], TBD

### Added
- Initial public release surface
- Multi-platform prebuilt binaries (macOS arm64/x86_64, Linux x86_64/aarch64 musl, Windows x86_64)
- Examples directory with runnable programs
- Documentation site at openie-dev.github.io/jouledb
- GitHub Discussions enabled
- Issue templates (bug report, feature request)

[Unreleased]: https://github.com/openIE-dev/jouledb/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/openIE-dev/jouledb/releases/tag/v0.1.0
