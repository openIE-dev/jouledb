# JouleDB

**An energy-metered database engine. Every operation, store, query, HDC bind, cascade tier dispatch, is measured in joules, and the receipt is cryptographically anchorable.**

[![Release](https://github.com/openIE-dev/jouledb/actions/workflows/release.yml/badge.svg)](https://github.com/openIE-dev/jouledb/actions/workflows/release.yml)
[![Quality](https://github.com/openIE-dev/jouledb/actions/workflows/quality.yml/badge.svg)](https://github.com/openIE-dev/jouledb/actions/workflows/quality.yml)
[![Docs](https://github.com/openIE-dev/jouledb/actions/workflows/docs.yml/badge.svg)](https://openie-dev.github.io/jouledb)
[![crates.io](https://img.shields.io/crates/v/jouledb.svg)](https://crates.io/crates/jouledb)


JouleDB takes one position: *the right unit of account for database + AI work is the verifiable joule, and a deterministic spine with the model at the leaves is how you minimize it.* It is not a competitor to a lakehouse or a warehouse, it is a different axis.

> Free and open research project, released under MIT OR Apache-2.0 (see `COPYRIGHT`). This is a **capability showcase**, published to demonstrate the architecture, not a managed service.

---

## What it is

One B-tree storage substrate carrying:

- **7 query languages**, SQL, Cypher, CQL, GraphQL, Datalog, SPARQL, Gremlin (+ InfluxQL/PromQL + `SigQL` biosignals), over one engine, not separate stores.
- **SSI MVCC copy-on-write + instant branching (`joule-db-branch`)** with Phase 5/6 atomic publish, depth-bounded cycle-safe traversal, `DBSP` incremental views, `TriModal RRF` (`BM25 ⋈ RaBitQ 1-bit POPCNT ⋈ HDC`) + ACORN 2-hop rescue, and `HippoRAG` Personalized PageRank.
- **A deterministic cascade as the substrate**, not a router bolted in front of an LLM: `LSN LangCache (95 nJ) -> Engram O(1) Memory (220 nJ) -> TriModal RRF / HDC -> LittleBit-2 Living Synapse`.
- **`JouleSynapse` & `JouleHarness` (`joule-db-weights` + `joule-db-gpu`)**: compiles HuggingFace `.safetensors` checkpoints into SQL-queryable sub-1-bit (`< 0.60` BPW) `LittleBit-2` bitplanes executed via `wgpu` Metal/Vulkan/WebGPU `XNOR + countOneBits` (`POPCNT`) shaders, paired with Sakana `SVF` online adaptation, Google `Titans` surprise-gated plasticity (with MVCC `rollback_to_lsn`), `AirLLM`/`PowerInfer` VRAM-capped streaming, `Mooncake` Radix KV blocks, and durable CRC32+`zstd` `JSYN` snapshots.
- **Post-Docker `JouleCapsule`, Post-Git `JouleSpace`, and `JouleWard` Capability Confinement**: daemonless pure-Rust OCI Distribution v2 pulls + $O(1)$ CoW rootfs branching, `Zed Delta`/`Jujutsu` operation logs + multimodal rooms, RFC-2104 `ThermoCap` offline-attenuatable capability macaroons, macOS `Seatbelt` / Linux `Landlock`+`bwrap`+`seccomp-bpf` / `WASI 0.2` confinement, `Dagger`-class 0-spawn Merkle action memoization, single-file SHA-256 `.jcap` bundles, and drop-in OpenAI (`/v1/*`) & Ollama (`/api/*`) wire endpoints.
- **High-resolution heterogeneous 6-rail energy accounting (`joule-db-energy` + `JouleWard`)**: 1-nanojoule (`nJ`) and 1-nanosecond (`ns`) attribution across `cpu_nj`, `gpu_nj`, `npu_ane_nj`, `dram_uma_nj`, `storage_nvme_nj`, and `network_wire_nj` spanning `McuBareMetal`, `WasmEdgeIsolate`, `MobileNpu`, `EdgeRobotic`, `AppleUmaWorkstation`, `LinuxAccelerator`, and `CloudCxlCluster`, Merkle-anchorable via `joule-db-ledger`.
- **An HDC runtime**, 15 irreducible primitives (bind / bundle / permute / Reflect / Inhibit / Spawn / ...), zero-train one-shot learning (70.0% Hits@1 / 88.1% Hits@10 on WN18RR at ~10 uJ/prediction; 100% on LongMemEval-60).
- Wire surface: native binary protocol (JWP), PostgreSQL pgwire compatibility, Arrow Flight SQL, WebSocket, HTTP, MCP Streamable HTTP, OpenAI `/v1/*`, and Ollama `/api/*`.

## What it is not (read this)

This repository is published with its limitations stated, because a credible showcase states them:

- **Self-hosted software, not a managed cloud service.** Raft, 2PC, sharding, and multi-node Heterogeneous Fabric routing ship and are verified by Jepsen-style fault-injection tests; OpenIE does not operate a commercial multi-region SaaS control plane from this repo.
- **Panic-hardened core & server (`#![deny(unsafe_code)]` in `joule-db-core`, `deny(clippy::unwrap_used)` across all 7 core crates: `core`, `local`, `weights`, `branch`, `ledger`, `query`, `server`)**, while auxiliary workspace crates continue ongoing clippy cleanup.
- **Private frontier reasoning models are separate.** All open-source engine crates here compile standalone from `crates.io` with zero proprietary dependencies (`joule-db-weights` natively ingests standard open `.safetensors` checkpoints); private frontier reasoning plugins are not in this repository.
- **The amorphic engine's pattern-language resolver is feature-gated off** in this build (it depends on a separate project). The storage-side amorphic path is intact; the pattern-lang-backed codegen path is not shipped.

For the full architectural assessment see `docs/WHITEPAPER-JOULEDB-2026-05.md` and `docs/jouledb/SOTA-SWOT-2026-11.md`.

## Why it exists

In 2026 the database+AI field converged on one move, collapse OLTP/OLAP/AI into one governed substrate, put an agent on top, and re-derived two of JouleDB's design points independently: copy-on-write versioned storage with instant branching, and deterministic complexity-routing in front of the model. Those arrivals were measured in **dollars**. JouleDB was built on the same structure because the unit of account is the **verifiable joule**.

## Quickstart

```bash
cargo build --release -p joule-db-core
cargo test  -p joule-db-core --lib
```

- Rust client SDK, `docs/jouledb/SDK-RUST.md`
- C ABI / WASM / ODBC / LangGraph, `docs/jouledb/SDK-*.md`
- Per-query-language references, `docs/jouledb/QUERY-*.md`
- Wire protocol, `docs/MGAI-JWP-PROTOCOL.md`; pgwire compatibility, `docs/MGAI-PGWIRE-COMPAT.md`
- HDC primitive reference, `docs/MGAI-HDC-REFERENCE.md`
- Architecture & honest assessment, `docs/WHITEPAPER-JOULEDB-2026-05.md`

## Layout

`crates/joule-db-*`, the engine (core, local, query, server, client, c, browser, edge, gpu, branch, crdt, ternary, features, energy, viz, weights, ledger, langgraph, hdc, domains, amorphic, odbc, test-harness, benches, quickstart).
Supporting infra crates: `jwp` (wire protocol), `sigql`, `inv-mcp-core`, `jpb-core`, `inv-energy-core`, `inv-hdc-core`, and the `inv-auth`/post-quantum-crypto cluster.

## License

Copyright (c) 2026 Open Interface Engineering, Inc. Licensed under either of Apache License, Version 2.0 ([LICENSE-APACHE](./LICENSE-APACHE)) or the MIT license ([LICENSE-MIT](./LICENSE-MIT)), at your option, for the binaries, the documentation and the examples. Earlier releases and earlier versions of this README described the binaries as Business Source License 1.1; that no longer applies.

