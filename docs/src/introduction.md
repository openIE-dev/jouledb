# Introduction

**JouleDB (`v0.4.0`)** is an open-source (`Apache-2.0 OR MIT`) database and living-weights (`.jsyn`) execution engine written in safe Rust (`40` workspace crates in [`crates/`](https://github.com/openIE-dev/jouledb/tree/main/crates)). Operations across SQL, CoW branch forks, causal `ASOF` sensor joins, `JouleCapsule` containers, `JouleKube` manifests, and sub-1-bit `.jsyn` forward passes emit explicit energy telemetry (`reported_j` on-chip counters where available, or `est_j` analytical byte/instruction models—see [Why JouleDB](./design/why.md)).

- **Website & Interactive Simulators:** [https://jouledb.org](https://jouledb.org)
- **Full Rust Workspace Source (`Apache-2.0 OR MIT`):** [https://github.com/openIE-dev/jouledb](https://github.com/openIE-dev/jouledb)
- **82 Public `.jsyn` Models on Hugging Face:** [https://huggingface.co/dcharlot65-openie](https://huggingface.co/dcharlot65-openie)
- **`cargo-binstall` Installer Package (`crates.io`):** [https://crates.io/crates/jouledb](https://crates.io/crates/jouledb) *(fetches GitHub Release binaries via `cargo binstall jouledb`; build from source via `cargo build --release --locked -p joule-db-server --bin jouledb`)*

## What You Will Find Here

- **[Install](./install.md)** — source builds from `crates/`, `cargo binstall jouledb`, Docker GHCR, and prebuilt release archives.
- **[Hello, JouleDB & .jsyn](./hello.md)** — launch the server, run SQL, and pull a `.jsyn` model in 60 seconds.
- **[.jsyn Living Model Weights](./jsyn.md)** — how `LittleBit-2 ⊕ Sakana SVF` shrinks `< 30 GB` models by `20x–560x` inside the database.
- **[Guided Tour](./tutorial/index.md)** — SQL, `TriModal GraphRAG`, CoW branching, `JouleKube`, and benchmark suites.
- **[API, Query & Wire Reference](./reference/index.md)** — 7 query languages, wire protocols, and REST/MCP/OpenAI routes.
- **[Cookbook Recipes](./cookbook/index.md)** — copy-paste recipes for hybrid search, rank sweeps, and embodied `ASOF` joins.
- **[Why JouleDB](./design/why.md)** — prior art (`JouleSort`, Harizopoulos et al., Tsirogiannis et al., `BitNet b1.58`), baseline comparisons (`DuckDB`, `ClickHouse`, `PostgreSQL + pgvector`, `vLLM`), the 15 HDC primitives, `WN18RR` `Hits@1` vs `Hits@10` protocol, and `Apache-2.0 OR MIT` / `crates.io` packaging disclosures.
