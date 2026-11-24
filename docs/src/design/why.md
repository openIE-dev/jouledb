# Why JouleDB

Data movement across disconnected infrastructure silos (OLTP, warehouse, vector store, graph DB, streaming cluster, Docker/Kubernetes daemon, and external LLM weight servers) consumes orders of magnitude more physical joules and tail latency than arithmetic.

JouleDB unifies those workloads on a single Copy-on-Write Serializable Snapshot Isolation (`SSI`) substrate governed by two principles:

1. **Verifiable Joules as the Unit of Account:** Operations route through a deterministic complexity cascade (`LangCache -> Engram -> TriModal BM25/RaBitQ/HDC -> .jsyn Living Bitplanes`) with 6-rail hardware/modeled energy attribution anchored in a Merkle ledger.
2. **Separating Neural Circuits (`.jsyn`) from Stored Facts (SQL + `Engram`):** Offloading factual storage to exact database structures allows `.jsyn` neural checkpoints to stay `20x–560x` smaller than `FP16` without sacrificing task accuracy.

Read the full architectural analysis in [`WHITEPAPER-JOULEDB-2026-11.md`](https://github.com/openIE-dev/jouledb/blob/main/docs/WHITEPAPER-JOULEDB-2026-11.md) and transparent boundaries at [jouledb.org/honest](https://jouledb.org/honest).
