# Guided Tour

This tour walks through JouleDB's core subsystems in five steps:

1. **ACID/SSI Relational & Multi-Model Storage:** Create tables via SQL (`jouledb query`) or PostgreSQL `pgwire` (`:8082`). Concurrent transactions run under Cahill Serializable Snapshot Isolation (`SSI`), certified against Kleppmann `Hermitage` (`POST /api/v1/certify/hermitage_elle`).
2. **Instant `O(1)` Copy-on-Write Branching & `JouleCapsule`:** Fork database tables, `JouleAgentFS`, and daemonless OCI container rootfses in `1.13 µs` with strict microjoule budgets (`jouledb capsule fork`).
3. **Tri-Modal Hybrid GraphRAG (`BM25 ⋈ RaBitQ ⋈ HDC`):** Query lexical terms, `RaBitQ` 1-bit `POPCNT` vectors, `ACORN` 2-hop filtered bridges, and `HippoRAG` Personalized PageRank inside one engine.
4. **`.jsyn` Living Model Weights:** Pull any of the [82 public `.jsyn` models](https://huggingface.co/dcharlot65-openie) via `POST /api/v1/synapse/hf/pull` and invoke drop-in OpenAI `/v1/chat/completions` or `/v1/embeddings` with 6-rail `x_jouledb_energy` receipts.
5. **`JouleKube` Kubernetes AI Control Plane:** Apply standard Kubernetes & CNCF AI-Conformance (`InferencePool`, `SandboxClaim`, `DRA`, `Kueue`, `Kagent`) YAML manifests directly to `POST /api/v1/kube/apply` (`0` `etcd`, `0.0W` warm pools).
