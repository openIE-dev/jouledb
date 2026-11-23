# joule-db-weights

LLM weight storage and retrieval for JouleDB — extent-based, mode-capped, zero-copy.

`joule-db-weights` stores model weight tensors in JouleDB using extent-based contiguous page allocation for fast sequential reads. The result: a B-tree-durable, transactional, zero-copy backing store for inference weights — readable directly from disk via `mmap` for unified-memory GPU access (Apple Silicon, integrated GPUs, edge accelerators).

## Capabilities

- **Ingest**: GGUF → JouleDB (one-time, sequential scan)
- **Query**: get tensor by name, get layer, get experts
- **Tier hints**: pin / evict / prefetch via the buffer pool
- **Zero-copy**: mmap-based reads for UMA GPU access
- **Default buffer pool**: 4096 pages = 256 MB to accommodate extent-based tensor storage

## Key scheme

Composite keys for predictable layout:

```text
model_id\0tensor_name              →  tensor data (raw bytes, extent-stored)
model_id\0_meta\0tensor_name       →  JSON metadata (inline, small)
model_id\0_index                   →  JSON model index (layer count, expert count, etc.)
```

## Module map

| Module | Role |
|---|---|
| [`ingest.rs`](src/ingest.rs) | GGUF → JouleDB ingest path |
| [`query.rs`](src/query.rs) | Get tensor / layer / experts |
| [`compress.rs`](src/compress.rs) | Compression hooks |
| [`bin/`](src/bin/) | CLI helpers |

## Why this lives in JouleDB

LLM weight files are large, mostly-read-only, and need fast sequential access — exactly what JouleDB's extent-based B-tree is designed for. Putting weights in the same engine that handles the surrounding query and energy data eliminates the "model file vs. database" split that every other AI stack has to manage.

## See also

- [joule-db-core](../joule-db-core/) — extent-based storage primitives
- [jouledb-ai-runtime](../jouledb-ai-runtime/) — uses this crate for the embedded-model tier of the cascade
