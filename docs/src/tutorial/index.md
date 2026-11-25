# Guided Tour (`v0.7.0` Preview)

Explore JouleDB `v0.7.0` locally in six steps:

1. **SQL & PostgreSQL `pgwire` (`:8082`):** Start `jouledb --port 8080 --data ./jdb` and run SQL via `jouledb query` or `psql -h 127.0.0.1 -p 8082 -U postgres`.
2. **Official ClickBench vs. DuckDB Harness (`bench/`):** Run `python3 bench/harness.py` or `python3 bench/clickbench_official.py` to compare `JouleDB` (`pgwire`) against `DuckDB` with hardware RAPL / `powermetrics` joules (`reported_j`), including queries where `DuckDB` wins.
3. **5-Node TCP Raft Fault Injection + `elle` (`0.2.8`):** Run `cargo test -p joule-db-server --test raft_tcp_elle_fault_injection` and verify the exported history (`bench/results/raft_tcp_elle_history.edn`) with official Clojure `elle/elle 0.2.8` (`bench/elle_check.clj`).
4. **Multi-Format Import (`jouledb migrate`):** Import a local SQLite `.db`, `.sql`, `.jsonl`, `_bulk`, or `.csv` file with order-independent `BLAKE3` multiset verification.
5. **Amorphous Continuum (`jouledb amorphic`):** Stream mixed-format records (`JSONL`, `CSV`, `XML`, `k=v`, prose, triples) into `10,240`-bit hypervectors with SI unit normalization.
6. **Verified `.jsyn` Sentence Embedder:** Pull [`dcharlot65-openie/all-MiniLM-L6-v2-LittleBit`](https://huggingface.co/dcharlot65-openie/all-MiniLM-L6-v2-LittleBit) (`1.95 MB`, `0.44 BPW`, `cos = 0.9441` vs upstream FP32) and inspect `synapse_tensors` in SQL.
