# Guided Tour (`v0.11.0` Preview)

Explore JouleDB `v0.11.0` locally in seven steps:

1. **Claude Desktop / MCP App (`jouledb mcp` & `.mcpb`):** Install the `.mcpb` bundle in Claude Desktop or configure `jouledb mcp --data ./jdb` to launch the interactive **JouleDB Studio** (`ui://jouledb/studio.html`, `50` MCP tools).
2. **SQL & PostgreSQL `pgwire` (`:8082`):** Start `jouledb --port 8080 --data ./jdb` and run SQL via `jouledb query` or `psql -h 127.0.0.1 -p 8082 -U postgres`.
3. **Official ClickBench vs. DuckDB Harness (`bench/`):** Run `python3 bench/harness.py` or `python3 bench/clickbench_official.py` to compare `JouleDB` (`pgwire`) against `DuckDB` with hardware RAPL / `powermetrics` joules (`reported_j`), including queries where `DuckDB` wins.
4. **5-Node Multi-Host WAN & Local TCP Raft + `elle` (`0.2.8`):** Run `cargo test -p joule-db-server --test raft_tcp_elle_fault_injection` or `python3 bench/raft_multihost_wan.py` and verify the exported history (`bench/results/raft_wan_multihost_elle_history.edn`) with official Clojure `elle/elle 0.2.8` (`bench/elle_check.clj`).
5. **Multi-Format Import (`jouledb migrate`):** Import a local SQLite `.db`, `.sql`, `.jsonl`, `_bulk`, or `.csv` file with order-independent `BLAKE3` multiset verification.
6. **Amorphous Continuum (`jouledb amorphic`):** Stream mixed-format records (`JSONL`, `CSV`, `XML`, `k=v`, prose, triples) into `10,240`-bit hypervectors with SI unit normalization.
7. **Verified `.jsyn` Sentence Embedder:** Pull [`dcharlot65-openie/all-MiniLM-L6-v2-LittleBit`](https://huggingface.co/dcharlot65-openie/all-MiniLM-L6-v2-LittleBit) (`1.95 MB`, `0.44 BPW`, `cos = 0.9441` vs upstream FP32) and inspect `synapse_tensors` in SQL.
