# Cookbook Recipes (`v0.13.0`)

## 1. Run the 5-Node Networked TCP Raft Fault-Injection Suite + Official `elle` (`0.2.8`)
```bash
cargo test -p joule-db-server --test raft_tcp_elle_fault_injection
clojure -Sdeps '{:mvn/repos {"clojars" {:url "https://repo.clojars.org/"}} :deps {elle/elle {:mvn/version "0.2.8"} spootnik/unilog {:mvn/version "0.7.31"}}}' \
  -M bench/elle_check.clj bench/results/raft_tcp_elle_history.edn rw-register
```

## 2. Run Official ClickBench Queries Against DuckDB (`pgwire` + Hardware RAPL / `powermetrics`)
```bash
./target/release/jouledb -p 28080 -d /tmp/jdb-cb &
python3 bench/clickbench_official.py --parquet /path/to/hits.parquet --queries /path/to/queries.sql --pg-port 28082
```

## 3. Import a SQLite Database with Multiset `BLAKE3` Verification
```bash
sqlite3 /tmp/demo.db "CREATE TABLE items(id INT PRIMARY KEY, name TEXT); INSERT INTO items VALUES (1,'laser-532nm');"
jouledb migrate --from sqlite --input /tmp/demo.db
```

## 4. Stream a Multi-Format Directory in Bounded Memory
```bash
jouledb amorphic stream-corpus --dir /tmp/unsilo_demo --generate-bytes 50000000 --entities 512 --cleanup
```

## 5. Run Technical Security-Control Self-Tests (Unaudited) & Export CycloneDX SBOM
```bash
jouledb trust
jouledb trust --sbom > sbom-v0.13.0.cdx.json
```
