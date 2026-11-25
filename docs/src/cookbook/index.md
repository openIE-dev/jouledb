# Cookbook Recipes (`v1.0.0`)

## 1. Run the Master 36-Pillar `JouleDB v1.0.0` Powerhouse Suite
```bash
jouledb powerhouse
# or over HTTP:
curl -s -X POST http://127.0.0.1:8080/api/v1/bench/v1_powerhouse | jq '.all_36_pillars_passed'
```

## 2. Zero-Downtime SQLite / PostgreSQL Migration with `ML-DSA-65` Certificate
```bash
sqlite3 /tmp/demo.db "CREATE TABLE items(id INT PRIMARY KEY, name TEXT); INSERT INTO items VALUES (1,'laser-532nm');"
jouledb migrate --from sqlite --input /tmp/demo.db
```

## 3. Stream a Heterogeneous Multi-Format Corpus in Bounded $O(E)$ Memory
```bash
jouledb amorphic stream-corpus --dir /tmp/unsilo_demo --generate-bytes 50000000 --entities 512 --cleanup
```

## 4. Evaluate an Out-of-Sample `.jsyn` Rank-Sweep (`r = 8 -> 128`)
```bash
curl -s -X POST http://127.0.0.1:8080/api/v1/synapse/bench/rank_sweep \
  -H "Content-Type: application/json" \
  -d '{"rows":512,"cols":512,"ranks":[8,32,64,128],"svf_sweeps":6}'
```

## 5. Continuous `SOC 2` / `HIPAA` / `GDPR` / `FedRAMP-MLS` Audit & `CycloneDX 1.6` SBOM
```bash
jouledb trust
jouledb trust --sbom > sbom-v1.0.0.cdx.json
```
