# Hello, JouleDB & `.jsyn`

Start a local JouleDB `v0.10.0` node, create a table, pull the verified `all-MiniLM-L6-v2-LittleBit` `.jsyn` sentence-embedding model (`cos = 0.9441` vs upstream `FP32`) from Hugging Face, and inspect both SQL rows and nanojoule receipts:

```bash
# 1. Start JouleDB v0.10.0 on port 8080 (HTTP :8080, PgWire :8082)
jouledb --port 8080 --data ./hello-jdb &

# 2. Run SQL with joule telemetry (`reported_j` on bare metal RAPL/SMC, `est_j` modeled fallback)
jouledb query "CREATE TABLE sensors (id INT PRIMARY KEY, temp REAL)"
jouledb query "INSERT INTO sensors VALUES (1, 21.4), (2, 39.8)"
jouledb query "SELECT * FROM sensors WHERE temp > 25.0"

# 3. Pull the verified 1.95 MB .jsyn sentence embedder (0.44 BPW, 22.2x smaller than FP16, cos = 0.9441)
curl -s http://127.0.0.1:8080/api/v1/synapse/hf/pull \
  -H "Content-Type: application/json" \
  -d '{"repo_id":"dcharlot65-openie/all-MiniLM-L6-v2-LittleBit","model_id":"minilm"}'

# 4. Query the model's factorized bitplanes as SQL rows
jouledb query "SELECT tensor_name, rows, cols, bits_per_weight FROM synapse_tensors LIMIT 5"
```

Runnable Rust workspace examples are also included under [`examples/`](https://github.com/openIE-dev/jouledb/tree/main/examples) (`hello`, `energy-per-query`, and `hdc-embeddings`).
