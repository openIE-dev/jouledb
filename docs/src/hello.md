# Hello, JouleDB & `.jsyn`

Start a local JouleDB node, create a table, pull a pre-factorized `.jsyn` living-weights model from Hugging Face, and inspect both SQL rows and 6-rail nanojoule receipts:

```bash
# 1. Start JouleDB on port 8080
jouledb --port 8080 --data ./hello-jdb &

# 2. Run SQL with hardware/modeled joule telemetry
jouledb query "CREATE TABLE sensors (id INT PRIMARY KEY, temp REAL)"
jouledb query "INSERT INTO sensors VALUES (1, 21.4), (2, 39.8)"
jouledb query "SELECT * FROM sensors WHERE temp > 25.0"

# 3. Pull a 1.95 MB sub-1-bit .jsyn embedding model (22.2x smaller than FP16) in < 1s
curl -s http://127.0.0.1:8080/api/v1/synapse/hf/pull \
  -H "Content-Type: application/json" \
  -d '{"repo_id":"dcharlot65-openie/all-MiniLM-L6-v2-LittleBit","model_id":"minilm"}'

# 4. Query the model's living bitplanes as SQL rows!
jouledb query "SELECT tensor_name, rows, cols, bits_per_weight FROM synapse_tensors LIMIT 5"
```

Runnable Rust workspace examples are also included under [`examples/`](https://github.com/openIE-dev/jouledb/tree/main/examples) (`hello`, `energy-per-query`, and `hdc-embeddings`).
