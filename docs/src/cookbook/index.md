# Cookbook Recipes

## 1. Run the 17-Pillar Database (`7`) + Frontier AI Exam (`10`) Suites
```bash
curl -s -X POST http://127.0.0.1:8080/api/v1/bench/industry_suite \
  -H "Content-Type: application/json" \
  -d '{"clickbench_rows":100000,"ann_vectors":2000,"tpcc_txns":500}'

curl -s -X POST http://127.0.0.1:8080/api/v1/bench/frontier_exam \
  -H "Content-Type: application/json" \
  -d '{"mrcr_updates":4000}'
```

## 2. Evaluate an Out-of-Sample `.jsyn` Rank-Sweep (`r = 8 -> 128`)
```bash
curl -s -X POST http://127.0.0.1:8080/api/v1/synapse/bench/rank_sweep \
  -H "Content-Type: application/json" \
  -d '{"rows":512,"cols":512,"ranks":[8,32,64,128],"svf_sweeps":6}'
```

## 3. Store & Recall Exact Long-Tail Facts in `Engram` (`O(1)`, `220 nJ`)
```bash
curl -s -X POST http://127.0.0.1:8080/api/v1/synapse/engram/upsert \
  -H "Content-Type: application/json" \
  -d '{"phrase":"landauer limit 300k","embedding":[1.0,0.0],"fact_payload":"2.87e-21 Joules/bit"}'

curl -s -X POST http://127.0.0.1:8080/api/v1/synapse/engram/lookup \
  -H "Content-Type: application/json" \
  -d '{"phrase":"Landauer Limit 300K"}'
```
