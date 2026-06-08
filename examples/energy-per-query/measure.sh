#!/usr/bin/env bash
# Run a query with the --energy-receipt flag and pipe the JSON receipt
set -e

jouledb init mydata 2>/dev/null || true

jouledb exec mydata \
  "CREATE TABLE IF NOT EXISTS metrics(t u64 pk, v f64)" 2>/dev/null

# Insert a million rows
jouledb exec mydata \
  "INSERT INTO metrics SELECT i, random() FROM generate_series(1, 1000000) i"

# Aggregate query with energy receipt
jouledb exec mydata \
  "SELECT avg(v), stddev(v) FROM metrics" \
  --energy-receipt | jq '{rows, energy_uj, op_breakdown}'
