# energy-per-query

Run an aggregate over a 1M-row table and inspect the per-query energy receipt.

```bash
./measure.sh
```

Sample output:

```json
{
  "rows": 1,
  "energy_uj": 142,
  "op_breakdown": {
    "scan":      120,
    "aggregate":  18,
    "result":      4
  }
}
```

What this shows:
- `--energy-receipt` flag attaches energy to the result envelope
- `energy_uj` (microjoules) is the total per-query cost
- `op_breakdown` decomposes the energy by execution-plan operator
