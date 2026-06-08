# hdc-embeddings

Use JouleDB's first-class HDC vector type and similarity query.

```bash
./embed.sh
```

Sample output:

```
id │ label  │ sim
───┼────────┼──────
 1 │ cat    │ 1.000
 2 │ kitten │ 0.832
 3 │ rocket │ 0.041
```

What this shows:
- `hdc[N]` typed column for fixed-size hypervectors
- `hdc_encode(text)` built-in encoder (text → 2048-d hypervector)
- `hdc_sim(a, b)` cosine similarity (normalized to [0,1])
- Sort + filter on the similarity result
