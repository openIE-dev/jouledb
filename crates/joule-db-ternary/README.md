# joule-db-ternary

Packed ternary encoding for HDC hypervectors and columnar storage.

Ternary (-1, 0, +1) is the natural alphabet of bipolar HDC. `joule-db-ternary` is the bit-packed encoding layer — five trits per byte (3⁵ = 243 < 256), with bind / bundle / similarity operations defined directly on the packed representation.

## Module map

| Module | Role |
|---|---|
| [`pack.rs`](src/pack.rs) | Pack / unpack 5-trit-per-byte encoding |

## Tests

13 tests in `src/`.

## See also

- [joule-db-hdc](../joule-db-hdc/) — the HDC substrate that uses ternary encoding
- [joule-db-domains](../joule-db-domains/) — domain-specific HDC encoders
