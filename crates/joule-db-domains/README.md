# joule-db-domains

Domain-specific HDC encoding modules for JouleDB.

`joule-db-domains` is the catalog of domain-tuned hypervector encoders. Each module knows how to turn structured input from its domain (a genomic sequence, a financial tick, a legal clause, a road graph) into a hyperdimensional vector that [`joule-db-hdc`](../joule-db-hdc/) can store, search, and reason over.

## Modules — 22 domains

| Module | Domain |
|---|---|
| [`adtech.rs`](src/adtech.rs) | Ad-tech (impressions, attribution, audience segments) |
| [`agri.rs`](src/agri.rs) | Agriculture (soil, crop, weather) |
| [`auto.rs`](src/auto.rs) | Automotive / vehicle telemetry |
| [`cyber.rs`](src/cyber.rs) | Cybersecurity (threats, indicators, attack chains) |
| [`edu.rs`](src/edu.rs) | Education (learners, curricula, assessment) |
| [`energy.rs`](src/energy.rs) | Energy (grid, generation, consumption) |
| [`gaming.rs`](src/gaming.rs) | Gaming (players, sessions, in-game state) |
| [`genomics.rs`](src/genomics.rs) | Genomics (sequences, variants, expression) |
| [`graph.rs`](src/graph.rs) | Generic graph (nodes, edges, paths) |
| [`health.rs`](src/health.rs) | Healthcare (patients, encounters, vitals) |
| [`insurance.rs`](src/insurance.rs) | Insurance (policies, claims, risk) |
| [`iot.rs`](src/iot.rs) | IoT (sensors, devices, streams) |
| [`legal.rs`](src/legal.rs) | Legal (cases, clauses, citations) |
| [`market/`](src/market/) | Financial markets (ticks, OHLCV, instruments) |
| [`media.rs`](src/media.rs) | Media (assets, metadata, transcoding) |
| [`multimodal.rs`](src/multimodal.rs) | Multimodal (joint text/image/audio) |
| [`retail.rs`](src/retail.rs) | Retail (SKUs, baskets, customers) |
| [`spatial.rs`](src/spatial.rs) | Spatial (geometry, geography, indexing) |
| [`supply.rs`](src/supply.rs) | Supply chain (shipments, inventory, routes) |
| [`telecom.rs`](src/telecom.rs) | Telecom (calls, sessions, network state) |
| [`temporal.rs`](src/temporal.rs) | Temporal (intervals, time-series semantics) |

## Tests

140 tests in `src/`.

## See also

- [joule-db-hdc](../joule-db-hdc/) — the HDC substrate these encoders target
- [joule-db-amorphic](../joule-db-amorphic/) — uses domain encoders for JIT schema inference
- [joule-db-features](../joule-db-features/) — generic feature modules (vector, graph, time-series, full-text)
