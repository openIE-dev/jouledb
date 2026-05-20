# joule-db-energy

Energy profiling and hardware-aware execution for JouleDB.

`joule-db-energy` is the metering substrate. Every metered operation in the JouleDB ecosystem — query, write, HDC bind, cascade tier dispatch — gets its joule count from this crate. The energy receipts that flow downstream into [`joule-db-ledger`](../joule-db-ledger/), [`jouledb-ai-runtime`](../jouledb-ai-runtime/), and the public ACID surface all originate here.

## Module map

| Module | Role |
|---|---|
| [`monitor.rs`](src/monitor.rs) | Per-thread / per-op energy monitor — samples hardware counters |
| [`tracker.rs`](src/tracker.rs) | Aggregates raw samples into per-op receipts |
| [`budget.rs`](src/budget.rs) | Budget enforcement — refuses ops projected to exceed a joule ceiling |
| [`controller.rs`](src/controller.rs) | High-level controller — combines monitor + tracker + budget |
| [`advisor.rs`](src/advisor.rs) | Plan-cost advisor for hardware-aware execution selection |
| [`router.rs`](src/router.rs) | Routes ops to the cheapest-in-joules hardware backend |
| [`bridge.rs`](src/bridge.rs) | Cross-crate bridge for downstream consumers |
| [`platform/`](src/platform/) | Per-platform hardware-counter readers (RAPL, IOKit, NVML, etc.) |

## Tests

84 tests in `src/`.

## See also

- [joule-db-ledger](../joule-db-ledger/) — anchors receipts to a verifiable ledger
- [jouledb-ai-runtime](../jouledb-ai-runtime/) — `EnergyReceipt` shape consumed by the cascade
- [joule-db-branch](../joule-db-branch/) — uses per-branch energy budgets
- [joule-db-server::energy](../joule-db-server/src/energy.rs) — per-query budget enforcement at the wire layer
