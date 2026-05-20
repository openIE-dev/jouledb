# JouleDB Time-Series Reference — InfluxQL & PromQL

**Version 1.0 — 2026-05-18**
**Source:** [`crates/joule-db-query/src/timeseries.rs`](../../crates/joule-db-query/src/timeseries.rs) (1,786 LOC)
**Feature:** Default-on via `timeseries` feature in [`joule-db-query`](../../crates/joule-db-query/)

JouleDB supports two time-series query languages — **InfluxQL** (InfluxDB / Telegraf compatibility) and **PromQL** (Prometheus / Grafana compatibility) — over the same storage substrate as SQL. Time-series-specific operators (downsampling, gap-fill, windowed aggregates) live in [`joule-db-features/timeseries/`](../../crates/joule-db-features/src/timeseries/).

## 1. InfluxQL

### 1.1 Supported

| Surface | Status |
|---|---|
| `SELECT … FROM measurement WHERE …` | yes |
| `GROUP BY time(interval)` (auto-bucketing) | yes |
| `GROUP BY tag` | yes |
| `FILL(null/none/previous/<value>)` | yes |
| Aggregates (`MEAN`, `SUM`, `COUNT`, `MAX`, `MIN`, `STDDEV`, `PERCENTILE`) | yes |
| `TIME` predicates (`time > now() - 1h`) | yes |
| `SHOW MEASUREMENTS`, `SHOW TAG KEYS`, `SHOW SERIES` | yes |
| Continuous queries | partial |
| Retention policies | partial |

### 1.2 Example

```sql
-- 5-minute mean CPU usage by host, last hour, filling gaps with previous value
SELECT MEAN(usage) FROM cpu
WHERE time > now() - 1h
GROUP BY time(5m), host FILL(previous)
```

## 2. PromQL

### 2.1 Supported

| Surface | Status |
|---|---|
| Instant vectors (`metric{label="value"}`) | yes |
| Range vectors (`metric[5m]`) | yes |
| Subqueries (`metric{}[1h:5m]`) | yes |
| Aggregation operators (`sum`, `avg`, `min`, `max`, `count`, `stddev`, `topk`, `bottomk`, `quantile`) | yes |
| Binary operators | yes |
| Functions (`rate`, `irate`, `increase`, `delta`, `idelta`, `predict_linear`, `histogram_quantile`, etc.) | yes |
| Recording rules | partial |
| Alerting rules | partial |

### 2.2 Examples

```promql
# Per-host 95th percentile request latency over last 5 minutes
histogram_quantile(0.95,
  rate(http_request_duration_seconds_bucket[5m])
)

# Per-second error rate aggregated by service
sum by (service) (rate(http_requests_total{status=~"5.."}[1m]))
```

## 3. Where it differs from upstream

- **Same substrate as SQL** — series data lives in JouleDB tables, accessible via both PromQL/InfluxQL and SQL.
- **Energy receipts** — every query returns total µWh cost.
- **Mixed predicate types** — you can join time-series queries with relational queries via CTEs / sub-queries.

## 4. See also

- [`crates/joule-db-query/README.md`](../../crates/joule-db-query/README.md)
- [`crates/joule-db-features/src/timeseries/`](../../crates/joule-db-features/src/timeseries/) — downsampling / gap-fill / windowed-aggregate operators
- [`QUERY-SQL.md`](QUERY-SQL.md) — relational queries on the same substrate
