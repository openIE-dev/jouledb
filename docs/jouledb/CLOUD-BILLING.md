# JouleDB Cloud Billing Runbook

**Version 1.0 — 2026-05-18**
**Source:** [`crates/joule-cloud-billing-service/`](../../crates/joule-cloud-billing-service/) (52 tests)
**Sister docs:** [`CLOUD-OPERATOR.md`](CLOUD-OPERATOR.md), [`CLOUD-API.md`](CLOUD-API.md)

The billing pipeline: usage events → aggregation → pricing → Stripe invoice. Energy-aware: every joule is metered upstream by [`joule-db-energy`](../../crates/joule-db-energy/) and [`joule-db-ledger`](../../crates/joule-db-ledger/); this pipeline turns those joules into dollars.

---

## 1. The pipeline

```
joule-db-energy (per-query µJ measurement)
   ↓
joule-db-ledger (Merkle-anchored receipts)
   ↓
joule-cloud-billing-service::usage   (event ingestion + per-cluster aggregation)
   ↓
joule-cloud-billing-service::pricing (tier-based pricing rules)
   ↓
joule-cloud-billing-service::persist (ledger persistence)
   ↓
joule-cloud-billing-service::stripe_integration (invoice push)
   ↓
Stripe API
   ↓
Customer invoice
```

---

## 2. Pricing model

Two pricing axes per cluster:

### 2.1 Reserved capacity (the cluster being up)

Per-hour rate based on tier:

| Tier | Per-hour | Per-month |
|---|---|---|
| `free` | $0 | $0 |
| `startup` | $0.15 | ~$108 |
| `business` | $1.20 | ~$870 |

Paused clusters bill at ~10% of running rate (storage only — PVC stays attached).

### 2.2 Usage (the queries you run)

Per-joule rate. Cheaper at lower cascade tiers:

| Cascade tier | µJ/query (typical) | $/M queries |
|---|---|---|
| Lookup (cache / HDC) | 0.2 µJ | $0.01 |
| Formula | 5 µJ | $0.05 |
| Extract (index scan) | 200 µJ | $0.20 |
| Aggregate | 5 mJ | $2.00 |
| Reason (LLM) | 1 J | $50.00 |
| Frontier API | (cost passthrough) | varies |

The reasoning behind the tier-pricing — see [`WHITEPAPER-JOULEDB-2026-05.md`](../WHITEPAPER-JOULEDB-2026-05.md) §2.4.3 for the cascade tier model.

### 2.3 Storage

Per-GiB-month flat rate on the cluster's PVC. ~$0.10 / GiB / month.

### 2.4 Network egress

Per-GiB egress out of the JouleDB region. ~$0.05-$0.10 / GiB depending on destination.

---

## 3. Usage events

Each metered operation produces an event:

```json
{
  "cluster_id": "cls_a1b2c3d4",
  "project_id": "proj_xyz",
  "timestamp": "2026-05-18T14:23:45.123Z",
  "operation": "query",
  "tier": "Extract",
  "joules": 0.000234,
  "rows_returned": 42,
  "elapsed_ms": 7,
  "user_id": "user_abc",
  "session_id": "sess_def"
}
```

Source: [`crates/joule-cloud-billing-service/src/usage.rs`](../../crates/joule-cloud-billing-service/src/usage.rs).

Events are batched and aggregated per cluster per timeslice (default: 5-minute buckets), then persisted via [`persist.rs`](../../crates/joule-cloud-billing-service/src/persist.rs).

---

## 4. Stripe integration

Source: [`crates/joule-cloud-billing-service/src/stripe_integration.rs`](../../crates/joule-cloud-billing-service/src/stripe_integration.rs).

### 4.1 Subscription model

Customer subscribes to a project at signup; the project's tier determines the base subscription. Usage charges are billed via metered subscription items.

### 4.2 Invoice cycle

Monthly. The billing service:

1. At end-of-cycle, computes `reserved + usage + storage + egress` for each cluster in the project
2. Pushes usage records to the Stripe metered-subscription-item
3. Stripe generates the invoice
4. Customer's payment method on file is charged
5. Webhook callback updates internal records

### 4.3 Webhooks

The service handles these Stripe webhook events:

| Event | Handler |
|---|---|
| `invoice.paid` | Mark invoice paid, extend service |
| `invoice.payment_failed` | Send notification, mark cluster `Failed-Payment` after grace period |
| `customer.subscription.updated` | Sync tier changes |
| `customer.subscription.deleted` | Pause / delete clusters |

---

## 5. Prepaid balance (alternate model)

Some customers prefer prepaid energy budgets — buy 1000 J of compute up-front, draw it down query by query. JWP exposes this directly:

| JWP frame | ID | Purpose |
|---|---|---|
| `BalanceQuery` | `0x21` | Check remaining balance |
| `BalanceResponse` | `0x22` | Server returns balance + projected runway |
| `TopupBegin` | `0x23` | Request a top-up (returns Stripe checkout URL) |
| `TopupResponse` | `0x24` | URL or confirmation |
| `UsageQuery` | `0x25` | Historical usage |
| `UsageResponse` | `0x26` | Returns usage entries |

See [`MGAI-JWP-PROTOCOL.md`](../MGAI-JWP-PROTOCOL.md) §3.9.

---

## 6. Cost dashboards

Customer-facing dashboard exposes:

- Current month's running total
- Per-cluster breakdown
- Per-tier breakdown (how many µJ on Lookup vs. mJ on Reason)
- Energy-vs-dollars correlation
- Projected month-end total

Internally, the same data feeds:

- Anomaly detection (sudden tier escalation = expensive query loop)
- Throttling decisions (approaching budget cap → rate-limit)
- Customer success outreach (customer paying for `business` tier but only using Lookup → suggest downgrade)

---

## 7. Operator tasks

### 7.1 Refunds

```bash
# Issue a credit on next invoice
curl -X POST https://api.jouledb.cloud/internal/credits \
     -H "Authorization: Bearer $OPERATOR_TOKEN" \
     -d '{"project_id": "proj_xyz", "amount_usd": 50.00, "reason": "incident-credit-2026-05-18"}'
```

### 7.2 Discrepancy investigation

Customer complains "I'm billed more than expected." Walk through:

1. Pull the cluster's ledger: `SELECT date_trunc('day', recorded_at), sum(joules) FROM receipts GROUP BY 1`
2. Pull the billing service totals: `GET /internal/usage?project_id=proj_xyz&from=...&to=...`
3. If they agree → show the customer the receipts. They're using more than they think.
4. If they disagree → there's a bug in `joule-cloud-billing-service`. Compare per-event details.

The receipts in [`joule-db-ledger`](../../crates/joule-db-ledger/) are Merkle-anchored — they're the source of truth.

### 7.3 Pricing updates

Pricing tables live in [`crates/joule-cloud-billing-service/src/pricing.rs`](../../crates/joule-cloud-billing-service/src/pricing.rs). Update there, redeploy, schedule customer notifications.

---

## 8. Test coverage

52 tests across the crate. Specific suites:

- 27 tests in `lib.rs` — top-level billing logic
- 16 tests in `stripe_integration.rs` — webhook handling, invoice push
- 6 tests in `usage.rs` — event ingestion + aggregation
- 3 tests in `persist.rs` — ledger persistence

---

## 9. See also

- [`CLOUD-OPERATOR.md`](CLOUD-OPERATOR.md)
- [`CLOUD-API.md`](CLOUD-API.md)
- [`crates/joule-db-energy/README.md`](../../crates/joule-db-energy/README.md) — joule measurement primitives
- [`crates/joule-db-ledger/README.md`](../../crates/joule-db-ledger/README.md) — Merkle-anchored receipts
- [`WHITEPAPER-JOULEDB-2026-05.md`](../WHITEPAPER-JOULEDB-2026-05.md) §2.4.3 — cascade tier model behind the pricing
