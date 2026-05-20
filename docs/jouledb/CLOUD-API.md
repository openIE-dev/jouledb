# JouleDB Cloud API Reference

**Version 1.0 — 2026-05-18**
**OpenAPI spec:** [`crates/joule-cloud-api-gateway/openapi.yaml`](../../crates/joule-cloud-api-gateway/openapi.yaml)
**Base URLs:**
- Production: `https://api.jouledb.cloud`
- Local dev: `http://localhost:8080`
**Sister docs:** [`CLOUD-OPERATOR.md`](CLOUD-OPERATOR.md), [`CLOUD-BILLING.md`](CLOUD-BILLING.md)

This is the narrative companion to the machine-readable OpenAPI spec. For tool-driven access, generate clients from the YAML directly.

---

## 1. Auth

All endpoints under `/v1/…` require auth. The flow:

1. `POST /v1/auth/login` with email + password → returns access token + refresh token
2. Use the access token in `Authorization: Bearer <token>` on every subsequent request
3. When access token expires, `POST /v1/auth/refresh` with the refresh token
4. `POST /v1/auth/logout` to invalidate

For machine-to-machine integrations, use API keys:

```bash
# Create an API key (returns it once — store it)
curl -X POST https://api.jouledb.cloud/v1/auth/api-keys \
     -H "Authorization: Bearer $TOKEN" \
     -d '{"name": "ci-deploy", "scopes": ["clusters:read", "queries:run"]}'

# Use it
curl https://api.jouledb.cloud/v1/clusters \
     -H "X-API-Key: $API_KEY"
```

API keys can be revoked via `DELETE /v1/auth/api-keys/{id}`.

---

## 2. Health

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness probe — returns 200 if the API gateway process is up |
| `GET /ready` | Readiness probe — returns 503 if downstream control-plane / provisioner unreachable |

Use for K8s probes and load-balancer health checks.

---

## 3. Projects

A project is a billing + IAM container for clusters. Most accounts have one or a few.

| Endpoint | Purpose |
|---|---|
| `GET /v1/projects` | List all projects in the account |
| `POST /v1/projects` | Create — `{ "name": "my-app", "region": "us-east-1" }` |
| `GET /v1/projects/{id}` | Get one |
| `PATCH /v1/projects/{id}` | Update name / region / metadata |
| `DELETE /v1/projects/{id}` | Delete (cascades to clusters — confirm prompt required) |

---

## 4. Clusters

The main control surface — provision, scale, pause, delete JouleDB clusters.

| Endpoint | Purpose |
|---|---|
| `GET /v1/projects/{project_id}/clusters` | List clusters in a project |
| `POST /v1/projects/{project_id}/clusters` | Create — body shape below |
| `GET /v1/clusters/{id}` | Get cluster details (incl. endpoint, status, current resources) |
| `PATCH /v1/clusters/{id}` | Scale — change tier or explicit resources |
| `DELETE /v1/clusters/{id}` | Delete |
| `POST /v1/clusters/{id}/pause` | Scale to 0 replicas, keep PVC |
| `POST /v1/clusters/{id}/resume` | Restart from paused state |

### Create body

```json
{
  "name": "prod-east",
  "tier": "business",
  "region": "us-east-1",
  "version": "1.0.0"
}
```

Tier resolves to `ResourceSpec` per [`CLOUD-OPERATOR.md`](CLOUD-OPERATOR.md) §3. For custom resources, pass `resources: { cpu_millicores, memory_mb, storage_gb, replicas }` instead of `tier`.

### Response

```json
{
  "id": "cls_a1b2c3d4e5f6",
  "name": "prod-east",
  "project_id": "proj_xyz",
  "tier": "business",
  "region": "us-east-1",
  "version": "1.0.0",
  "status": "Provisioning",
  "endpoint": null,
  "created_at": "2026-05-18T14:23:45Z",
  "updated_at": "2026-05-18T14:23:45Z"
}
```

Poll `GET /v1/clusters/{id}` until `status: Running` and `endpoint` is populated. Typical provisioning time: 30-60 seconds.

---

## 5. Query proxy

Run queries through the control plane (no separate JWP / pgwire connection needed) — useful for one-off operations or thin web clients.

```bash
curl -X POST https://api.jouledb.cloud/v1/clusters/cls_a1b2c3d4/query \
     -H "Authorization: Bearer $TOKEN" \
     -d '{"sql": "SELECT count(*) FROM users", "language": "sql"}'
```

```json
{
  "columns": ["count"],
  "rows": [[42]],
  "row_count": 1,
  "energy_uwh": 234,
  "tier": "Lookup",
  "elapsed_ms": 2
}
```

For high-throughput / streaming, connect directly to the cluster's JWP / pgwire endpoint instead.

---

## 6. Billing

| Endpoint | Purpose |
|---|---|
| `GET /v1/projects/{id}/usage` | Usage for a project — joules, query count, storage, network |
| `GET /v1/projects/{id}/usage?from=YYYY-MM-DD&to=YYYY-MM-DD` | Date-range usage |
| `GET /v1/projects/{id}/invoices` | Stripe invoice history |
| `GET /v1/projects/{id}/balance` | Prepaid balance (if on prepaid tier) |
| `POST /v1/projects/{id}/topup` | Add to prepaid balance — returns Stripe checkout URL |

See [`CLOUD-BILLING.md`](CLOUD-BILLING.md) for the metering and pricing model.

---

## 7. Rate limits

The API gateway enforces rate limits per API key / token:

| Tier | RPM | Concurrent connections |
|---|---|---|
| free | 60 | 10 |
| startup | 600 | 50 |
| business | 6000 | 500 |
| enterprise | negotiated | negotiated |

429 responses include `Retry-After` header. Burst capacity = 2× rate limit for short windows.

---

## 8. Error format

All error responses follow:

```json
{
  "error": {
    "code": "CLUSTER_NOT_FOUND",
    "message": "Cluster cls_xxx does not exist or you don't have access",
    "details": { "cluster_id": "cls_xxx" },
    "trace_id": "01H..."
  }
}
```

Standard HTTP status codes apply (400 / 401 / 403 / 404 / 409 / 429 / 500 / 503).

---

## 9. SDKs

Auto-generated from the OpenAPI spec. The canonical generators:

```bash
# Python
openapi-generator generate -i openapi.yaml -g python -o jouledb-py/

# TypeScript
openapi-generator generate -i openapi.yaml -g typescript-axios -o jouledb-ts/

# Go
openapi-generator generate -i openapi.yaml -g go -o jouledb-go/

# Rust
openapi-generator generate -i openapi.yaml -g rust -o jouledb-rs/
```

A first-party Rust SDK is in [`crates/joule-cloud-control-plane`](../../crates/joule-cloud-control-plane/) — the same types the server uses internally.

---

## 10. See also

- [`CLOUD-OPERATOR.md`](CLOUD-OPERATOR.md) — operator runbook (Kubernetes-side)
- [`CLOUD-BILLING.md`](CLOUD-BILLING.md) — billing pipeline
- [`crates/joule-cloud-api-gateway/openapi.yaml`](../../crates/joule-cloud-api-gateway/openapi.yaml) — machine-readable spec
- [`MGAI-JWP-PROTOCOL.md`](../MGAI-JWP-PROTOCOL.md) — direct JWP access (bypass the API)
