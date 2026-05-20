# JouleDB Cloud — Operator Runbook

**Version 1.0 — 2026-05-18**
**Scope:** the four `joule-cloud-*` crates that package JouleDB as a managed Kubernetes service
**Sister docs:** [`MGAI-JWP-PROTOCOL.md`](../MGAI-JWP-PROTOCOL.md), [`MGAI-SPEC-DOMAIN-JOULEDB.md`](../MGAI-SPEC-DOMAIN-JOULEDB.md), [`WHITEPAPER-JOULEDB-2026-05.md`](../WHITEPAPER-JOULEDB-2026-05.md) §2.5

---

## 1. What the operator gives you

A four-component pipeline that takes a customer request through HTTP and ends with a JouleDB pod running JWP on port 9000:

```
Customer
   │  HTTPS
   ▼
api-gateway        (joule-cloud-api-gateway)   axum + tower, auth, rate-limit
   │
   ▼
control-plane      (joule-cloud-control-plane) orchestrator HTTP API
   │
   ▼
provisioner        (joule-cloud-provisioner)   K8s operator
   │  kube-rs
   ▼
k8s API: JouleDBCluster CRD (jouledb.cloud/v1)
   │
   ▼
StatefulSet — joule-db-server pod
   │  JWP port 9000
   ▼
PVC (fast-ssd storage class)
   │
joule-db-local on disk

(orthogonal pipeline)
billing-service    (joule-cloud-billing-service) usage → Stripe
```

| Crate | Binary | Purpose |
|---|---|---|
| [`joule-cloud-control-plane`](../../crates/joule-cloud-control-plane/) | `control-plane` | Orchestrator HTTP API |
| [`joule-cloud-provisioner`](../../crates/joule-cloud-provisioner/) | (library + `provisioner`) | K8s operator: reconciles `JouleDBCluster` CRDs |
| [`joule-cloud-api-gateway`](../../crates/joule-cloud-api-gateway/) | `api-gateway` | Customer-facing edge: auth, rate-limit, OpenAPI |
| [`joule-cloud-billing-service`](../../crates/joule-cloud-billing-service/) | `billing-service` | Usage metering + Stripe |

---

## 2. The `JouleDBCluster` CRD

API group: **`jouledb.cloud/v1`**. Kind: **`JouleDBCluster`**. Plural: **`jouledbclusters`**. Namespace: **`jouledb`** (per [`crates/joule-cloud-provisioner/src/kubernetes.rs:18`](../../crates/joule-cloud-provisioner/src/kubernetes.rs#L18)).

### 2.1 Spec shape

```yaml
apiVersion: jouledb.cloud/v1
kind: JouleDBCluster
metadata:
  name: my-cluster
  namespace: jouledb
  labels:
    app: jouledb
    cluster-id: cls_a1b2c3d4e5f6
    project-id: proj_123
    tier: startup
spec:
  replicas: 1
  version: "1.0.0"
  resources:
    requests:
      cpu: "2000m"
      memory: "4096Mi"
    limits:
      cpu: "4000m"        # 2× the requests, set automatically
      memory: "8192Mi"
  storage:
    size: "10Gi"
    storageClass: "fast-ssd"
  networking:
    serviceType: "LoadBalancer"
    tlsEnabled: true
```

Definition: [`crates/joule-cloud-provisioner/src/kubernetes.rs:36-67`](../../crates/joule-cloud-provisioner/src/kubernetes.rs#L36).

### 2.2 What the provisioner does on reconcile

[`KubeClient::create_cluster`](../../crates/joule-cloud-provisioner/src/kubernetes.rs#L144) calls the K8s API and creates the resource as a `DynamicObject`. Downstream controllers (in production: a JouleDB-specific K8s controller; in test: mocks) translate the CRD into a `StatefulSet` + `Service` + `PVC`.

Lifecycle states (from [`crates/joule-cloud-provisioner/src/lib.rs:78-86`](../../crates/joule-cloud-provisioner/src/lib.rs#L78)):

```text
Provisioning → Running → Scaling → Running
              ↘ Paused ↗
              ↘ Deleting → (resource gone)
              ↘ Failed
```

---

## 3. Tier presets

The control plane exposes three default tiers ([`crates/joule-cloud-provisioner/src/lib.rs:34-74`](../../crates/joule-cloud-provisioner/src/lib.rs#L34)):

| Tier | CPU (millicores) | Memory | Storage | Replicas |
|---|---|---|---|---|
| **free** | 250 | 512 MiB | 1 GiB | 1 |
| **startup** | 2000 | 4 GiB | 10 GiB | 1 |
| **business** | 8000 | 32 GiB | 100 GiB | 3 |

`ResourceSpec::for_tier(name)` resolves the tier name to a `ResourceSpec`. Unknown tiers default to `startup` (intentional — avoids silent free-tier provisioning when a customer mistypes).

CPU / memory **limits** are set to **2× requests** automatically by `KubernetesResources::from`. This gives bursty headroom without permanent over-provisioning.

Storage class is hardcoded to `fast-ssd`. If your cluster doesn't have a `fast-ssd` `StorageClass`, create one pointing at NVMe-class persistent volumes (or alias your existing class).

---

## 4. Deploy the operator (cluster-side install)

### 4.1 Prereqs

- Kubernetes ≥ 1.24 (`kube-rs` supports the v1 CRD API)
- `kubectl` access with permissions to create CRDs in the cluster
- A `StorageClass` named `fast-ssd` (or aliased) — backed by NVMe / fast SSD volumes
- A container registry path for the JouleDB server image

### 4.2 Install the CRD

```yaml
# crd.yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: jouledbclusters.jouledb.cloud
spec:
  group: jouledb.cloud
  names:
    kind: JouleDBCluster
    plural: jouledbclusters
    singular: jouledbcluster
    shortNames: [jdbc]
  scope: Namespaced
  versions:
    - name: v1
      served: true
      storage: true
      schema:
        openAPIV3Schema:
          type: object
          properties:
            spec:
              type: object
              properties:
                replicas: { type: integer }
                version: { type: string }
                resources:
                  type: object
                  properties:
                    requests: { type: object }
                    limits: { type: object }
                storage:
                  type: object
                  properties:
                    size: { type: string }
                    storageClass: { type: string }
                networking:
                  type: object
                  properties:
                    serviceType: { type: string }
                    tlsEnabled: { type: boolean }
```

```bash
kubectl create namespace jouledb
kubectl apply -f crd.yaml
```

### 4.3 Install the provisioner

Run the provisioner binary as a `Deployment` in the `jouledb` namespace, with a service account that has rights on the CRD:

```yaml
# rbac.yaml — minimum permissions for the operator
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: jouledb-operator
rules:
  - apiGroups: ["jouledb.cloud"]
    resources: ["jouledbclusters"]
    verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
  - apiGroups: [""]
    resources: ["pods", "services", "persistentvolumeclaims", "configmaps", "secrets"]
    verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
  - apiGroups: ["apps"]
    resources: ["statefulsets", "deployments"]
    verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
```

(Binding, ServiceAccount, and Deployment manifests are mechanical — pattern after any controller-runtime setup.)

The provisioner's K8s client is initialized via `Client::try_default().await` ([`kubernetes.rs:127`](../../crates/joule-cloud-provisioner/src/kubernetes.rs#L127)) — pick up service-account credentials from the in-cluster default path.

---

## 5. Provision a cluster — three paths

### 5.1 Direct `kubectl apply`

For one-off / test deployments, write a `JouleDBCluster` resource and apply it directly:

```yaml
apiVersion: jouledb.cloud/v1
kind: JouleDBCluster
metadata:
  name: my-cluster
  namespace: jouledb
spec:
  replicas: 1
  version: "1.0.0"
  resources:
    requests: { cpu: "2000m", memory: "4096Mi" }
    limits:   { cpu: "4000m", memory: "8192Mi" }
  storage: { size: "10Gi", storageClass: "fast-ssd" }
  networking: { serviceType: "LoadBalancer", tlsEnabled: true }
```

```bash
kubectl apply -f my-cluster.yaml
kubectl get jouledbcluster -n jouledb
```

### 5.2 Control-plane HTTP API

For programmatic / customer-self-serve flows, go through the control plane:

```bash
curl -X POST https://api.jouledb.cloud/v1/projects/proj_123/clusters \
     -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
       "name": "my-cluster",
       "tier": "startup",
       "region": "us-east-1",
       "version": "1.0.0"
     }'
```

The control plane resolves the tier to a `ResourceSpec`, generates a cluster id (`cls_<12 hex chars>`), and calls into the provisioner. It also persists the cluster record to disk via [`crates/joule-cloud-control-plane/src/persist.rs`](../../crates/joule-cloud-control-plane/src/persist.rs).

OpenAPI spec for the full surface: [`crates/joule-cloud-api-gateway/openapi.yaml`](../../crates/joule-cloud-api-gateway/openapi.yaml).

### 5.3 Rust SDK

```rust,ignore
use joule_cloud_provisioner::{Provisioner, ClusterSpec, ResourceSpec};

let mut prov = Provisioner::with_persistence("/var/lib/jouledb-prov/clusters.json")?;
let spec = ClusterSpec {
    name: "my-cluster".into(),
    project_id: "proj_123".into(),
    tier: "startup".into(),
    region: "us-east-1".into(),
    version: "1.0.0".into(),
    resources: ResourceSpec::startup_tier(),
};
let cluster = prov.create_cluster(spec).await?;
println!("endpoint: {}", cluster.endpoint.unwrap());
// endpoint: my-cluster.jouledb.cloud:9000
```

---

## 6. Lifecycle operations

### 6.1 Scale

```bash
# Via kubectl
kubectl patch jouledbcluster my-cluster -n jouledb --type=merge -p '{
  "spec": { "replicas": 3, "resources": { "requests": { "cpu": "8000m", "memory": "32768Mi" } } }
}'

# Via control plane
curl -X PATCH https://api.jouledb.cloud/v1/clusters/cls_a1b2c3d4 \
     -H "Authorization: Bearer $TOKEN" \
     -d '{ "tier": "business" }'
```

Rust: `Provisioner::scale_cluster(cluster_id, new_resources)` ([`lib.rs:170`](../../crates/joule-cloud-provisioner/src/lib.rs#L170)).

### 6.2 Pause / resume

Pause stops the StatefulSet (scale to 0) but preserves the PVC. Resume scales back up.

```rust,ignore
prov.pause_cluster(&cluster_id).await?;   // → Paused
prov.resume_cluster(&cluster_id).await?;  // → Running
```

Use for cost optimization on staging / dev clusters that don't need 24/7 availability.

### 6.3 Delete

```bash
kubectl delete jouledbcluster my-cluster -n jouledb
```

Deletes the K8s resource. The PVC is **not** automatically deleted (StatefulSet semantics) — set the storage class's `reclaimPolicy` or use a finalizer if you want auto-cleanup.

Rust: `Provisioner::delete_cluster(cluster_id)` ([`lib.rs:226`](../../crates/joule-cloud-provisioner/src/lib.rs#L226)).

---

## 7. Endpoints, ports, TLS

Every cluster gets a public endpoint of the form `{name}.jouledb.cloud:9000`. Port 9000 is the JouleDB JWP wire protocol (see [`MGAI-JWP-PROTOCOL.md`](../MGAI-JWP-PROTOCOL.md)). Other ports the StatefulSet typically exposes inside the cluster:

| Port | Protocol | Surface |
|---|---|---|
| 9000 | TCP | JWP (external) |
| 9090 | TCP | JWP (local-default; not normally exposed) |
| 5432 | TCP | pgwire (optional) |
| 8080 | HTTP | REST / health |
| 4317 | gRPC | OpenTelemetry export (optional) |

TLS is **on by default** at the API gateway and the public LoadBalancer; the pod itself can run plain JWP behind a TLS-terminating service. The provisioner sets `tls_enabled: true` in the cluster spec when creating the resource.

---

## 8. State persistence

The provisioner is stateful — it tracks cluster records on disk so it survives restarts without losing what it provisioned:

```rust,ignore
let prov = Provisioner::with_persistence("/var/lib/jouledb-prov/clusters.json")?;
```

State file is JSON. Reloaded on startup via `StatePersister::load`. Every lifecycle operation calls `persist()` to flush ([`crates/joule-cloud-provisioner/src/lib.rs:122-129`](../../crates/joule-cloud-provisioner/src/lib.rs#L122)).

**Operations note:** mount the persistence path on a PVC so it survives pod reschedule, not just process restart.

---

## 9. Billing — usage → Stripe

`joule-cloud-billing-service` is the orthogonal pipeline. It subscribes to usage events (joule consumption per cluster per timeslice), aggregates them per project/customer, applies the pricing table, and pushes invoices via Stripe.

| Module | Role |
|---|---|
| [`crates/joule-cloud-billing-service/src/usage.rs`](../../crates/joule-cloud-billing-service/src/usage.rs) | Usage event ingestion + per-cluster aggregation |
| [`crates/joule-cloud-billing-service/src/pricing.rs`](../../crates/joule-cloud-billing-service/src/pricing.rs) | Tier-based pricing rules |
| [`crates/joule-cloud-billing-service/src/persist.rs`](../../crates/joule-cloud-billing-service/src/persist.rs) | Ledger persistence |
| [`crates/joule-cloud-billing-service/src/stripe_integration.rs`](../../crates/joule-cloud-billing-service/src/stripe_integration.rs) | Stripe webhook + invoice push |

Test coverage: 52 tests in [`joule-cloud-billing-service`](../../crates/joule-cloud-billing-service/).

A narrative `docs/jouledb/CLOUD-BILLING.md` runbook is queued in the punch list — for now, the test suite is the most up-to-date reference.

---

## 10. Auth at the gateway

`joule-cloud-api-gateway` handles auth before any request reaches the control plane:

| Module | Role |
|---|---|
| [`crates/joule-cloud-api-gateway/src/auth.rs`](../../crates/joule-cloud-api-gateway/src/auth.rs) | Token-based auth — JWT + API keys |
| [`crates/joule-cloud-api-gateway/src/middleware_mod.rs`](../../crates/joule-cloud-api-gateway/src/middleware_mod.rs) | Rate-limit (tower `limit` + `load-shed`), timeout (tower `timeout`) |
| [`crates/joule-cloud-api-gateway/src/config.rs`](../../crates/joule-cloud-api-gateway/src/config.rs) | Configuration |
| [`crates/joule-cloud-api-gateway/src/error.rs`](../../crates/joule-cloud-api-gateway/src/error.rs) | Error mapping (gRPC-style status codes → HTTP) |

Tower middleware stack: `timeout → limit → load-shed → auth → router`.

---

## 11. Troubleshooting

### 11.1 Cluster stuck in `Provisioning`

Check the underlying K8s resources:

```bash
kubectl get jouledbcluster my-cluster -n jouledb -o yaml
kubectl describe statefulset -n jouledb -l cluster-id=cls_a1b2c3d4
kubectl get pvc -n jouledb -l cluster-id=cls_a1b2c3d4
```

Most common cause: no `StorageClass` named `fast-ssd` exists. Either create one or alias.

### 11.2 Cluster in `Failed`

The provisioner doesn't auto-retry — it reports the failure and waits. To recover:

```bash
# Inspect why
kubectl logs -n jouledb deploy/joule-provisioner
# Delete the bad cluster and recreate
kubectl delete jouledbcluster my-cluster -n jouledb
# Re-create via your preferred path (kubectl, control plane, SDK)
```

### 11.3 Pod corruption — use `joule-db-recover`

If the JouleDB pod can't start because `meta.wdb` is corrupted but `data.wdb` is intact, exec into the pod and run the forensic recovery tool:

```bash
kubectl exec -it my-cluster-0 -n jouledb -- \
  joule-db-recover scan /data/joule.db --write-recovery-meta
```

See [`MGAI-CLI-REFERENCE.md`](../MGAI-CLI-REFERENCE.md) (forthcoming JouleDB section) for full options. The Scholar incident on 2026-05-02 validated this flow end-to-end.

### 11.4 Customer says "I'm being billed more than expected"

The flow: per-query energy receipts (`EnergyReceipt` in `joule-db-ledger`) → usage events → billing aggregator. Verify against the source:

```bash
# Query the cluster's own ledger
psql -h my-cluster.jouledb.cloud -p 5432 -c "SELECT date_trunc('day', recorded_at), sum(joules) FROM receipts GROUP BY 1 ORDER BY 1"

# Compare to billing service's totals
curl https://api.jouledb.cloud/v1/projects/proj_123/usage?from=2026-05-01&to=2026-05-18 \
  -H "Authorization: Bearer $TOKEN"
```

If they disagree, the discrepancy is in `joule-cloud-billing-service`. If they agree, the customer is using more than they think they are — show them the receipts.

---

## 12. Tests

| Crate | Test count |
|---|---|
| `joule-cloud-provisioner` | 23 |
| `joule-cloud-control-plane` | 26 |
| `joule-cloud-api-gateway` | 9 |
| `joule-cloud-billing-service` | 52 |
| **Total** | **110** |

`joule-cloud-provisioner` has a `kubernetes` feature flag — when off, `KubeClient` falls back to a mock that logs but doesn't make K8s API calls. That's how the lifecycle tests can run without an actual cluster.

---

## 13. See also

- [`MGAI-JWP-PROTOCOL.md`](../MGAI-JWP-PROTOCOL.md) — the wire protocol every cluster speaks
- [`MGAI-SPEC-DOMAIN-JOULEDB.md`](../MGAI-SPEC-DOMAIN-JOULEDB.md) — domain audit
- [`WHITEPAPER-JOULEDB-2026-05.md`](../WHITEPAPER-JOULEDB-2026-05.md) §2.5 — cloud layer in the architecture narrative
- [`crates/joule-cloud-api-gateway/openapi.yaml`](../../crates/joule-cloud-api-gateway/openapi.yaml) — machine-readable API spec
- [`docs/jouledb/CLOUD-API.md`](CLOUD-API.md) *(in progress)* — narrative wrapper over the OpenAPI
- [`docs/jouledb/CLOUD-BILLING.md`](CLOUD-BILLING.md) *(in progress)* — billing pipeline runbook

---

*Drafted 2026-05-18 as wave 2 of the JouleDB documentation parity pass.*
