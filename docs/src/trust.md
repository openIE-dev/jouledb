# Implemented Security Controls (`trust_compliance.rs` — Unaudited)

> **Mandatory Compliance Disclosure:** Software cannot self-certify or cryptographically attest compliance with `SOC 2 Type II`, `ISO/IEC 27001`, `HIPAA`, `PCI-DSS`, `CMVP FIPS 140-3`, or `FedRAMP`. **No third-party audit, certification, CMVP validation, or agency authorization has been performed on JouleDB.** This chapter documents only the technical security primitives implemented in `crates/joule-db-server/src/trust_compliance.rs` and `crates/inv-pqcrypto`.

---

## Implemented Technical Primitives

| Control Area | What the Code Actually Implements | Verification Endpoint / Test |
| :--- | :--- | :--- |
| **Hash-Chained Audit Log** | Append-only in-engine log where each entry hashes `(seq, ts_ms, actor, control_ref, action, prev_hash)` with `SHA-256`; single-character mutation fails `verify_audit_chain()` | `POST /api/v1/trust/compliance_audit` |
| **Retention-Lock Flag (WORM Guard)** | Optional `worm_retain_until_ms` timestamp on audit records; `attempt_delete_worm_protected` rejects deletion prior to that timestamp | Unit & HTTP self-test in `trust_compliance.rs` |
| **Per-Subject DEK Zeroization** | Encrypts per-subject byte payloads under a per-subject `AES-256-GCM` key; `POST /api/v1/trust/crypto_shred` overwrites and drops the in-memory key entry | `POST /api/v1/trust/crypto_shred` |
| **Regex & Luhn Text Redaction** | Scans strings for Luhn-valid payment card numbers, `NNN-NN-NNNN` patterns, email addresses, and `sk-*` tokens, replacing matches with salted hashes | `POST /api/v1/trust/redact` |
| **Laplace Noise Budget Counter** | Adds calibrated `Laplace(sensitivity / epsilon)` noise to scalar aggregates and decrements a per-tenant `epsilon` counter | `evaluate_dp_laplace` in `trust_compliance.rs` |
| **Bell-LaPadula Lattice Check** | Pure function checking clearance level dominance and compartment subset inclusion (`No Read Up` / `No Write Down`) on labeled requests | `POST /api/v1/trust/mls_guard` |
| **Hybrid `Ed25519 + ML-DSA-65` Signing** | Signs JSON digests/manifests using `ed25519-dalek` and `fips204` (`ML-DSA-65`, non-CMVP Rust crate) | `crates/inv-pqcrypto` |
| **CycloneDX v1.6 JSON Manifest** | Emits a JSON Software & loaded-model Bill of Materials summary | `GET /api/v1/trust/sbom` / `jouledb trust --sbom` |

---

## Running the Technical Control Self-Tests

```bash
# Run the in-process unit self-tests for the primitives above (returns external_audit_performed: false)
jouledb trust

# Emit CycloneDX v1.6 JSON summary
jouledb trust --sbom > sbom-v0.7.0.cdx.json
```
