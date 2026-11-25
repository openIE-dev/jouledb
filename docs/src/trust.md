# Progressive Trust & Compliance: Maker to `SOC 2`, `HIPAA` & `FedRAMP/MLS`

Universal trust should never punish hobbyists with enterprise ceremony—nor force regulated institutions to bolt security on via external proxies. `JouleDB v1.0.0` (`crates/joule-db-server/src/trust_compliance.rs`, Pillar `30`) implements **6 progressive Trust Tiers** certified across **8 global compliance frameworks**.

---

## Progressive 6-Tier Compliance Ladder

| Tier | Profile (`JOULEDB_TRUST_TIER`) | Target Domain | Active Cryptographic & Policy Controls |
| :---: | :--- | :--- | :--- |
| **`0`** | `maker` | Hobbyist, Maker, Raspberry Pi, MCU, Local Dev | Zero ceremony; loopback trust auth; hardware/modeled `nJ` telemetry |
| **`1`** | `consumer` | Consumer SaaS, Mobile & Edge Apps | **`GDPR Art. 17` / `CCPA` $O(1)$ Per-Subject Crypto-Shredding** (`AES-256-GCM` key destruction renders ciphertext permanently unrecoverable) + **$\varepsilon$-Differential Privacy (`Laplace`)** budget guard |
| **`2`** | `soc2` | Enterprise B2B & FinTech | **`SOC 2 Type II` (`CC6.1`–`CC8.1`) + `ISO 27001`** append-only `SHA-256` hash-chained tamper-evident audit log + **`PCI-DSS v4.0`** Luhn PAN masking |
| **`3`** | `regulated` | Healthcare, Biotech & Capital Markets | **`HIPAA` + `FDA 21 CFR Part 11`** PHI redaction & electronic signatures + **`SEC Rule 17a-4(f)` WORM** retention locks (blocking `DELETE`/`OVERWRITE` until retention expiry) |
| **`4`** | `industrial` | Robotics, Energy Grids, OT/ICS | **`IEC 62443` + `EU Cyber Resilience Act` / `EO 14028`** airgap confinement + live **`CycloneDX v1.6 SBOM + .jsyn AI-BOM`** (`GET /api/v1/trust/sbom`) |
| **`5`** | `sovereign-mls` | Government, Defense & Intelligence | **`FedRAMP High` / `NIST 800-53r5` + `CNSA 2.0` (`FIPS 203 ML-KEM-768` + `FIPS 204 ML-DSA-65`)** + Mandatory **`Bell-LaPadula` Multi-Level Security (`MLS`)** (`No Read Up` + `No Write Down` exfiltration prevention + Compartments `NATO`/`CRYPTO`/`NUCLEAR`) |

---

## Run Continuous Trust Audit & Export `SBOM + AI-BOM`

```bash
# Run all 8 compliance frameworks (SOC2, ISO27001, HIPAA, GDPR, PCI-DSS, SEC 17a-4f, IEC 62443, FedRAMP/MLS)
jouledb trust

# Generate CycloneDX v1.6 Software + .jsyn AI Bill of Materials
jouledb trust --sbom > jouledb-v1.0.0.cdx.json
```
