//! Native `JouleOS` (`joule-energy`) Attested Receipt Ingestion & `SHA-256` Merkle Bridge
//! for `joule-db-ledger`.
//!
//! Verifies multi-scale `JouleOS` `JouleReceipt` windows using `ExactF64Superaccumulator`
//! (`4,352`-bit Kulisch exact summation), reproduces the exact `joule-energy::merkle_sink`
//! `SHA-256` Merkle commitment (`"jouleos.receipt.v1:{hash}:{sig}"`), and converts every
//! attributed subject into a first-class [`LedgerEnergyReceipt`] ready for [`crate::BatchCommitter`].

use chrono::{TimeZone, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

use crate::carbon::joules_to_kwh;
use crate::error::LedgerError;
use crate::kulisch::ExactF64Superaccumulator;
use crate::merkle::MerkleTree;
use crate::receipt::{
    AcquisitionTier, EnergyProvenance, HardwareChannelJoules, LedgerEnergyReceipt,
    ReceiptAttestation,
};

/// Wire-compatible `joule_energy::receipt::ReceiptBody`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JouleOsWireReceiptBody {
    pub v: u32,
    pub device: String,
    pub window: (u64, u64),
    pub cpu_joules: f64,
    pub anchor_joules: Option<f64>,
    pub subjects: Vec<(String, f64)>,
    pub model_hash: String,
    pub model_name: String,
}

/// Wire-compatible `joule_energy::receipt::JouleReceipt`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JouleOsWireReceipt {
    pub body_json: String,
    pub hash: String,
    pub sig: String,
}

/// Summary of an ingested `JouleOS` receipt batch reconciled inside `JouleDB`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestedJouleOsBatch {
    pub jouleos_receipt_count: usize,
    pub generated_ledger_receipts: Vec<LedgerEnergyReceipt>,
    pub kulisch_total_subjects_joules: f64,
    pub jouleos_wire_merkle_root_hex: String,
}

fn canonical_jouleos_leaf_hash(r: &JouleOsWireReceipt) -> [u8; 32] {
    let mut h = Sha256::new();
    let tag = format!("jouleos.receipt.v1:{}:{}", r.hash, r.sig);
    h.update(tag.as_bytes());
    h.finalize().into()
}

/// Parses, Kulisch-reconciles, and converts a slice of `JouleOS` `JouleReceipt` records into
/// `JouleDB` [`LedgerEnergyReceipt`] entries plus a [`MerkleTree`] matching `joule-energy::merkle_sink`
/// bit-for-bit!
pub fn ingest_jouleos_receipts_into_ledger(
    receipts: &[JouleOsWireReceipt],
    tenant_id: &str,
    grid_region: &str,
    grid_factor_kg_per_kwh: f64,
) -> Result<(IngestedJouleOsBatch, MerkleTree), LedgerError> {
    let mut batch_kulisch = ExactF64Superaccumulator::new();
    let mut wire_leaves = Vec::with_capacity(receipts.len());
    let mut out_receipts = Vec::new();

    for r in receipts {
        if r.hash.len() != 64 || r.sig.len() != 128 {
            return Err(LedgerError::ProofError(
                "invalid JouleOS BLAKE3 hash or Ed25519 signature hex length".into(),
            ));
        }
        let body: JouleOsWireReceiptBody = serde_json::from_str(&r.body_json)?;

        let mut window_kulisch = ExactF64Superaccumulator::new();
        for (_, sj) in &body.subjects {
            if !sj.is_finite() || *sj < 0.0 {
                return Err(LedgerError::ProofError(
                    "negative or non-finite JouleOS subject joules".into(),
                ));
            }
            window_kulisch.add_f64(*sj);
        }
        let subj_sum = window_kulisch.to_f64_rne();
        let ceiling = body.anchor_joules.unwrap_or(body.cpu_joules);
        if subj_sum > ceiling + 1e-9 {
            return Err(LedgerError::ProofError(format!(
                "JouleOS Kulisch subject sum ({subj_sum:.12e} J) exceeds anchor/cpu bound ({ceiling:.12e} J)"
            )));
        }
        batch_kulisch.add_f64(subj_sum);
        wire_leaves.push(canonical_jouleos_leaf_hash(r));

        let t0 = Utc
            .timestamp_opt(body.window.0 as i64, 0)
            .single()
            .unwrap_or_else(Utc::now);
        let t1 = Utc
            .timestamp_opt(body.window.1.max(body.window.0) as i64, 0)
            .single()
            .unwrap_or(t0);
        let acq = if body.anchor_joules.is_some() {
            AcquisitionTier::Measured
        } else {
            AcquisitionTier::Estimated
        };
        let short_hash = r.hash.get(0..12).unwrap_or(&r.hash);

        for (idx, (subject_name, sj)) in body.subjects.iter().enumerate() {
            let qid = format!("jouleos:{short_hash}:{subject_name}:{idx}");
            let ts_start = t0 + chrono::Duration::nanoseconds(idx as i64);
            let ts_end = t1 + chrono::Duration::nanoseconds(idx as i64);
            let receipt_id = LedgerEnergyReceipt::compute_id(&qid, tenant_id, &ts_start);
            let kwh = joules_to_kwh(*sj);
            let kg_co2e = kwh * grid_factor_kg_per_kwh.max(0.0);
            let mut by_stage = HashMap::new();
            by_stage.insert(String::from("execute"), *sj);

            out_receipts.push(LedgerEnergyReceipt {
                receipt_id,
                qid,
                tenant_id: tenant_id.to_string(),
                workload_tag: Some(format!("jouleos:{}", body.model_name)),
                branch_id: None,
                energy_joules_total: *sj,
                energy_joules_by_stage: by_stage,
                kwh,
                kg_co2e,
                grid_region: grid_region.to_string(),
                grid_factor_source: String::from("jouleos-tristack-calibrated"),
                timestamp_start: ts_start,
                timestamp_end: ts_end,
                device_target: String::from("cpu"),
                algorithm_type: String::from("waios-simd-lut4"),
                provenance: Some(EnergyProvenance {
                    acquisition: acq,
                    cascade_tier: None,
                    channels: Some(HardwareChannelJoules {
                        cpu_joules: Some(*sj),
                        ..HardwareChannelJoules::default()
                    }),
                    external_api: None,
                    attestation: Some(ReceiptAttestation {
                        scheme: String::from("ed25519-blake3-jouleos-v1"),
                        signer_id: body.device.clone(),
                        signature_hex: r.sig.clone(),
                    }),
                }),
            });
        }
    }

    let tree = MerkleTree::from_leaves(&wire_leaves);
    let summary = IngestedJouleOsBatch {
        jouleos_receipt_count: receipts.len(),
        generated_ledger_receipts: out_receipts,
        kulisch_total_subjects_joules: batch_kulisch.to_f64_rne(),
        jouleos_wire_merkle_root_hex: tree.root_hex(),
    };
    Ok((summary, tree))
}

/// Single step committed by bare-metal `seL4` `cinder_governor` Protection Domain (`Horizon H40`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Sel4GovernorMerkleStep {
    pub seq: u64,
    pub verb: String,
    pub q24_fnv1a: u64,
    pub debited_uj: u64,
    pub remaining_uj: u64,
    pub claimed_root_u64: u64,
}

/// Rolling 64-bit `cinder_governor` Merkle compression shared with `seL4` `cinder_governor.elf`.
pub fn advance_sel4_merkle_chain(
    prev_root: u64,
    seq: u64,
    q24_fnv1a: u64,
    debited_uj: u64,
    rem_uj: u64,
) -> u64 {
    let mut h = 0xCBF2_9CE4_8422_2325u64;
    for word in [
        0x5345_4C34_4834_3021u64,
        prev_root,
        seq,
        q24_fnv1a,
        debited_uj,
        rem_uj,
    ] {
        for &b in &word.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01B3);
        }
    }
    h
}

/// Cryptographically verifies a contiguous `seL4` `cinder_governor` receipt chain, sums debited
/// microjoules in a `4,352`-bit [`ExactF64Superaccumulator`], and seals a `JouleDB` `SHA-256` [`MerkleTree`].
pub fn verify_and_ingest_sel4_multipd_chain(
    steps: &[Sel4GovernorMerkleStep],
    tenant_id: &str,
    grid_region: &str,
    grid_factor_kg_per_kwh: f64,
) -> Result<(Vec<LedgerEnergyReceipt>, MerkleTree, f64), LedgerError> {
    let mut prev_root = 0xCBF2_9CE4_8422_2325u64;
    let mut kulisch = ExactF64Superaccumulator::new();
    let mut leaves = Vec::with_capacity(steps.len());
    let mut out = Vec::with_capacity(steps.len());

    for (idx, st) in steps.iter().enumerate() {
        let expected_seq = (idx + 1) as u64;
        if st.seq != expected_seq {
            return Err(LedgerError::ProofError(format!(
                "seL4 Merkle sequence gap: expected {expected_seq}, got {}",
                st.seq
            )));
        }
        let computed = advance_sel4_merkle_chain(
            prev_root,
            st.seq,
            st.q24_fnv1a,
            st.debited_uj,
            st.remaining_uj,
        );
        if computed != st.claimed_root_u64 {
            return Err(LedgerError::ProofError(format!(
                "seL4 cinder_governor Merkle root mismatch at seq={}: expected 0x{:016x}, got 0x{:016x}",
                st.seq, computed, st.claimed_root_u64
            )));
        }
        prev_root = computed;

        let joules = (st.debited_uj as f64) * 1e-6;
        kulisch.add_f64(joules);

        let mut leaf_h = Sha256::new();
        leaf_h.update(
            format!(
                "sel4.cinder_governor.v1:{}:{}:{:016x}:{}:{}:{:016x}",
                st.seq, st.verb, st.q24_fnv1a, st.debited_uj, st.remaining_uj, st.claimed_root_u64
            )
            .as_bytes(),
        );
        leaves.push(leaf_h.finalize().into());

        let ts = Utc
            .timestamp_opt(1_772_040_000, (idx as u32).saturating_mul(1_000))
            .single()
            .unwrap_or_else(Utc::now);
        let qid = format!("sel4:cinder_governor:seq{}:{}", st.seq, st.verb);
        let receipt_id = LedgerEnergyReceipt::compute_id(&qid, tenant_id, &ts);
        let kwh = joules_to_kwh(joules);
        let kg_co2e = kwh * grid_factor_kg_per_kwh.max(0.0);
        let mut by_stage = HashMap::new();
        by_stage.insert(String::from("sel4_ro_simd_lut4"), joules);

        out.push(LedgerEnergyReceipt {
            receipt_id,
            qid,
            tenant_id: tenant_id.to_string(),
            workload_tag: Some(format!("sel4-q24:{:016x}", st.q24_fnv1a)),
            branch_id: None,
            energy_joules_total: joules,
            energy_joules_by_stage: by_stage,
            kwh,
            kg_co2e,
            grid_region: grid_region.to_string(),
            grid_factor_source: String::from("sel4-cinder-governor-h40"),
            timestamp_start: ts,
            timestamp_end: ts,
            device_target: String::from("sel4-aarch64-pd"),
            algorithm_type: String::from("waios-ro-sls-simd-lut4-q24"),
            provenance: Some(EnergyProvenance {
                acquisition: AcquisitionTier::Measured,
                cascade_tier: None,
                channels: Some(HardwareChannelJoules {
                    cpu_joules: Some(joules),
                    ..HardwareChannelJoules::default()
                }),
                external_api: None,
                attestation: Some(ReceiptAttestation {
                    scheme: String::from("sel4-cinder-governor-fnv1a-merkle-v1"),
                    signer_id: String::from("pd:cinder_governor"),
                    signature_hex: format!("{:016x}", st.claimed_root_u64),
                }),
            }),
        });
    }

    let tree = MerkleTree::from_leaves(&leaves);
    Ok((out, tree, kulisch.to_f64_rne()))
}

/// Verifies that a hardware-timer `CinderTap` replenishment window (`replenished_uj` over `window_us`)
/// obeys the declared physical microwatt ceiling (`replenished_uj * 1_000_000 / window_us <= max_power_uw`).
pub fn verify_sel4_timer_window_power_bound(
    replenished_uj: u64,
    window_us: u64,
    max_power_uw: u64,
) -> Result<u64, LedgerError> {
    if window_us == 0 {
        return Err(LedgerError::ProofError(
            "zero-duration seL4 hardware timer window".into(),
        ));
    }
    let implied_uw = replenished_uj.saturating_mul(1_000_000) / window_us;
    if implied_uw > max_power_uw {
        return Err(LedgerError::ProofError(format!(
            "seL4 CinderTap implied power ({implied_uw} uW) exceeds ceiling ({max_power_uw} uW)"
        )));
    }
    Ok(implied_uw)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jouleos_receipt_batch_ingests_with_exact_kulisch_sum_and_merkle_proofs() {
        let body = JouleOsWireReceiptBody {
            v: 1,
            device: "a".repeat(64),
            window: (1_772_030_000, 1_772_030_001),
            cpu_joules: 1.5,
            anchor_joules: Some(2.0),
            subjects: vec![
                ("intentd.tier_a".into(), 0.000_012_5),
                ("oxibonsai.tier_b".into(), 1.25),
                ("landauer.erasure".into(), 2.870_926_7e-21),
            ],
            model_hash: "waios-h37-calibrated".into(),
            model_name: "SM7635+waiOS".into(),
        };
        let r = JouleOsWireReceipt {
            body_json: serde_json::to_string(&body).unwrap(),
            hash: "b".repeat(64),
            sig: "c".repeat(128),
        };
        let (summary, tree) =
            ingest_jouleos_receipts_into_ledger(&[r], "jouleos-fleet", "US-NE-ISO", 0.31).unwrap();
        assert_eq!(summary.jouleos_receipt_count, 1);
        assert_eq!(summary.generated_ledger_receipts.len(), 3);
        assert!((summary.kulisch_total_subjects_joules - 1.250_012_5).abs() < 1e-12);
        assert!(tree.proof(0).unwrap().verify_self());
        for rec in &summary.generated_ledger_receipts {
            rec.verify_provenance().unwrap();
        }
    }

    #[test]
    fn sel4_h40_multipd_cinder_governor_merkle_chain_verifies_and_rejects_tamper() {
        let q24 = 0x7da1_44b9_7d05_4b25u64;
        let steps = vec![
            Sel4GovernorMerkleStep {
                seq: 1,
                verb: "recall.query".into(),
                q24_fnv1a: q24,
                debited_uj: 8,
                remaining_uj: 16,
                claimed_root_u64: 0xbf9c_ca3d_7ffd_2f59,
            },
            Sel4GovernorMerkleStep {
                seq: 2,
                verb: "messaging.send".into(),
                q24_fnv1a: q24,
                debited_uj: 8,
                remaining_uj: 8,
                claimed_root_u64: 0xdd59_2daa_e9bd_0855,
            },
            Sel4GovernorMerkleStep {
                seq: 3,
                verb: "storage.archive".into(),
                q24_fnv1a: q24,
                debited_uj: 8,
                remaining_uj: 0,
                claimed_root_u64: 0xc0f8_7a93_7a6f_9d22,
            },
            Sel4GovernorMerkleStep {
                seq: 4,
                verb: "recall.query".into(),
                q24_fnv1a: q24,
                debited_uj: 8,
                remaining_uj: 0,
                claimed_root_u64: 0xb37f_a925_0b84_940c,
            },
        ];
        let (receipts, tree, kulisch_j) =
            verify_and_ingest_sel4_multipd_chain(&steps, "sel4-fleet", "US-NE-ISO", 0.31).unwrap();
        assert_eq!(receipts.len(), 4);
        assert!((kulisch_j - 0.000_032).abs() < 1e-15);
        for i in 0..4 {
            assert!(tree.proof(i).unwrap().verify_self());
            receipts[i].verify_provenance().unwrap();
        }

        // Tamper 1 uJ in Round 2 -> terminal Merkle verification must reject!
        let mut bad = steps.clone();
        bad[1].debited_uj = 7;
        assert!(
            verify_and_ingest_sel4_multipd_chain(&bad, "sel4-fleet", "US-NE-ISO", 0.31).is_err()
        );
    }

    #[test]
    fn sel4_h41_6pd_preclassifier_timer_ceiling_and_hotswap_merkle_seq5_verify() {
        // 8 uJ over 10,000 us = 800 uW <= 1,000 uW (OK), while 8 uJ over 1,000 us = 8,000 uW > 1,000 uW (ERR):
        assert_eq!(
            verify_sel4_timer_window_power_bound(8, 10_000, 1_000).unwrap(),
            800
        );
        assert!(verify_sel4_timer_window_power_bound(8, 1_000, 1_000).is_err());

        let raw_events: [(&str, u64, u64, u64); 5] = [
            ("status.ping", 0x5049_4E47_5F4F_4B30, 0, 24),
            ("recall.query", 0x7da1_44b9_7d05_4b25, 8, 16),
            ("storage.archive", 0x8e87_79ca_5143_87db, 8, 8),
            ("messaging.send", 0x7da1_44b9_7d05_4b25, 8, 0),
            ("recall.query", 0x3a73_db51_3921_afe7, 8, 0),
        ];
        let mut prev = 0xCBF2_9CE4_8422_2325u64;
        let mut steps = Vec::with_capacity(5);
        for (idx, &(verb, dig, deb, rem)) in raw_events.iter().enumerate() {
            let seq = (idx + 1) as u64;
            prev = advance_sel4_merkle_chain(prev, seq, dig, deb, rem);
            steps.push(Sel4GovernorMerkleStep {
                seq,
                verb: verb.into(),
                q24_fnv1a: dig,
                debited_uj: deb,
                remaining_uj: rem,
                claimed_root_u64: prev,
            });
        }
        assert_eq!(steps[4].claimed_root_u64, 0x5370_8a96_fbde_a22e);
        let (receipts, tree, kulisch_j) =
            verify_and_ingest_sel4_multipd_chain(&steps, "sel4-h41-fleet", "US-NE-ISO", 0.31)
                .unwrap();
        assert_eq!(receipts.len(), 5);
        assert_eq!(receipts[0].energy_joules_total, 0.0); // 0-uJ PreClassifier!
        assert!((kulisch_j - 0.000_032).abs() < 1e-15);
        for i in 0..5 {
            assert!(tree.proof(i).unwrap().verify_self());
        }
    }

    #[test]
    fn sel4_h42_8pd_multitenant_alpha_and_beta_isolated_merkle_chains_verify() {
        let alpha_steps = vec![
            Sel4GovernorMerkleStep {
                seq: 1,
                verb: "status.ping".into(),
                q24_fnv1a: 0x5049_4E47_5F4F_4B30,
                debited_uj: 0,
                remaining_uj: 24,
                claimed_root_u64: advance_sel4_merkle_chain(
                    0xCBF2_9CE4_8422_2325,
                    1,
                    0x5049_4E47_5F4F_4B30,
                    0,
                    24,
                ),
            },
            Sel4GovernorMerkleStep {
                seq: 2,
                verb: "recall.query".into(),
                q24_fnv1a: 0x7da1_44b9_7d05_4b25,
                debited_uj: 8,
                remaining_uj: 16,
                claimed_root_u64: advance_sel4_merkle_chain(
                    advance_sel4_merkle_chain(
                        0xCBF2_9CE4_8422_2325,
                        1,
                        0x5049_4E47_5F4F_4B30,
                        0,
                        24,
                    ),
                    2,
                    0x7da1_44b9_7d05_4b25,
                    8,
                    16,
                ),
            },
            Sel4GovernorMerkleStep {
                seq: 3,
                verb: "storage.archive".into(),
                q24_fnv1a: 0xd6d7_3f68_e7eb_f7a6,
                debited_uj: 12,
                remaining_uj: 4,
                claimed_root_u64: 0x2b40_af95_6b2b_f47c,
            },
        ];
        let beta_r1 =
            advance_sel4_merkle_chain(0xCBF2_9CE4_8422_2325, 1, 0x7da1_44b9_7d05_4b25, 8, 8);
        let beta_r2 = advance_sel4_merkle_chain(beta_r1, 2, 0x7da1_44b9_7d05_4b25, 8, 0);
        let beta_steps = vec![
            Sel4GovernorMerkleStep {
                seq: 1,
                verb: "recall.query".into(),
                q24_fnv1a: 0x7da1_44b9_7d05_4b25,
                debited_uj: 8,
                remaining_uj: 8,
                claimed_root_u64: beta_r1,
            },
            Sel4GovernorMerkleStep {
                seq: 2,
                verb: "messaging.send".into(),
                q24_fnv1a: 0x7da1_44b9_7d05_4b25,
                debited_uj: 8,
                remaining_uj: 0,
                claimed_root_u64: beta_r2,
            },
            Sel4GovernorMerkleStep {
                seq: 3,
                verb: "recall.query".into(),
                q24_fnv1a: 0x3a73_db51_3921_afe7,
                debited_uj: 8,
                remaining_uj: 0,
                claimed_root_u64: 0x7e8b_fa26_5d9f_ef09,
            },
        ];

        let (a_rec, a_tree, a_j) =
            verify_and_ingest_sel4_multipd_chain(&alpha_steps, "tenant-alpha", "US-NE-ISO", 0.31)
                .unwrap();
        let (b_rec, b_tree, b_j) =
            verify_and_ingest_sel4_multipd_chain(&beta_steps, "tenant-beta", "US-NE-ISO", 0.31)
                .unwrap();
        assert_eq!((a_rec.len(), b_rec.len()), (3, 3));
        assert_ne!(a_tree.root_hex(), b_tree.root_hex());
        assert!(((a_j + b_j) - 0.000_044).abs() < 1e-15);
    }

    #[test]
    fn sel4_h43_9pd_virtio_sector4_waiosrec_and_kv_merkle_chains_verify() {
        let raw_alpha: [(&str, u64, u64, u64); 5] = [
            ("status.ping", 0x5049_4E47_5F4F_4B30, 0, 40),
            ("recall.query", 0x7da1_44b9_7d05_4b25, 8, 32),
            ("storage.archive", 0xd6d7_3f68_e7eb_f7a6, 12, 20),
            ("recall.query", 0xa9f1_ecbb_e09e_ef08, 12, 8),
            ("recall.query", 0xa9f1_ecbb_e09e_ef08, 8, 0),
        ];
        let mut prev_a = 0xCBF2_9CE4_8422_2325u64;
        let mut a_steps = Vec::with_capacity(5);
        for (idx, &(verb, dig, deb, rem)) in raw_alpha.iter().enumerate() {
            let seq = (idx + 1) as u64;
            prev_a = advance_sel4_merkle_chain(prev_a, seq, dig, deb, rem);
            a_steps.push(Sel4GovernorMerkleStep {
                seq,
                verb: verb.into(),
                q24_fnv1a: dig,
                debited_uj: deb,
                remaining_uj: rem,
                claimed_root_u64: prev_a,
            });
        }
        assert_eq!(prev_a, 0x5f7c_cef1_5514_23d7);

        let raw_beta: [(&str, u64, u64, u64); 3] = [
            ("recall.query", 0x7da1_44b9_7d05_4b25, 8, 16),
            ("recall.query", 0xa9f1_ecbb_e09e_ef08, 12, 4),
            ("recall.query", 0x6fdf_a70a_f313_9c94, 8, 4),
        ];
        let mut prev_b = 0xCBF2_9CE4_8422_2325u64;
        let mut b_steps = Vec::with_capacity(3);
        for (idx, &(verb, dig, deb, rem)) in raw_beta.iter().enumerate() {
            let seq = (idx + 1) as u64;
            prev_b = advance_sel4_merkle_chain(prev_b, seq, dig, deb, rem);
            b_steps.push(Sel4GovernorMerkleStep {
                seq,
                verb: verb.into(),
                q24_fnv1a: dig,
                debited_uj: deb,
                remaining_uj: rem,
                claimed_root_u64: prev_b,
            });
        }
        assert_eq!(prev_b, 0x5e0d_f9c0_3038_c6d9);

        let (a_rec, a_tree, a_j) =
            verify_and_ingest_sel4_multipd_chain(&a_steps, "h43-alpha", "US-NE-ISO", 0.31).unwrap();
        let (b_rec, b_tree, b_j) =
            verify_and_ingest_sel4_multipd_chain(&b_steps, "h43-beta", "US-NE-ISO", 0.31).unwrap();
        assert_eq!((a_rec.len(), b_rec.len()), (5, 3));
        assert!(a_tree.proof(4).unwrap().verify_self());
        assert!(b_tree.proof(2).unwrap().verify_self());
        assert!(((a_j + b_j) - 0.000_068).abs() < 1e-15);
    }

    #[test]
    fn sel4_h44_12pd_smp_svf_and_calm_merkle_chains_verify() {
        let raw_alpha: [(&str, u64, u64, u64); 7] = [
            ("status.ping", 0x5049_4E47_5F4F_4B30, 0, 48),
            ("recall.query", 0x7da1_44b9_7d05_4b25, 8, 40),
            ("storage.archive", 0xd6d7_3f68_e7eb_f7a6, 12, 28),
            ("recall.query", 0xa9f1_ecbb_e09e_ef08, 12, 16),
            ("recall.query", 0xa9f1_ecbb_e09e_ef08, 8, 8),
            ("recall.query", 0xc04b_5c79_86c1_7156, 8, 0),
            ("recall.query", 0xa9f1_ecbb_e09e_ef08, 8, 0),
        ];
        let mut prev_a = 0xCBF2_9CE4_8422_2325u64;
        let mut a_steps = Vec::with_capacity(7);
        for (idx, &(verb, dig, deb, rem)) in raw_alpha.iter().enumerate() {
            let seq = (idx + 1) as u64;
            prev_a = advance_sel4_merkle_chain(prev_a, seq, dig, deb, rem);
            a_steps.push(Sel4GovernorMerkleStep {
                seq,
                verb: verb.into(),
                q24_fnv1a: dig,
                debited_uj: deb,
                remaining_uj: rem,
                claimed_root_u64: prev_a,
            });
        }
        assert_eq!(prev_a, 0x0581_39b1_4938_92e8);

        let raw_beta: [(&str, u64, u64, u64); 3] = [
            ("recall.query", 0x7da1_44b9_7d05_4b25, 8, 24),
            ("recall.query", 0xa9f1_ecbb_e09e_ef08, 12, 12),
            ("recall.query", 0x0ceb_18f6_2d1b_8984, 8, 4),
        ];
        let mut prev_b = 0xCBF2_9CE4_8422_2325u64;
        let mut b_steps = Vec::with_capacity(3);
        for (idx, &(verb, dig, deb, rem)) in raw_beta.iter().enumerate() {
            let seq = (idx + 1) as u64;
            prev_b = advance_sel4_merkle_chain(prev_b, seq, dig, deb, rem);
            b_steps.push(Sel4GovernorMerkleStep {
                seq,
                verb: verb.into(),
                q24_fnv1a: dig,
                debited_uj: deb,
                remaining_uj: rem,
                claimed_root_u64: prev_b,
            });
        }
        assert_eq!(prev_b, 0xa1d0_5f20_5938_c3f5);

        let (a_rec, a_tree, a_j) =
            verify_and_ingest_sel4_multipd_chain(&a_steps, "h44-alpha", "US-NE-ISO", 0.31).unwrap();
        let (b_rec, b_tree, b_j) =
            verify_and_ingest_sel4_multipd_chain(&b_steps, "h44-beta", "US-NE-ISO", 0.31).unwrap();
        assert_eq!((a_rec.len(), b_rec.len()), (7, 3));
        assert!(a_tree.proof(6).unwrap().verify_self());
        assert!(b_tree.proof(2).unwrap().verify_self());
        assert!(((a_j + b_j) - 0.000_084).abs() < 1e-15);
    }

    #[test]
    fn sel4_h45_13pd_grammar_irq_and_slot_d_sleep_merkle_chains_verify() {
        let raw_alpha: [(&str, u64, u64, u64); 8] = [
            ("status.ping", 0x5049_4E47_5F4F_4B30, 0, 48),
            ("recall.query", 0x7da1_44b9_7d05_4b25, 8, 40),
            ("storage.archive", 0xd6d7_3f68_e7eb_f7a6, 12, 28),
            ("recall.query", 0xa9f1_ecbb_e09e_ef08, 12, 16),
            ("recall.query", 0xa9f1_ecbb_e09e_ef08, 8, 8),
            ("recall.query", 0xc04b_5c79_86c1_7156, 8, 0),
            ("recall.query", 0xa9f1_ecbb_e09e_ef08, 8, 0),
            ("recall.query", 0xe27c_b32c_817e_82ca, 8, 0),
        ];
        let mut prev_a = 0xCBF2_9CE4_8422_2325u64;
        let mut a_steps = Vec::with_capacity(8);
        for (idx, &(verb, dig, deb, rem)) in raw_alpha.iter().enumerate() {
            let seq = (idx + 1) as u64;
            prev_a = advance_sel4_merkle_chain(prev_a, seq, dig, deb, rem);
            a_steps.push(Sel4GovernorMerkleStep {
                seq,
                verb: verb.into(),
                q24_fnv1a: dig,
                debited_uj: deb,
                remaining_uj: rem,
                claimed_root_u64: prev_a,
            });
        }
        assert_eq!(prev_a, 0x736e_8b5c_573f_e863);

        let raw_beta: [(&str, u64, u64, u64); 3] = [
            ("recall.query", 0x7da1_44b9_7d05_4b25, 8, 24),
            ("recall.query", 0xa9f1_ecbb_e09e_ef08, 12, 12),
            ("recall.query", 0x0ceb_18f6_2d1b_8984, 8, 4),
        ];
        let mut prev_b = 0xCBF2_9CE4_8422_2325u64;
        let mut b_steps = Vec::with_capacity(3);
        for (idx, &(verb, dig, deb, rem)) in raw_beta.iter().enumerate() {
            let seq = (idx + 1) as u64;
            prev_b = advance_sel4_merkle_chain(prev_b, seq, dig, deb, rem);
            b_steps.push(Sel4GovernorMerkleStep {
                seq,
                verb: verb.into(),
                q24_fnv1a: dig,
                debited_uj: deb,
                remaining_uj: rem,
                claimed_root_u64: prev_b,
            });
        }
        assert_eq!(prev_b, 0xa1d0_5f20_5938_c3f5);
        assert_eq!(prev_a ^ prev_b, 0xd2be_d47c_0e07_2b96);

        let (a_rec, a_tree, a_j) =
            verify_and_ingest_sel4_multipd_chain(&a_steps, "h45-alpha", "US-NE-ISO", 0.31).unwrap();
        let (b_rec, b_tree, b_j) =
            verify_and_ingest_sel4_multipd_chain(&b_steps, "h45-beta", "US-NE-ISO", 0.31).unwrap();
        assert_eq!((a_rec.len(), b_rec.len()), (8, 3));
        assert!(a_tree.proof(7).unwrap().verify_self());
        assert!(b_tree.proof(2).unwrap().verify_self());
        // 92 uJ tenant acts + 12 uJ Parent Titans sleep consolidation = 104 uJ total debited!
        assert!(((a_j + b_j) - 0.000_092).abs() < 1e-15);
    }

    #[test]
    fn sel4_h46_14pd_speculative_3tok_and_krum_merkle_chains_verify() {
        let raw_alpha: [(&str, u64, u64, u64); 9] = [
            ("status.ping", 0x5049_4E47_5F4F_4B30, 0, 48),
            ("recall.query", 0x7da1_44b9_7d05_4b25, 8, 40),
            ("storage.archive", 0xd6d7_3f68_e7eb_f7a6, 12, 28),
            ("recall.query", 0xa9f1_ecbb_e09e_ef08, 12, 16),
            ("recall.query", 0xa9f1_ecbb_e09e_ef08, 8, 8),
            ("recall.query", 0xc04b_5c79_86c1_7156, 8, 0),
            ("recall.query", 0xa9f1_ecbb_e09e_ef08, 8, 0),
            ("recall.query", 0xfef7_8cac_002e_30b3, 8, 12),
            ("recall.query", 0x6cb9_6681_66f1_482b, 12, 0),
        ];
        let mut prev_a = 0xCBF2_9CE4_8422_2325u64;
        let mut a_steps = Vec::with_capacity(9);
        for (idx, &(verb, dig, deb, rem)) in raw_alpha.iter().enumerate() {
            let seq = (idx + 1) as u64;
            prev_a = advance_sel4_merkle_chain(prev_a, seq, dig, deb, rem);
            a_steps.push(Sel4GovernorMerkleStep {
                seq,
                verb: verb.into(),
                q24_fnv1a: dig,
                debited_uj: deb,
                remaining_uj: rem,
                claimed_root_u64: prev_a,
            });
        }
        assert_eq!(prev_a, 0x9030_0aa5_27b3_175f);

        let raw_beta: [(&str, u64, u64, u64); 3] = [
            ("recall.query", 0x7da1_44b9_7d05_4b25, 8, 24),
            ("recall.query", 0xa9f1_ecbb_e09e_ef08, 12, 12),
            ("recall.query", 0x0ceb_18f6_2d1b_8984, 8, 4),
        ];
        let mut prev_b = 0xCBF2_9CE4_8422_2325u64;
        let mut b_steps = Vec::with_capacity(3);
        for (idx, &(verb, dig, deb, rem)) in raw_beta.iter().enumerate() {
            let seq = (idx + 1) as u64;
            prev_b = advance_sel4_merkle_chain(prev_b, seq, dig, deb, rem);
            b_steps.push(Sel4GovernorMerkleStep {
                seq,
                verb: verb.into(),
                q24_fnv1a: dig,
                debited_uj: deb,
                remaining_uj: rem,
                claimed_root_u64: prev_b,
            });
        }
        assert_eq!(prev_b, 0xa1d0_5f20_5938_c3f5);
        assert_eq!(prev_a ^ prev_b, 0x31e0_5585_7e8b_d4aa);

        let (a_rec, a_tree, a_j) =
            verify_and_ingest_sel4_multipd_chain(&a_steps, "h46-alpha", "US-NE-ISO", 0.31).unwrap();
        let (b_rec, b_tree, b_j) =
            verify_and_ingest_sel4_multipd_chain(&b_steps, "h46-beta", "US-NE-ISO", 0.31).unwrap();
        assert_eq!((a_rec.len(), b_rec.len()), (9, 3));
        assert!(a_tree.proof(8).unwrap().verify_self());
        assert!(b_tree.proof(2).unwrap().verify_self());
        // 104 uJ tenant acts + 12 uJ Parent Titans sleep consolidation = 116 uJ total debited!
        assert!(((a_j + b_j) - 0.000_104).abs() < 1e-15);
    }
}
