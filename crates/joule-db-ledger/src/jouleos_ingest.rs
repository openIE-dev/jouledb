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
}
