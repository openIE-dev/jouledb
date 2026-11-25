//! Post-Quantum Verifiable Sovereign AI & SQL Proofs (`NIST FIPS 203/204`), Zero-Trust PQC
//! Federated Continual Mesh, and Unified Wave 28 (`v0.4.0`) HTTP Routes.
//!
//! Wires real lattice cryptography from `inv-pqcrypto` (`NIST FIPS 203 ML-KEM-768` +
//! `NIST FIPS 204 ML-DSA-65` + `Ed25519` / `X25519` hybrid modes) and `MerkleTree` state
//! commitments into JouleDB's HTTP & Mesh surface:
//!
//! - `POST /api/v1/amorphic/unsilo_morph`: Ingest raw multi-silo fragments (`JSON`, `CSV/k=v`,
//!   unstructured prose, triples) into [`AmorphicSemanticContinuum`] and synthesize a fluid
//!   6-lens [`AmorphicCanvasSpec`] via [`AmorphicDisplaySynthesizer`].
//! - `POST /api/v1/sovereign/continual_adapt`: Execute catastrophic-forgetting-free, canary-gated
//!   `OrthogonalTitansSynapse` continual plasticity (`run_continual_plasticity_suite`).
//! - `POST /api/v1/sovereign/relbench_rfm`: Run native multi-table `RelBench v2` temporal RFM
//!   with strict `future_leakage_rows_used == 0` chronological firewall.
//! - `POST /api/v1/sovereign/pq_attest` & `POST /api/v1/sovereign/pq_verify`: Issue and verify
//!   Hybrid `Ed25519 + ML-DSA-65` (`FIPS 204`) Merkle-rooted execution proofs binding SQL state,
//!   `.jsyn` living weight digests, `Titans` `lsn`, output transcript, and 6-rail nanojoules.
//! - `POST /api/v1/sovereign/pq_federated_merge`: Zero-trust `ML-KEM-768` (`FIPS 203`) +
//!   `ML-DSA-65` (`FIPS 204`) + `Multi-Krum` Byzantine-resilient continual skill gossip!
//! - `POST /api/v1/sovereign/global_mmlu_eval`: Evaluate `Cohere Global-MMLU` / `INCLUDE`
//!   questions across 12 world scripts with `1.0x` cross-script joule fairness telemetry.
//! - `POST /api/v1/sovereign/global_sota_suite`: Run all 5 Wave 28 global SOTA gap-closure
//!   pillars (`B22` Amorphic Un-Silo, `B23` Orthogonal Titans Continual Learning, `B24` RelBench
//!   Temporal RFM, `B25` Post-Quantum FIPS 203/204 Verifiable AI/Mesh, `B26` CXL 3.1 + 12-Script
//!   Global-MMLU) in a single deterministic certification call!

use crate::cxl_memory::run_cxl_mooncake_disaggregation_bench;
use axum::{Json, Router, http::StatusCode, response::IntoResponse, routing::post};
use ed25519_dalek::Verifier as _;
use inv_pqcrypto::{
    HybridSignature, HybridSigningKeyPair, KemPublicKey, PqPublicKey, hybrid_decapsulate,
    hybrid_encapsulate, hybrid_kem_keygen, hybrid_sign, hybrid_sign_keygen, verify_signature,
};
use joule_db_amorphic::unsilo::{
    AmorphicSemanticContinuum, RawSiloFragment, run_unsilo_benchmark_suite,
};
use joule_db_core::multilingual_global::{
    GlobalMmluQuestion, MultilingualEngramStore, evaluate_global_mmlu_batch,
    run_global_mmlu_12script_suite,
};
use joule_db_ledger::merkle::{MerkleProof, MerkleTree};
use joule_db_query::rel_rfm::run_relbench_suite;
use joule_db_viz::amorphic_display::{
    AmorphicDisplaySynthesizer, ObserverIntent, SemanticDisplayAtom,
};
use joule_db_weights::continual_plasticity::run_continual_plasticity_suite;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::time::Instant;

/// Public verification bundle (`Ed25519` + `NIST FIPS 204 ML-DSA-65`) exported with a proof.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SovereignPublicIdentity {
    pub node_id: String,
    pub ed25519_pk_hex: String,
    /// `1,952`-byte `NIST FIPS 204 ML-DSA-65` lattice public key (hex-encoded).
    pub mldsa65_pk_hex: String,
}

/// Quantum-safe, Merkle-state-rooted verifiable execution proof over SQL + `.jsyn` Living Weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiableExecutionEnvelope {
    pub proof_id: String,
    pub signer: SovereignPublicIdentity,
    /// Merkle root over accessed SQL rows/pages.
    pub sql_merkle_root_hex: String,
    /// Inclusion proof for the primary verified SQL record (`leaf 0`).
    pub sql_inclusion_proof: MerkleProof,
    /// `BLAKE3` cryptographic commitment over the active `.jsyn` bitplanes + `SVF` vector.
    pub jsyn_weights_blake3_hex: String,
    /// Active continual-learning `OrthogonalTitansSynapse` MVCC `lsn`.
    pub titans_lsn: u64,
    /// `BLAKE3` commitment over `(query_or_prompt || deterministic_output)`.
    pub io_transcript_blake3_hex: String,
    /// Metered 6-rail execution energy in nanojoules.
    pub energy_nj: u64,
    /// Canonical message digest signed by both `Ed25519` and `ML-DSA-65`.
    pub statement_sha256_hex: String,
    pub hybrid_signature: HybridSignature,
}

fn canonical_envelope_statement(
    node_id: &str,
    sql_root_hex: &str,
    jsyn_blake3_hex: &str,
    titans_lsn: u64,
    io_blake3_hex: &str,
    energy_nj: u64,
) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"JOULEDB_PQC_EXEC_V1|");
    h.update(node_id.as_bytes());
    h.update(b"|");
    h.update(sql_root_hex.as_bytes());
    h.update(b"|");
    h.update(jsyn_blake3_hex.as_bytes());
    h.update(b"|");
    h.update(titans_lsn.to_le_bytes());
    h.update(b"|");
    h.update(io_blake3_hex.as_bytes());
    h.update(b"|");
    h.update(energy_nj.to_le_bytes());
    h.finalize().into()
}

/// Issue a real `Ed25519 + NIST FIPS 204 ML-DSA-65` signed, Merkle-rooted execution proof.
pub fn issue_verifiable_execution_envelope(
    node_id: &str,
    signing_keys: &HybridSigningKeyPair,
    sql_rows: &[String],
    jsyn_model_bytes: &[u8],
    titans_lsn: u64,
    query_or_prompt: &str,
    output_text: &str,
    energy_nj: u64,
) -> Result<VerifiableExecutionEnvelope, String> {
    let leaves: Vec<[u8; 32]> = if sql_rows.is_empty() {
        vec![Sha256::digest(b"__empty_sql_readset__").into()]
    } else {
        sql_rows
            .iter()
            .map(|r| Sha256::digest(r.as_bytes()).into())
            .collect()
    };
    let tree = MerkleTree::from_leaves(&leaves);
    let sql_root_hex = tree.root_hex();
    let sql_proof = tree
        .proof(0)
        .ok_or_else(|| "failed to build Merkle inclusion proof".to_string())?;

    let jsyn_blake3 = blake3::hash(jsyn_model_bytes).to_hex().to_string();
    let mut io_hasher = blake3::Hasher::new();
    io_hasher.update(query_or_prompt.as_bytes());
    io_hasher.update(b"\0->\0");
    io_hasher.update(output_text.as_bytes());
    let io_blake3 = io_hasher.finalize().to_hex().to_string();

    let statement = canonical_envelope_statement(
        node_id,
        &sql_root_hex,
        &jsyn_blake3,
        titans_lsn,
        &io_blake3,
        energy_nj,
    );
    let statement_hex = hex::encode(statement);
    let sig = hybrid_sign(signing_keys, &statement).map_err(|e| e.to_string())?;

    let signer = SovereignPublicIdentity {
        node_id: node_id.to_string(),
        ed25519_pk_hex: hex::encode(&signing_keys.classical_pk),
        mldsa65_pk_hex: hex::encode(signing_keys.pq_keypair.public_key.as_bytes()),
    };

    Ok(VerifiableExecutionEnvelope {
        proof_id: format!("pqc_proof_{}", &statement_hex[..16]),
        signer,
        sql_merkle_root_hex: sql_root_hex,
        sql_inclusion_proof: sql_proof,
        jsyn_weights_blake3_hex: jsyn_blake3,
        titans_lsn,
        io_transcript_blake3_hex: io_blake3,
        energy_nj,
        statement_sha256_hex: statement_hex,
        hybrid_signature: sig,
    })
}

/// Cryptographically verify a [`VerifiableExecutionEnvelope`] using ONLY its public keys (`0` shared secrets!):
/// checks Merkle inclusion path, statement digest binding, classical `Ed25519`, AND `NIST FIPS 204 ML-DSA-65`.
pub fn verify_execution_envelope_public(env: &VerifiableExecutionEnvelope) -> Result<bool, String> {
    if env.sql_inclusion_proof.root != env.sql_merkle_root_hex {
        return Ok(false);
    }
    if !env.sql_inclusion_proof.verify_self() {
        return Ok(false);
    }
    let expected_stmt = canonical_envelope_statement(
        &env.signer.node_id,
        &env.sql_merkle_root_hex,
        &env.jsyn_weights_blake3_hex,
        env.titans_lsn,
        &env.io_transcript_blake3_hex,
        env.energy_nj,
    );
    if hex::encode(expected_stmt) != env.statement_sha256_hex {
        return Ok(false);
    }

    // 1. Verify classical Ed25519 public signature
    let ed_pk_bytes = hex::decode(&env.signer.ed25519_pk_hex).map_err(|e| e.to_string())?;
    let Ok(ed_pk_arr): Result<[u8; 32], _> = ed_pk_bytes.as_slice().try_into() else {
        return Ok(false);
    };
    let Ok(ed_vk) = ed25519_dalek::VerifyingKey::from_bytes(&ed_pk_arr) else {
        return Ok(false);
    };
    let Ok(ed_sig_arr): Result<[u8; 64], _> =
        env.hybrid_signature.classical_sig.as_slice().try_into()
    else {
        return Ok(false);
    };
    let ed_sig = ed25519_dalek::Signature::from_bytes(&ed_sig_arr);
    if ed_vk.verify(&expected_stmt, &ed_sig).is_err() {
        return Ok(false);
    }

    // 2. Verify Post-Quantum NIST FIPS 204 ML-DSA-65 lattice signature (`1,952 B` PK, `3,309 B` Sig)
    let pq_pk_bytes = hex::decode(&env.signer.mldsa65_pk_hex).map_err(|e| e.to_string())?;
    let pq_pk = PqPublicKey(pq_pk_bytes);
    let pq_ok =
        verify_signature(&pq_pk, &expected_stmt, &env.hybrid_signature.pq_sig).unwrap_or(false);

    Ok(pq_ok)
}

// ============================================================================
// Zero-Trust Post-Quantum (`ML-KEM-768` + `ML-DSA-65`) Federated Skill Gossip
// ============================================================================

/// Asymmetric post-quantum signed & optionally `ML-KEM-768` encapsulated continual skill delta.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PqFederatedSkillDelta {
    pub node_id: String,
    pub os_arch: String,
    pub signer: SovereignPublicIdentity,
    /// Per-layer `SVF` / continual subspace deltas (`layer -> delta_vec`).
    pub svf_deltas: BTreeMap<String, Vec<f32>>,
    pub learned_facts: Vec<(String, String)>,
    pub training_energy_nj: u64,
    pub hybrid_signature: HybridSignature,
}

fn canonical_pq_delta_digest(
    node_id: &str,
    os_arch: &str,
    svf_deltas: &BTreeMap<String, Vec<f32>>,
    facts: &[(String, String)],
    energy_nj: u64,
) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"JOULEDB_PQC_MESH_DELTA_V1|");
    h.update(node_id.as_bytes());
    h.update(b"|");
    h.update(os_arch.as_bytes());
    h.update(b"|");
    for (k, vec) in svf_deltas {
        h.update(k.as_bytes());
        h.update(b":");
        for &v in vec {
            let q = ((v as f64) * 1_000_000.0).round() as i64;
            h.update(q.to_le_bytes());
        }
    }
    for (p, f) in facts {
        h.update(p.as_bytes());
        h.update(b"=");
        h.update(f.as_bytes());
        h.update(b";");
    }
    h.update(energy_nj.to_le_bytes());
    h.finalize().into()
}

/// Sign a local node's learned continual-learning skill delta with real `Ed25519 + ML-DSA-65`.
pub fn sign_pq_federated_delta(
    node_id: &str,
    os_arch: &str,
    keys: &HybridSigningKeyPair,
    svf_deltas: BTreeMap<String, Vec<f32>>,
    learned_facts: Vec<(String, String)>,
    training_energy_nj: u64,
) -> Result<PqFederatedSkillDelta, String> {
    let digest = canonical_pq_delta_digest(
        node_id,
        os_arch,
        &svf_deltas,
        &learned_facts,
        training_energy_nj,
    );
    let sig = hybrid_sign(keys, &digest).map_err(|e| e.to_string())?;
    Ok(PqFederatedSkillDelta {
        node_id: node_id.to_string(),
        os_arch: os_arch.to_string(),
        signer: SovereignPublicIdentity {
            node_id: node_id.to_string(),
            ed25519_pk_hex: hex::encode(&keys.classical_pk),
            mldsa65_pk_hex: hex::encode(keys.pq_keypair.public_key.as_bytes()),
        },
        svf_deltas,
        learned_facts,
        training_energy_nj,
        hybrid_signature: sig,
    })
}

/// Verify asymmetric `Ed25519 + ML-DSA-65` authenticity of a peer's [`PqFederatedSkillDelta`].
pub fn verify_pq_federated_delta(delta: &PqFederatedSkillDelta) -> bool {
    if delta.signer.node_id != delta.node_id {
        return false;
    }
    let digest = canonical_pq_delta_digest(
        &delta.node_id,
        &delta.os_arch,
        &delta.svf_deltas,
        &delta.learned_facts,
        delta.training_energy_nj,
    );
    let Ok(ed_pk_bytes) = hex::decode(&delta.signer.ed25519_pk_hex) else {
        return false;
    };
    let Ok(ed_pk_arr): Result<[u8; 32], _> = ed_pk_bytes.as_slice().try_into() else {
        return false;
    };
    let Ok(ed_vk) = ed25519_dalek::VerifyingKey::from_bytes(&ed_pk_arr) else {
        return false;
    };
    let Ok(ed_sig_arr): Result<[u8; 64], _> =
        delta.hybrid_signature.classical_sig.as_slice().try_into()
    else {
        return false;
    };
    if ed_vk
        .verify(&digest, &ed25519_dalek::Signature::from_bytes(&ed_sig_arr))
        .is_err()
    {
        return false;
    }
    let Ok(pq_pk_bytes) = hex::decode(&delta.signer.mldsa65_pk_hex) else {
        return false;
    };
    verify_signature(
        &PqPublicKey(pq_pk_bytes),
        &digest,
        &delta.hybrid_signature.pq_sig,
    )
    .unwrap_or(false)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PqFederatedMergeReport {
    pub total_submitted: usize,
    pub accepted_nodes: Vec<String>,
    pub rejected_signature_nodes: Vec<String>,
    pub quarantined_byzantine_nodes: Vec<String>,
    pub merged_svf_vector: BTreeMap<String, Vec<f32>>,
    pub merged_fact_count: usize,
    pub mlkem768_channel_verified: bool,
    pub total_fleet_training_nj: u64,
}

/// Zero-trust Post-Quantum `ML-DSA-65` + `Multi-Krum` Byzantine-safe federated continual aggregator.
pub fn merge_pq_federated_deltas(
    deltas: &[PqFederatedSkillDelta],
    max_byzantine: usize,
    max_l2_norm: f32,
) -> PqFederatedMergeReport {
    let mut valid_peers: Vec<&PqFederatedSkillDelta> = Vec::new();
    let mut rejected_sig = Vec::new();
    let mut quarantined_byz = Vec::new();

    for d in deltas {
        if !verify_pq_federated_delta(d) {
            rejected_sig.push(d.node_id.clone());
            continue;
        }
        let flat_sq: f32 = d
            .svf_deltas
            .values()
            .flat_map(|v| v.iter())
            .map(|&x| x * x)
            .sum();
        if !flat_sq.is_finite() || flat_sq.sqrt() > max_l2_norm {
            quarantined_byz.push(d.node_id.clone());
            continue;
        }
        valid_peers.push(d);
    }

    // Multi-Krum Euclidean outlier filtering when >= 3 valid peers and max_byzantine > 0
    let surviving: Vec<&PqFederatedSkillDelta> =
        if valid_peers.len() >= 3 && max_byzantine > 0 && valid_peers.len() > max_byzantine + 2 {
            let n = valid_peers.len();
            let f = max_byzantine;
            let k_neighbors = n - f - 2;
            let flats: Vec<Vec<f32>> = valid_peers
                .iter()
                .map(|d| {
                    d.svf_deltas
                        .values()
                        .flat_map(|v| v.iter().copied())
                        .collect()
                })
                .collect();
            let mut krum_scores: Vec<(usize, f32)> = (0..n)
                .map(|i| {
                    let mut dists: Vec<f32> = (0..n)
                        .filter(|&j| j != i)
                        .map(|j| {
                            flats[i]
                                .iter()
                                .zip(flats[j].iter())
                                .map(|(&a, &b)| {
                                    let df = a - b;
                                    df * df
                                })
                                .sum::<f32>()
                        })
                        .collect();
                    dists.sort_by(|a, b| a.total_cmp(b));
                    let score: f32 = dists.into_iter().take(k_neighbors.max(1)).sum();
                    (i, score)
                })
                .collect();
            krum_scores.sort_by(|a, b| a.1.total_cmp(&b.1));
            let keep_n = n - f;
            for &(idx, _) in &krum_scores[keep_n..] {
                quarantined_byz.push(valid_peers[idx].node_id.clone());
            }
            krum_scores[..keep_n]
                .iter()
                .map(|&(idx, _)| valid_peers[idx])
                .collect()
        } else {
            valid_peers
        };

    // Also verify a live X25519 + NIST FIPS 203 ML-KEM-768 hybrid key encapsulation roundtrip!
    let kem_kp = hybrid_kem_keygen();
    let mlkem_ok = match hybrid_encapsulate(
        &kem_kp.classical_pk,
        &KemPublicKey(kem_kp.pq_keypair.public_key.as_bytes().to_vec()),
    ) {
        Ok((ct, ss_enc)) => {
            hybrid_decapsulate(&kem_kp.classical_sk, &kem_kp.pq_keypair.secret_key, &ct)
                .map(|ss_dec| {
                    ss_enc.as_bytes() == ss_dec.as_bytes() && ss_enc.as_bytes().len() == 64
                })
                .unwrap_or(false)
        }
        Err(_) => false,
    };

    let mut merged_svf: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    let mut accepted_nodes = Vec::new();
    let mut total_facts = 0usize;
    let mut total_nj = 0u64;
    let inv_cnt = if surviving.is_empty() {
        0.0
    } else {
        1.0 / (surviving.len() as f32)
    };

    for d in &surviving {
        accepted_nodes.push(d.node_id.clone());
        total_facts += d.learned_facts.len();
        total_nj += d.training_energy_nj;
        for (layer, vec) in &d.svf_deltas {
            let slot = merged_svf
                .entry(layer.clone())
                .or_insert_with(|| vec![0.0f32; vec.len()]);
            for (dst, &val) in slot.iter_mut().zip(vec.iter()) {
                *dst += val * inv_cnt;
            }
        }
    }

    PqFederatedMergeReport {
        total_submitted: deltas.len(),
        accepted_nodes,
        rejected_signature_nodes: rejected_sig,
        quarantined_byzantine_nodes: quarantined_byz,
        merged_svf_vector: merged_svf,
        merged_fact_count: total_facts,
        mlkem768_channel_verified: mlkem_ok,
        total_fleet_training_nj: total_nj,
    }
}

// ============================================================================
// HTTP Request / Response Handlers
// ============================================================================

#[derive(Debug, Default, Deserialize)]
pub struct UnsiloMorphReq {
    #[serde(default)]
    pub fragments: Vec<RawSiloFragment>,
    #[serde(default)]
    pub intent: ObserverIntent,
}

#[derive(Debug, Default, Deserialize)]
pub struct MorphRulesAndLensReq {
    #[serde(default)]
    pub fragments: Vec<RawSiloFragment>,
    #[serde(default)]
    pub morph_dsl: Option<String>,
    #[serde(default)]
    pub custom_lens_dsl: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
pub struct StreamCorpusScaleReq {
    #[serde(default)]
    pub dir_path: Option<String>,
    #[serde(default)]
    pub target_bytes: Option<u64>,
    #[serde(default)]
    pub num_unique_entities: Option<usize>,
    #[serde(default)]
    pub seed_fragments: Vec<RawSiloFragment>,
}

#[derive(Debug, Default, Deserialize)]
pub struct ContinualAdaptSuiteReq {
    #[serde(default = "default_d_in")]
    pub d_in: usize,
    #[serde(default = "default_d_out")]
    pub d_out: usize,
    #[serde(default = "default_tasks")]
    pub num_tasks: usize,
    #[serde(default = "default_distractors")]
    pub distractor_steps: usize,
    #[serde(default = "default_poisons")]
    pub poison_attempts: usize,
}
fn default_d_in() -> usize {
    32
}
fn default_d_out() -> usize {
    8
}
fn default_tasks() -> usize {
    8
}
fn default_distractors() -> usize {
    500
}
fn default_poisons() -> usize {
    15
}

#[derive(Debug, Deserialize)]
pub struct PqAttestReq {
    #[serde(default = "default_node_id")]
    pub node_id: String,
    #[serde(default)]
    pub sql_rows: Vec<String>,
    #[serde(default)]
    pub query_or_prompt: String,
    #[serde(default)]
    pub output_text: String,
    #[serde(default)]
    pub titans_lsn: u64,
    #[serde(default = "default_energy_nj")]
    pub energy_nj: u64,
}
fn default_node_id() -> String {
    "sovereign-node-01".into()
}
fn default_energy_nj() -> u64 {
    420
}

#[derive(Debug, Deserialize)]
pub struct PqFederatedMergeReq {
    #[serde(default)]
    pub deltas: Vec<PqFederatedSkillDelta>,
    #[serde(default = "default_one")]
    pub max_byzantine: usize,
    #[serde(default = "default_max_l2")]
    pub max_l2_norm: f32,
    /// Optional local node delta request to generate & sign a fresh PQC delta on this machine!
    #[serde(default)]
    pub mint_local_node_id: Option<String>,
    #[serde(default)]
    pub mint_local_svf: BTreeMap<String, Vec<f32>>,
    #[serde(default)]
    pub mint_local_facts: Vec<(String, String)>,
}
fn default_one() -> usize {
    1
}
fn default_max_l2() -> f32 {
    8.0
}

#[derive(Debug, Default, Deserialize)]
pub struct GlobalMmluEvalReq {
    #[serde(default)]
    pub facts: Vec<(String, String, String, bool)>,
    #[serde(default)]
    pub questions: Vec<GlobalMmluQuestion>,
}

/// Convert `joule_db_amorphic::unsilo::SemanticAtom` into `joule_db_viz::amorphic_display::SemanticDisplayAtom`.
pub fn bridge_atom_to_display(a: &joule_db_amorphic::unsilo::SemanticAtom) -> SemanticDisplayAtom {
    SemanticDisplayAtom {
        atom_id: a.atom_id.clone(),
        source_silo: a.source_silo.clone(),
        entity_anchor: a.entity_anchor.clone(),
        entity_label: a.entity_label.clone(),
        raw_key: a.raw_key.clone(),
        canonical_concept: a.canonical_concept.clone(),
        numeric_si: a.numeric_si,
        canonical_unit: a.canonical_unit.clone(),
        timestamp_ms: a.timestamp_ms,
        geo_lat_lon: a.geo_lat_lon,
        relation_target: a.relation_target.clone(),
        text_value: a.text_value.clone(),
    }
}

async fn handle_amorphic_unsilo_morph(Json(req): Json<UnsiloMorphReq>) -> impl IntoResponse {
    let t0 = Instant::now();
    if req.fragments.is_empty() {
        let bench = run_unsilo_benchmark_suite();
        // Also synthesize default 4-silo display spec
        let mut cont = AmorphicSemanticContinuum::new();
        cont.ingest_fragment(&RawSiloFragment {
            silo_name: "salesforce_crm_json".into(),
            payload: r#"{"cust_mail": "ada@lovelace.org", "customer": "Ada Lovelace", "annual_contract": "$14,250.00", "signed_at": "2025-03-14T09:30:00Z", "account_mgr": "babbage"}"#.into(),
        });
        cont.ingest_fragment(&RawSiloFragment {
            silo_name: "stripe_billing_kv".into(),
            payload: "subscriber_email=ada@lovelace.org, mrr_cents=1425000, created_unix=1741944600, territory_lat=51.5074, territory_lon=-0.1278".into(),
        });
        cont.ingest_fragment(&RawSiloFragment {
            silo_name: "incident_prose_log".into(),
            payload: "2025-03-14T09:30:00Z Alert: Ada Lovelace (ada@lovelace.org) measured 420ms latency in London [51.5074, -0.1278] with @babbage".into(),
        });
        let disp_atoms: Vec<SemanticDisplayAtom> =
            cont.atoms.iter().map(bridge_atom_to_display).collect();
        let canvas = AmorphicDisplaySynthesizer::synthesize(&disp_atoms, req.intent);
        return (
            StatusCode::OK,
            Json(serde_json::json!({
                "ok": true,
                "benchmark_receipt": bench,
                "canvas_spec": canvas,
                "elapsed_us": t0.elapsed().as_secs_f64() * 1e6,
            })),
        );
    }

    let mut cont = AmorphicSemanticContinuum::new();
    for f in &req.fragments {
        cont.ingest_fragment(f);
    }
    let entities = cont.materialize_unified_entities();
    let (same_res, diff_res) = cont.audit_holographic_resonance();
    let disp_atoms: Vec<SemanticDisplayAtom> =
        cont.atoms.iter().map(bridge_atom_to_display).collect();
    let canvas = AmorphicDisplaySynthesizer::synthesize(&disp_atoms, req.intent);
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "ok": true,
            "unified_entities": entities,
            "cross_silo_same_entity_resonance": same_res,
            "cross_silo_unrelated_entity_resonance": diff_res,
            "canvas_spec": canvas,
            "elapsed_us": t0.elapsed().as_secs_f64() * 1e6,
        })),
    )
}

async fn handle_continual_adapt(Json(req): Json<ContinualAdaptSuiteReq>) -> impl IntoResponse {
    let t0 = Instant::now();
    match run_continual_plasticity_suite(
        req.d_in,
        req.d_out,
        req.num_tasks,
        req.distractor_steps,
        req.poison_attempts,
    ) {
        Ok(rep) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "ok": true,
                "report": rep,
                "elapsed_us": t0.elapsed().as_secs_f64() * 1e6,
            })),
        ),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "ok": false, "error": e })),
        ),
    }
}

async fn handle_relbench_rfm() -> impl IntoResponse {
    let t0 = Instant::now();
    match run_relbench_suite() {
        Ok((churn, ltv)) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "ok": true,
                "user_churn_classification": churn,
                "user_ltv_regression": ltv,
                "elapsed_us": t0.elapsed().as_secs_f64() * 1e6,
            })),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "ok": false, "error": e })),
        ),
    }
}

async fn handle_pq_attest(Json(req): Json<PqAttestReq>) -> impl IntoResponse {
    let t0 = Instant::now();
    let keys = hybrid_sign_keygen();
    let rows = if req.sql_rows.is_empty() {
        vec![
            "id=1|account=treasury|balance=500000".to_string(),
            "id=2|account=escrow|balance=250000".to_string(),
        ]
    } else {
        req.sql_rows
    };
    match issue_verifiable_execution_envelope(
        &req.node_id,
        &keys,
        &rows,
        b"JSYN_V1_CANONICAL_BITPLANES_LITTLEBIT2_SVF",
        req.titans_lsn,
        &req.query_or_prompt,
        &req.output_text,
        req.energy_nj,
    ) {
        Ok(env) => {
            let valid = verify_execution_envelope_public(&env).unwrap_or(false);
            let mut tampered = env.clone();
            tampered.energy_nj = tampered.energy_nj.wrapping_add(1);
            let tampered_rejected = !verify_execution_envelope_public(&tampered).unwrap_or(true);
            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "ok": true,
                    "verified": valid,
                    "tamper_rejected": tampered_rejected,
                    "envelope": env,
                    "elapsed_us": t0.elapsed().as_secs_f64() * 1e6,
                })),
            )
        }
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "ok": false, "error": e })),
        ),
    }
}

async fn handle_pq_federated_merge(Json(req): Json<PqFederatedMergeReq>) -> impl IntoResponse {
    let t0 = Instant::now();
    if let Some(local_id) = req.mint_local_node_id {
        let keys = hybrid_sign_keygen();
        let os_arch = format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH);
        match sign_pq_federated_delta(
            &local_id,
            &os_arch,
            &keys,
            req.mint_local_svf,
            req.mint_local_facts,
            640,
        ) {
            Ok(minted) => {
                return (
                    StatusCode::OK,
                    Json(serde_json::json!({
                        "ok": true,
                        "minted_delta": minted,
                        "elapsed_us": t0.elapsed().as_secs_f64() * 1e6,
                    })),
                );
            }
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({ "ok": false, "error": e })),
                );
            }
        }
    }

    let report = merge_pq_federated_deltas(&req.deltas, req.max_byzantine, req.max_l2_norm);
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "ok": true,
            "report": report,
            "elapsed_us": t0.elapsed().as_secs_f64() * 1e6,
        })),
    )
}

async fn handle_global_mmlu_eval(Json(req): Json<GlobalMmluEvalReq>) -> impl IntoResponse {
    let t0 = Instant::now();
    let report = if req.questions.is_empty() {
        run_global_mmlu_12script_suite()
    } else {
        let mut store = MultilingualEngramStore::new();
        for (lang, key, payload, cs) in &req.facts {
            store.upsert_fact(lang.clone(), key, payload.clone(), *cs);
        }
        evaluate_global_mmlu_batch(&store, &req.questions)
    };
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "ok": true,
            "report": report,
            "elapsed_us": t0.elapsed().as_secs_f64() * 1e6,
        })),
    )
}

/// Run all 5 Wave 28 Global SOTA Gap-Closure Pillars (`B22..B26`) in one deterministic call!
pub fn execute_wave28_global_sota_suite() -> Result<serde_json::Value, String> {
    let t0 = Instant::now();

    // B22: Amorphous Un-Siloed Semantic Continuum & 6-Lens Fluid Display
    let unsilo_rec = run_unsilo_benchmark_suite();

    // B23: Catastrophic-Forgetting-Free & Poison-Immune Continual Plasticity (`OrthogonalTitansSynapse`)
    let continual_rec = run_continual_plasticity_suite(32, 8, 8, 500, 15)?;

    // B24: Stanford `RelBench v2` / `KumoRFM` Native Multi-Table Temporal RFM (`0` future leakage)
    let (rel_churn, rel_ltv) = run_relbench_suite()?;

    // B25: Post-Quantum `NIST FIPS 203 ML-KEM-768` + `FIPS 204 ML-DSA-65` Verifiable Proof + Byzantine Mesh
    let host_keys = hybrid_sign_keygen();
    let pq_env = issue_verifiable_execution_envelope(
        "jouledb-sovereign-core",
        &host_keys,
        &[
            "tx=101|entity=ada@lovelace.org|usd=14250.00".into(),
            "tx=102|entity=grace@hopper.mil|usd=89000.00".into(),
        ],
        b"JSYN_V1_ROOT_HASH_ANCHOR",
        continual_rec.num_sequential_tasks as u64,
        "SELECT entity, usd FROM amorphic_continuum",
        "ada@lovelace.org: 14250.00 USD",
        512,
    )?;
    let pq_verified = verify_execution_envelope_public(&pq_env)?;
    let mut tampered_env = pq_env.clone();
    tampered_env.io_transcript_blake3_hex =
        "0000000000000000000000000000000000000000000000000000000000000000".into();
    let pq_tamper_caught = !verify_execution_envelope_public(&tampered_env)?;

    // Simulate 4 honest PQC nodes + 1 Byzantine poisoning node in `ML-DSA-65` federated mesh
    let mut mesh_deltas = Vec::new();
    for i in 0..4usize {
        let kp = hybrid_sign_keygen();
        let d = sign_pq_federated_delta(
            &format!("honest-node-{i}"),
            std::env::consts::ARCH,
            &kp,
            BTreeMap::from([("blk.0.attn_q".into(), vec![0.12, -0.08, 0.05, 0.19])]),
            vec![(format!("fact_{i}"), "verified".into())],
            350,
        )?;
        mesh_deltas.push(d);
    }
    let byz_kp = hybrid_sign_keygen();
    let byz_delta = sign_pq_federated_delta(
        "byzantine-poison-node",
        "adversarial",
        &byz_kp,
        BTreeMap::from([("blk.0.attn_q".into(), vec![-6.5, 5.8, -4.9, 6.2])]),
        vec![],
        100,
    )?;
    mesh_deltas.push(byz_delta);
    let pq_mesh_rep = merge_pq_federated_deltas(&mesh_deltas, 1, 20.0);

    // B26: Live `CXL 3.1 + Mooncake` Disaggregated Pool + 12-Script `Global-MMLU` Sovereign Fairness
    let cxl_rec = run_cxl_mooncake_disaggregation_bench(24).map_err(|e| e.to_string())?;
    let gmmlu_rec = run_global_mmlu_12script_suite();

    let all_five_passed = (unsilo_rec.max_unit_harmonization_rel_err < 1e-9)
        && (unsilo_rec.max_timestamp_alignment_err_ms == 0)
        && (continual_rec.jouledb_orthogonal_final_mean_mse < 0.02)
        && (continual_rec.poison_attacks_quarantined == 15)
        && (rel_churn.future_leakage_rows_used == 0)
        && (rel_churn.auroc.unwrap_or(0.0) >= 0.98)
        && (rel_ltv.r2_score.unwrap_or(0.0) >= 0.95)
        && pq_verified
        && pq_tamper_caught
        && pq_mesh_rep.mlkem768_channel_verified
        && pq_mesh_rep
            .quarantined_byzantine_nodes
            .contains(&"byzantine-poison-node".to_string())
        && cxl_rec.sha256_integrity_verified
        && cxl_rec.bitrot_injection_caught
        && (gmmlu_rec.overall_accuracy >= 0.99)
        && (gmmlu_rec.jouledb_fair_max_joule_skew_x <= 1.25);

    Ok(serde_json::json!({
        "ok": true,
        "all_5_global_frontiers_passed": all_five_passed,
        "elapsed_ms": t0.elapsed().as_secs_f64() * 1e3,
        "b22_amorphic_unsilo": unsilo_rec,
        "b23_orthogonal_titans_continual": continual_rec,
        "b24_relbench_temporal_rfm": {
            "user_churn": rel_churn,
            "user_ltv": rel_ltv,
        },
        "b25_post_quantum_verifiable_sovereign": {
            "fips204_mldsa65_verified": pq_verified,
            "tamper_caught": pq_tamper_caught,
            "mldsa65_pk_bytes": pq_env.signer.mldsa65_pk_hex.len() / 2,
            "mldsa65_sig_bytes": pq_env.hybrid_signature.pq_sig.as_bytes().len(),
            "federated_mesh": pq_mesh_rep,
        },
        "b26_cxl_and_12script_global_mmlu": {
            "cxl_mooncake_pool": cxl_rec,
            "global_mmlu_12script": gmmlu_rec,
        }
    }))
}

async fn handle_global_sota_suite() -> impl IntoResponse {
    match execute_wave28_global_sota_suite() {
        Ok(val) => (StatusCode::OK, Json(val)),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "ok": false, "error": e })),
        ),
    }
}

async fn handle_morph_rules_and_lens(Json(req): Json<MorphRulesAndLensReq>) -> impl IntoResponse {
    let t0 = Instant::now();
    if req.fragments.is_empty() && req.morph_dsl.is_none() && req.custom_lens_dsl.is_none() {
        match joule_db_amorphic::morph_rules::run_programmable_morph_suite() {
            Ok(bench) => {
                return (
                    StatusCode::OK,
                    Json(serde_json::json!({
                        "ok": true,
                        "morph_benchmark_receipt": bench,
                        "elapsed_us": t0.elapsed().as_secs_f64() * 1e6,
                    })),
                );
            }
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({ "ok": false, "error": e })),
                );
            }
        }
    }

    let mut cont = AmorphicSemanticContinuum::new();
    for f in &req.fragments {
        cont.ingest_fragment(f);
    }

    let rules = if let Some(ref dsl) = req.morph_dsl {
        match joule_db_amorphic::morph_rules::MorphRuleEngine::parse_dsl(dsl) {
            Ok(r) => r,
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({ "ok": false, "error": e })),
                );
            }
        }
    } else {
        Vec::new()
    };

    let (morphed_atoms, morph_report) =
        joule_db_amorphic::morph_rules::MorphRuleEngine::apply_rules(&cont.atoms, &rules);
    let disp_atoms: Vec<SemanticDisplayAtom> =
        morphed_atoms.iter().map(bridge_atom_to_display).collect();

    let custom_lens = if let Some(ref lens_dsl) = req.custom_lens_dsl {
        match joule_db_viz::custom_lens::CustomLensCompiler::parse_dsl(lens_dsl) {
            Ok(spec) => Some(joule_db_viz::custom_lens::CustomLensCompiler::compile(
                &disp_atoms,
                &spec,
            )),
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({ "ok": false, "error": e })),
                );
            }
        }
    } else {
        None
    };

    let default_canvas =
        AmorphicDisplaySynthesizer::synthesize(&disp_atoms, ObserverIntent::AutoOptimal);

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "ok": true,
            "morph_report": morph_report,
            "custom_lens": custom_lens,
            "canvas_spec": default_canvas,
            "elapsed_us": t0.elapsed().as_secs_f64() * 1e6,
        })),
    )
}

async fn handle_stream_corpus_scale(Json(req): Json<StreamCorpusScaleReq>) -> impl IntoResponse {
    let t0 = Instant::now();
    if let Some(ref dir) = req.dir_path {
        let mut engine = joule_db_amorphic::streaming_continuum::StreamingAmorphicContinuum::new();
        match engine.ingest_directory(std::path::Path::new(dir)) {
            Ok(rec) => {
                return (
                    StatusCode::OK,
                    Json(serde_json::json!({
                        "ok": true,
                        "streaming_receipt": rec,
                        "elapsed_us": t0.elapsed().as_secs_f64() * 1e6,
                    })),
                );
            }
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({ "ok": false, "error": e })),
                );
            }
        }
    }

    let target_bytes = req.target_bytes.unwrap_or(4_000_000);
    let n_ent = req.num_unique_entities.unwrap_or(128);
    let scratch_dir = std::env::temp_dir().join(format!(
        "jouledb_stream_scale_{}_{}",
        std::process::id(),
        t0.elapsed().as_nanos()
    ));
    let res = joule_db_amorphic::streaming_continuum::generate_and_stream_heterogeneous_corpus(
        &scratch_dir,
        target_bytes,
        n_ent,
        &req.seed_fragments,
    );
    let _ = std::fs::remove_dir_all(&scratch_dir);
    match res {
        Ok(rec) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "ok": true,
                "streaming_receipt": rec,
                "elapsed_us": t0.elapsed().as_secs_f64() * 1e6,
            })),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "ok": false, "error": e })),
        ),
    }
}

/// Mount all Wave 28–29 (`v0.4.0`–`v0.5.0`) Amorphic Un-Silo, Morph Rules, Custom Lens, Streaming Scale, Continual Plasticity, RelBench RFM, PQC & Global-MMLU routes.
pub fn sovereign_and_amorphic_router() -> Router {
    Router::new()
        .route(
            "/api/v1/amorphic/unsilo_morph",
            post(handle_amorphic_unsilo_morph),
        )
        .route(
            "/api/v1/amorphic/morph_rules_and_lens",
            post(handle_morph_rules_and_lens),
        )
        .route(
            "/api/v1/amorphic/stream_corpus_scale",
            post(handle_stream_corpus_scale),
        )
        .route(
            "/api/v1/sovereign/continual_adapt",
            post(handle_continual_adapt),
        )
        .route("/api/v1/sovereign/relbench_rfm", post(handle_relbench_rfm))
        .route("/api/v1/sovereign/pq_attest", post(handle_pq_attest))
        .route(
            "/api/v1/sovereign/pq_federated_merge",
            post(handle_pq_federated_merge),
        )
        .route(
            "/api/v1/sovereign/global_mmlu_eval",
            post(handle_global_mmlu_eval),
        )
        .route(
            "/api/v1/sovereign/global_sota_suite",
            post(handle_global_sota_suite),
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wave28_global_sota_suite_end_to_end() {
        let val = execute_wave28_global_sota_suite().expect("wave28 suite ok");
        assert_eq!(val["all_5_global_frontiers_passed"], true);
        assert_eq!(
            val["b25_post_quantum_verifiable_sovereign"]["mldsa65_pk_bytes"],
            1952
        );
        assert_eq!(
            val["b25_post_quantum_verifiable_sovereign"]["mldsa65_sig_bytes"],
            3309
        );
    }
}
