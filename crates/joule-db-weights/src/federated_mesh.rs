//! `JouleSynapseMesh` — Byzantine-Resilient Federated Living-Weight Skill Gossip
//!
//! Disintermediates standalone federated-learning frameworks (`Flower`, `Ray` parameter servers)
//! and multi-GB HuggingFace LoRA adapter sync by exchanging **sub-4 KB cryptographically signed
//! differential synaptic deltas** (`Sakana SVF` singular-value adjustments + `Engram` conditional
//! facts + L2 sensitivity clipping + 6-rail nanojoule training receipts) across every tier of the
//! compute fabric (`MCU`, `WASM`, `Mobile`, `EdgeRobotic`, `AppleUMA`, `LinuxGPU`, `CloudCXL`).
//!
//! Poisoned or Byzantine outlier nodes are automatically quarantined via **Multi-Krum Euclidean
//! distance filtering + L2 norm clipping** before a single living weight is touched.

use crate::synapse::JouleHarnessEngine;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

/// Compact federated synaptic skill delta (`< 4 KB` vs multi-GB `.safetensors` / LoRA slabs).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynapseSkillDelta {
    pub delta_id: String,
    pub source_node_id: String,
    pub source_fabric_class: String,
    pub expert_name: String,
    /// Per-layer Sakana `SVF` singular-value adjustments (`tensor_name -> delta_z`, where `z_new = z_base + delta_z`).
    pub svf_deltas: BTreeMap<String, Vec<f32>>,
    /// Newly learned canonical `(phrase, fact_payload)` conditional memory entries.
    pub engram_facts: Vec<(String, String)>,
    /// L2 clipping bound enforced at export time.
    pub clip_l2_norm: f32,
    /// Deterministic differential-privacy noise scale applied at export (`0.0` = exact).
    pub dp_noise_sigma: f32,
    /// Total 6-rail nanojoules spent locally learning this delta.
    pub training_energy_nj: u64,
    /// Peak local milliwatts observed while learning this delta.
    pub peak_power_mw: u64,
    /// RFC-2104 `HMAC-SHA256` signature authenticating `(delta_id, source_node_id, expert_name, svf_deltas, engram_facts)`.
    pub hmac_hex: String,
}

/// Summary receipt returned after Byzantine-safe federated aggregation into [`JouleHarnessEngine`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedMergeReceipt {
    pub expert_name: String,
    pub accepted_nodes: Vec<String>,
    pub rejected_invalid_hmac_nodes: Vec<String>,
    pub rejected_byzantine_nodes: Vec<String>,
    pub updated_tensors: usize,
    pub merged_engram_facts: usize,
    pub total_peer_training_nj: u64,
}

/// Pure-Rust RFC-2104 `HMAC-SHA256` for authenticating federated `SynapseSkillDelta`s.
pub fn mesh_hmac_sha256(secret: &[u8], msg: &[u8]) -> [u8; 32] {
    const BLOCK: usize = 64;
    let mut k = [0u8; BLOCK];
    if secret.len() > BLOCK {
        let d = Sha256::digest(secret);
        k[..32].copy_from_slice(&d);
    } else {
        k[..secret.len()].copy_from_slice(secret);
    }
    let mut ipad = [0x36u8; BLOCK];
    let mut opad = [0x5cu8; BLOCK];
    for i in 0..BLOCK {
        ipad[i] ^= k[i];
        opad[i] ^= k[i];
    }
    let mut inner = Sha256::new();
    inner.update(ipad);
    inner.update(msg);
    let inner_out = inner.finalize();

    let mut outer = Sha256::new();
    outer.update(opad);
    outer.update(inner_out);
    let digest = outer.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}

fn canonical_delta_payload(
    delta_id: &str,
    source_node_id: &str,
    expert_name: &str,
    svf_deltas: &BTreeMap<String, Vec<f32>>,
    engram_facts: &[(String, String)],
) -> Vec<u8> {
    let mut buf = Vec::with_capacity(256);
    buf.extend_from_slice(b"JSYN_MESH_V1:");
    buf.extend_from_slice(delta_id.as_bytes());
    buf.push(b'|');
    buf.extend_from_slice(source_node_id.as_bytes());
    buf.push(b'|');
    buf.extend_from_slice(expert_name.as_bytes());
    buf.push(b'|');
    for (tname, vec) in svf_deltas {
        buf.extend_from_slice(tname.as_bytes());
        buf.push(b':');
        for &v in vec {
            // Quantize canonical MAC float bits to 1e-6 integer ticks for cross-ISA stability
            let q = (v as f64 * 1_000_000.0).round() as i64;
            buf.extend_from_slice(&q.to_le_bytes());
        }
        buf.push(b';');
    }
    for (p, f) in engram_facts {
        buf.extend_from_slice(p.as_bytes());
        buf.push(b'=');
        buf.extend_from_slice(f.as_bytes());
        buf.push(b';');
    }
    buf
}

fn ct_eq_32(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let mut d = 0u8;
    for i in 0..32 {
        d |= a[i] ^ b[i];
    }
    d == 0
}

fn bytes_to_hex(bytes: &[u8]) -> String {
    const LUT: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(LUT[(b >> 4) as usize] as char);
        s.push(LUT[(b & 0x0f) as usize] as char);
    }
    s
}

fn hex_to_bytes_32(hex_s: &str) -> Option<[u8; 32]> {
    if hex_s.len() != 64 {
        return None;
    }
    let nibble = |c: u8| -> Option<u8> {
        match c {
            b'0'..=b'9' => Some(c - b'0'),
            b'a'..=b'f' => Some(c - b'a' + 10),
            b'A'..=b'F' => Some(c - b'A' + 10),
            _ => None,
        }
    };
    let raw = hex_s.as_bytes();
    let mut out = [0u8; 32];
    for (i, slot) in out.iter_mut().enumerate() {
        let hi = nibble(raw[i * 2])?;
        let lo = nibble(raw[i * 2 + 1])?;
        *slot = (hi << 4) | lo;
    }
    Some(out)
}

fn clip_l2_in_place(vec: &mut [f32], max_l2: f32) {
    let cap = max_l2.max(1e-4);
    let norm = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > cap && norm > 0.0 {
        let scale = cap / norm;
        for v in vec.iter_mut() {
            *v *= scale;
        }
    }
}

fn add_deterministic_dp_noise(vec: &mut [f32], seed_tag: &str, sigma: f32) {
    if sigma <= 0.0 {
        return;
    }
    let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
    for &b in seed_tag.as_bytes() {
        state ^= b as u64;
        state = state.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    }
    for (idx, val) in vec.iter_mut().enumerate() {
        // Box-Muller / Irwin-Hall 12-uniform sum -> zero-mean unit-variance Gaussian approximation!
        let mut irwin_hall = 0.0f32;
        for s in 0..12u64 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407 ^ ((idx as u64) << 4) ^ s);
            let u = ((state >> 40) as f32) / ((1u32 << 24) as f32);
            irwin_hall += u;
        }
        let z_std = irwin_hall - 6.0;
        *val += z_std * sigma;
    }
}

/// Export a local node's learned Sakana `SVF` expert adjustments (`z - 1.0`) + selected `Engram`
/// facts as an L2-clipped, optionally DP-noised, `HMAC-SHA256` signed [`SynapseSkillDelta`].
#[allow(clippy::too_many_arguments)]
pub fn export_skill_delta(
    engine: &JouleHarnessEngine,
    cluster_secret: &[u8],
    delta_id: impl Into<String>,
    source_node_id: impl Into<String>,
    source_fabric_class: impl Into<String>,
    expert_name: impl Into<String>,
    clip_l2_norm: f32,
    dp_noise_sigma: f32,
    engram_facts: Vec<(String, String)>,
    training_energy_nj: u64,
    peak_power_mw: u64,
) -> SynapseSkillDelta {
    let delta_id = delta_id.into();
    let source_node_id = source_node_id.into();
    let source_fabric_class = source_fabric_class.into();
    let expert_name = expert_name.into();
    let eff_clip = clip_l2_norm.clamp(0.01, 16.0);
    let eff_sigma = dp_noise_sigma.clamp(0.0, 2.0);

    let mut svf_deltas = BTreeMap::new();
    for (tname, tensor) in &engine.littlebit_tensors {
        if let Some(z_vec) = tensor.svf_experts.get(&expert_name) {
            // `SVF` identity baseline is `1.0`; export additive delta `z - 1.0`
            let mut dz: Vec<f32> = z_vec.iter().map(|&z| z - 1.0).collect();
            clip_l2_in_place(&mut dz, eff_clip);
            add_deterministic_dp_noise(
                &mut dz,
                &format!("{delta_id}:{source_node_id}:{tname}"),
                eff_sigma,
            );
            svf_deltas.insert(tname.clone(), dz);
        }
    }

    let payload = canonical_delta_payload(
        &delta_id,
        &source_node_id,
        &expert_name,
        &svf_deltas,
        &engram_facts,
    );
    let mac = mesh_hmac_sha256(cluster_secret, &payload);

    SynapseSkillDelta {
        delta_id,
        source_node_id,
        source_fabric_class,
        expert_name,
        svf_deltas,
        engram_facts,
        clip_l2_norm: eff_clip,
        dp_noise_sigma: eff_sigma,
        training_energy_nj,
        peak_power_mw,
        hmac_hex: bytes_to_hex(&mac),
    }
}

/// Verify `HMAC-SHA256` authenticity of a [`SynapseSkillDelta`].
pub fn verify_skill_delta(cluster_secret: &[u8], delta: &SynapseSkillDelta) -> bool {
    let Some(claimed) = hex_to_bytes_32(&delta.hmac_hex) else {
        return false;
    };
    let payload = canonical_delta_payload(
        &delta.delta_id,
        &delta.source_node_id,
        &delta.expert_name,
        &delta.svf_deltas,
        &delta.engram_facts,
    );
    let expected = mesh_hmac_sha256(cluster_secret, &payload);
    ct_eq_32(&claimed, &expected)
}

fn flatten_delta_vector(delta: &SynapseSkillDelta) -> Vec<f32> {
    let mut flat = Vec::new();
    for vec in delta.svf_deltas.values() {
        flat.extend_from_slice(vec);
    }
    flat
}

fn l2_dist_sq(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().max(b.len());
    let mut sum = 0.0f32;
    for i in 0..len {
        let va = a.get(i).copied().unwrap_or(0.0);
        let vb = b.get(i).copied().unwrap_or(0.0);
        let d = va - vb;
        sum += d * d;
    }
    sum
}

/// Byzantine-resilient Multi-Krum + Norm-Bound federated skill aggregator.
///
/// 1. Rejects any peer delta with a forged/corrupted `HMAC-SHA256` signature.
/// 2. Rejects any peer delta whose per-tensor L2 norm exceeds `max_allowed_l2` (norm-attack defense).
/// 3. When `>= 3` valid peers submit deltas and `max_byzantine_peers > 0`, computes each peer's
///    Krum score (sum of squared distances to its `k = n - f - 2` closest neighbors) and drops the
///    `f = max_byzantine_peers` highest-distance outliers!
/// 4. Averages the surviving honest `SVF` deltas onto `engine.littlebit_tensors` (`0` base bitplane
///    mutations) and upserts honest `Engram` facts.
pub fn merge_federated_skill_deltas(
    engine: &mut JouleHarnessEngine,
    cluster_secret: &[u8],
    expert_name: &str,
    deltas: &[SynapseSkillDelta],
    max_byzantine_peers: usize,
    max_allowed_l2: f32,
) -> FederatedMergeReceipt {
    let mut valid: Vec<&SynapseSkillDelta> = Vec::new();
    let mut rejected_hmac = Vec::new();
    let mut rejected_byz = Vec::new();
    let norm_cap = max_allowed_l2.max(0.1);

    for d in deltas {
        if d.expert_name != expert_name || !verify_skill_delta(cluster_secret, d) {
            rejected_hmac.push(d.source_node_id.clone());
            continue;
        }
        let exceeds_norm = d.svf_deltas.values().any(|v| {
            let l2 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            !l2.is_finite() || l2 > norm_cap * 1.05
        });
        if exceeds_norm {
            rejected_byz.push(d.source_node_id.clone());
            continue;
        }
        valid.push(d);
    }

    // Multi-Krum outlier pruning when we have >= 3 candidate peers and f >= 1
    let n = valid.len();
    let eff_f = if n >= 3 {
        max_byzantine_peers.min((n - 1) / 2)
    } else {
        0
    };

    let honest: Vec<&SynapseSkillDelta> = if eff_f > 0 {
        let flats: Vec<Vec<f32>> = valid.iter().map(|d| flatten_delta_vector(d)).collect();
        let neighbors_to_sum = n.saturating_sub(eff_f).saturating_sub(2).max(1);
        let mut scored: Vec<(usize, f32)> = (0..n)
            .map(|i| {
                let mut dists: Vec<f32> = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| l2_dist_sq(&flats[i], &flats[j]))
                    .collect();
                dists.sort_by(|a, b| a.total_cmp(b));
                let krum_score: f32 = dists.into_iter().take(neighbors_to_sum).sum();
                (i, krum_score)
            })
            .collect();
        scored.sort_by(|a, b| a.1.total_cmp(&b.1));

        let keep_count = n - eff_f;
        for &(drop_idx, _) in &scored[keep_count..] {
            rejected_byz.push(valid[drop_idx].source_node_id.clone());
        }
        scored[..keep_count]
            .iter()
            .map(|&(idx, _)| valid[idx])
            .collect()
    } else {
        valid
    };

    let mut accepted_nodes = Vec::with_capacity(honest.len());
    let mut total_peer_nj = 0u64;
    let mut merged_engram = 0usize;
    let mut updated_tensors = 0usize;

    if !honest.is_empty() {
        let inv_count = 1.0f32 / (honest.len() as f32);
        let mut accum_svf: BTreeMap<String, Vec<f32>> = BTreeMap::new();

        for d in &honest {
            accepted_nodes.push(d.source_node_id.clone());
            total_peer_nj = total_peer_nj.saturating_add(d.training_energy_nj);
            for (tname, dz) in &d.svf_deltas {
                let slot = accum_svf
                    .entry(tname.clone())
                    .or_insert_with(|| vec![0.0; dz.len()]);
                for (i, &val) in dz.iter().enumerate() {
                    if let Some(s) = slot.get_mut(i) {
                        *s += val * inv_count;
                    }
                }
            }
            for (phrase, fact) in &d.engram_facts {
                engine.engram.upsert(phrase, Vec::new(), fact.clone());
                merged_engram += 1;
            }
        }

        for (tname, mean_dz) in accum_svf {
            if let Some(tensor) = engine.littlebit_tensors.get_mut(&tname) {
                let rank = tensor.rank;
                let entry = tensor
                    .svf_experts
                    .entry(expert_name.to_string())
                    .or_insert_with(|| vec![1.0; rank]);
                for (k, slot) in entry.iter_mut().enumerate() {
                    let delta_k = mean_dz.get(k).copied().unwrap_or(0.0);
                    *slot = (*slot + delta_k).clamp(-8.0, 8.0);
                }
                updated_tensors += 1;
            }
        }
    }

    FederatedMergeReceipt {
        expert_name: expert_name.to_string(),
        accepted_nodes,
        rejected_invalid_hmac_nodes: rejected_hmac,
        rejected_byzantine_nodes: rejected_byz,
        updated_tensors,
        merged_engram_facts: merged_engram,
        total_peer_training_nj: total_peer_nj,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synapse::LittleBitSVFTensor;

    #[test]
    fn test_federated_mesh_delta_hmac_dp_clipping_and_byzantine_krum_rejection() {
        let secret = b"fleet-shared-synapse-secret";
        let weights: Vec<f32> = (0..(64 * 32)).map(|i| ((i as f32) * 0.17).cos()).collect();

        let make_node_engine = |expert_z: Vec<f32>| {
            let mut eng = JouleHarnessEngine::new(16, 16, 8 * 1024 * 1024).unwrap();
            let mut t = LittleBitSVFTensor::factorize("m::l0", &weights, 64, 32, 4, 6, 0).unwrap();
            t.svf_experts.insert("robotics_grasp".into(), expert_z);
            eng.littlebit_tensors.insert("m::l0".into(), t);
            eng
        };

        // 3 honest robot/UMA nodes learn coherent grasp adjustments around +0.25 (`z = 1.25`)
        let node1 = make_node_engine(vec![1.24, 1.26, 0.80, 1.10]);
        let node2 = make_node_engine(vec![1.26, 1.24, 0.82, 1.08]);
        let node3 = make_node_engine(vec![1.25, 1.25, 0.81, 1.09]);
        // 1 Byzantine poisoned node tries to inject a wild adversarial direction within norm cap (`z = -1.9, -1.9, +2.8, -2.5`)
        let byz_node = make_node_engine(vec![-0.8, -0.8, 2.7, -0.7]);

        let d1 = export_skill_delta(
            &node1,
            secret,
            "d1",
            "edge-robot-01",
            "edge_robotic",
            "robotics_grasp",
            4.0,
            0.0,
            vec![("torque safety".into(), "max 14.2 Nm".into())],
            420_000,
            3_200,
        );
        let d2 = export_skill_delta(
            &node2,
            secret,
            "d2",
            "edge-robot-02",
            "edge_robotic",
            "robotics_grasp",
            4.0,
            0.0,
            vec![],
            410_000,
            3_100,
        );
        let d3 = export_skill_delta(
            &node3,
            secret,
            "d3",
            "uma-primary-01",
            "apple_uma_workstation",
            "robotics_grasp",
            4.0,
            0.0,
            vec![],
            190_000,
            8_500,
        );
        let d_byz = export_skill_delta(
            &byz_node,
            secret,
            "d_byz",
            "compromised-peer-99",
            "edge_robotic",
            "robotics_grasp",
            4.0,
            0.0,
            vec![("poison".into(), "disable brakes".into())],
            10_000,
            500,
        );
        let mut d_forged = d1.clone();
        d_forged.source_node_id = "forged-mitm".into(); // breaks HMAC!

        let mut coordinator = make_node_engine(vec![1.0, 1.0, 1.0, 1.0]);
        let receipt = merge_federated_skill_deltas(
            &mut coordinator,
            secret,
            "robotics_grasp",
            &[d1, d2, d3, d_byz, d_forged],
            1, // tolerate 1 Byzantine node via Multi-Krum
            4.0,
        );

        assert_eq!(receipt.rejected_invalid_hmac_nodes, vec!["forged-mitm"]);
        assert_eq!(
            receipt.rejected_byzantine_nodes,
            vec!["compromised-peer-99"]
        );
        assert_eq!(receipt.accepted_nodes.len(), 3);
        assert_eq!(receipt.updated_tensors, 1);

        // Verify poisoned Engram fact was NEVER merged, while honest fact WAS merged!
        assert!(coordinator.engram.lookup("poison").is_none());
        assert_eq!(
            coordinator
                .engram
                .lookup("torque safety")
                .unwrap()
                .fact_payload,
            "max 14.2 Nm"
        );

        // Verify merged SVF vector equals `1.0 + mean(0.25, 0.25, -0.19, 0.09)` = `[1.25, 1.25, 0.81, 1.09]`!
        let merged_z = &coordinator.littlebit_tensors["m::l0"].svf_experts["robotics_grasp"];
        assert!((merged_z[0] - 1.25).abs() < 0.02);
        assert!((merged_z[2] - 0.81).abs() < 0.02);
    }
}
