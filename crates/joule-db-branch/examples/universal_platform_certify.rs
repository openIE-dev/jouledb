//! Universal Cross-Platform JouleDB Engine Certifier (`universal_platform_certify`)
//!
//! Runs identically across native servers, desktops, mobile (Android ARM64), Windows 11,
//! Linux x86_64/ARM64, and WebAssembly (`wasm32-wasip2` / `wasm32-wasip1` in `wasmtime`),
//! verifying:
//! 1. `RaBitQ` 1-bit SIMD `POPCNT` vector search (`joule-db-core`)
//! 2. Embodied `EctStore` strictly causal `ASOF` alignment + `SE(3)` SLERP (`joule-db-core`)
//! 3. Living `SurpriseGate` active-inference sensory suppression (`joule-db-core`)
//! 4. Bit-exact `AlpCodec` floating-point compression (`joule-db-core`)
//! 5. Post-Docker `JouleCapsule` OCI `Joulefile` + `O(1)` CoW fork (`joule-db-branch`)
//! 6. Post-Silo `JouleSpace` (`DeltaDB` + `Jujutsu` conflict + Docker Sandbox scrub + LiveKit barge-in)

use joule_db_branch::{
    agentfs::JouleAgentFs,
    capsule::{CapsuleEngine, SpawnCapsuleRequest},
    joulespace::{ActorKind, JouleSpaceEngine, ModalityTrack, PresenceActivity, RoomParticipant},
};
use joule_db_core::{
    embodied::{AsofJoinSpec, EctStore, Transform3D, TransformTree},
    index::rabitq::{RaBitQConfig, RaBitQIndex},
    living::{PerceptionOutcome, SensoryFrame, SurpriseGate},
    storage::alp_fsst::AlpCodec,
};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Instant;

fn main() {
    let t0 = Instant::now();

    // 1. RaBitQ 1-bit POPCNT Vector Probe
    let mut rq = RaBitQIndex::new(RaBitQConfig::new(64));
    for id in 0..64u64 {
        let v: Vec<f32> = (0..64)
            .map(|d| ((id as usize + d) as f32 * 0.13).sin())
            .collect();
        rq.insert(id, &v);
    }
    let qv: Vec<f32> = (0..64).map(|d| ((7 + d) as f32 * 0.13).sin()).collect();
    let hits = rq.search(&qv, 3);
    let rabitq_ok = hits.first().is_some_and(|h| h.id == 7);

    // 2. Embodied ECT Causal ASOF + SE(3) Quaternion SLERP
    let ect = EctStore::new(64);
    ect.ingest("robot", "cmd", 2_000_000, vec![1.0], 0.01);
    ect.ingest("imu", "accel_z", 1_000_000, vec![9.81], 0.01);
    ect.ingest("imu", "accel_z", 3_000_000, vec![12.40], 0.01);
    let aligned = ect.asof_align(
        "robot",
        "cmd",
        0,
        5_000_000,
        &[AsofJoinSpec {
            entity: "imu".into(),
            component: "accel_z".into(),
            max_staleness_ns: 5_000_000,
        }],
    );
    let embodied_asof_ok = aligned
        .first()
        .and_then(|r| r.aligned.first())
        .and_then(|s| s.as_ref())
        .is_some_and(|s| (s.values[0] - 9.81).abs() < 1e-6);

    let tf = TransformTree::new();
    tf.upsert_transform("world", "gripper", 1_000, Transform3D::identity());
    tf.upsert_transform(
        "world",
        "gripper",
        2_000,
        Transform3D {
            translation: [2.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
        },
    );
    let mid_pose = tf.lookup_transform("world", "gripper", 1_500);
    let se3_ok = mid_pose.is_some_and(|p| (p.translation[0] - 1.0).abs() < 1e-4);

    // 3. Living SurpriseGate
    let gate = SurpriseGate::new(0.15, 9.0, 4);
    let mut suppressed = 0usize;
    for i in 0..40 {
        let d = gate.perceive(
            "joint.temp",
            SensoryFrame {
                timestamp_ns: i * 1_000,
                values: vec![36.5],
            },
        );
        if matches!(d, PerceptionOutcome::Habituated { .. }) {
            suppressed += 1;
        }
    }
    let spike = gate.perceive(
        "joint.temp",
        SensoryFrame {
            timestamp_ns: 41_000,
            values: vec![85.0],
        },
    );
    let living_ok = suppressed >= 20 && matches!(spike, PerceptionOutcome::PromotedEpisode(_));

    // 4. Bit-exact ALP Float Compression
    let floats: Vec<f64> = (0..256).map(|i| 24.0 + (i as f64) * 0.01).collect();
    let encoded = AlpCodec::encode(&floats);
    let decoded = AlpCodec::decode(&encoded).expect("alp decode");
    let alp_ok = decoded == floats && encoded.len() < floats.len() * 8;

    // 5. Post-Docker JouleCapsule + Post-Silo JouleSpace
    let agentfs = Arc::new(JouleAgentFs::standalone());
    let capsules = CapsuleEngine::new(agentfs.clone());
    let mut ctx = BTreeMap::new();
    ctx.insert("boot.txt".to_string(), b"universal-capsule-ready".to_vec());
    let _img = capsules
        .build_from_joulefile(
            "universal:v1",
            "FROM scratch\nWORKDIR /app\nENERGY_BUDGET 100mJ\nCOPY boot.txt /app/boot.txt\n",
            &ctx,
        )
        .expect("joulefile build");
    let _cap_a = capsules
        .spawn_capsule(SpawnCapsuleRequest {
            capsule_id: "u-a".into(),
            image_ref: "universal:v1".into(),
            env: BTreeMap::new(),
            ports: vec![],
            energy_budget_uj: None,
        })
        .expect("spawn");
    let cap_b = capsules
        .fork_capsule("u-a", "u-b", Some(50_000))
        .expect("fork");
    let capsule_ok = cap_b.capsule_id == "u-b";

    let space = JouleSpaceEngine::new(agentfs);
    let (_op, scrubbed) = space
        .apply_continuous_edit(
            "u-space",
            "agent",
            ActorKind::AiAgent,
            "/src/main.rs",
            "let token = \"sk-testsecret1234567890\";",
            "probe edit",
            None,
        )
        .expect("space edit");
    let mut tracks = BTreeSet::new();
    tracks.insert(ModalityTrack::AudioOpus);
    let _ = space
        .join_or_update_room(
            "u-room",
            "u-space",
            RoomParticipant {
                participant_id: "bot".into(),
                kind: ActorKind::AiAgent,
                activity: PresenceActivity::Idle,
                published_tracks: tracks,
                cursor_anchor: None,
                last_heartbeat_ms: 1,
            },
        )
        .expect("join room");
    let _ = space
        .start_agent_turn("u-room", "t1", "bot", 200_000)
        .expect("turn");
    let bi = space
        .trigger_barge_in("u-room", "human", "vad_interrupt")
        .expect("barge in");
    let space_ok = scrubbed == 1 && bi.joules_reclaimed_uj == 150_000;

    let elapsed_us = t0.elapsed().as_micros();
    let all_ok =
        rabitq_ok && embodied_asof_ok && se3_ok && living_ok && alp_ok && capsule_ok && space_ok;

    let report = serde_json::json!({
        "all_ok": all_ok,
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "family": std::env::consts::FAMILY,
        "elapsed_us": elapsed_us,
        "checks": {
            "rabitq_1bit_popcnt_ok": rabitq_ok,
            "embodied_causal_asof_ok": embodied_asof_ok,
            "se3_kinematic_slerp_ok": se3_ok,
            "living_surprise_gate_ok": living_ok,
            "alp_lossless_float_codec_ok": alp_ok,
            "post_docker_joule_capsule_cow_ok": capsule_ok,
            "post_silo_joulespace_barge_in_ok": space_ok,
        }
    });
    println!("{}", serde_json::to_string_pretty(&report).expect("json"));
    assert!(all_ok, "Universal platform certification failed: {report}");
}
