//! Wave 18 End-to-End Integration Suite:
//! 1. **Frontier 1 (Multimodal `SigLIP`/`ViT` Vision & `Whisper` Log-Mel Audio Living Bitplanes)**:
//!    - Ingests a `SigLIP` vision checkpoint (`vision_encoder_layers == 1`) into SQL `synapse_tensors`
//!      and verifies `POST /api/v1/synapse/vision/encode` + OpenAI multimodal `image_url` (`ChatContent::Parts`)
//!      on `POST /v1/chat/completions`.
//!    - Ingests a `Whisper` audio checkpoint (`audio_encoder_layers == 1`) and verifies BOTH JSON
//!      and `multipart/form-data` binary `.wav` uploads on OpenAI `POST /v1/audio/transcriptions`
//!      with 6-rail nanojoule receipts!
//! 2. **Frontier 2 (Lossless Self-Speculative `LayerSkip` + Rank-8 `LittleBit-2` Draft-and-Verify)**:
//!    - Verifies `speculative_draft_k: 4` produces **100% bit-identical greedy tokens** to
//!      `speculative_draft_k: 0` on `POST /api/v1/synapse/decoder/generate` while reporting
//!      `speculative_drafted_tokens > 0`.

use base64::Engine;
use joule_db_server::{Server, ServerConfig};
use joule_db_weights::{
    checkpoint_ingest::{SafetensorBuildEntry, SafetensorDtype, build_safetensors_bytes},
    multimodal_encoder::{build_pcm16_wav_bytes, build_ppm_p6_bytes},
};
use serde_json::{Value as JsonValue, json};
use std::collections::BTreeMap;
use std::f32::consts::PI;
use tokio::net::TcpListener;

#[tokio::test]
async fn test_wave18_multimodal_siglip_whisper_and_lossless_speculative_decoder_e2e() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let cfg = ServerConfig {
        db_path: tmp.path().to_string_lossy().to_string(),
        auth_enabled: false,
        rate_limiting_enabled: false,
        ..ServerConfig::default()
    };
    let server = Server::new(cfg).expect("boot server");
    let app = server.router();

    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let port = listener.local_addr().expect("addr").port();
    let srv_task = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });

    let base_url = format!("http://127.0.0.1:{port}");
    let http = reqwest::Client::new();
    let mk =
        |n: usize, s: f32| -> Vec<f32> { (0..n).map(|i| ((i as f32) * s).sin() * 0.45).collect() };

    // -------------------------------------------------------------------------
    // 1. Ingest `SigLIP` Vision Checkpoint & Encode P6 Image + Multimodal Chat
    // -------------------------------------------------------------------------
    let vis_entries = vec![
        SafetensorBuildEntry {
            name: "vision_model.embeddings.patch_embedding.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![32, 3, 4, 4],
            values_f32: mk(32 * 48, 0.11),
        },
        SafetensorBuildEntry {
            name: "vision_model.embeddings.position_embedding.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![16, 32],
            values_f32: mk(16 * 32, 0.17),
        },
        SafetensorBuildEntry {
            name: "vision_model.encoder.layers.0.self_attn.q_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![32, 32],
            values_f32: mk(32 * 32, 0.19),
        },
        SafetensorBuildEntry {
            name: "vision_model.encoder.layers.0.self_attn.k_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![32, 32],
            values_f32: mk(32 * 32, 0.23),
        },
        SafetensorBuildEntry {
            name: "vision_model.encoder.layers.0.self_attn.v_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![32, 32],
            values_f32: mk(32 * 32, 0.29),
        },
        SafetensorBuildEntry {
            name: "vision_model.encoder.layers.0.self_attn.out_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![32, 32],
            values_f32: mk(32 * 32, 0.31),
        },
        SafetensorBuildEntry {
            name: "vision_model.encoder.layers.0.mlp.fc1.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![64, 32],
            values_f32: mk(64 * 32, 0.37),
        },
        SafetensorBuildEntry {
            name: "vision_model.encoder.layers.0.mlp.fc2.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![32, 64],
            values_f32: mk(32 * 64, 0.41),
        },
    ];
    let vis_st = build_safetensors_bytes(&BTreeMap::new(), &vis_entries).expect("vis st");
    let vis_ing: JsonValue = http
        .post(format!("{base_url}/api/v1/synapse/ingest/safetensors"))
        .json(&json!({
            "model_id": "siglip-e2e",
            "bytes_base64": base64::engine::general_purpose::STANDARD.encode(&vis_st),
            "target_rank": 8,
            "distill_svf": true,
        }))
        .send()
        .await
        .expect("vis ing")
        .json()
        .await
        .expect("vis ing json");
    assert_eq!(vis_ing["vision_encoder_layers"], 1);

    let rgb: Vec<u8> = (0..16 * 16)
        .flat_map(|i| [(i % 255) as u8, 120, 40])
        .collect();
    let ppm = build_ppm_p6_bytes(16, 16, &rgb).expect("ppm");
    let ppm_b64 = base64::engine::general_purpose::STANDARD.encode(&ppm);

    let v_enc: JsonValue = http
        .post(format!("{base_url}/api/v1/synapse/vision/encode"))
        .json(&json!({
            "model_id": "siglip-e2e",
            "image_base64": ppm_b64,
        }))
        .send()
        .await
        .expect("v_enc")
        .json()
        .await
        .expect("v_enc json");
    assert_eq!(v_enc["model_id"], "siglip-e2e");
    assert_eq!(v_enc["num_patches"], 16);
    assert_eq!(v_enc["embedding"].as_array().map(Vec::len), Some(32));

    // OpenAI multimodal `image_url` (`ChatContent::Parts`) on `/v1/chat/completions`
    let mm_chat: JsonValue = http
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&json!({
            "model": "jouledb-cascade-v1",
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "text", "text": "Inspect thermal camera frame:" },
                    { "type": "image_url", "image_url": { "url": format!("data:image/x-portable-pixmap;base64,{ppm_b64}") } }
                ]
            }]
        }))
        .send()
        .await
        .expect("mm chat")
        .json()
        .await
        .expect("mm chat json");
    assert_eq!(mm_chat["object"], "chat.completion");

    // -------------------------------------------------------------------------
    // 2. Ingest `Whisper` Audio Checkpoint & Verify `/v1/audio/transcriptions` (JSON + Multipart!)
    // -------------------------------------------------------------------------
    let aud_entries = vec![
        SafetensorBuildEntry {
            name: "model.encoder.conv1.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![32, 80, 3],
            values_f32: mk(32 * 240, 0.13),
        },
        SafetensorBuildEntry {
            name: "model.encoder.layers.0.self_attn.q_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![32, 32],
            values_f32: mk(32 * 32, 0.17),
        },
        SafetensorBuildEntry {
            name: "model.encoder.layers.0.self_attn.k_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![32, 32],
            values_f32: mk(32 * 32, 0.19),
        },
        SafetensorBuildEntry {
            name: "model.encoder.layers.0.self_attn.v_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![32, 32],
            values_f32: mk(32 * 32, 0.23),
        },
        SafetensorBuildEntry {
            name: "model.encoder.layers.0.self_attn.out_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![32, 32],
            values_f32: mk(32 * 32, 0.29),
        },
        SafetensorBuildEntry {
            name: "model.encoder.layers.0.fc1.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![64, 32],
            values_f32: mk(64 * 32, 0.31),
        },
        SafetensorBuildEntry {
            name: "model.encoder.layers.0.fc2.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![32, 64],
            values_f32: mk(32 * 64, 0.37),
        },
        SafetensorBuildEntry {
            name: "proj_out.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![16, 32],
            values_f32: mk(16 * 32, 0.41),
        },
    ];
    let aud_st = build_safetensors_bytes(&BTreeMap::new(), &aud_entries).expect("aud st");
    let aud_ing: JsonValue = http
        .post(format!("{base_url}/api/v1/synapse/ingest/safetensors"))
        .json(&json!({
            "model_id": "whisper-e2e",
            "bytes_base64": base64::engine::general_purpose::STANDARD.encode(&aud_st),
            "target_rank": 8,
            "distill_svf": true,
        }))
        .send()
        .await
        .expect("aud ing")
        .json()
        .await
        .expect("aud ing json");
    assert_eq!(aud_ing["audio_encoder_layers"], 1);

    let samples: Vec<f32> = (0..4_000)
        .map(|i| (2.0 * PI * 440.0 * (i as f32) / 16_000.0).sin() * 0.5)
        .collect();
    let wav = build_pcm16_wav_bytes(16_000, &samples);
    let wav_b64 = base64::engine::general_purpose::STANDARD.encode(&wav);

    let tr_json: JsonValue = http
        .post(format!("{base_url}/v1/audio/transcriptions"))
        .json(&json!({
            "model": "whisper-e2e",
            "audio_base64": wav_b64,
        }))
        .send()
        .await
        .expect("tr json")
        .json()
        .await
        .expect("tr json parse");
    assert_eq!(tr_json["model"], "whisper-e2e");
    assert_eq!(tr_json["mel_bins"], 80);
    assert!(
        tr_json["x_jouledb_energy"]["aggregate_rails"]["wall_duration_ns"]
            .as_u64()
            .unwrap_or(0)
            > 0
    );

    // Also verify raw `multipart/form-data` binary WAV upload (`curl -F file=@...`)
    let boundary = "----JouleDBWave18Boundary";
    let mut mp_body = Vec::new();
    mp_body.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper-e2e\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"tone.wav\"\r\nContent-Type: audio/wav\r\n\r\n"
        )
        .as_bytes(),
    );
    mp_body.extend_from_slice(&wav);
    mp_body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());

    let tr_mp: JsonValue = http
        .post(format!("{base_url}/v1/audio/transcriptions"))
        .header(
            "content-type",
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(mp_body)
        .send()
        .await
        .expect("tr mp")
        .json()
        .await
        .expect("tr mp json");
    assert_eq!(tr_mp["model"], "whisper-e2e");
    assert_eq!(tr_mp["mel_bins"], 80);

    // -------------------------------------------------------------------------
    // 3. Frontier 2: Lossless Self-Speculative `LayerSkip` + Rank-8 Causal Decode
    // -------------------------------------------------------------------------
    let dec_entries = vec![
        SafetensorBuildEntry {
            name: "model.embed_tokens.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![64, 64],
            values_f32: mk(64 * 64, 0.11),
        },
        SafetensorBuildEntry {
            name: "model.layers.0.input_layernorm.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![64],
            values_f32: vec![1.0; 64],
        },
        SafetensorBuildEntry {
            name: "model.layers.0.self_attn.q_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![64, 64],
            values_f32: mk(64 * 64, 0.13),
        },
        SafetensorBuildEntry {
            name: "model.layers.0.self_attn.k_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![32, 64],
            values_f32: mk(32 * 64, 0.17),
        },
        SafetensorBuildEntry {
            name: "model.layers.0.self_attn.v_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![32, 64],
            values_f32: mk(32 * 64, 0.19),
        },
        SafetensorBuildEntry {
            name: "model.layers.0.self_attn.o_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![64, 64],
            values_f32: mk(64 * 64, 0.23),
        },
        SafetensorBuildEntry {
            name: "model.layers.0.post_attention_layernorm.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![64],
            values_f32: vec![1.0; 64],
        },
        SafetensorBuildEntry {
            name: "model.layers.0.mlp.gate_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![128, 64],
            values_f32: mk(128 * 64, 0.29),
        },
        SafetensorBuildEntry {
            name: "model.layers.0.mlp.up_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![128, 64],
            values_f32: mk(128 * 64, 0.31),
        },
        SafetensorBuildEntry {
            name: "model.layers.0.mlp.down_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![64, 128],
            values_f32: mk(64 * 128, 0.37),
        },
        SafetensorBuildEntry {
            name: "model.norm.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![64],
            values_f32: vec![1.0; 64],
        },
    ];
    let dec_st = build_safetensors_bytes(&BTreeMap::new(), &dec_entries).expect("dec st");
    let _ = http
        .post(format!("{base_url}/api/v1/synapse/ingest/safetensors"))
        .json(&json!({
            "model_id": "spec-causal",
            "bytes_base64": base64::engine::general_purpose::STANDARD.encode(&dec_st),
            "target_rank": 16,
            "distill_svf": true,
        }))
        .send()
        .await
        .expect("dec ing");

    let base_dec: JsonValue = http
        .post(format!("{base_url}/api/v1/synapse/decoder/generate"))
        .json(&json!({
            "model_id": "spec-causal",
            "prompt": "Verify lossless self-speculative LayerSkip and rank-8 LittleBit draft verification",
            "max_new_tokens": 10,
            "temperature": 0.0,
            "speculative_draft_k": 0,
        }))
        .send()
        .await
        .expect("base dec")
        .json()
        .await
        .expect("base dec json");

    let spec_dec: JsonValue = http
        .post(format!("{base_url}/api/v1/synapse/decoder/generate"))
        .json(&json!({
            "model_id": "spec-causal",
            "prompt": "Verify lossless self-speculative LayerSkip and rank-8 LittleBit draft verification",
            "max_new_tokens": 10,
            "temperature": 0.0,
            "speculative_draft_k": 4,
        }))
        .send()
        .await
        .expect("spec dec")
        .json()
        .await
        .expect("spec dec json");

    assert_eq!(
        spec_dec["generated_token_ids"],
        base_dec["generated_token_ids"]
    );
    assert!(spec_dec["speculative_drafted_tokens"].as_u64().unwrap_or(0) > 0);

    srv_task.abort();
}
