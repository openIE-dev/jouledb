/*
 * Wire Protocol Stress Tests
 * Exercises the binary protocol encoder/decoder with adversarial inputs:
 * malformed headers, corrupted magic, invalid message types, truncation,
 * payload size limits, varint edge cases, and roundtrip fuzz patterns.
 */

use joule_db_server::binary_protocol::{
    BatchOp, BinaryMessage, BinaryProtocol, BinaryProtocolError, Flags, HEADER_SIZE, MAGIC,
    MAX_PAYLOAD_SIZE, MessageType, VERSION,
};

fn proto() -> BinaryProtocol {
    BinaryProtocol::new()
}

/// Helper: decode must fail with the expected error.
fn assert_decode_err(data: &[u8], expected: BinaryProtocolError) {
    let result = proto().decode(data);
    assert_eq!(result.unwrap_err(), expected);
}

// ── Magic & Version ──────────────────────────────────────────────────

#[test]
fn proto_wrong_magic_byte_0() {
    let p = proto();
    let msg = BinaryMessage::ping(1);
    let mut enc = p.encode(&msg).unwrap();
    enc[0] = 0x00;
    assert_decode_err(&enc, BinaryProtocolError::InvalidMagic);
}

#[test]
fn proto_wrong_magic_all_zeros() {
    let mut buf = vec![0u8; HEADER_SIZE];
    buf[4] = VERSION;
    assert_decode_err(&buf, BinaryProtocolError::InvalidMagic);
}

#[test]
fn proto_wrong_magic_all_ff() {
    let buf = vec![0xFF; HEADER_SIZE];
    assert_decode_err(&buf, BinaryProtocolError::InvalidMagic);
}

#[test]
fn proto_wrong_version() {
    let p = proto();
    let msg = BinaryMessage::ping(1);
    let mut enc = p.encode(&msg).unwrap();
    enc[4] = 99;
    assert_decode_err(&enc, BinaryProtocolError::UnsupportedVersion(99));
}

#[test]
fn proto_version_zero() {
    let p = proto();
    let msg = BinaryMessage::ping(1);
    let mut enc = p.encode(&msg).unwrap();
    enc[4] = 0;
    assert_decode_err(&enc, BinaryProtocolError::UnsupportedVersion(0));
}

#[test]
fn proto_version_max() {
    let p = proto();
    let msg = BinaryMessage::ping(1);
    let mut enc = p.encode(&msg).unwrap();
    enc[4] = 255;
    assert_decode_err(&enc, BinaryProtocolError::UnsupportedVersion(255));
}

// ── Truncation ───────────────────────────────────────────────────────

#[test]
fn proto_empty_buffer() {
    assert_decode_err(&[], BinaryProtocolError::TruncatedMessage);
}

#[test]
fn proto_one_byte() {
    assert_decode_err(&[0x57], BinaryProtocolError::TruncatedMessage);
}

#[test]
fn proto_header_minus_one() {
    let buf = vec![0u8; HEADER_SIZE - 1];
    assert_decode_err(&buf, BinaryProtocolError::TruncatedMessage);
}

#[test]
fn proto_header_only_with_nonzero_payload_len() {
    let p = proto();
    let msg = BinaryMessage::get(1, b"key");
    let enc = p.encode(&msg).unwrap();
    let header_only = &enc[..HEADER_SIZE];
    assert_decode_err(header_only, BinaryProtocolError::TruncatedMessage);
}

#[test]
fn proto_payload_one_byte_short() {
    let p = proto();
    let msg = BinaryMessage::get(1, b"hello");
    let enc = p.encode(&msg).unwrap();
    let truncated = &enc[..enc.len() - 1];
    assert_decode_err(truncated, BinaryProtocolError::TruncatedMessage);
}

// ── Unknown Message Types ────────────────────────────────────────────

#[test]
fn proto_unknown_message_type_0x0000() {
    let mut buf = vec![0u8; HEADER_SIZE];
    buf[..4].copy_from_slice(&MAGIC);
    buf[4] = VERSION;
    buf[6] = 0x00;
    buf[7] = 0x00;
    assert_decode_err(&buf, BinaryProtocolError::UnknownMessageType(0x0000));
}

#[test]
fn proto_unknown_message_type_0xFFFF() {
    let mut buf = vec![0u8; HEADER_SIZE];
    buf[..4].copy_from_slice(&MAGIC);
    buf[4] = VERSION;
    buf[6] = 0xFF;
    buf[7] = 0xFF;
    assert_decode_err(&buf, BinaryProtocolError::UnknownMessageType(0xFFFF));
}

#[test]
fn proto_unknown_message_type_0x7FFF() {
    let mut buf = vec![0u8; HEADER_SIZE];
    buf[..4].copy_from_slice(&MAGIC);
    buf[4] = VERSION;
    buf[6] = 0xFF;
    buf[7] = 0x7F;
    assert_decode_err(&buf, BinaryProtocolError::UnknownMessageType(0x7FFF));
}

#[test]
fn proto_message_type_from_u16_exhaustive() {
    let defined: Vec<u16> = vec![
        0x0001, 0x0002, 0x0003, 0x0004, 0x0005, 0x0006, 0x0007, 0x0008, 0x0009, 0x0010, 0x0011,
        0x0012, 0x0013, 0x0014, 0x0015, 0x8001, 0x8002, 0x8003, 0x8004, 0x8005, 0x8006, 0x8007,
        0x8008, 0x8010, 0x8011, 0x8012, 0x8013, 0x8014, 0x8015, 0x80FF,
    ];
    for &v in &defined {
        assert!(
            MessageType::from_u16(v).is_some(),
            "Expected Some for 0x{:04X}",
            v
        );
    }
    for v in [
        0u16, 0x000A, 0x000F, 0x0016, 0x7FFF, 0x8000, 0x8009, 0x80FE, 0xFFFF,
    ] {
        assert!(
            MessageType::from_u16(v).is_none(),
            "Expected None for 0x{:04X}",
            v
        );
    }
}

// ── Payload Size Limits ──────────────────────────────────────────────

#[test]
fn proto_payload_at_max_size() {
    let mut buf = vec![0u8; HEADER_SIZE];
    buf[..4].copy_from_slice(&MAGIC);
    buf[4] = VERSION;
    buf[6] = 0x06; // Ping
    buf[7] = 0x00;
    let max_len = MAX_PAYLOAD_SIZE;
    buf[12..16].copy_from_slice(&max_len.to_le_bytes());
    assert_decode_err(&buf, BinaryProtocolError::TruncatedMessage);
}

#[test]
fn proto_payload_exceeds_max() {
    let mut buf = vec![0u8; HEADER_SIZE];
    buf[..4].copy_from_slice(&MAGIC);
    buf[4] = VERSION;
    buf[6] = 0x06; // Ping
    buf[7] = 0x00;
    let over = MAX_PAYLOAD_SIZE + 1;
    buf[12..16].copy_from_slice(&over.to_le_bytes());
    assert_decode_err(&buf, BinaryProtocolError::PayloadTooLarge(over));
}

#[test]
fn proto_encode_rejects_oversized_payload() {
    let p = proto();
    let msg = BinaryMessage::new(
        MessageType::Query,
        1,
        vec![0u8; (MAX_PAYLOAD_SIZE + 1) as usize],
    );
    assert!(matches!(
        p.encode(&msg),
        Err(BinaryProtocolError::PayloadTooLarge(_))
    ));
}

// ── Roundtrip All Message Types ──────────────────────────────────────

#[test]
fn proto_roundtrip_all_request_types() {
    let p = proto();
    let messages: Vec<BinaryMessage> = vec![
        BinaryMessage::get(1, b"key"),
        BinaryMessage::set(2, b"key", b"val", None),
        BinaryMessage::set(3, b"key", b"val", Some(60)),
        BinaryMessage::delete(4, b"key"),
        BinaryMessage::ping(5),
        BinaryMessage::subscribe(6, "events:*"),
        BinaryMessage::unsubscribe(7, 42),
        BinaryMessage::auth(8, "token123"),
        BinaryMessage::query(9, "SELECT 1", None),
        BinaryMessage::query(10, "SELECT ?", Some(b"param")),
        BinaryMessage::batch(
            11,
            vec![
                BatchOp::Set {
                    key: b"k".to_vec(),
                    value: b"v".to_vec(),
                    ttl: None,
                },
                BatchOp::Delete { key: b"d".to_vec() },
            ],
        ),
    ];
    for msg in &messages {
        let enc = p.encode(msg).unwrap();
        let dec = p.decode(&enc).unwrap();
        assert_eq!(dec.msg_type, msg.msg_type);
        assert_eq!(dec.request_id, msg.request_id);
        assert_eq!(dec.payload, msg.payload);
    }
}

#[test]
fn proto_roundtrip_all_response_types() {
    let p = proto();
    let messages: Vec<BinaryMessage> = vec![
        BinaryMessage::get_response(1, Some(b"value")),
        BinaryMessage::get_response(2, None),
        BinaryMessage::set_response(3, true),
        BinaryMessage::set_response(4, false),
        BinaryMessage::delete_response(5, true),
        BinaryMessage::delete_response(6, false),
        BinaryMessage::pong(7),
        BinaryMessage::error(8, "ERR001", "something went wrong"),
        BinaryMessage::auth_response(9, true, "admin"),
        BinaryMessage::auth_response(10, false, "bad token"),
        BinaryMessage::notification(11, 999, 0, "my-key", Some(b"new"), None, 1234567890),
    ];
    for msg in &messages {
        let enc = p.encode(msg).unwrap();
        let dec = p.decode(&enc).unwrap();
        assert_eq!(dec.msg_type, msg.msg_type);
        assert_eq!(dec.request_id, msg.request_id);
        assert_eq!(dec.payload, msg.payload);
    }
}

// ── Request ID Boundaries ────────────────────────────────────────────

#[test]
fn proto_request_id_zero() {
    let p = proto();
    let msg = BinaryMessage::ping(0);
    let enc = p.encode(&msg).unwrap();
    let dec = p.decode(&enc).unwrap();
    assert_eq!(dec.request_id, 0);
}

#[test]
fn proto_request_id_max() {
    let p = proto();
    let msg = BinaryMessage::ping(u32::MAX);
    let enc = p.encode(&msg).unwrap();
    let dec = p.decode(&enc).unwrap();
    assert_eq!(dec.request_id, u32::MAX);
}

// ── Flags ────────────────────────────────────────────────────────────

#[test]
fn proto_all_flags_set() {
    let p = proto();
    let mut msg = BinaryMessage::ping(1);
    msg.flags = Flags::from_bits(0xFF);
    let enc = p.encode(&msg).unwrap();
    let dec = p.decode(&enc).unwrap();
    assert_eq!(dec.flags.bits(), 0xFF);
    assert!(dec.flags.has(Flags::COMPRESSED));
    assert!(dec.flags.has(Flags::EXPECT_RESPONSE));
    assert!(dec.flags.has(Flags::FINAL));
}

#[test]
fn proto_flags_clear() {
    let mut flags = Flags::from_bits(0xFF);
    flags.clear(Flags::COMPRESSED);
    assert!(!flags.has(Flags::COMPRESSED));
    assert!(flags.has(Flags::EXPECT_RESPONSE));
    assert!(flags.has(Flags::FINAL));
}

// ── Payload Parsing Edge Cases ───────────────────────────────────────

#[test]
fn proto_parse_get_empty_payload() {
    assert!(proto().parse_get(&[]).is_err());
}

#[test]
fn proto_parse_set_empty_payload() {
    assert!(proto().parse_set(&[]).is_err());
}

#[test]
fn proto_parse_set_missing_ttl_flag() {
    let p = proto();
    let msg = BinaryMessage::set(1, b"key", b"val", Some(60));
    let payload = &msg.payload[..msg.payload.len() - 9];
    assert!(p.parse_set(payload).is_err());
}

#[test]
fn proto_parse_get_response_empty() {
    assert!(proto().parse_get_response(&[]).is_err());
}

#[test]
fn proto_parse_subscribe_empty() {
    assert!(proto().parse_subscribe(&[]).is_err());
}

#[test]
fn proto_parse_unsubscribe_short() {
    assert!(proto().parse_unsubscribe(&[0; 7]).is_err());
}

#[test]
fn proto_parse_auth_token_empty() {
    assert_eq!(BinaryMessage::parse_auth_token(&[]), None);
}

// ── Empty Key/Value ──────────────────────────────────────────────────

#[test]
fn proto_get_empty_key() {
    let p = proto();
    let msg = BinaryMessage::get(1, b"");
    let enc = p.encode(&msg).unwrap();
    let dec = p.decode(&enc).unwrap();
    let key = p.parse_get(&dec.payload).unwrap();
    assert!(key.is_empty());
}

#[test]
fn proto_set_empty_key_empty_value() {
    let p = proto();
    let msg = BinaryMessage::set(1, b"", b"", None);
    let enc = p.encode(&msg).unwrap();
    let dec = p.decode(&enc).unwrap();
    let (key, value, ttl) = p.parse_set(&dec.payload).unwrap();
    assert!(key.is_empty());
    assert!(value.is_empty());
    assert!(ttl.is_none());
}

// ── Large Payloads ───────────────────────────────────────────────────

#[test]
fn proto_large_key_1mb() {
    let p = proto();
    let big_key = vec![b'A'; 1024 * 1024];
    let msg = BinaryMessage::get(1, &big_key);
    let enc = p.encode(&msg).unwrap();
    let dec = p.decode(&enc).unwrap();
    let key = p.parse_get(&dec.payload).unwrap();
    assert_eq!(key.len(), 1024 * 1024);
}

#[test]
fn proto_query_large_sql() {
    let p = proto();
    let huge_sql = "SELECT ".to_string() + &"a, ".repeat(100_000) + "b FROM t";
    let msg = BinaryMessage::query(1, &huge_sql, None);
    let enc = p.encode(&msg).unwrap();
    let dec = p.decode(&enc).unwrap();
    assert_eq!(dec.msg_type, MessageType::Query);
    assert_eq!(dec.payload, msg.payload);
}

// ── Batch Edge Cases ─────────────────────────────────────────────────

#[test]
fn proto_batch_empty() {
    let p = proto();
    let msg = BinaryMessage::batch(1, vec![]);
    let enc = p.encode(&msg).unwrap();
    let dec = p.decode(&enc).unwrap();
    assert_eq!(dec.msg_type, MessageType::Batch);
}

#[test]
fn proto_batch_1000_ops() {
    let p = proto();
    let ops: Vec<BatchOp> = (0..1000)
        .map(|i| {
            if i % 3 == 0 {
                BatchOp::Delete {
                    key: format!("key_{}", i).into_bytes(),
                }
            } else {
                BatchOp::Set {
                    key: format!("key_{}", i).into_bytes(),
                    value: format!("value_{}", i).into_bytes(),
                    ttl: if i % 2 == 0 { Some(i as u64) } else { None },
                }
            }
        })
        .collect();
    let msg = BinaryMessage::batch(1, ops);
    let enc = p.encode(&msg).unwrap();
    let dec = p.decode(&enc).unwrap();
    assert_eq!(dec.msg_type, MessageType::Batch);
    assert_eq!(dec.payload, msg.payload);
}

// ── TTL Boundaries ───────────────────────────────────────────────────

#[test]
fn proto_set_ttl_zero() {
    let p = proto();
    let msg = BinaryMessage::set(1, b"k", b"v", Some(0));
    let enc = p.encode(&msg).unwrap();
    let dec = p.decode(&enc).unwrap();
    let (_, _, ttl) = p.parse_set(&dec.payload).unwrap();
    assert_eq!(ttl, Some(0));
}

#[test]
fn proto_set_ttl_max_u64() {
    let p = proto();
    let msg = BinaryMessage::set(1, b"k", b"v", Some(u64::MAX));
    let enc = p.encode(&msg).unwrap();
    let dec = p.decode(&enc).unwrap();
    let (_, _, ttl) = p.parse_set(&dec.payload).unwrap();
    assert_eq!(ttl, Some(u64::MAX));
}

// ── Notification Edge Cases ──────────────────────────────────────────

#[test]
fn proto_notification_no_values() {
    let p = proto();
    let msg = BinaryMessage::notification(1, 0, 2, "key", None, None, 0);
    let enc = p.encode(&msg).unwrap();
    let dec = p.decode(&enc).unwrap();
    assert_eq!(dec.msg_type, MessageType::Notification);
}

#[test]
fn proto_notification_both_values() {
    let p = proto();
    let msg = BinaryMessage::notification(
        1,
        u64::MAX,
        1,
        "my-key",
        Some(b"new_val"),
        Some(b"old_val"),
        u64::MAX,
    );
    let enc = p.encode(&msg).unwrap();
    let dec = p.decode(&enc).unwrap();
    assert_eq!(dec.payload, msg.payload);
}

// ── Corrupted Payloads ───────────────────────────────────────────────

#[test]
fn proto_corrupted_payload_len_too_large() {
    let p = proto();
    let msg = BinaryMessage::ping(1);
    let mut enc = p.encode(&msg).unwrap();
    enc[12..16].copy_from_slice(&1000u32.to_le_bytes());
    assert_decode_err(&enc, BinaryProtocolError::TruncatedMessage);
}

#[test]
fn proto_payload_len_zero_but_data_appended() {
    let p = proto();
    let msg = BinaryMessage::ping(1);
    let mut enc = p.encode(&msg).unwrap();
    enc.extend_from_slice(b"garbage_data");
    let dec = p.decode(&enc).unwrap();
    assert_eq!(dec.msg_type, MessageType::Ping);
    assert!(dec.payload.is_empty());
}

// ── Binary Data in Keys/Values ───────────────────────────────────────

#[test]
fn proto_binary_key_with_nulls() {
    let p = proto();
    let key = b"\x00\x01\x02\xFF\xFE\xFD\x00\x00";
    let msg = BinaryMessage::get(1, key);
    let enc = p.encode(&msg).unwrap();
    let dec = p.decode(&enc).unwrap();
    let parsed_key = p.parse_get(&dec.payload).unwrap();
    assert_eq!(parsed_key, key);
}

#[test]
fn proto_utf8_multibyte_in_query() {
    let p = proto();
    let sql = "SELECT * FROM 表 WHERE 名前 = '日本語テスト' AND emoji = '🔥🚀💎'";
    let msg = BinaryMessage::query(1, sql, None);
    let enc = p.encode(&msg).unwrap();
    let dec = p.decode(&enc).unwrap();
    assert_eq!(dec.payload, msg.payload);
}

// ── parse_header Stress ──────────────────────────────────────────────

#[test]
fn proto_parse_header_valid() {
    let p = proto();
    let msg = BinaryMessage::query(42, "SELECT 1", None);
    let enc = p.encode(&msg).unwrap();
    let (msg_type, req_id, payload_len) = p.parse_header(&enc).unwrap();
    assert_eq!(msg_type, MessageType::Query);
    assert_eq!(req_id, 42);
    assert_eq!(payload_len as usize + HEADER_SIZE, enc.len());
}

#[test]
fn proto_parse_header_too_short() {
    assert!(proto().parse_header(&[0u8; 15]).is_err());
}

// ── Message Type Properties ──────────────────────────────────────────

#[test]
fn proto_request_response_classification() {
    let requests = [
        MessageType::Get,
        MessageType::Set,
        MessageType::Delete,
        MessageType::Query,
        MessageType::Batch,
        MessageType::Ping,
        MessageType::Subscribe,
        MessageType::Unsubscribe,
        MessageType::Auth,
        MessageType::BeginTx,
        MessageType::Commit,
        MessageType::Rollback,
        MessageType::Savepoint,
        MessageType::Prepare,
        MessageType::Execute,
    ];
    for req in requests {
        assert!(req.is_request(), "{:?} should be request", req);
        assert!(!req.is_response(), "{:?} should not be response", req);
    }
    let responses = [
        MessageType::GetResponse,
        MessageType::SetResponse,
        MessageType::DeleteResponse,
        MessageType::QueryResponse,
        MessageType::BatchResponse,
        MessageType::Pong,
        MessageType::Notification,
        MessageType::AuthResponse,
        MessageType::BeginTxResponse,
        MessageType::CommitResponse,
        MessageType::RollbackResponse,
        MessageType::SavepointResponse,
        MessageType::PrepareResponse,
        MessageType::ExecuteResponse,
        MessageType::Error,
    ];
    for resp in responses {
        assert!(resp.is_response(), "{:?} should be response", resp);
        assert!(!resp.is_request(), "{:?} should not be request", resp);
    }
}

// ── Rapid Encode/Decode Throughput ───────────────────────────────────

#[test]
fn proto_encode_decode_10k_messages() {
    let p = proto();
    for i in 0..10_000u32 {
        let msg = BinaryMessage::set(
            i,
            format!("key_{}", i).as_bytes(),
            format!("value_{}", i).as_bytes(),
            if i % 2 == 0 {
                Some(i as u64 * 1000)
            } else {
                None
            },
        );
        let enc = p.encode(&msg).unwrap();
        let dec = p.decode(&enc).unwrap();
        assert_eq!(dec.request_id, i);
        assert_eq!(dec.msg_type, MessageType::Set);
    }
}

// ── Error Message Roundtrip ──────────────────────────────────────────

#[test]
fn proto_error_with_unicode() {
    let p = proto();
    let msg = BinaryMessage::error(1, "UNICODE_ERR", "Erreur: données invalides 🔥");
    let enc = p.encode(&msg).unwrap();
    let dec = p.decode(&enc).unwrap();
    assert_eq!(dec.msg_type, MessageType::Error);
    assert_eq!(dec.payload, msg.payload);
}

#[test]
fn proto_error_empty_strings() {
    let p = proto();
    let msg = BinaryMessage::error(1, "", "");
    let enc = p.encode(&msg).unwrap();
    let dec = p.decode(&enc).unwrap();
    assert_eq!(dec.msg_type, MessageType::Error);
}
