//! JWP (Joule Wire Protocol) transport for JouleDB.
//!
//! Energy-aware binary wire protocol over TCP. Every frame header carries
//! cumulative µWh energy cost. Supports v1 (static 21-byte headers) and
//! v2 (adaptive amorphous headers, compression negotiation).
//!
//! ## Connection lifecycle
//!
//! ```text
//! Client                            Server
//! ──────                            ──────
//! Handshake ──────────────────────► HandshakeAck
//! Query ──────────────────────────► Meta + Result* + Done
//! Heartbeat ──────────────────────► Heartbeat (echo)
//! Cancel ─────────────────────────► (abort streaming / unsubscribe)
//! ```

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;
use tokio::sync::{Notify, RwLock, Semaphore, mpsc};

use jwp::TcpTransport;
use jwp::Transport;
use jwp::frame::{
    ErrorPayload, FrameType, HandshakePayload, JwpFrame, PROTOCOL_VERSION, cbor_decode, cbor_encode,
};
use jwp::state_machine::ProtocolStateMachine;

use crate::query::{QueryExecutor, QueryRequest};
use crate::subscriptions::{ChangeEvent, ChangeOperation, SubscriptionManager};

use joule_db_ledger::ReceiptCollector;
use joule_db_ledger::ReceiptStore;

// ── Database-specific CBOR payloads ──────────────────────────────────

/// Client's SQL query (carried in a Query frame payload).
#[derive(Debug, Serialize, Deserialize)]
struct DbQueryPayload {
    sql: String,
    #[serde(default)]
    args: Vec<serde_json::Value>,
    #[serde(default)]
    named: BTreeMap<String, serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    limit: Option<usize>,
    #[serde(default)]
    explain: bool,
}

/// Column metadata (carried in a Meta frame payload).
#[derive(Debug, Serialize, Deserialize)]
struct DbMetaPayload {
    columns: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    session_id: Option<String>,
}

/// A batch of rows (carried in a Result frame payload).
#[derive(Debug, Serialize, Deserialize)]
struct DbResultPayload {
    rows: Vec<Vec<serde_json::Value>>,
}

/// Query completion summary (carried in a Done frame payload).
#[derive(Debug, Serialize, Deserialize)]
struct DbDonePayload {
    row_count: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    affected_rows: Option<u64>,
    total_cost_uwh: u64,
    elapsed_ms: u64,
}

// ── Config ───────────────────────────────────────────────────────────

/// JWP server configuration.
#[derive(Debug, Clone)]
pub struct JwpServerConfig {
    pub bind_addr: String,
    pub max_connections: usize,
    pub connection_timeout_secs: u64,
}

impl Default for JwpServerConfig {
    fn default() -> Self {
        Self {
            bind_addr: "0.0.0.0:9200".to_string(),
            max_connections: 1000,
            connection_timeout_secs: 300,
        }
    }
}

// ── Stats ────────────────────────────────────────────────────────────

/// Atomic counters for the JWP server.
#[derive(Debug, Default)]
pub struct JwpServerStats {
    pub connections_accepted: AtomicU64,
    pub connections_active: AtomicU64,
    pub frames_received: AtomicU64,
    pub frames_sent: AtomicU64,
    pub queries_executed: AtomicU64,
    pub errors: AtomicU64,
}

// ── Server ───────────────────────────────────────────────────────────

/// Ledger context for JWP ledger dispatch.
pub struct JwpLedgerContext {
    pub collector: Arc<ReceiptCollector>,
    pub store: Arc<RwLock<ReceiptStore>>,
}

/// JWP transport server for JouleDB.
pub struct JwpServer {
    config: JwpServerConfig,
    query_executor: Arc<dyn QueryExecutor>,
    subscription_manager: Arc<SubscriptionManager>,
    stats: Arc<JwpServerStats>,
    shutdown: Arc<Notify>,
    connection_semaphore: Arc<Semaphore>,
    energy_snapshot: Arc<std::sync::RwLock<joule_db_energy::EnergySnapshot>>,
    ledger: Option<Arc<JwpLedgerContext>>,
}

impl JwpServer {
    pub fn new(
        config: JwpServerConfig,
        query_executor: Arc<dyn QueryExecutor>,
        subscription_manager: Arc<SubscriptionManager>,
        energy_snapshot: Arc<std::sync::RwLock<joule_db_energy::EnergySnapshot>>,
    ) -> Self {
        let connection_semaphore = Arc::new(Semaphore::new(config.max_connections));
        Self {
            config,
            query_executor,
            subscription_manager,
            stats: Arc::new(JwpServerStats::default()),
            shutdown: Arc::new(Notify::new()),
            connection_semaphore,
            energy_snapshot,
            ledger: None,
        }
    }

    /// Attach ledger context for LEDGER command dispatch.
    pub fn with_ledger(
        mut self,
        collector: Arc<ReceiptCollector>,
        store: Arc<RwLock<ReceiptStore>>,
    ) -> Self {
        self.ledger = Some(Arc::new(JwpLedgerContext { collector, store }));
        self
    }

    pub fn stats(&self) -> &JwpServerStats {
        &self.stats
    }

    pub fn shutdown(&self) {
        self.shutdown.notify_one();
    }

    pub async fn run(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let listener = TcpListener::bind(&self.config.bind_addr).await?;
        tracing::info!("JWP server listening on {}", self.config.bind_addr);

        loop {
            tokio::select! {
                accept = listener.accept() => {
                    let (stream, addr) = accept?;

                    // Apply TCP socket options
                    let _ = stream.set_nodelay(true);

                    // Back-pressure: acquire semaphore permit
                    let permit = match self.connection_semaphore.clone().try_acquire_owned() {
                        Ok(p) => p,
                        Err(_) => {
                            tracing::warn!("JWP max connections reached, rejecting {}", addr);
                            drop(stream);
                            continue;
                        }
                    };

                    self.stats.connections_accepted.fetch_add(1, Ordering::Relaxed);
                    self.stats.connections_active.fetch_add(1, Ordering::Relaxed);

                    let query_executor = self.query_executor.clone();
                    let subscription_manager = self.subscription_manager.clone();
                    let stats = self.stats.clone();
                    let energy_snapshot = self.energy_snapshot.clone();
                    let ledger = self.ledger.clone();

                    tokio::spawn(async move {
                        if let Err(e) = handle_connection(
                            stream,
                            query_executor,
                            subscription_manager,
                            stats.clone(),
                            energy_snapshot,
                            ledger,
                        ).await {
                            tracing::debug!("JWP connection {} closed: {}", addr, e);
                        }
                        stats.connections_active.fetch_sub(1, Ordering::Relaxed);
                        drop(permit);
                    });
                }
                _ = self.shutdown.notified() => {
                    tracing::info!("JWP server shutting down");
                    break;
                }
            }
        }

        Ok(())
    }
}

// ── Connection handler ───────────────────────────────────────────────

async fn handle_connection(
    stream: tokio::net::TcpStream,
    query_executor: Arc<dyn QueryExecutor>,
    subscription_manager: Arc<SubscriptionManager>,
    stats: Arc<JwpServerStats>,
    energy_snapshot: Arc<std::sync::RwLock<joule_db_energy::EnergySnapshot>>,
    ledger: Option<Arc<JwpLedgerContext>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut transport = TcpTransport::new(stream);
    let mut state_machine = ProtocolStateMachine::new();
    let mut cumulative_uwh: u64 = 0;
    let mut sequence: u32 = 0;
    let mut active_subscription: Option<(u64, mpsc::UnboundedReceiver<ChangeEvent>)> = None;

    loop {
        tokio::select! {
            incoming = transport.recv_frame() => {
                let frame = match incoming {
                    Ok(Some(f)) => f,
                    Ok(None) => break, // clean close
                    Err(e) => {
                        tracing::debug!("JWP recv error: {}", e);
                        break;
                    }
                };

                stats.frames_received.fetch_add(1, Ordering::Relaxed);

                // State machine transition
                if let Err(e) = state_machine.transition(frame.header.frame_type) {
                    send_error(&mut transport, &mut sequence, cumulative_uwh, "PROTOCOL_ERROR", &e.to_string(), &stats).await;
                    break;
                }

                match frame.header.frame_type {
                    FrameType::Handshake => {
                        dispatch_handshake(&mut transport, &mut sequence, cumulative_uwh, &stats).await?;
                    }
                    FrameType::Query => {
                        dispatch_query(
                            &mut transport,
                            &mut sequence,
                            &mut cumulative_uwh,
                            &frame,
                            &query_executor,
                            &subscription_manager,
                            &energy_snapshot,
                            &mut active_subscription,
                            &stats,
                            &ledger,
                        ).await;
                        // Reset state machine to Ready after query completes
                        // (dispatch_query sends Done or Error internally)
                        let _ = state_machine.transition(FrameType::Done);
                    }
                    FrameType::Heartbeat => {
                        sequence += 1;
                        let reply = JwpFrame::new(FrameType::Heartbeat, sequence, cumulative_uwh, vec![]);
                        transport.send_frame(reply).await?;
                        stats.frames_sent.fetch_add(1, Ordering::Relaxed);
                    }
                    FrameType::Cancel => {
                        // Drop active subscription
                        if let Some((sub_id, _rx)) = active_subscription.take() {
                            subscription_manager.unsubscribe(sub_id).await;
                        }
                    }
                    _ => {
                        // Negotiate, ProfileUpdate, EnergyGradient — log and ignore
                        tracing::debug!("Ignoring frame type {:?}", frame.header.frame_type);
                    }
                }
            }

            // Stream subscription change events as Result frames
            Some(event) = async {
                match active_subscription {
                    Some((_, ref mut rx)) => rx.recv().await,
                    None => std::future::pending().await,
                }
            } => {
                let payload = encode_change_event(&event);
                sequence += 1;
                let frame = JwpFrame::new(FrameType::Result, sequence, cumulative_uwh, payload);
                if transport.send_frame(frame).await.is_err() {
                    break;
                }
                stats.frames_sent.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    // Clean up active subscription on disconnect
    if let Some((sub_id, _rx)) = active_subscription.take() {
        subscription_manager.unsubscribe(sub_id).await;
    }

    Ok(())
}

// ── Handshake ────────────────────────────────────────────────────────

async fn dispatch_handshake(
    transport: &mut TcpTransport,
    sequence: &mut u32,
    cumulative_uwh: u64,
    stats: &JwpServerStats,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let server_hs = HandshakePayload {
        version: PROTOCOL_VERSION,
        capabilities: vec!["sql".to_string(), "subscribe".to_string()],
    };
    let payload = cbor_encode(&server_hs)?;
    *sequence += 1;
    let reply = JwpFrame::new(FrameType::Handshake, *sequence, cumulative_uwh, payload);
    transport.send_frame(reply).await?;
    stats.frames_sent.fetch_add(1, Ordering::Relaxed);
    Ok(())
}

// ── Query dispatch ───────────────────────────────────────────────────

async fn dispatch_query(
    transport: &mut TcpTransport,
    sequence: &mut u32,
    cumulative_uwh: &mut u64,
    frame: &JwpFrame,
    query_executor: &Arc<dyn QueryExecutor>,
    subscription_manager: &Arc<SubscriptionManager>,
    energy_snapshot: &Arc<std::sync::RwLock<joule_db_energy::EnergySnapshot>>,
    active_subscription: &mut Option<(u64, mpsc::UnboundedReceiver<ChangeEvent>)>,
    stats: &JwpServerStats,
    ledger: &Option<Arc<JwpLedgerContext>>,
) {
    // Decode the query payload
    let db_query: DbQueryPayload = match cbor_decode(&frame.payload) {
        Ok(q) => q,
        Err(e) => {
            send_error(
                transport,
                sequence,
                *cumulative_uwh,
                "DECODE_ERROR",
                &e.to_string(),
                stats,
            )
            .await;
            return;
        }
    };

    // Check for LEDGER prefix
    let sql_trimmed = db_query.sql.trim();
    if sql_trimmed.starts_with("LEDGER ") || sql_trimmed.starts_with("ledger ") {
        dispatch_ledger(
            transport,
            sequence,
            cumulative_uwh,
            sql_trimmed,
            ledger,
            stats,
        )
        .await;
        return;
    }

    // Check for SUBSCRIBE prefix
    if let Some(pattern) = parse_subscribe(sql_trimmed) {
        handle_subscribe(
            transport,
            sequence,
            cumulative_uwh,
            subscription_manager,
            active_subscription,
            &pattern,
            stats,
        )
        .await;
        return;
    }

    // Snapshot energy before execution
    let energy_before = read_cumulative_joules(energy_snapshot);
    let t_start = Instant::now();

    // Build QueryRequest from the JWP payload
    let request = QueryRequest {
        sql: db_query.sql,
        params: db_query.named.into_iter().collect(),
        args: db_query.args,
        explain: db_query.explain,
        limit: db_query.limit,
        session_id: db_query.session_id,
        query_timeout_ms: None,
        branch_id: None,
        tenant_id: None,
    };

    let result = query_executor.execute(&request);

    // Compute energy delta
    let energy_after = read_cumulative_joules(energy_snapshot);
    let energy_delta_joules = (energy_after - energy_before).max(0.0);
    let query_uwh = joules_to_uwh(energy_delta_joules);
    *cumulative_uwh = cumulative_uwh.saturating_add(query_uwh);
    let elapsed_ms = t_start.elapsed().as_millis() as u64;

    match result {
        Ok(response) => {
            let columns = response.columns.clone();
            let rows = response.rows;
            let row_count = rows.len() as u64;
            let affected = response.affected_rows.map(|n| n as u64);

            // Send Meta frame (column names)
            *sequence += 1;
            let meta_payload = match cbor_encode(&DbMetaPayload {
                columns,
                session_id: response.session_id.clone(),
            }) {
                Ok(p) => p,
                Err(_) => vec![],
            };
            let meta_frame =
                JwpFrame::new(FrameType::Meta, *sequence, *cumulative_uwh, meta_payload);
            if let Err(e) = transport.send_frame(meta_frame).await {
                tracing::debug!("Failed to send Meta frame: {}", e);
                return;
            }
            stats.frames_sent.fetch_add(1, Ordering::Relaxed);

            // Stream rows in batches
            let batch_size = 50;
            if rows.is_empty() {
                // Send one empty Result frame
                *sequence += 1;
                let result_payload =
                    cbor_encode(&DbResultPayload { rows: vec![] }).unwrap_or_default();
                let result_frame = JwpFrame::new(
                    FrameType::Result,
                    *sequence,
                    *cumulative_uwh,
                    result_payload,
                );
                let _ = transport.send_frame(result_frame).await;
                stats.frames_sent.fetch_add(1, Ordering::Relaxed);
            } else {
                for chunk in rows.chunks(batch_size) {
                    *sequence += 1;
                    let result_payload = cbor_encode(&DbResultPayload {
                        rows: chunk.to_vec(),
                    })
                    .unwrap_or_default();
                    let result_frame = JwpFrame::new(
                        FrameType::Result,
                        *sequence,
                        *cumulative_uwh,
                        result_payload,
                    );
                    if let Err(e) = transport.send_frame(result_frame).await {
                        tracing::debug!("Failed to send Result frame: {}", e);
                        return;
                    }
                    stats.frames_sent.fetch_add(1, Ordering::Relaxed);
                }
            }

            // Send Done frame with energy total
            *sequence += 1;
            let done_payload = cbor_encode(&DbDonePayload {
                row_count,
                affected_rows: affected,
                total_cost_uwh: query_uwh,
                elapsed_ms,
            })
            .unwrap_or_default();
            let done_frame =
                JwpFrame::new_final(FrameType::Done, *sequence, *cumulative_uwh, done_payload);
            let _ = transport.send_frame(done_frame).await;
            stats.frames_sent.fetch_add(1, Ordering::Relaxed);
            stats.queries_executed.fetch_add(1, Ordering::Relaxed);
        }
        Err(error) => {
            send_error(
                transport,
                sequence,
                *cumulative_uwh,
                &error.code,
                &error.message,
                stats,
            )
            .await;
        }
    }
}

// ── Subscribe ────────────────────────────────────────────────────────

/// Parse `SUBSCRIBE 'pattern'` or `SUBSCRIBE "pattern"` from SQL string.
fn parse_subscribe(sql: &str) -> Option<String> {
    let upper = sql.to_uppercase();
    if !upper.starts_with("SUBSCRIBE ") {
        return None;
    }
    let rest = sql["SUBSCRIBE ".len()..].trim();
    // Strip surrounding quotes
    if (rest.starts_with('\'') && rest.ends_with('\''))
        || (rest.starts_with('"') && rest.ends_with('"'))
    {
        Some(rest[1..rest.len() - 1].to_string())
    } else {
        Some(rest.to_string())
    }
}

async fn handle_subscribe(
    transport: &mut TcpTransport,
    sequence: &mut u32,
    cumulative_uwh: &mut u64,
    subscription_manager: &Arc<SubscriptionManager>,
    active_subscription: &mut Option<(u64, mpsc::UnboundedReceiver<ChangeEvent>)>,
    pattern: &str,
    stats: &JwpServerStats,
) {
    // Drop previous subscription if any
    if let Some((old_id, _rx)) = active_subscription.take() {
        subscription_manager.unsubscribe(old_id).await;
    }

    let (sub_id, receiver) = match subscription_manager.subscribe(pattern).await {
        Ok(pair) => pair,
        Err(e) => {
            tracing::warn!("Subscribe failed: {}", e);
            return;
        }
    };

    // Send Meta with subscription schema
    *sequence += 1;
    let meta = cbor_encode(&DbMetaPayload {
        columns: vec!["subscription_id".to_string(), "pattern".to_string()],
        session_id: None,
    })
    .unwrap_or_default();
    let _ = transport
        .send_frame(JwpFrame::new(
            FrameType::Meta,
            *sequence,
            *cumulative_uwh,
            meta,
        ))
        .await;
    stats.frames_sent.fetch_add(1, Ordering::Relaxed);

    // Send one Result row with the subscription ID
    *sequence += 1;
    let result = cbor_encode(&DbResultPayload {
        rows: vec![vec![
            serde_json::Value::Number(sub_id.into()),
            serde_json::Value::String(pattern.to_string()),
        ]],
    })
    .unwrap_or_default();
    let _ = transport
        .send_frame(JwpFrame::new(
            FrameType::Result,
            *sequence,
            *cumulative_uwh,
            result,
        ))
        .await;
    stats.frames_sent.fetch_add(1, Ordering::Relaxed);

    // Store the receiver — future ChangeEvents will stream as Result frames
    *active_subscription = Some((sub_id, receiver));
}

// ── Ledger dispatch ─────────────────────────────────────────────────

/// CBOR payload for a Cloud Insights ledger receipt submitted via JWP.
#[derive(Debug, Serialize, Deserialize)]
struct CloudInsightsReceipt {
    receipt_id: String,
    tenant_id: String,
    resource_id: String,
    region: String,
    energy_kwh: f64,
    carbon_gco2e: f64,
    interval_start: String,
    interval_end: String,
    timestamp: String,
}

/// Handle LEDGER commands: SUBMIT, VERIFY, STATS.
async fn dispatch_ledger(
    transport: &mut TcpTransport,
    sequence: &mut u32,
    cumulative_uwh: &mut u64,
    sql: &str,
    ledger: &Option<Arc<JwpLedgerContext>>,
    stats: &JwpServerStats,
) {
    let ledger = match ledger {
        Some(l) => l,
        None => {
            send_error(
                transport,
                sequence,
                *cumulative_uwh,
                "LEDGER_UNAVAILABLE",
                "Ledger not enabled on this server",
                stats,
            )
            .await;
            return;
        }
    };

    // Strip the "LEDGER " prefix (case-insensitive)
    let rest = sql[7..].trim();
    let t_start = Instant::now();

    if rest.starts_with("SUBMIT ") || rest.starts_with("submit ") {
        let json_str = rest[7..].trim();
        let receipt: CloudInsightsReceipt = match serde_json::from_str(json_str) {
            Ok(r) => r,
            Err(e) => {
                send_error(
                    transport,
                    sequence,
                    *cumulative_uwh,
                    "LEDGER_PARSE_ERROR",
                    &format!("Invalid receipt JSON: {e}"),
                    stats,
                )
                .await;
                return;
            }
        };

        // Convert kWh to joules (1 kWh = 3,600,000 J)
        let energy_joules = receipt.energy_kwh * 3_600_000.0;

        // Parse timestamps
        let ts_start = chrono::DateTime::parse_from_rfc3339(&receipt.interval_start)
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .unwrap_or_else(|_| chrono::Utc::now());
        let ts_end = chrono::DateTime::parse_from_rfc3339(&receipt.interval_end)
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .unwrap_or_else(|_| chrono::Utc::now());

        // Record via the collector (builds receipt, batches, Merkle tree)
        match ledger.collector.record(
            &receipt.resource_id,
            &receipt.tenant_id,
            Some(&receipt.region),
            energy_joules,
            "cloud",
            "cloud-estimate",
            ts_start,
            ts_end,
        ) {
            Ok(db_receipt) => {
                let elapsed_ms = t_start.elapsed().as_millis() as u64;
                let query_uwh = joules_to_uwh(0.0001); // small overhead
                *cumulative_uwh = cumulative_uwh.saturating_add(query_uwh);

                // Meta frame
                *sequence += 1;
                let meta = cbor_encode(&DbMetaPayload {
                    columns: vec!["receipt_id".into(), "status".into()],
                    session_id: None,
                })
                .unwrap_or_default();
                let _ = transport
                    .send_frame(JwpFrame::new(
                        FrameType::Meta,
                        *sequence,
                        *cumulative_uwh,
                        meta,
                    ))
                    .await;
                stats.frames_sent.fetch_add(1, Ordering::Relaxed);

                // Result frame
                *sequence += 1;
                let result = cbor_encode(&DbResultPayload {
                    rows: vec![vec![
                        serde_json::Value::String(db_receipt.receipt_id),
                        serde_json::Value::String("committed".into()),
                    ]],
                })
                .unwrap_or_default();
                let _ = transport
                    .send_frame(JwpFrame::new(
                        FrameType::Result,
                        *sequence,
                        *cumulative_uwh,
                        result,
                    ))
                    .await;
                stats.frames_sent.fetch_add(1, Ordering::Relaxed);

                // Done frame
                *sequence += 1;
                let done = cbor_encode(&DbDonePayload {
                    row_count: 1,
                    affected_rows: Some(1),
                    total_cost_uwh: query_uwh,
                    elapsed_ms,
                })
                .unwrap_or_default();
                let _ = transport
                    .send_frame(JwpFrame::new_final(
                        FrameType::Done,
                        *sequence,
                        *cumulative_uwh,
                        done,
                    ))
                    .await;
                stats.frames_sent.fetch_add(1, Ordering::Relaxed);
                stats.queries_executed.fetch_add(1, Ordering::Relaxed);
            }
            Err(e) => {
                send_error(
                    transport,
                    sequence,
                    *cumulative_uwh,
                    "LEDGER_SUBMIT_ERROR",
                    &e.to_string(),
                    stats,
                )
                .await;
            }
        }
    } else if rest.starts_with("VERIFY ") || rest.starts_with("verify ") {
        let receipt_id = rest[7..].trim().trim_matches(|c| c == '\'' || c == '"');

        let store = ledger.store.read().await;
        let elapsed_ms = t_start.elapsed().as_millis() as u64;
        let query_uwh = joules_to_uwh(0.00005);
        *cumulative_uwh = cumulative_uwh.saturating_add(query_uwh);

        if let Some((receipt, batch_id, leaf_index)) = store.receipts.get(receipt_id) {
            let tree = store.trees.get(batch_id);
            let proof = tree.and_then(|t| t.proof(*leaf_index));
            let root = tree.map(|t| hex::encode(t.root())).unwrap_or_default();

            // Meta frame
            *sequence += 1;
            let meta = cbor_encode(&DbMetaPayload {
                columns: vec![
                    "valid".into(),
                    "receipt_id".into(),
                    "merkle_root".into(),
                    "leaf_hash".into(),
                    "leaf_index".into(),
                    "tree_size".into(),
                ],
                session_id: None,
            })
            .unwrap_or_default();
            let _ = transport
                .send_frame(JwpFrame::new(
                    FrameType::Meta,
                    *sequence,
                    *cumulative_uwh,
                    meta,
                ))
                .await;
            stats.frames_sent.fetch_add(1, Ordering::Relaxed);

            // Result frame
            *sequence += 1;
            let leaf_hash = hex::encode(receipt.content_hash());
            let tree_size = tree.map(|t| t.leaf_count()).unwrap_or(0);
            let result = cbor_encode(&DbResultPayload {
                rows: vec![vec![
                    serde_json::Value::Bool(proof.is_some()),
                    serde_json::Value::String(receipt_id.to_string()),
                    serde_json::Value::String(root),
                    serde_json::Value::String(leaf_hash),
                    serde_json::json!(*leaf_index),
                    serde_json::json!(tree_size),
                ]],
            })
            .unwrap_or_default();
            let _ = transport
                .send_frame(JwpFrame::new(
                    FrameType::Result,
                    *sequence,
                    *cumulative_uwh,
                    result,
                ))
                .await;
            stats.frames_sent.fetch_add(1, Ordering::Relaxed);

            // Done frame
            *sequence += 1;
            let done = cbor_encode(&DbDonePayload {
                row_count: 1,
                affected_rows: None,
                total_cost_uwh: query_uwh,
                elapsed_ms,
            })
            .unwrap_or_default();
            let _ = transport
                .send_frame(JwpFrame::new_final(
                    FrameType::Done,
                    *sequence,
                    *cumulative_uwh,
                    done,
                ))
                .await;
            stats.frames_sent.fetch_add(1, Ordering::Relaxed);
            stats.queries_executed.fetch_add(1, Ordering::Relaxed);
        } else {
            send_error(
                transport,
                sequence,
                *cumulative_uwh,
                "LEDGER_NOT_FOUND",
                &format!("Receipt not found: {receipt_id}"),
                stats,
            )
            .await;
        }
    } else if rest.eq_ignore_ascii_case("STATS") {
        let store = ledger.store.read().await;
        let elapsed_ms = t_start.elapsed().as_millis() as u64;
        let query_uwh = joules_to_uwh(0.00005);
        *cumulative_uwh = cumulative_uwh.saturating_add(query_uwh);

        let total_receipts = store.receipts.len() as u64;
        let total_batches = store.commitments.len() as u64;
        let total_energy_joules: f64 = store
            .receipts
            .values()
            .map(|(r, _, _)| r.energy_joules_total)
            .sum();
        let total_carbon_kg: f64 = store.receipts.values().map(|(r, _, _)| r.kg_co2e).sum();

        // Meta frame
        *sequence += 1;
        let meta = cbor_encode(&DbMetaPayload {
            columns: vec![
                "total_receipts".into(),
                "total_batches".into(),
                "total_energy_joules".into(),
                "total_carbon_kg_co2e".into(),
            ],
            session_id: None,
        })
        .unwrap_or_default();
        let _ = transport
            .send_frame(JwpFrame::new(
                FrameType::Meta,
                *sequence,
                *cumulative_uwh,
                meta,
            ))
            .await;
        stats.frames_sent.fetch_add(1, Ordering::Relaxed);

        // Result frame
        *sequence += 1;
        let result = cbor_encode(&DbResultPayload {
            rows: vec![vec![
                serde_json::json!(total_receipts),
                serde_json::json!(total_batches),
                serde_json::json!(total_energy_joules),
                serde_json::json!(total_carbon_kg),
            ]],
        })
        .unwrap_or_default();
        let _ = transport
            .send_frame(JwpFrame::new(
                FrameType::Result,
                *sequence,
                *cumulative_uwh,
                result,
            ))
            .await;
        stats.frames_sent.fetch_add(1, Ordering::Relaxed);

        // Done frame
        *sequence += 1;
        let done = cbor_encode(&DbDonePayload {
            row_count: 1,
            affected_rows: None,
            total_cost_uwh: query_uwh,
            elapsed_ms,
        })
        .unwrap_or_default();
        let _ = transport
            .send_frame(JwpFrame::new_final(
                FrameType::Done,
                *sequence,
                *cumulative_uwh,
                done,
            ))
            .await;
        stats.frames_sent.fetch_add(1, Ordering::Relaxed);
        stats.queries_executed.fetch_add(1, Ordering::Relaxed);
    } else {
        send_error(
            transport,
            sequence,
            *cumulative_uwh,
            "LEDGER_UNKNOWN_CMD",
            &format!("Unknown LEDGER command: {rest}"),
            stats,
        )
        .await;
    }
}

// ── Helpers ──────────────────────────────────────────────────────────

/// Convert joules (f64) to micro-watt-hours (u64).
fn joules_to_uwh(joules: f64) -> u64 {
    if joules <= 0.0 {
        return 0;
    }
    (joules * 1_000_000.0 / 3_600.0).round() as u64
}

/// Read cumulative joules from the shared energy snapshot.
fn read_cumulative_joules(
    snapshot: &Arc<std::sync::RwLock<joule_db_energy::EnergySnapshot>>,
) -> f64 {
    snapshot.read().map(|s| s.cumulative_joules).unwrap_or(0.0)
}

/// Send a JWP Error frame.
async fn send_error(
    transport: &mut TcpTransport,
    sequence: &mut u32,
    energy_uwh: u64,
    code: &str,
    message: &str,
    stats: &JwpServerStats,
) {
    *sequence += 1;
    let payload = cbor_encode(&ErrorPayload {
        code: code.to_string(),
        message: message.to_string(),
    })
    .unwrap_or_default();
    let frame = JwpFrame::new(FrameType::Error, *sequence, energy_uwh, payload);
    let _ = transport.send_frame(frame).await;
    stats.frames_sent.fetch_add(1, Ordering::Relaxed);
    stats.errors.fetch_add(1, Ordering::Relaxed);
}

/// Encode a ChangeEvent as a CBOR byte vector for a Result frame payload.
fn encode_change_event(event: &ChangeEvent) -> Vec<u8> {
    let operation = match event.operation {
        ChangeOperation::Insert => "insert",
        ChangeOperation::Update => "update",
        ChangeOperation::Delete => "delete",
    };
    let value_json = event
        .value
        .as_ref()
        .and_then(|v| serde_json::from_slice::<serde_json::Value>(v).ok())
        .unwrap_or(serde_json::Value::Null);
    let row = vec![
        serde_json::Value::Number(event.id.into()),
        serde_json::Value::String(operation.to_string()),
        serde_json::Value::String(event.key.clone()),
        value_json,
        serde_json::Value::Number(event.timestamp.into()),
    ];
    cbor_encode(&DbResultPayload { rows: vec![row] }).unwrap_or_default()
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::{QueryErrorResponse, QueryResponse};
    use std::collections::HashMap;

    /// Mock query executor that echoes SQL back as a single-row result.
    struct EchoExecutor;

    impl QueryExecutor for EchoExecutor {
        fn execute(&self, request: &QueryRequest) -> Result<QueryResponse, QueryErrorResponse> {
            if request.sql.trim().is_empty() {
                return Err(QueryErrorResponse {
                    code: "SYNTAX_ERROR".to_string(),
                    message: "Empty query".to_string(),
                    line: None,
                    column: None,
                });
            }
            Ok(QueryResponse {
                columns: vec!["echo".to_string()],
                rows: vec![vec![serde_json::Value::String(request.sql.clone())]],
                affected_rows: None,
                execution_time_ms: 0,
                truncated: false,
                warnings: vec![],
                energy_joules: Some(0.0036), // 1 µWh
                power_watts: Some(5.0),
                device_target: Some("cpu".to_string()),
                algorithm_type: None,
                session_id: None,
                #[cfg(feature = "viz")]
                viz_hint: None,
            })
        }
    }

    /// Spawn a JWP test server and return the address.
    async fn spawn_test_server() -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap().to_string();

        let executor: Arc<dyn QueryExecutor> = Arc::new(EchoExecutor);
        let sub_mgr = Arc::new(SubscriptionManager::new());
        let energy = Arc::new(std::sync::RwLock::new(
            joule_db_energy::EnergySnapshot::default(),
        ));

        tokio::spawn(async move {
            loop {
                let (stream, _) = listener.accept().await.unwrap();
                let exec = executor.clone();
                let subs = sub_mgr.clone();
                let snap = energy.clone();
                tokio::spawn(async move {
                    let _ = handle_connection(
                        stream,
                        exec,
                        subs,
                        Arc::new(JwpServerStats::default()),
                        snap,
                        None,
                    )
                    .await;
                });
            }
        });

        addr
    }

    #[tokio::test]
    async fn test_handshake_roundtrip() {
        let addr = spawn_test_server().await;
        let stream = tokio::net::TcpStream::connect(&addr).await.unwrap();
        let mut transport = TcpTransport::new(stream);

        // Send handshake
        let hs = HandshakePayload {
            version: PROTOCOL_VERSION,
            capabilities: vec!["sql".to_string()],
        };
        let payload = cbor_encode(&hs).unwrap();
        let frame = JwpFrame::new(FrameType::Handshake, 1, 0, payload);
        transport.send_frame(frame).await.unwrap();

        // Receive handshake ack
        let reply = transport.recv_frame().await.unwrap().unwrap();
        assert_eq!(reply.header.frame_type, FrameType::Handshake);
        let server_hs: HandshakePayload = cbor_decode(&reply.payload).unwrap();
        assert_eq!(server_hs.version, PROTOCOL_VERSION);
        assert!(server_hs.capabilities.contains(&"sql".to_string()));
        assert!(server_hs.capabilities.contains(&"subscribe".to_string()));
    }

    #[tokio::test]
    async fn test_query_meta_result_done() {
        let addr = spawn_test_server().await;
        let stream = tokio::net::TcpStream::connect(&addr).await.unwrap();
        let mut transport = TcpTransport::new(stream);

        // Handshake first
        let hs = HandshakePayload {
            version: PROTOCOL_VERSION,
            capabilities: vec![],
        };
        transport
            .send_frame(JwpFrame::new(
                FrameType::Handshake,
                1,
                0,
                cbor_encode(&hs).unwrap(),
            ))
            .await
            .unwrap();
        let _ = transport.recv_frame().await.unwrap().unwrap();

        // Send query
        let query = DbQueryPayload {
            sql: "SELECT 1".to_string(),
            args: vec![],
            named: BTreeMap::new(),
            session_id: None,
            limit: None,
            explain: false,
        };
        transport
            .send_frame(JwpFrame::new(
                FrameType::Query,
                2,
                0,
                cbor_encode(&query).unwrap(),
            ))
            .await
            .unwrap();

        // Meta frame
        let meta = transport.recv_frame().await.unwrap().unwrap();
        assert_eq!(meta.header.frame_type, FrameType::Meta);
        let meta_payload: DbMetaPayload = cbor_decode(&meta.payload).unwrap();
        assert_eq!(meta_payload.columns, vec!["echo"]);

        // Result frame
        let result = transport.recv_frame().await.unwrap().unwrap();
        assert_eq!(result.header.frame_type, FrameType::Result);
        let result_payload: DbResultPayload = cbor_decode(&result.payload).unwrap();
        assert_eq!(result_payload.rows.len(), 1);
        assert_eq!(
            result_payload.rows[0][0],
            serde_json::Value::String("SELECT 1".to_string())
        );

        // Done frame
        let done = transport.recv_frame().await.unwrap().unwrap();
        assert_eq!(done.header.frame_type, FrameType::Done);
        let done_payload: DbDonePayload = cbor_decode(&done.payload).unwrap();
        assert_eq!(done_payload.row_count, 1);
    }

    #[tokio::test]
    async fn test_heartbeat_echo() {
        let addr = spawn_test_server().await;
        let stream = tokio::net::TcpStream::connect(&addr).await.unwrap();
        let mut transport = TcpTransport::new(stream);

        // Handshake
        let hs = HandshakePayload {
            version: PROTOCOL_VERSION,
            capabilities: vec![],
        };
        transport
            .send_frame(JwpFrame::new(
                FrameType::Handshake,
                1,
                0,
                cbor_encode(&hs).unwrap(),
            ))
            .await
            .unwrap();
        let _ = transport.recv_frame().await.unwrap().unwrap();

        // Send heartbeat
        transport
            .send_frame(JwpFrame::new(FrameType::Heartbeat, 2, 0, vec![]))
            .await
            .unwrap();

        // Receive heartbeat echo
        let reply = transport.recv_frame().await.unwrap().unwrap();
        assert_eq!(reply.header.frame_type, FrameType::Heartbeat);
    }

    #[tokio::test]
    async fn test_error_on_empty_query() {
        let addr = spawn_test_server().await;
        let stream = tokio::net::TcpStream::connect(&addr).await.unwrap();
        let mut transport = TcpTransport::new(stream);

        // Handshake
        let hs = HandshakePayload {
            version: PROTOCOL_VERSION,
            capabilities: vec![],
        };
        transport
            .send_frame(JwpFrame::new(
                FrameType::Handshake,
                1,
                0,
                cbor_encode(&hs).unwrap(),
            ))
            .await
            .unwrap();
        let _ = transport.recv_frame().await.unwrap().unwrap();

        // Send empty query
        let query = DbQueryPayload {
            sql: "  ".to_string(),
            args: vec![],
            named: BTreeMap::new(),
            session_id: None,
            limit: None,
            explain: false,
        };
        transport
            .send_frame(JwpFrame::new(
                FrameType::Query,
                2,
                0,
                cbor_encode(&query).unwrap(),
            ))
            .await
            .unwrap();

        // Should get Error frame
        let reply = transport.recv_frame().await.unwrap().unwrap();
        assert_eq!(reply.header.frame_type, FrameType::Error);
        let error: ErrorPayload = cbor_decode(&reply.payload).unwrap();
        assert_eq!(error.code, "SYNTAX_ERROR");
    }

    #[test]
    fn test_energy_conversion() {
        assert_eq!(joules_to_uwh(0.0), 0);
        assert_eq!(joules_to_uwh(-1.0), 0);
        assert_eq!(joules_to_uwh(0.0036), 1); // 1 µWh = 0.0036 J
        assert_eq!(joules_to_uwh(3.6), 1_000); // 3.6 J = 1 mWh = 1000 µWh
        assert_eq!(joules_to_uwh(3_600.0), 1_000_000); // 3600 J = 1 Wh = 1,000,000 µWh
    }

    #[test]
    fn test_parse_subscribe() {
        assert_eq!(
            parse_subscribe("SUBSCRIBE 'users:*'"),
            Some("users:*".to_string())
        );
        assert_eq!(
            parse_subscribe("subscribe 'data'"),
            Some("data".to_string())
        );
        assert_eq!(
            parse_subscribe("SUBSCRIBE \"orders:*\""),
            Some("orders:*".to_string())
        );
        assert_eq!(
            parse_subscribe("SUBSCRIBE raw_pattern"),
            Some("raw_pattern".to_string())
        );
        assert_eq!(parse_subscribe("SELECT 1"), None);
    }
}
