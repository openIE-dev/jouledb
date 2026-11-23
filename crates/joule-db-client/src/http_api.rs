//! Zero-Dependency Async HTTP/1.1 SDK (`HttpFabricClient`) for `JouleSynapse`, `JouleWard`,
//! OpenAI (`/v1/*`), Anthropic (`/v1/messages`), and `JouleFlow` Disintermediation APIs.

use crate::error::{ClientError, Result};
use serde_json::{Value as JsonValue, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// Lightweight async HTTP/1.1 client for JouleDB's living-weights, confinement, 6-rail `nJ`
/// ledger, OpenAI/Anthropic model endpoints, and durable `JouleFlow` workflows.
#[derive(Debug, Clone)]
pub struct HttpFabricClient {
    host: String,
    port: u16,
    bearer_token: Option<String>,
}

impl HttpFabricClient {
    /// Parse an `http://host:port` URL into an [`HttpFabricClient`].
    pub fn from_url(url: &str) -> Result<Self> {
        let stripped = url
            .trim()
            .strip_prefix("http://")
            .ok_or_else(|| ClientError::protocol("URL must start with http://"))?;
        let host_port = stripped.split('/').next().unwrap_or(stripped);
        let (host, port) = match host_port.rsplit_once(':') {
            Some((h, p_str)) => {
                let p: u16 = p_str
                    .parse()
                    .map_err(|_| ClientError::protocol(format!("invalid port in '{url}'")))?;
                (h.to_string(), p)
            }
            None => (host_port.to_string(), 8080),
        };
        if host.is_empty() {
            return Err(ClientError::protocol("empty host in URL"));
        }
        Ok(Self {
            host,
            port,
            bearer_token: None,
        })
    }

    /// Attach an optional Bearer authentication token.
    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.bearer_token = Some(token.into());
        self
    }

    /// Low-level JSON HTTP/1.1 request (`GET` or `POST`).
    pub async fn request_json(
        &self,
        method: &str,
        path: &str,
        body: Option<&JsonValue>,
    ) -> Result<JsonValue> {
        let addr = format!("{}:{}", self.host, self.port);
        let mut stream = TcpStream::connect(&addr)
            .await
            .map_err(ClientError::connection_failed)?;

        let payload = match body {
            Some(v) => v.to_string().into_bytes(),
            None => Vec::new(),
        };
        let auth_hdr = match &self.bearer_token {
            Some(t) => format!("Authorization: Bearer {t}\r\n"),
            None => String::new(),
        };
        let content_hdr = if body.is_some() {
            format!(
                "Content-Type: application/json\r\nContent-Length: {}\r\n",
                payload.len()
            )
        } else {
            String::new()
        };
        let req_head = format!(
            "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n{auth_hdr}{content_hdr}\r\n"
        );
        stream.write_all(req_head.as_bytes()).await?;
        if !payload.is_empty() {
            stream.write_all(&payload).await?;
        }
        stream.flush().await?;

        let mut raw = Vec::new();
        stream.read_to_end(&mut raw).await?;

        let split_pos = raw
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .ok_or_else(|| ClientError::invalid_response("malformed HTTP response headers"))?;
        let head_str = String::from_utf8_lossy(&raw[..split_pos]);
        let body_bytes = &raw[split_pos + 4..];

        let first_line = head_str.lines().next().unwrap_or("");
        let status_code: u16 = first_line
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| ClientError::invalid_response("missing HTTP status code"))?;

        let is_chunked = head_str.lines().any(|l| {
            l.to_ascii_lowercase()
                .contains("transfer-encoding: chunked")
        });
        let decoded_body = if is_chunked {
            decode_chunked_body(body_bytes)?
        } else {
            body_bytes.to_vec()
        };

        let parsed: JsonValue = serde_json::from_slice(&decoded_body)
            .map_err(|e| ClientError::invalid_response(format!("invalid JSON body: {e}")))?;

        if status_code >= 400 {
            let msg = parsed
                .get("error")
                .map(|e| e.to_string())
                .unwrap_or_else(|| parsed.to_string());
            return Err(ClientError::server(status_code.to_string(), msg));
        }
        Ok(parsed)
    }

    /// Inspect `JouleSynapse` living tensors, GPU kernel, `Titans`, `Engram`, and loaded vocabs.
    pub async fn synapse_status(&self) -> Result<JsonValue> {
        self.request_json("GET", "/api/v1/synapse/status", None)
            .await
    }

    /// Inspect the high-resolution 6-rail `nJ` `JouleWard` fabric energy ledger + live hardware telemetry.
    pub async fn ward_ledger(&self) -> Result<JsonValue> {
        self.request_json("GET", "/api/v1/ward/ledger", None).await
    }

    /// Mint a root `HMAC-SHA256` `ThermoCap` capability macaroon.
    pub async fn ward_mint_token(
        &self,
        root_token_id: &str,
        subject: &str,
        caveats: JsonValue,
    ) -> Result<JsonValue> {
        let body = json!({
            "root_token_id": root_token_id,
            "subject": subject,
            "caveats": caveats,
        });
        self.request_json("POST", "/api/v1/ward/tokens/mint", Some(&body))
            .await
    }

    /// Execute a confined command under `JouleWard` with Merkle memoization and optional cross-fabric placement.
    pub async fn ward_exec(&self, exec_req: &JsonValue) -> Result<JsonValue> {
        self.request_json("POST", "/api/v1/ward/exec", Some(exec_req))
            .await
    }

    /// Generate 384-D semantic embeddings directly from living sub-1-bit `JouleSynapse` Transformer layers (`POST /v1/embeddings`).
    pub async fn embed(&self, model: &str, input: JsonValue) -> Result<JsonValue> {
        let body = json!({
            "model": model,
            "input": input,
        });
        self.request_json("POST", "/v1/embeddings", Some(&body))
            .await
    }

    /// Run OpenAI-compatible 4-tier thermodynamic chat completion (`POST /v1/chat/completions`).
    pub async fn chat(&self, model: &str, user_prompt: &str) -> Result<JsonValue> {
        let body = json!({
            "model": model,
            "messages": [{ "role": "user", "content": user_prompt }],
        });
        self.request_json("POST", "/v1/chat/completions", Some(&body))
            .await
    }

    /// Run Anthropic-compatible `/v1/messages` with inline `45 nJ` `JoulePulse` guardrail firewall.
    pub async fn messages(
        &self,
        model: &str,
        user_prompt: &str,
        max_tokens: usize,
    ) -> Result<JsonValue> {
        let body = json!({
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{ "role": "user", "content": user_prompt }],
        });
        self.request_json("POST", "/v1/messages", Some(&body)).await
    }

    /// Execute a crash-idempotent durable `JouleFlow` DAG (`POST /api/v1/flow/execute`).
    pub async fn flow_execute(&self, flow_req: &JsonValue) -> Result<JsonValue> {
        self.request_json("POST", "/api/v1/flow/execute", Some(flow_req))
            .await
    }

    /// Execute autoregressive causal token generation directly over sub-1-bit `LittleBit-2` bitplanes (`POST /api/v1/synapse/decoder/generate`).
    pub async fn decoder_generate(
        &self,
        model_id: &str,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
    ) -> Result<JsonValue> {
        let body = json!({
            "model_id": model_id,
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        });
        self.request_json("POST", "/api/v1/synapse/decoder/generate", Some(&body))
            .await
    }

    /// Run schema-constrained JSON chat completion (`response_format: {"type": "json_schema", ...}`).
    pub async fn chat_structured(
        &self,
        model: &str,
        user_prompt: &str,
        json_schema: JsonValue,
    ) -> Result<JsonValue> {
        let body = json!({
            "model": model,
            "messages": [{ "role": "user", "content": user_prompt }],
            "response_format": {
                "type": "json_schema",
                "json_schema": json_schema,
            }
        });
        self.request_json("POST", "/v1/chat/completions", Some(&body))
            .await
    }

    /// Run tool-constrained chat completion (`tools` -> `tool_calls`).
    pub async fn chat_tools(
        &self,
        model: &str,
        user_prompt: &str,
        tools: JsonValue,
    ) -> Result<JsonValue> {
        let body = json!({
            "model": model,
            "messages": [{ "role": "user", "content": user_prompt }],
            "tools": tools,
        });
        self.request_json("POST", "/v1/chat/completions", Some(&body))
            .await
    }

    /// Stream SSE `chat.completion.chunk` events (`stream: true`), returning all parsed JSON chunks before `[DONE]`.
    pub async fn chat_stream(&self, model: &str, user_prompt: &str) -> Result<Vec<JsonValue>> {
        let addr = format!("{}:{}", self.host, self.port);
        let mut stream = TcpStream::connect(&addr)
            .await
            .map_err(ClientError::connection_failed)?;
        let payload = json!({
            "model": model,
            "stream": true,
            "messages": [{ "role": "user", "content": user_prompt }],
        })
        .to_string()
        .into_bytes();
        let auth_hdr = match &self.bearer_token {
            Some(t) => format!("Authorization: Bearer {t}\r\n"),
            None => String::new(),
        };
        let req_head = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n{auth_hdr}Content-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            payload.len()
        );
        stream.write_all(req_head.as_bytes()).await?;
        stream.write_all(&payload).await?;
        stream.flush().await?;

        let mut raw = Vec::new();
        stream.read_to_end(&mut raw).await?;
        let split_pos = raw
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .ok_or_else(|| ClientError::invalid_response("malformed SSE HTTP headers"))?;
        let head_str = String::from_utf8_lossy(&raw[..split_pos]);
        let body_bytes = &raw[split_pos + 4..];
        let is_chunked = head_str.lines().any(|l| {
            l.to_ascii_lowercase()
                .contains("transfer-encoding: chunked")
        });
        let decoded = if is_chunked {
            decode_chunked_body(body_bytes)?
        } else {
            body_bytes.to_vec()
        };
        let text = String::from_utf8_lossy(&decoded);
        let mut events = Vec::new();
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("data: ") {
                let trimmed = rest.trim();
                if trimmed == "[DONE]" || trimmed.is_empty() {
                    continue;
                }
                if let Ok(v) = serde_json::from_str::<JsonValue>(trimmed) {
                    events.push(v);
                }
            }
        }
        Ok(events)
    }

    /// Register an external Streamable-HTTP MCP server (`POST /api/v1/mcp/host/servers`).
    pub async fn mcp_register_server(
        &self,
        server_id: &str,
        endpoint_url: &str,
        discover_live: bool,
    ) -> Result<JsonValue> {
        let body = json!({
            "server_id": server_id,
            "endpoint_url": endpoint_url,
            "discover_live": discover_live,
        });
        self.request_json("POST", "/api/v1/mcp/host/servers", Some(&body))
            .await
    }

    /// List all native + remote MCP tools registered on JouleDB's MCP Host (`GET /api/v1/mcp/host/tools`).
    pub async fn mcp_list_tools(&self) -> Result<JsonValue> {
        self.request_json("GET", "/api/v1/mcp/host/tools", None)
            .await
    }

    /// Invoke a native or remote MCP tool under `JoulePulse` + `JouleWard` (`POST /api/v1/mcp/host/call`).
    pub async fn mcp_call_tool(&self, tool_name: &str, arguments: JsonValue) -> Result<JsonValue> {
        let body = json!({
            "tool_name": tool_name,
            "arguments": arguments,
        });
        self.request_json("POST", "/api/v1/mcp/host/call", Some(&body))
            .await
    }

    /// Execute an autonomous multi-turn SQL/MCP/`Titans` agent loop (`POST /api/v1/agent/run`).
    pub async fn agent_run(&self, goal: &str, max_turns: usize) -> Result<JsonValue> {
        let body = json!({
            "goal": goal,
            "max_turns": max_turns,
            "commit_to_titans": true,
        });
        self.request_json("POST", "/api/v1/agent/run", Some(&body))
            .await
    }

    /// Encode an image (`image_base64`) via sub-1-bit `SigLIP` / `ViT` (`POST /api/v1/synapse/vision/encode`).
    pub async fn vision_encode(
        &self,
        model_id: Option<&str>,
        image_base64: &str,
    ) -> Result<JsonValue> {
        let body = json!({
            "model_id": model_id,
            "image_base64": image_base64,
        });
        self.request_json("POST", "/api/v1/synapse/vision/encode", Some(&body))
            .await
    }

    /// Transcribe base64 `.wav` audio via OpenAI-compatible `POST /v1/audio/transcriptions`.
    pub async fn audio_transcribe(
        &self,
        model_id: Option<&str>,
        audio_base64: &str,
    ) -> Result<JsonValue> {
        let body = json!({
            "model": model_id.unwrap_or("whisper-1"),
            "audio_base64": audio_base64,
        });
        self.request_json("POST", "/v1/audio/transcriptions", Some(&body))
            .await
    }

    /// Run self-speculative `LayerSkip` + Rank-8 draft-and-verify causal generation (`speculative_draft_k`).
    pub async fn decoder_generate_speculative(
        &self,
        model_id: &str,
        prompt: &str,
        max_new_tokens: usize,
        speculative_draft_k: usize,
    ) -> Result<JsonValue> {
        let body = json!({
            "model_id": model_id,
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
            "speculative_draft_k": speculative_draft_k,
        });
        self.request_json("POST", "/api/v1/synapse/decoder/generate", Some(&body))
            .await
    }
}

fn decode_chunked_body(raw: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let mut cursor = 0usize;
    while cursor < raw.len() {
        let rel_end = raw[cursor..]
            .windows(2)
            .position(|w| w == b"\r\n")
            .ok_or_else(|| ClientError::invalid_response("malformed chunk header"))?;
        let size_str = String::from_utf8_lossy(&raw[cursor..cursor + rel_end]);
        let hex_part = size_str.split(';').next().unwrap_or("").trim();
        let chunk_len = usize::from_str_radix(hex_part, 16)
            .map_err(|_| ClientError::invalid_response("invalid chunk hex length"))?;
        cursor += rel_end + 2;
        if chunk_len == 0 {
            break;
        }
        if cursor + chunk_len > raw.len() {
            return Err(ClientError::invalid_response("truncated HTTP chunk body"));
        }
        out.extend_from_slice(&raw[cursor..cursor + chunk_len]);
        cursor = (cursor + chunk_len + 2).min(raw.len());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::net::TcpListener;

    #[tokio::test]
    async fn test_http_fabric_client_embed_and_ward_ledger_roundtrip() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let port = listener.local_addr().expect("addr").port();

        let srv = tokio::spawn(async move {
            for _ in 0..2 {
                let (mut sock, _) = listener.accept().await.expect("accept");
                let mut buf = vec![0u8; 4096];
                let n = sock.read(&mut buf).await.expect("read");
                let req_str = String::from_utf8_lossy(&buf[..n]);
                let body = if req_str.starts_with("POST /v1/embeddings") {
                    r#"{"object":"list","data":[{"index":0,"embedding":[0.6,0.8]}]}"#
                } else {
                    r#"{"ok":true,"hardware_sensor_fused":true,"ledger":{"total_traces":1}}"#
                };
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                    body.len(),
                    body
                );
                sock.write_all(resp.as_bytes()).await.expect("write");
            }
        });

        let client = HttpFabricClient::from_url(&format!("http://127.0.0.1:{port}")).expect("url");
        let emb = client
            .embed("all-minilm-l6-v2", json!("thermodynamic joules"))
            .await
            .expect("embed");
        assert_eq!(emb["data"][0]["embedding"], json!([0.6, 0.8]));

        let ledger = client.ward_ledger().await.expect("ledger");
        assert_eq!(ledger["hardware_sensor_fused"], true);
        srv.await.expect("srv join");
    }
}
