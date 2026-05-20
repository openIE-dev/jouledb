//! High-level JouleDB client.
//!
//! [`Client`] wraps a single [`Connection`] and provides a convenient,
//! ergonomic API. For connection-pooled access see
//! [`ConnectionPool`](crate::pool::ConnectionPool).

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::connection::{Connection, ConnectionConfig, Transaction};
use crate::error::{ClientError, Result};

// ============================================================================
// QueryResult
// ============================================================================

/// The result of a SQL query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    /// Column names in result order.
    pub columns: Vec<String>,
    /// Row data. Each inner `Vec` has one element per column.
    pub rows: Vec<Vec<serde_json::Value>>,
    /// Convenience: `rows.len()`.
    pub row_count: usize,
    /// Server-reported execution time in milliseconds (0 if not reported).
    pub execution_time_ms: u64,
}

impl QueryResult {
    /// Parse a `QueryResult` from the raw JSON bytes in a `QueryResponse`
    /// payload.
    pub(crate) fn from_json(data: &[u8]) -> Result<Self> {
        // The server may send a structured JSON object or a simple result.
        // We attempt to deserialise into our canonical shape first; if that
        // fails we fall back to treating the whole thing as an opaque JSON
        // value and wrapping it.
        if let Ok(qr) = serde_json::from_slice::<QueryResult>(data) {
            return Ok(qr);
        }

        // Fallback: wrap the raw JSON value.
        let value: serde_json::Value = serde_json::from_slice(data).map_err(|e| {
            ClientError::protocol(format!("QueryResponse is not valid JSON: {}", e))
        })?;

        // Try to extract common shapes.
        if let Some(obj) = value.as_object() {
            let columns = obj
                .get("columns")
                .and_then(|v| serde_json::from_value::<Vec<String>>(v.clone()).ok())
                .unwrap_or_default();
            let rows = obj
                .get("rows")
                .and_then(|v| serde_json::from_value::<Vec<Vec<serde_json::Value>>>(v.clone()).ok())
                .unwrap_or_default();
            let row_count = obj
                .get("row_count")
                .and_then(|v| v.as_u64())
                .unwrap_or(rows.len() as u64) as usize;
            let execution_time_ms = obj
                .get("execution_time_ms")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            return Ok(QueryResult {
                columns,
                rows,
                row_count,
                execution_time_ms,
            });
        }

        // Absolute fallback: single-cell result.
        Ok(QueryResult {
            columns: vec!["result".to_string()],
            rows: vec![vec![value]],
            row_count: 1,
            execution_time_ms: 0,
        })
    }
}

// ============================================================================
// Client
// ============================================================================

/// A high-level JouleDB client backed by a single TCP connection.
///
/// # Example
///
/// ```no_run
/// # async fn example() -> joule_db_client::error::Result<()> {
/// use joule_db_client::client::Client;
///
/// let client = Client::connect("127.0.0.1", 9000).await?;
/// client.put("greeting", b"hello", None).await?;
/// let val = client.get("greeting").await?;
/// assert_eq!(val, Some(b"hello".to_vec()));
/// client.close().await;
/// # Ok(())
/// # }
/// ```
pub struct Client {
    config: ConnectionConfig,
    conn: Connection,
}

impl std::fmt::Debug for Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl Client {
    /// Connect to an JouleDB server using default timeouts.
    pub async fn connect(host: &str, port: u16) -> Result<Self> {
        let config = ConnectionConfig {
            host: host.to_string(),
            port,
            ..Default::default()
        };
        Self::connect_with_config(config).await
    }

    /// Connect with a fully-specified [`ConnectionConfig`].
    pub async fn connect_with_config(config: ConnectionConfig) -> Result<Self> {
        let conn = Connection::connect(config.clone()).await?;
        Ok(Self { config, conn })
    }

    /// Send a `Ping` to the server. Returns the round-trip latency.
    pub async fn ping(&self) -> Result<Duration> {
        self.conn.ping().await
    }

    /// Retrieve the value for `key`. Returns `None` if the key is absent.
    pub async fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        self.conn.get(key).await
    }

    /// Store `value` under `key`, optionally with a TTL in seconds.
    pub async fn put(&self, key: &str, value: &[u8], ttl: Option<u64>) -> Result<bool> {
        self.conn.put(key, value, ttl).await
    }

    /// Delete the value for `key`. Returns `true` if the key existed.
    pub async fn delete(&self, key: &str) -> Result<bool> {
        self.conn.delete(key).await
    }

    /// Execute a SQL query and return the result set.
    pub async fn query(&self, sql: &str, params: &[serde_json::Value]) -> Result<QueryResult> {
        self.conn.query(sql, params).await
    }

    /// Execute a SQL statement and return the number of affected rows.
    pub async fn execute(&self, sql: &str, params: &[serde_json::Value]) -> Result<u64> {
        self.conn.execute(sql, params).await
    }

    /// Begin a transaction.
    pub async fn begin(&self) -> Result<Transaction<'_>> {
        self.conn.begin().await
    }

    /// Gracefully shut down the connection (currently a no-op; the socket is
    /// closed when the `Client` is dropped).
    pub async fn close(&self) {
        // The TcpStream is dropped when the Mutex and Client go out of scope.
        // A future version might send a "close" message to the server.
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_result_from_json_canonical() {
        let json = serde_json::json!({
            "columns": ["id", "name"],
            "rows": [[1, "Alice"], [2, "Bob"]],
            "row_count": 2,
            "execution_time_ms": 5
        });
        let data = serde_json::to_vec(&json).unwrap();
        let qr = QueryResult::from_json(&data).unwrap();
        assert_eq!(qr.columns, vec!["id", "name"]);
        assert_eq!(qr.row_count, 2);
        assert_eq!(qr.execution_time_ms, 5);
        assert_eq!(qr.rows.len(), 2);
    }

    #[test]
    fn test_query_result_from_json_partial() {
        // Server sends only columns + rows, no row_count.
        let json = serde_json::json!({
            "columns": ["x"],
            "rows": [[42]]
        });
        let data = serde_json::to_vec(&json).unwrap();
        let qr = QueryResult::from_json(&data).unwrap();
        assert_eq!(qr.columns, vec!["x"]);
        assert_eq!(qr.row_count, 1);
        assert_eq!(qr.execution_time_ms, 0);
    }

    #[test]
    fn test_query_result_from_json_scalar() {
        // Server sends a raw scalar (edge case).
        let data = b"42";
        let qr = QueryResult::from_json(data).unwrap();
        assert_eq!(qr.columns, vec!["result"]);
        assert_eq!(qr.row_count, 1);
        assert_eq!(qr.rows[0][0], serde_json::json!(42));
    }

    #[test]
    fn test_query_result_from_json_invalid() {
        let data = b"not json at all {{{";
        assert!(QueryResult::from_json(data).is_err());
    }
}
