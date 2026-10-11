//! Arrow Flight SQL endpoint (gRPC, default HTTP port + 4).
//!
//! Works with ADBC's Flight SQL driver, the Arrow Flight SQL JDBC driver and
//! any `arrow-flight` client:
//!
//! - `GetFlightInfo(CommandStatementQuery)` runs the SQL and returns its
//!   schema and one endpoint whose ticket holds a result handle;
//! - `DoGet(TicketStatementQuery)` streams that result as Arrow batches;
//! - `DoPut(CommandStatementUpdate)` runs DML/DDL and returns the row count.
//!
//! The energy receipt (energy_joules, power_watts, device_target,
//! algorithm_type, execution_time_ms) rides in the FlightInfo's
//! `app_metadata` as JSON. Results are held for 10 minutes or until fetched.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use arrow::array::RecordBatch;
use arrow_flight::encode::FlightDataEncoderBuilder;
use arrow_flight::flight_service_server::{FlightService, FlightServiceServer};
use arrow_flight::sql::server::{FlightSqlService, PeekableFlightDataStream};
use arrow_flight::sql::{
    CommandStatementQuery, CommandStatementUpdate, ProstMessageExt, SqlInfo, TicketStatementQuery,
};
use arrow_flight::{FlightDescriptor, FlightEndpoint, FlightInfo, HandshakeRequest, HandshakeResponse, Ticket};
use futures::{Stream, TryStreamExt};
use prost::Message;
use tonic::{Request, Response, Status, Streaming};

use crate::query::{QueryExecutor, QueryRequest, QueryResponse};

const RESULT_TTL: Duration = Duration::from_secs(600);

#[derive(Clone)]
pub struct JouleFlightSql {
    executor: Arc<dyn QueryExecutor>,
    results: Arc<Mutex<HashMap<String, (Instant, RecordBatch)>>>,
}

impl JouleFlightSql {
    pub fn new(executor: Arc<dyn QueryExecutor>) -> Self {
        Self { executor, results: Arc::new(Mutex::new(HashMap::new())) }
    }

    async fn run(&self, sql: String) -> Result<QueryResponse, Status> {
        let ex = self.executor.clone();
        tokio::task::spawn_blocking(move || ex.execute(&QueryRequest { sql, ..Default::default() }))
            .await
            .map_err(|e| Status::internal(e.to_string()))?
            .map_err(|e| Status::invalid_argument(e.message))
    }
}

fn receipt_json(r: &QueryResponse) -> Vec<u8> {
    serde_json::json!({
        "energy_joules": r.energy_joules,
        "power_watts": r.power_watts,
        "device_target": r.device_target,
        "algorithm_type": r.algorithm_type,
        "execution_time_ms": r.execution_time_ms,
        "affected_rows": r.affected_rows,
    })
    .to_string()
    .into_bytes()
}

#[tonic::async_trait]
impl FlightSqlService for JouleFlightSql {
    type FlightService = JouleFlightSql;

    /// No auth handshake needed (loopback trust, as for pgwire); answer with
    /// an empty token so clients that always handshake proceed.
    async fn do_handshake(
        &self,
        _request: Request<Streaming<HandshakeRequest>>,
    ) -> Result<Response<Pin<Box<dyn Stream<Item = Result<HandshakeResponse, Status>> + Send>>>, Status>
    {
        let out = futures::stream::iter(vec![Ok(HandshakeResponse { protocol_version: 0, payload: Default::default() })]);
        Ok(Response::new(Box::pin(out)))
    }

    async fn get_flight_info_statement(
        &self,
        query: CommandStatementQuery,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        let resp = self.run(query.query).await?;
        let batch = crate::file_formats::rows_to_batch(&resp.columns, &resp.rows)
            .map_err(Status::internal)?;
        let handle = uuid::Uuid::new_v4().to_string();
        {
            let mut m = self.results.lock().unwrap();
            m.retain(|_, (t, _)| t.elapsed() < RESULT_TTL);
            m.insert(handle.clone(), (Instant::now(), batch.clone()));
        }
        let ticket = TicketStatementQuery { statement_handle: handle.into_bytes().into() };
        let endpoint = FlightEndpoint::new().with_ticket(Ticket::new(ticket.as_any().encode_to_vec()));
        let info = FlightInfo::new()
            .try_with_schema(&batch.schema())
            .map_err(|e| Status::internal(e.to_string()))?
            .with_endpoint(endpoint)
            .with_descriptor(request.into_inner())
            .with_total_records(batch.num_rows() as i64)
            .with_total_bytes(batch.get_array_memory_size() as i64)
            .with_ordered(true)
            .with_app_metadata(receipt_json(&resp));
        Ok(Response::new(info))
    }

    async fn do_get_statement(
        &self,
        ticket: TicketStatementQuery,
        _request: Request<Ticket>,
    ) -> Result<Response<<Self as FlightService>::DoGetStream>, Status> {
        let handle = String::from_utf8_lossy(&ticket.statement_handle).to_string();
        let (_, batch) = self
            .results
            .lock()
            .unwrap()
            .remove(&handle)
            .ok_or_else(|| Status::not_found("result handle expired or already fetched"))?;
        let schema = batch.schema();
        let stream = FlightDataEncoderBuilder::new()
            .with_schema(schema)
            .build(futures::stream::iter(vec![Ok(batch)]))
            .map_err(Status::from);
        Ok(Response::new(Box::pin(stream)))
    }

    async fn do_put_statement_update(
        &self,
        cmd: CommandStatementUpdate,
        _request: Request<PeekableFlightDataStream>,
    ) -> Result<i64, Status> {
        let resp = self.run(cmd.query).await?;
        Ok(resp.affected_rows.unwrap_or(0) as i64)
    }

    async fn register_sql_info(&self, _id: i32, _result: &SqlInfo) {}
}

/// Serve Flight SQL on `addr` until the process exits.
pub async fn serve(addr: std::net::SocketAddr, executor: Arc<dyn QueryExecutor>) -> Result<(), String> {
    tonic::transport::Server::builder()
        .add_service(FlightServiceServer::new(JouleFlightSql::new(executor)))
        .serve(addr)
        .await
        .map_err(|e| e.to_string())
}
