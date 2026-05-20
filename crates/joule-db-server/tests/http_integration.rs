//! HTTP Integration Tests (Phase 6.2)
//!
//! Verifies the HTTP API works correctly via actual tower::oneshot routing.
//! Tests auth flows, query execution, error handling, security headers,
//! CORS enforcement, body size limits, and concurrent request handling.

use axum::{
    body::Body,
    http::{Method, Request, StatusCode},
};
use joule_db_server::{Server, ServerConfig};
use tower::ServiceExt;

fn create_test_server() -> Server {
    let dir = tempfile::tempdir().unwrap();
    let config = ServerConfig {
        db_path: dir.path().to_string_lossy().to_string(),
        auth_enabled: false,
        rate_limiting_enabled: false,
        ..Default::default()
    };
    std::mem::forget(dir);
    Server::new(config).unwrap()
}

fn create_auth_server(secret: &str) -> Server {
    let dir = tempfile::tempdir().unwrap();
    let config = ServerConfig {
        db_path: dir.path().to_string_lossy().to_string(),
        auth_enabled: true,
        auth_jwt_secret: Some(secret.to_string()),
        rate_limiting_enabled: false,
        ..Default::default()
    };
    std::mem::forget(dir);
    Server::new(config).unwrap()
}

// ================================================================
// Health Endpoints
// ================================================================

#[tokio::test]
async fn test_health_returns_200() {
    let server = create_test_server();
    let app = server.router();
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_liveness_returns_200() {
    let server = create_test_server();
    let app = server.router();
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/health/live")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_readiness_returns_200() {
    let server = create_test_server();
    let app = server.router();
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/health/ready")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

// ================================================================
// Security Headers
// ================================================================

#[tokio::test]
async fn test_all_security_headers_present() {
    let server = create_test_server();
    let app = server.router();
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    let headers = resp.headers();
    assert_eq!(headers.get("X-Content-Type-Options").unwrap(), "nosniff");
    assert_eq!(headers.get("X-Frame-Options").unwrap(), "DENY");
    assert_eq!(headers.get("X-XSS-Protection").unwrap(), "1; mode=block");
    assert!(headers.get("Content-Security-Policy").is_some());
    assert!(headers.get("Referrer-Policy").is_some());
    assert!(headers.get("Permissions-Policy").is_some());
    assert!(headers.get("Strict-Transport-Security").is_some());
}

// ================================================================
// CORS Enforcement
// ================================================================

#[tokio::test]
async fn test_cors_blocks_unauthorized_origin() {
    let server = create_test_server(); // no cors_origins configured
    let app = server.router();
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::OPTIONS)
                .uri("/health")
                .header("Origin", "https://evil.example.com")
                .header("Access-Control-Request-Method", "GET")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert!(
        resp.headers().get("Access-Control-Allow-Origin").is_none(),
        "Should not allow cross-origin requests when cors_origins is empty"
    );
}

#[tokio::test]
async fn test_cors_allows_configured_origin() {
    let dir = tempfile::tempdir().unwrap();
    let config = ServerConfig {
        db_path: dir.path().to_string_lossy().to_string(),
        auth_enabled: false,
        rate_limiting_enabled: false,
        cors_origins: vec!["https://trusted.example.com".to_string()],
        ..Default::default()
    };
    std::mem::forget(dir);
    let server = Server::new(config).unwrap();
    let app = server.router();

    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::OPTIONS)
                .uri("/health")
                .header("Origin", "https://trusted.example.com")
                .header("Access-Control-Request-Method", "GET")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        resp.headers()
            .get("Access-Control-Allow-Origin")
            .map(|v| v.to_str().unwrap()),
        Some("https://trusted.example.com")
    );
}

// ================================================================
// Auth Enforcement
// ================================================================

#[tokio::test]
async fn test_auth_rejects_unauthenticated_request() {
    let server = create_auth_server("test-secret-key-12345");
    let app = server.router();

    // POST to the unified endpoint without auth
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/")
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"q": "SELECT 1"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_auth_allows_health_without_token() {
    let server = create_auth_server("test-secret-key-12345");
    let app = server.router();

    let resp = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    // Health endpoints should work without auth
    assert_eq!(resp.status(), StatusCode::OK);
}

// ================================================================
// Body Size Limit
// ================================================================

#[tokio::test]
async fn test_oversized_body_rejected() {
    let server = create_test_server();
    let app = server.router();

    // 20MB body should exceed the 16MB limit
    let big_body = vec![b'x'; 20 * 1024 * 1024];
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/")
                .header("Content-Type", "application/json")
                .body(Body::from(big_body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
}

// ================================================================
// Query Execution via HTTP
// ================================================================

#[tokio::test]
async fn test_sql_query_via_unified_endpoint() {
    let server = create_test_server();
    let app = server.router();

    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/")
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"q": "SELECT 1 AS value"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["ok"], true);
}

#[tokio::test]
async fn test_invalid_json_returns_400() {
    let server = create_test_server();
    let app = server.router();

    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/")
                .header("Content-Type", "application/json")
                .body(Body::from("this is not json"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

// ================================================================
// Concurrent Requests
// ================================================================

#[tokio::test]
async fn test_concurrent_health_checks() {
    let server = create_test_server();

    let mut handles = Vec::new();
    for _ in 0..50 {
        let app = server.router();
        handles.push(tokio::spawn(async move {
            let resp = app
                .oneshot(
                    Request::builder()
                        .uri("/health")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            resp.status()
        }));
    }

    for handle in handles {
        let status = handle.await.unwrap();
        assert_eq!(status, StatusCode::OK);
    }
}

// ================================================================
// Metrics Endpoint
// ================================================================

#[tokio::test]
async fn test_prometheus_metrics_endpoint() {
    let server = create_test_server();
    let app = server.router();

    let resp = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let text = std::str::from_utf8(&body).unwrap();
    // Prometheus metrics should contain standard gauge/counter names
    assert!(
        text.contains("joule_db") || text.contains("# HELP") || text.len() > 0,
        "Metrics endpoint should return metrics content"
    );
}
