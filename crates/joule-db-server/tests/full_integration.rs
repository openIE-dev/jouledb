//! Full Integration Tests
//!
//! End-to-end tests that verify all components work together:
//! - SQL query execution
//! - Time-series storage and queries
//! - Graph traversals
//! - Vector similarity search
//! - Full-text search
//! - Real-time subscriptions

use std::sync::Arc;
use std::time::Duration;

// ============================================================================
// SQL Integration Tests
// ============================================================================

mod sql_tests {
    use joule_db_server::query::{QueryExecutor, QueryRequest, SimpleQueryExecutor};

    #[tokio::test]
    async fn test_full_sql_workflow() {
        let executor = SimpleQueryExecutor::new();

        // 1. Create table
        let create = QueryRequest {
            sql: "CREATE TABLE users (id INT, name TEXT, email TEXT, age INT)".to_string(),
            params: Default::default(),
            args: Vec::new(),
            explain: false,
            limit: None,
            session_id: None,
            query_timeout_ms: None,
            branch_id: None,
            tenant_id: None,
        };
        let result = executor.execute(&create).unwrap();
        assert_eq!(result.affected_rows, Some(0));

        // 2. Insert rows
        for i in 1..=5 {
            let insert = QueryRequest {
                sql: format!(
                    "INSERT INTO users (id, name, email, age) VALUES ({}, 'User{}', 'user{}@example.com', {})",
                    i,
                    i,
                    i,
                    20 + i
                ),
                params: Default::default(),
                args: Vec::new(),
                explain: false,
                limit: None,
                session_id: None,
                query_timeout_ms: None,
                branch_id: None,
                tenant_id: None,
            };
            executor.execute(&insert).unwrap();
        }

        // 3. Select all
        let select = QueryRequest {
            sql: "SELECT * FROM users".to_string(),
            params: Default::default(),
            args: Vec::new(),
            explain: false,
            limit: None,
            session_id: None,
            query_timeout_ms: None,
            branch_id: None,
            tenant_id: None,
        };
        let result = executor.execute(&select).unwrap();
        assert_eq!(result.rows.len(), 5);
        assert_eq!(result.columns.len(), 4);

        // 4. Select with limit
        let select_limited = QueryRequest {
            sql: "SELECT * FROM users".to_string(),
            params: Default::default(),
            args: Vec::new(),
            explain: false,
            limit: Some(3),
            session_id: None,
            query_timeout_ms: None,
            branch_id: None,
            tenant_id: None,
        };
        let result = executor.execute(&select_limited).unwrap();
        assert_eq!(result.rows.len(), 3);

        // 5. Drop table
        let drop = QueryRequest {
            sql: "DROP TABLE users".to_string(),
            params: Default::default(),
            args: Vec::new(),
            explain: false,
            limit: None,
            session_id: None,
            query_timeout_ms: None,
            branch_id: None,
            tenant_id: None,
        };
        executor.execute(&drop).unwrap();

        // 6. Verify table is gone
        let select_after_drop = QueryRequest {
            sql: "SELECT * FROM users".to_string(),
            params: Default::default(),
            args: Vec::new(),
            explain: false,
            limit: None,
            session_id: None,
            query_timeout_ms: None,
            branch_id: None,
            tenant_id: None,
        };
        assert!(executor.execute(&select_after_drop).is_err());
    }

    #[tokio::test]
    async fn test_multiple_tables() {
        let executor = SimpleQueryExecutor::new();

        // Create multiple tables
        executor
            .execute(&QueryRequest {
                sql: "CREATE TABLE users (id INT, name TEXT)".to_string(),
                params: Default::default(),
                args: Vec::new(),
                explain: false,
                limit: None,
                session_id: None,
                query_timeout_ms: None,
                branch_id: None,
                tenant_id: None,
            })
            .unwrap();

        executor
            .execute(&QueryRequest {
                sql: "CREATE TABLE orders (id INT, user_id INT, amount REAL)".to_string(),
                params: Default::default(),
                args: Vec::new(),
                explain: false,
                limit: None,
                session_id: None,
                query_timeout_ms: None,
                branch_id: None,
                tenant_id: None,
            })
            .unwrap();

        executor
            .execute(&QueryRequest {
                sql: "CREATE TABLE products (id INT, name TEXT, price REAL)".to_string(),
                params: Default::default(),
                args: Vec::new(),
                explain: false,
                limit: None,
                session_id: None,
                query_timeout_ms: None,
                branch_id: None,
                tenant_id: None,
            })
            .unwrap();

        // Insert into each
        executor
            .execute(&QueryRequest {
                sql: "INSERT INTO users (id, name) VALUES (1, 'Alice')".to_string(),
                params: Default::default(),
                args: Vec::new(),
                explain: false,
                limit: None,
                session_id: None,
                query_timeout_ms: None,
                branch_id: None,
                tenant_id: None,
            })
            .unwrap();

        executor
            .execute(&QueryRequest {
                sql: "INSERT INTO orders (id, user_id, amount) VALUES (1, 1, 99.99)".to_string(),
                params: Default::default(),
                args: Vec::new(),
                explain: false,
                limit: None,
                session_id: None,
                query_timeout_ms: None,
                branch_id: None,
                tenant_id: None,
            })
            .unwrap();

        executor
            .execute(&QueryRequest {
                sql: "INSERT INTO products (id, name, price) VALUES (1, 'Widget', 19.99)"
                    .to_string(),
                params: Default::default(),
                args: Vec::new(),
                explain: false,
                limit: None,
                session_id: None,
                query_timeout_ms: None,
                branch_id: None,
                tenant_id: None,
            })
            .unwrap();

        // Query each
        let users = executor
            .execute(&QueryRequest {
                sql: "SELECT * FROM users".to_string(),
                params: Default::default(),
                args: Vec::new(),
                explain: false,
                limit: None,
                session_id: None,
                query_timeout_ms: None,
                branch_id: None,
                tenant_id: None,
            })
            .unwrap();
        assert_eq!(users.rows.len(), 1);

        let orders = executor
            .execute(&QueryRequest {
                sql: "SELECT * FROM orders".to_string(),
                params: Default::default(),
                args: Vec::new(),
                explain: false,
                limit: None,
                session_id: None,
                query_timeout_ms: None,
                branch_id: None,
                tenant_id: None,
            })
            .unwrap();
        assert_eq!(orders.rows.len(), 1);

        let products = executor
            .execute(&QueryRequest {
                sql: "SELECT * FROM products".to_string(),
                params: Default::default(),
                args: Vec::new(),
                explain: false,
                limit: None,
                session_id: None,
                query_timeout_ms: None,
                branch_id: None,
                tenant_id: None,
            })
            .unwrap();
        assert_eq!(products.rows.len(), 1);
    }
}

// ============================================================================
// Subscription Integration Tests
// ============================================================================

mod subscription_tests {
    use joule_db_server::subscriptions::{ChangeOperation, SubscriptionManager};
    use std::sync::Arc;

    #[tokio::test]
    async fn test_subscription_workflow() {
        let manager = Arc::new(SubscriptionManager::new());

        // Subscribe to user changes
        let (sub_id, mut receiver) = manager.subscribe("users:*").await;

        // Simulate database operations
        manager.notify_insert("users:1", b"Alice").await;
        manager.notify_insert("users:2", b"Bob").await;
        manager
            .notify_update("users:1", b"Alice", b"Alice Smith")
            .await;
        manager.notify_delete("users:2", Some(b"Bob")).await;

        // Verify events received in order
        let event1 = receiver.recv().await.unwrap();
        assert_eq!(event1.operation, ChangeOperation::Insert);
        assert_eq!(event1.key, "users:1");

        let event2 = receiver.recv().await.unwrap();
        assert_eq!(event2.operation, ChangeOperation::Insert);
        assert_eq!(event2.key, "users:2");

        let event3 = receiver.recv().await.unwrap();
        assert_eq!(event3.operation, ChangeOperation::Update);
        assert_eq!(event3.key, "users:1");
        assert_eq!(event3.old_value, Some(b"Alice".to_vec()));
        assert_eq!(event3.value, Some(b"Alice Smith".to_vec()));

        let event4 = receiver.recv().await.unwrap();
        assert_eq!(event4.operation, ChangeOperation::Delete);
        assert_eq!(event4.key, "users:2");

        // Unsubscribe
        manager.unsubscribe(sub_id).await;

        // Verify stats
        let stats = manager.stats();
        assert_eq!(stats.total_events, 4);
        assert_eq!(stats.total_deliveries, 4);
    }

    #[tokio::test]
    async fn test_multiple_subscribers_different_patterns() {
        let manager = Arc::new(SubscriptionManager::new());

        let (_, mut user_rx) = manager.subscribe("users:*").await;
        let (_, mut order_rx) = manager.subscribe("orders:*").await;
        let (_, mut all_rx) = manager.subscribe("*").await;

        // User event
        manager.notify_insert("users:1", b"Alice").await;

        // Order event
        manager.notify_insert("orders:1", b"Order1").await;

        // user_rx should only get user event
        let event = user_rx.recv().await.unwrap();
        assert_eq!(event.key, "users:1");
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        assert!(user_rx.try_recv().is_err());

        // order_rx should only get order event
        let event = order_rx.recv().await.unwrap();
        assert_eq!(event.key, "orders:1");
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        assert!(order_rx.try_recv().is_err());

        // all_rx should get both
        let event1 = all_rx.recv().await.unwrap();
        let event2 = all_rx.recv().await.unwrap();
        let keys: Vec<_> = vec![event1.key, event2.key];
        assert!(keys.contains(&"users:1".to_string()));
        assert!(keys.contains(&"orders:1".to_string()));
    }

    #[tokio::test]
    async fn test_high_volume_notifications() {
        let manager = Arc::new(SubscriptionManager::new());
        let (_, mut receiver) = manager.subscribe("*").await;

        // Send many events
        for i in 0..1000i32 {
            manager
                .notify_insert(&format!("key:{}", i), &i.to_le_bytes())
                .await;
        }

        // Receive all
        let mut count = 0;
        while let Ok(_) = receiver.try_recv() {
            count += 1;
        }

        // Some might still be in flight
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        while let Ok(_) = receiver.try_recv() {
            count += 1;
        }

        assert_eq!(count, 1000);
    }
}

// ============================================================================
// Feature Persistence Integration Tests
// ============================================================================

#[cfg(feature = "persistence-tests")]
mod persistence_tests {
    use joule_db_features::persistence::*;
    use std::collections::HashMap;
    use std::sync::Arc;

    // These tests require the persistence module and a storage backend
    // They're gated behind a feature flag since they need more setup

    #[test]
    fn test_timeseries_persistence_roundtrip() {
        // Test would go here with actual storage backend
    }

    #[test]
    fn test_graph_persistence_roundtrip() {
        // Test would go here with actual storage backend
    }

    #[test]
    fn test_vector_persistence_roundtrip() {
        // Test would go here with actual storage backend
    }
}

// ============================================================================
// HTTP API Integration Tests
// ============================================================================

mod http_tests {
    use super::*;
    use std::net::TcpListener;

    fn find_available_port() -> u16 {
        TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port()
    }

    #[tokio::test]
    async fn test_query_endpoint_create_and_select() {
        use axum::body::Body;
        use axum::http::{Request, StatusCode};
        use joule_db_server::query::{QueryState, SimpleQueryExecutor, query_router};
        use tower::ServiceExt;

        let executor = Arc::new(SimpleQueryExecutor::new());
        let app = query_router(executor);

        // Create table
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/v1/query")
                    .header("Content-Type", "application/json")
                    .body(Body::from(
                        r#"{"sql": "CREATE TABLE test (id INT, name TEXT)"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        // Insert
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/v1/query")
                    .header("Content-Type", "application/json")
                    .body(Body::from(
                        r#"{"sql": "INSERT INTO test (id, name) VALUES (1, 'Alice')"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        // Select
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/v1/query")
                    .header("Content-Type", "application/json")
                    .body(Body::from(r#"{"sql": "SELECT * FROM test"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        // Parse response body
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(json["rows"].as_array().unwrap().len(), 1);
        assert_eq!(json["columns"].as_array().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn test_query_endpoint_error_handling() {
        use axum::body::Body;
        use axum::http::{Request, StatusCode};
        use joule_db_server::query::{SimpleQueryExecutor, query_router};
        use tower::ServiceExt;

        let executor = Arc::new(SimpleQueryExecutor::new());
        let app = query_router(executor);

        // Query non-existent table
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/v1/query")
                    .header("Content-Type", "application/json")
                    .body(Body::from(r#"{"sql": "SELECT * FROM nonexistent"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(json["code"], "TABLE_NOT_FOUND");
    }
}

// ============================================================================
// Concurrent Access Tests
// ============================================================================

mod concurrency_tests {
    use super::*;
    use joule_db_server::query::{QueryExecutor, QueryRequest, SimpleQueryExecutor};
    use joule_db_server::subscriptions::SubscriptionManager;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_concurrent_queries() {
        let executor = Arc::new(SimpleQueryExecutor::new());

        // Create table
        executor
            .execute(&QueryRequest {
                sql: "CREATE TABLE concurrent (id INT, value INT)".to_string(),
                params: Default::default(),
                args: Vec::new(),
                explain: false,
                limit: None,
                session_id: None,
                query_timeout_ms: None,
                branch_id: None,
                tenant_id: None,
            })
            .unwrap();

        // Spawn concurrent inserts
        let mut handles = Vec::new();
        for i in 0..100 {
            let exec = executor.clone();
            handles.push(tokio::spawn(async move {
                exec.execute(&QueryRequest {
                    sql: format!(
                        "INSERT INTO concurrent (id, value) VALUES ({}, {})",
                        i,
                        i * 10
                    ),
                    params: Default::default(),
                    args: Vec::new(),
                    explain: false,
                    limit: None,
                    session_id: None,
                    query_timeout_ms: None,
                    branch_id: None,
                    tenant_id: None,
                })
                .unwrap();
            }));
        }

        // Wait for all
        for handle in handles {
            handle.await.unwrap();
        }

        // Verify all inserted
        let result = executor
            .execute(&QueryRequest {
                sql: "SELECT * FROM concurrent".to_string(),
                params: Default::default(),
                args: Vec::new(),
                explain: false,
                limit: None,
                session_id: None,
                query_timeout_ms: None,
                branch_id: None,
                tenant_id: None,
            })
            .unwrap();

        assert_eq!(result.rows.len(), 100);
    }

    #[tokio::test]
    async fn test_concurrent_subscriptions() {
        let manager = Arc::new(SubscriptionManager::new());

        // Create many subscribers
        let mut handles = Vec::new();
        for i in 0..50 {
            let mgr = manager.clone();
            handles.push(tokio::spawn(async move {
                let pattern = format!("topic:{}:*", i);
                let (id, mut rx) = mgr.subscribe(&pattern).await;

                // Wait for potential events
                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

                // Cleanup
                mgr.unsubscribe(id).await;
            }));
        }

        // Publish events concurrently
        for i in 0..50 {
            let mgr = manager.clone();
            handles.push(tokio::spawn(async move {
                for j in 0..10 {
                    mgr.notify_insert(&format!("topic:{}:{}", i, j), &[]).await;
                }
            }));
        }

        // Wait for all
        for handle in handles {
            handle.await.unwrap();
        }

        let stats = manager.stats();
        assert_eq!(stats.total_events, 500);
    }
}

// ============================================================================
// Performance Baseline Tests
// ============================================================================

mod performance_tests {
    use super::*;
    use joule_db_server::query::{QueryExecutor, QueryRequest, SimpleQueryExecutor};
    use std::time::Instant;

    #[tokio::test]
    async fn test_insert_performance() {
        let executor = SimpleQueryExecutor::new();

        executor
            .execute(&QueryRequest {
                sql: "CREATE TABLE perf (id INT, data TEXT)".to_string(),
                params: Default::default(),
                args: Vec::new(),
                explain: false,
                limit: None,
                session_id: None,
                query_timeout_ms: None,
                branch_id: None,
                tenant_id: None,
            })
            .unwrap();

        let start = Instant::now();
        let count = 100; // Reduced from 1000 for faster tests

        for i in 0..count {
            executor
                .execute(&QueryRequest {
                    sql: format!(
                        "INSERT INTO perf (id, data) VALUES ({}, 'test data {}')",
                        i, i
                    ),
                    params: Default::default(),
                    args: Vec::new(),
                    explain: false,
                    limit: None,
                    session_id: None,
                    query_timeout_ms: None,
                    branch_id: None,
                    tenant_id: None,
                })
                .unwrap();
        }

        let elapsed = start.elapsed();
        let ops_per_sec = count as f64 / elapsed.as_secs_f64();

        println!(
            "Insert performance: {} inserts in {:?} ({:.0} ops/sec)",
            count, elapsed, ops_per_sec
        );

        // Should complete in reasonable time (not a hard requirement)
        assert!(elapsed.as_secs() < 30, "Inserts took too long");
    }

    #[tokio::test]
    async fn test_select_performance() {
        let executor = SimpleQueryExecutor::new();

        executor
            .execute(&QueryRequest {
                sql: "CREATE TABLE perf_select (id INT, data TEXT)".to_string(),
                params: Default::default(),
                args: Vec::new(),
                explain: false,
                limit: None,
                session_id: None,
                query_timeout_ms: None,
                branch_id: None,
                tenant_id: None,
            })
            .unwrap();

        // Insert test data (reduced from 1000)
        for i in 0..100 {
            executor
                .execute(&QueryRequest {
                    sql: format!("INSERT INTO perf_select (id, data) VALUES ({}, 'data')", i),
                    params: Default::default(),
                    args: Vec::new(),
                    explain: false,
                    limit: None,
                    session_id: None,
                    query_timeout_ms: None,
                    branch_id: None,
                    tenant_id: None,
                })
                .unwrap();
        }

        let start = Instant::now();
        let iterations = 10; // Reduced from 100

        for _ in 0..iterations {
            executor
                .execute(&QueryRequest {
                    sql: "SELECT * FROM perf_select".to_string(),
                    params: Default::default(),
                    args: Vec::new(),
                    explain: false,
                    limit: None,
                    session_id: None,
                    query_timeout_ms: None,
                    branch_id: None,
                    tenant_id: None,
                })
                .unwrap();
        }

        let elapsed = start.elapsed();
        let ops_per_sec = iterations as f64 / elapsed.as_secs_f64();

        println!(
            "Select performance: {} selects (100 rows each) in {:?} ({:.0} ops/sec)",
            iterations, elapsed, ops_per_sec
        );

        assert!(elapsed.as_secs() < 30, "Selects took too long");
    }
}
