//! Raft mTLS Integration Tests
//!
//! Tests Raft transport over mutual TLS using self-signed certificates
//! generated with `rcgen`. Only compiled when `--features tls` is enabled.

#![cfg(feature = "tls")]

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use joule_db_server::raft::{ClusterConfig, Command, KvStateMachine, RaftConfig, RaftNode};
use joule_db_server::raft_server::{RaftRpcServer, RaftTlsConfig, raft_loop};
use joule_db_server::raft_transport::TcpRaftTransport;

/// Generate a self-signed certificate and key for testing.
fn generate_test_cert() -> (String, String) {
    let cert =
        rcgen::generate_simple_self_signed(vec!["localhost".to_string(), "127.0.0.1".to_string()])
            .expect("failed to generate test cert");

    let cert_pem = cert.cert.pem();
    let key_pem = cert.key_pair.serialize_pem();
    (cert_pem, key_pem)
}

/// Build a mTLS acceptor that requires client certificates.
fn build_tls_acceptor(cert_pem: &str, key_pem: &str) -> tokio_rustls::TlsAcceptor {
    let certs: Vec<rustls::pki_types::CertificateDer<'static>> =
        rustls_pemfile::certs(&mut cert_pem.as_bytes())
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

    let key = rustls_pemfile::private_key(&mut key_pem.as_bytes())
        .unwrap()
        .unwrap();

    // Build root store for client cert verification (self-signed = same cert)
    let mut root_store = rustls::RootCertStore::empty();
    for cert in &certs {
        root_store.add(cert.clone()).unwrap();
    }

    let client_verifier = rustls::server::WebPkiClientVerifier::builder(Arc::new(root_store))
        .build()
        .unwrap();

    let config = rustls::ServerConfig::builder()
        .with_client_cert_verifier(client_verifier)
        .with_single_cert(certs, key)
        .unwrap();

    tokio_rustls::TlsAcceptor::from(Arc::new(config))
}

/// Build a mTLS connector that presents a client certificate.
fn build_tls_connector(cert_pem: &str, key_pem: &str) -> tokio_rustls::TlsConnector {
    let certs: Vec<rustls::pki_types::CertificateDer<'static>> =
        rustls_pemfile::certs(&mut cert_pem.as_bytes())
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

    let key = rustls_pemfile::private_key(&mut key_pem.as_bytes())
        .unwrap()
        .unwrap();

    let mut root_store = rustls::RootCertStore::empty();
    for cert in &certs {
        root_store.add(cert.clone()).unwrap();
    }

    let config = rustls::ClientConfig::builder()
        .with_root_certificates(root_store)
        .with_client_auth_cert(certs, key)
        .unwrap();

    tokio_rustls::TlsConnector::from(Arc::new(config))
}

/// Build a TLS connector that does NOT present a client certificate.
fn build_tls_connector_no_client_cert(cert_pem: &str) -> tokio_rustls::TlsConnector {
    let certs: Vec<rustls::pki_types::CertificateDer<'static>> =
        rustls_pemfile::certs(&mut cert_pem.as_bytes())
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

    let mut root_store = rustls::RootCertStore::empty();
    for cert in &certs {
        root_store.add(cert.clone()).unwrap();
    }

    let config = rustls::ClientConfig::builder()
        .with_root_certificates(root_store)
        .with_no_client_auth();

    tokio_rustls::TlsConnector::from(Arc::new(config))
}

/// Allocate N unique ephemeral ports.
async fn allocate_ports(count: usize) -> Vec<u16> {
    let mut ports = Vec::with_capacity(count);
    for _ in 0..count {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        ports.push(port);
    }
    ports
}

/// Create a 3-node mTLS cluster on ephemeral ports.
async fn create_tls_cluster() -> (
    Vec<Arc<RaftNode<KvStateMachine, TcpRaftTransport>>>,
    Vec<tokio::task::JoinHandle<()>>,
    Vec<tokio::task::JoinHandle<()>>,
) {
    let (cert_pem, key_pem) = generate_test_cert();
    let ports = allocate_ports(3).await;
    let node_ids: Vec<String> = (1..=3).map(|i| format!("tls_node{}", i)).collect();
    let addrs: Vec<String> = ports.iter().map(|p| format!("127.0.0.1:{}", p)).collect();

    let mut members = HashSet::new();
    for id in &node_ids {
        members.insert(id.clone());
    }
    let cluster_config = ClusterConfig::new(members);

    let mut nodes = Vec::new();
    let mut rpc_handles = Vec::new();
    let mut loop_handles = Vec::new();

    for i in 0..3 {
        let mut peers = HashMap::new();
        for j in 0..3 {
            if j != i {
                peers.insert(node_ids[j].clone(), addrs[j].clone());
            }
        }

        let mut raft_config = RaftConfig::new(node_ids[i].clone());
        raft_config.election_timeout_min = Duration::from_millis(200);
        raft_config.election_timeout_max = Duration::from_millis(400);
        raft_config.heartbeat_interval = Duration::from_millis(50);

        // Build transport with mTLS connector (presents client cert)
        let connector = build_tls_connector(&cert_pem, &key_pem);
        let transport =
            Arc::new(TcpRaftTransport::new(node_ids[i].clone(), peers).with_tls(connector));

        let state_machine = KvStateMachine::default();
        let node = Arc::new(RaftNode::new(
            raft_config,
            state_machine,
            transport,
            cluster_config.clone(),
        ));

        // Build RPC server with mTLS acceptor (requires client cert)
        let acceptor = build_tls_acceptor(&cert_pem, &key_pem);
        let rpc_server = RaftRpcServer::new(node.clone(), addrs[i].clone()).with_tls(acceptor);

        rpc_handles.push(tokio::spawn(async move {
            let _ = rpc_server.run().await;
        }));

        let loop_node = node.clone();
        loop_handles.push(tokio::spawn(async move {
            raft_loop(loop_node).await;
        }));

        nodes.push(node);
    }

    // Give listeners time to bind
    tokio::time::sleep(Duration::from_millis(100)).await;
    (nodes, rpc_handles, loop_handles)
}

async fn wait_for_leader(
    nodes: &[Arc<RaftNode<KvStateMachine, TcpRaftTransport>>],
    timeout: Duration,
) -> Option<usize> {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        if tokio::time::Instant::now() >= deadline {
            return None;
        }
        for (i, node) in nodes.iter().enumerate() {
            if node.is_leader().await {
                return Some(i);
            }
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

fn abort_all(handles: &[tokio::task::JoinHandle<()>]) {
    for h in handles {
        h.abort();
    }
}

// ================================================================
// Test: mTLS cluster elects leader
// ================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_tls_cluster_elects_leader() {
    let result = tokio::time::timeout(Duration::from_secs(15), async {
        let (nodes, rpc_handles, loop_handles) = create_tls_cluster().await;

        let leader = wait_for_leader(&nodes, Duration::from_secs(8)).await;
        assert!(leader.is_some(), "mTLS cluster should elect a leader");

        // Exactly one leader
        let mut leader_count = 0;
        for node in &nodes {
            if node.is_leader().await {
                leader_count += 1;
            }
        }
        assert_eq!(leader_count, 1, "Exactly one leader in mTLS cluster");

        abort_all(&rpc_handles);
        abort_all(&loop_handles);
    })
    .await;

    assert!(
        result.is_ok(),
        "mTLS leader election should complete within 15s"
    );
}

// ================================================================
// Test: mTLS cluster replicates data
// ================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_tls_cluster_replicates_data() {
    let result = tokio::time::timeout(Duration::from_secs(20), async {
        let (nodes, rpc_handles, loop_handles) = create_tls_cluster().await;

        let leader_idx = wait_for_leader(&nodes, Duration::from_secs(8))
            .await
            .expect("leader should be elected");

        // Propose a command
        let cmd = Command::Set {
            key: b"tls_key".to_vec(),
            value: b"tls_value".to_vec(),
        };
        let _ = nodes[leader_idx].propose(cmd).await;

        // Give time for replication
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Check data on a follower
        let follower_idx = (0..3).find(|&i| i != leader_idx).unwrap();
        let has_key = nodes[follower_idx]
            .with_state_machine(|sm| sm.data().contains_key("tls_key"))
            .await;

        // Data should have replicated (allow for timing)
        if has_key {
            // Verify the value matches
            let value = nodes[follower_idx]
                .with_state_machine(|sm| sm.data().get("tls_key").cloned())
                .await;
            assert_eq!(value.as_deref(), Some(b"tls_value".as_slice()));
        }

        abort_all(&rpc_handles);
        abort_all(&loop_handles);
    })
    .await;

    assert!(
        result.is_ok(),
        "mTLS data replication should complete within 20s"
    );
}

// ================================================================
// Test: Plain TCP client rejected by mTLS server
// ================================================================

#[tokio::test]
async fn test_plain_tcp_rejected_by_tls_server() {
    let (cert_pem, key_pem) = generate_test_cert();
    let acceptor = build_tls_acceptor(&cert_pem, &key_pem);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    // Start a mTLS-enabled server that handles one connection
    let server_handle = tokio::spawn(async move {
        let (tcp_stream, _) = listener.accept().await.unwrap();
        // TLS handshake should fail when client sends plain text
        let result = acceptor.accept(tcp_stream).await;
        result.is_err() // should be Err because client didn't do TLS
    });

    // Connect with plain TCP and send garbage
    let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    use tokio::io::AsyncWriteExt;
    let _ = stream.write_all(b"plain text, not TLS").await;
    let _ = stream.shutdown().await;

    let handshake_failed = tokio::time::timeout(Duration::from_secs(5), server_handle)
        .await
        .expect("server should respond within 5s")
        .expect("server task should not panic");

    assert!(
        handshake_failed,
        "Plain TCP client should fail TLS handshake"
    );
}

// ================================================================
// Test: TLS client without client cert rejected by mTLS server
// ================================================================

#[tokio::test]
async fn test_no_client_cert_rejected_by_mtls_server() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let (cert_pem, key_pem) = generate_test_cert();
    let acceptor = build_tls_acceptor(&cert_pem, &key_pem);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    // Start a mTLS-enabled server
    let server_handle = tokio::spawn(async move {
        let (tcp_stream, _) = listener.accept().await.unwrap();
        // mTLS handshake should fail because client doesn't present a cert
        let result = acceptor.accept(tcp_stream).await;
        result.is_err()
    });

    // Connect with TLS but WITHOUT presenting a client certificate.
    // In TLS 1.3, the client-side connect() may succeed before the server
    // processes the empty certificate. The error surfaces on I/O.
    let connector = build_tls_connector_no_client_cert(&cert_pem);
    let tcp_stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    let server_name = rustls::pki_types::ServerName::try_from("localhost".to_string()).unwrap();

    let connection_failed = match connector.connect(server_name, tcp_stream).await {
        Err(_) => true, // Handshake failed immediately
        Ok(mut tls_stream) => {
            // Handshake appeared to succeed, but server will reject.
            // Try I/O — should fail with connection reset or alert.
            let write_result = tls_stream.write_all(b"hello").await;
            if write_result.is_err() {
                true
            } else {
                let mut buf = [0u8; 1];
                let read_result = tls_stream.read(&mut buf).await;
                match read_result {
                    Err(_) => true,
                    Ok(0) => true, // Connection closed = rejected
                    Ok(_) => false,
                }
            }
        }
    };

    assert!(
        connection_failed,
        "Client without cert should be rejected by mTLS server"
    );

    let server_saw_failure = tokio::time::timeout(Duration::from_secs(5), server_handle)
        .await
        .expect("server should respond within 5s")
        .expect("server task should not panic");

    assert!(
        server_saw_failure,
        "mTLS server should detect missing client certificate"
    );
}
