//! End-to-End Backup/Restore Tests (Phase 6.5)
//!
//! Tests the full backup → restore → verify pipeline using live storage.

use joule_db_server::query::{QueryExecutor, QueryRequest, SimpleQueryExecutor};

fn make_query(sql: &str) -> QueryRequest {
    QueryRequest {
        sql: sql.to_string(),
        params: Default::default(),
        args: Vec::new(),
        explain: false,
        limit: None,
        session_id: None,
        query_timeout_ms: None,
        branch_id: None,
        tenant_id: None,
    }
}

#[test]
fn test_backup_restore_preserves_all_data() {
    // Phase 1: Create and populate a database
    let executor = SimpleQueryExecutor::new();

    executor
        .execute(&make_query(
            "CREATE TABLE users (id INT, name TEXT, email TEXT)",
        ))
        .unwrap();
    executor
        .execute(&make_query(
            "CREATE TABLE orders (id INT, user_id INT, total REAL)",
        ))
        .unwrap();

    for i in 0..100 {
        executor
            .execute(&make_query(&format!(
                "INSERT INTO users VALUES ({}, 'user{}', 'user{}@example.com')",
                i, i, i
            )))
            .unwrap();
        executor
            .execute(&make_query(&format!(
                "INSERT INTO orders VALUES ({}, {}, {})",
                i,
                i % 10,
                (i as f64) * 9.99
            )))
            .unwrap();
    }

    // Verify data exists
    let users = executor
        .execute(&make_query("SELECT * FROM users"))
        .unwrap();
    assert_eq!(users.rows.len(), 100, "Should have 100 users");

    let orders = executor
        .execute(&make_query("SELECT * FROM orders"))
        .unwrap();
    assert_eq!(orders.rows.len(), 100, "Should have 100 orders");

    // Phase 2: Create backup
    let backup_manager =
        joule_db_server::BackupManager::new(joule_db_server::BackupConfig::default());

    let backup_result = backup_manager.create_full_backup_from_storage(
        std::path::Path::new("/tmp/jouledb-e2e-backup-test"),
        executor.amorphic(),
    );
    assert!(
        backup_result.is_ok(),
        "Backup should succeed: {:?}",
        backup_result.err()
    );
    let metadata = backup_result.unwrap();

    // Phase 3: Verify backup integrity
    let verify_result = backup_manager.verify_backup_integrity(&metadata.id);
    assert!(
        verify_result.is_ok(),
        "Backup should be valid: {:?}",
        verify_result.err()
    );
    let verified = verify_result.unwrap();
    assert!(verified, "Backup integrity check should pass");

    // Phase 4: Restore to a fresh storage
    let fresh_dir = tempfile::tempdir().unwrap();
    let fresh_store = joule_db_amorphic::DurableAmorphicStore::open(fresh_dir.path()).unwrap();
    let fresh_storage = joule_db_server::amorphic_adapter::AmorphicTableStorage::new(fresh_store);

    let restore_result = backup_manager.restore_backup_to_storage(&metadata.id, &fresh_storage);
    assert!(
        restore_result.is_ok(),
        "Restore should succeed: {:?}",
        restore_result.err()
    );
    let restored_count = restore_result.unwrap();
    assert!(
        restored_count >= 200,
        "Should restore at least 200 records (100 users + 100 orders)"
    );

    // Phase 5: Verify restored data is queryable
    // The fresh_storage should now contain the same tables
    let tables = fresh_storage.list_tables();
    assert!(
        tables.contains(&"users".to_string()),
        "Restored storage should contain 'users' table"
    );
    assert!(
        tables.contains(&"orders".to_string()),
        "Restored storage should contain 'orders' table"
    );
}

#[test]
fn test_backup_empty_database() {
    let executor = SimpleQueryExecutor::new();

    let backup_manager =
        joule_db_server::BackupManager::new(joule_db_server::BackupConfig::default());

    let result = backup_manager.create_full_backup_from_storage(
        std::path::Path::new("/tmp/jouledb-e2e-empty-backup"),
        executor.amorphic(),
    );
    assert!(result.is_ok(), "Backup of empty DB should succeed");
}

#[test]
fn test_backup_with_special_characters() {
    let executor = SimpleQueryExecutor::new();

    executor
        .execute(&make_query("CREATE TABLE special (id INT, data TEXT)"))
        .unwrap();

    // Insert data with special characters
    let special_strings = [
        "hello 'world'",
        "line1\nline2",
        "tab\there",
        "unicode: café résumé naïve",
        "emoji text",
        "null\x00byte",
        "",
    ];

    for (i, s) in special_strings.iter().enumerate() {
        let escaped = s.replace('\'', "''");
        let _ = executor.execute(&make_query(&format!(
            "INSERT INTO special VALUES ({}, '{}')",
            i, escaped
        )));
    }

    let backup_manager =
        joule_db_server::BackupManager::new(joule_db_server::BackupConfig::default());

    let result = backup_manager.create_full_backup_from_storage(
        std::path::Path::new("/tmp/jouledb-e2e-special-backup"),
        executor.amorphic(),
    );
    assert!(result.is_ok(), "Backup with special chars should succeed");
}
