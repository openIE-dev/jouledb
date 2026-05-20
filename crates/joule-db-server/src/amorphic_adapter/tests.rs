use super::*;
use super::conversions::{coerce_row_types, infer_column_type};
use tempfile::TempDir;

fn setup() -> (TempDir, AmorphicTableStorage) {
    let dir = TempDir::new().unwrap();
    let store = DurableAmorphicStore::open(dir.path()).unwrap();
    let adapter = AmorphicTableStorage::new(store);
    (dir, adapter)
}

#[test]
fn test_create_table_and_exists() {
    let (_dir, adapter) = setup();

    assert!(!adapter.table_exists("users").unwrap());

    adapter
        .create_table("users", &["id".into(), "name".into(), "age".into()])
        .unwrap();

    assert!(adapter.table_exists("users").unwrap());
}

#[test]
fn test_create_table_duplicate() {
    let (_dir, adapter) = setup();

    adapter.create_table("t", &["x".into()]).unwrap();
    let err = adapter.create_table("t", &["x".into()]);
    assert!(err.is_err());
}

#[test]
fn test_insert_and_scan() {
    let (_dir, adapter) = setup();

    adapter
        .create_table("users", &["name".into(), "age".into()])
        .unwrap();

    let row = RowData::new(
        vec!["name".into(), "age".into()],
        vec![AstValue::String("Alice".into()), AstValue::Int(30)],
    );
    adapter.insert("users", &row).unwrap();

    let rows = adapter.scan("users").unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("name"), Some(&AstValue::String("Alice".into())));
    assert_eq!(rows[0].get("age"), Some(&AstValue::Int(30)));
}

#[test]
fn test_columns() {
    let (_dir, adapter) = setup();
    adapter
        .create_table("t", &["a".into(), "b".into(), "c".into()])
        .unwrap();
    let cols = adapter.columns("t").unwrap();
    assert_eq!(cols, vec!["a", "b", "c"]);
}

#[test]
fn test_delete_with_predicate() {
    let (_dir, adapter) = setup();
    adapter
        .create_table("users", &["name".into(), "age".into()])
        .unwrap();

    adapter
        .insert(
            "users",
            &RowData::new(
                vec!["name".into(), "age".into()],
                vec![AstValue::String("Alice".into()), AstValue::Int(30)],
            ),
        )
        .unwrap();

    adapter
        .insert(
            "users",
            &RowData::new(
                vec!["name".into(), "age".into()],
                vec![AstValue::String("Bob".into()), AstValue::Int(25)],
            ),
        )
        .unwrap();

    // DELETE WHERE name = 'Alice'
    let predicate = Expression::Binary {
        left: Box::new(Expression::Column("name".into())),
        op: Operator::Eq,
        right: Box::new(Expression::Literal(AstValue::String("Alice".into()))),
    };

    let deleted = adapter.delete("users", Some(&predicate)).unwrap();
    assert_eq!(deleted, 1);

    let rows = adapter.scan("users").unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("name"), Some(&AstValue::String("Bob".into())));
}

#[test]
fn test_update_with_predicate() {
    let (_dir, adapter) = setup();
    adapter
        .create_table("users", &["name".into(), "age".into()])
        .unwrap();

    adapter
        .insert(
            "users",
            &RowData::new(
                vec!["name".into(), "age".into()],
                vec![AstValue::String("Alice".into()), AstValue::Int(30)],
            ),
        )
        .unwrap();

    // UPDATE users SET age = 31 WHERE name = 'Alice'
    let mut assignments = HashMap::new();
    assignments.insert("age".to_string(), AstValue::Int(31));

    let predicate = Expression::Binary {
        left: Box::new(Expression::Column("name".into())),
        op: Operator::Eq,
        right: Box::new(Expression::Literal(AstValue::String("Alice".into()))),
    };

    let updated = adapter
        .update("users", &assignments, Some(&predicate))
        .unwrap();
    assert_eq!(updated, 1);

    let rows = adapter.scan("users").unwrap();
    assert_eq!(rows[0].get("age"), Some(&AstValue::Int(31)));
}

#[test]
fn test_drop_table() {
    let (_dir, adapter) = setup();
    adapter.create_table("users", &["name".into()]).unwrap();
    adapter
        .insert(
            "users",
            &RowData::new(vec!["name".into()], vec![AstValue::String("Alice".into())]),
        )
        .unwrap();

    assert!(adapter.drop_table("users").unwrap());
    assert!(!adapter.table_exists("users").unwrap());

    // Scanning a dropped table should fail
    assert!(adapter.scan("users").is_err());
}

#[test]
fn test_insert_into_nonexistent_table() {
    let (_dir, adapter) = setup();
    let row = RowData::new(vec!["x".into()], vec![AstValue::Int(1)]);
    assert!(adapter.insert("nope", &row).is_err());
}

#[test]
fn test_multi_table_isolation() {
    let (_dir, adapter) = setup();
    adapter.create_table("a", &["x".into()]).unwrap();
    adapter.create_table("b", &["y".into()]).unwrap();

    adapter
        .insert("a", &RowData::new(vec!["x".into()], vec![AstValue::Int(1)]))
        .unwrap();
    adapter
        .insert("b", &RowData::new(vec!["y".into()], vec![AstValue::Int(2)]))
        .unwrap();

    let a_rows = adapter.scan("a").unwrap();
    let b_rows = adapter.scan("b").unwrap();

    assert_eq!(a_rows.len(), 1);
    assert_eq!(b_rows.len(), 1);
    assert_eq!(a_rows[0].get("x"), Some(&AstValue::Int(1)));
    assert_eq!(b_rows[0].get("y"), Some(&AstValue::Int(2)));
}

#[test]
fn test_durability_across_reopen() {
    let dir = TempDir::new().unwrap();

    // Insert data
    {
        let store = DurableAmorphicStore::open(dir.path()).unwrap();
        let adapter = AmorphicTableStorage::new(store);
        adapter
            .create_table("users", &["name".into(), "age".into()])
            .unwrap();
        adapter
            .insert(
                "users",
                &RowData::new(
                    vec!["name".into(), "age".into()],
                    vec![AstValue::String("Alice".into()), AstValue::Int(30)],
                ),
            )
            .unwrap();
    }

    // Reopen and verify
    {
        let store = DurableAmorphicStore::open(dir.path()).unwrap();
        let adapter = AmorphicTableStorage::new(store);
        assert!(adapter.table_exists("users").unwrap());
        let rows = adapter.scan("users").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("name"), Some(&AstValue::String("Alice".into())));
    }
}

// ==================== Phase 2: Index Management Tests ====================

#[test]
fn test_create_index() {
    let (_dir, adapter) = setup();
    adapter
        .create_table("users", &["name".into(), "age".into()])
        .unwrap();

    adapter
        .create_index("idx_age", "users", &["age".into()], false, false)
        .unwrap();

    let indexes = adapter.list_indexes("users").unwrap();
    assert_eq!(indexes.len(), 1);
    assert_eq!(indexes[0].0, "idx_age");
    assert_eq!(indexes[0].1, vec!["age"]);
    assert!(!indexes[0].2); // not unique
}

#[test]
fn test_create_unique_index() {
    let (_dir, adapter) = setup();
    adapter
        .create_table("users", &["name".into(), "age".into()])
        .unwrap();

    adapter
        .create_index("idx_name_unique", "users", &["name".into()], true, false)
        .unwrap();

    let indexes = adapter.list_indexes("users").unwrap();
    assert_eq!(indexes.len(), 1);
    assert!(indexes[0].2); // unique
}

#[test]
fn test_create_index_duplicate() {
    let (_dir, adapter) = setup();
    adapter.create_table("users", &["name".into()]).unwrap();

    adapter
        .create_index("idx_name", "users", &["name".into()], false, false)
        .unwrap();

    // Duplicate should fail
    assert!(
        adapter
            .create_index("idx_name", "users", &["name".into()], false, false)
            .is_err()
    );

    // With IF NOT EXISTS should succeed silently
    adapter
        .create_index("idx_name", "users", &["name".into()], false, true)
        .unwrap();
}

#[test]
fn test_create_index_nonexistent_table() {
    let (_dir, adapter) = setup();
    assert!(
        adapter
            .create_index("idx_x", "nope", &["x".into()], false, false)
            .is_err()
    );
}

#[test]
fn test_drop_index() {
    let (_dir, adapter) = setup();
    adapter.create_table("users", &["name".into()]).unwrap();

    adapter
        .create_index("idx_name", "users", &["name".into()], false, false)
        .unwrap();

    assert!(adapter.drop_index("idx_name").unwrap());
    assert!(!adapter.drop_index("idx_name").unwrap()); // already dropped

    let indexes = adapter.list_indexes("users").unwrap();
    assert!(indexes.is_empty());
}

// ==================== Phase 2: Index Scan Tests ====================

#[test]
fn test_index_scan_equality() {
    let (_dir, adapter) = setup();
    adapter
        .create_table("users", &["name".into(), "age".into()])
        .unwrap();

    adapter
        .insert(
            "users",
            &RowData::new(
                vec!["name".into(), "age".into()],
                vec![AstValue::String("Alice".into()), AstValue::Int(30)],
            ),
        )
        .unwrap();
    adapter
        .insert(
            "users",
            &RowData::new(
                vec!["name".into(), "age".into()],
                vec![AstValue::String("Bob".into()), AstValue::Int(25)],
            ),
        )
        .unwrap();

    let predicate = Expression::Binary {
        left: Box::new(Expression::Column("name".into())),
        op: Operator::Eq,
        right: Box::new(Expression::Literal(AstValue::String("Alice".into()))),
    };

    let rows = adapter.index_scan("users", "name", &predicate).unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("name"), Some(&AstValue::String("Alice".into())));
}

#[test]
fn test_index_scan_range() {
    let (_dir, adapter) = setup();
    adapter
        .create_table("items", &["name".into(), "price".into()])
        .unwrap();

    for (name, price) in &[("A", 10), ("B", 20), ("C", 30), ("D", 40), ("E", 50)] {
        adapter
            .insert(
                "items",
                &RowData::new(
                    vec!["name".into(), "price".into()],
                    vec![AstValue::String(name.to_string()), AstValue::Int(*price)],
                ),
            )
            .unwrap();
    }

    // price >= 20 AND price <= 30 (amorphic query_range is inclusive on both ends)
    let predicate = Expression::Binary {
        left: Box::new(Expression::Binary {
            left: Box::new(Expression::Column("price".into())),
            op: Operator::Ge,
            right: Box::new(Expression::Literal(AstValue::Int(20))),
        }),
        op: Operator::And,
        right: Box::new(Expression::Binary {
            left: Box::new(Expression::Column("price".into())),
            op: Operator::Le,
            right: Box::new(Expression::Literal(AstValue::Int(30))),
        }),
    };

    let rows = adapter.index_scan("items", "price", &predicate).unwrap();
    assert_eq!(rows.len(), 2);
}

// ==================== Phase 2: Statistics Tests ====================

#[test]
fn test_table_statistics() {
    let (_dir, adapter) = setup();
    adapter
        .create_table("users", &["name".into(), "age".into()])
        .unwrap();

    adapter
        .insert(
            "users",
            &RowData::new(
                vec!["name".into(), "age".into()],
                vec![AstValue::String("Alice".into()), AstValue::Int(30)],
            ),
        )
        .unwrap();
    adapter
        .insert(
            "users",
            &RowData::new(
                vec!["name".into(), "age".into()],
                vec![AstValue::String("Bob".into()), AstValue::Int(25)],
            ),
        )
        .unwrap();
    adapter
        .insert(
            "users",
            &RowData::new(
                vec!["name".into(), "age".into()],
                vec![AstValue::String("Charlie".into()), AstValue::Int(30)],
            ),
        )
        .unwrap();

    let stats = adapter.get_table_statistics("users").unwrap();
    assert_eq!(stats.row_count, 3);

    let age_stats = stats.columns.get("age").unwrap();
    assert_eq!(age_stats.distinct_count, 2); // 25, 30
    assert_eq!(age_stats.null_fraction, 0.0);
    assert_eq!(age_stats.min_value, Some("25".to_string()));
    assert_eq!(age_stats.max_value, Some("30".to_string()));

    let name_stats = stats.columns.get("name").unwrap();
    assert_eq!(name_stats.distinct_count, 3);
}

#[test]
fn test_planner_statistics_collection() {
    let (_dir, adapter) = setup();
    adapter
        .create_table("users", &["name".into(), "age".into()])
        .unwrap();
    adapter
        .insert(
            "users",
            &RowData::new(
                vec!["name".into(), "age".into()],
                vec![AstValue::String("Alice".into()), AstValue::Int(30)],
            ),
        )
        .unwrap();

    adapter
        .create_index("idx_age", "users", &["age".into()], false, false)
        .unwrap();

    let stats = adapter.collect_planner_statistics();
    assert_eq!(stats.table_rows("users"), 1);
}

// ==================== Phase 2: Columnar Aggregate Tests ====================

#[test]
fn test_columnar_aggregate_sum() {
    let (_dir, adapter) = setup();
    adapter
        .create_table("scores", &["name".into(), "score".into()])
        .unwrap();

    for (name, score) in &[("A", 10), ("B", 20), ("C", 30)] {
        adapter
            .insert(
                "scores",
                &RowData::new(
                    vec!["name".into(), "score".into()],
                    vec![AstValue::String(name.to_string()), AstValue::Int(*score)],
                ),
            )
            .unwrap();
    }

    let sum = adapter
        .columnar_aggregate("scores", "SUM", "score")
        .unwrap();
    assert_eq!(sum, Some(60.0));
}

#[test]
fn test_columnar_aggregate_avg() {
    let (_dir, adapter) = setup();
    adapter
        .create_table("scores", &["name".into(), "score".into()])
        .unwrap();

    for (name, score) in &[("A", 10), ("B", 20), ("C", 30)] {
        adapter
            .insert(
                "scores",
                &RowData::new(
                    vec!["name".into(), "score".into()],
                    vec![AstValue::String(name.to_string()), AstValue::Int(*score)],
                ),
            )
            .unwrap();
    }

    let avg = adapter
        .columnar_aggregate("scores", "AVG", "score")
        .unwrap();
    assert_eq!(avg, Some(20.0));
}

#[test]
fn test_columnar_aggregate_min_max() {
    let (_dir, adapter) = setup();
    adapter
        .create_table("scores", &["name".into(), "score".into()])
        .unwrap();

    for (name, score) in &[("A", 10), ("B", 20), ("C", 30)] {
        adapter
            .insert(
                "scores",
                &RowData::new(
                    vec!["name".into(), "score".into()],
                    vec![AstValue::String(name.to_string()), AstValue::Int(*score)],
                ),
            )
            .unwrap();
    }

    let min = adapter
        .columnar_aggregate("scores", "MIN", "score")
        .unwrap();
    assert_eq!(min, Some(10.0));

    let max = adapter
        .columnar_aggregate("scores", "MAX", "score")
        .unwrap();
    assert_eq!(max, Some(30.0));
}

#[test]
fn test_columnar_aggregate_count() {
    let (_dir, adapter) = setup();
    adapter
        .create_table("scores", &["name".into(), "score".into()])
        .unwrap();

    for (name, score) in &[("A", 10), ("B", 20), ("C", 30)] {
        adapter
            .insert(
                "scores",
                &RowData::new(
                    vec!["name".into(), "score".into()],
                    vec![AstValue::String(name.to_string()), AstValue::Int(*score)],
                ),
            )
            .unwrap();
    }

    let count = adapter
        .columnar_aggregate("scores", "COUNT", "score")
        .unwrap();
    assert_eq!(count, Some(3.0));
}

#[test]
fn test_columnar_aggregate_multi_table_isolation() {
    let (_dir, adapter) = setup();
    adapter.create_table("a", &["val".into()]).unwrap();
    adapter.create_table("b", &["val".into()]).unwrap();

    adapter
        .insert(
            "a",
            &RowData::new(vec!["val".into()], vec![AstValue::Int(10)]),
        )
        .unwrap();
    adapter
        .insert(
            "a",
            &RowData::new(vec!["val".into()], vec![AstValue::Int(20)]),
        )
        .unwrap();

    adapter
        .insert(
            "b",
            &RowData::new(vec!["val".into()], vec![AstValue::Int(100)]),
        )
        .unwrap();

    // SUM(val) for table 'a' should be 30, not 130
    let sum_a = adapter.columnar_aggregate("a", "SUM", "val").unwrap();
    assert_eq!(sum_a, Some(30.0));

    let sum_b = adapter.columnar_aggregate("b", "SUM", "val").unwrap();
    assert_eq!(sum_b, Some(100.0));
}

#[test]
fn test_alter_add_column() {
    let (_dir, adapter) = setup();
    adapter
        .create_table("users", &["name".into(), "age".into()])
        .unwrap();

    adapter.alter_add_column("users", "email").unwrap();

    let cols = adapter.columns("users").unwrap();
    assert_eq!(cols, vec!["name", "age", "email"]);
}

#[test]
fn test_alter_add_column_duplicate() {
    let (_dir, adapter) = setup();
    adapter
        .create_table("users", &["name".into(), "age".into()])
        .unwrap();

    let err = adapter.alter_add_column("users", "name");
    assert!(err.is_err());
    assert!(err.unwrap_err().to_string().contains("already exists"));
}

#[test]
fn test_alter_drop_column() {
    let (_dir, adapter) = setup();
    adapter
        .create_table("users", &["name".into(), "age".into(), "email".into()])
        .unwrap();

    adapter.alter_drop_column("users", "email").unwrap();

    let cols = adapter.columns("users").unwrap();
    assert_eq!(cols, vec!["name", "age"]);
}

#[test]
fn test_alter_drop_column_nonexistent() {
    let (_dir, adapter) = setup();
    adapter.create_table("users", &["name".into()]).unwrap();

    let err = adapter.alter_drop_column("users", "nope");
    assert!(err.is_err());
    assert!(err.unwrap_err().to_string().contains("does not exist"));
}

#[test]
fn test_alter_drop_column_blocked_by_index() {
    let (_dir, adapter) = setup();
    adapter
        .create_table("users", &["name".into(), "age".into()])
        .unwrap();

    adapter
        .create_index("idx_name", "users", &["name".into()], false, false)
        .unwrap();

    let err = adapter.alter_drop_column("users", "name");
    assert!(err.is_err());
    assert!(err.unwrap_err().to_string().contains("referenced by index"));
}

#[test]
fn test_alter_rename_column() {
    let (_dir, adapter) = setup();
    adapter
        .create_table("users", &["name".into(), "age".into()])
        .unwrap();

    adapter
        .insert(
            "users",
            &RowData::new(
                vec!["name".into(), "age".into()],
                vec![AstValue::String("Alice".into()), AstValue::Int(30)],
            ),
        )
        .unwrap();

    adapter
        .alter_rename_column("users", "name", "full_name")
        .unwrap();

    // Schema updated
    let cols = adapter.columns("users").unwrap();
    assert_eq!(cols, vec!["full_name", "age"]);

    // Data rows updated
    let rows = adapter.scan("users").unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].get("full_name"),
        Some(&AstValue::String("Alice".into()))
    );
}

#[test]
fn test_alter_rename_column_updates_index() {
    let (_dir, adapter) = setup();
    adapter
        .create_table("t", &["a".into(), "b".into()])
        .unwrap();

    adapter
        .create_index("idx_a", "t", &["a".into()], false, false)
        .unwrap();

    adapter.alter_rename_column("t", "a", "c").unwrap();

    let indexes = adapter.list_indexes("t").unwrap();
    assert_eq!(indexes.len(), 1);
    assert_eq!(indexes[0].1, vec!["c".to_string()]);
}

#[test]
fn test_schema_column_defs_stored_and_retrieved() {
    let (_dir, adapter) = setup();
    let defs = vec![
        SqlColumnDef {
            name: "id".into(),
            data_type: "INTEGER".into(),
            nullable: false,
            primary_key: true,
            unique: false,
            default: None,
            check: None,
            auto_increment: false,
            foreign_key: None,
            column_family: None,
            computed: None,
        },
        SqlColumnDef {
            name: "name".into(),
            data_type: "TEXT".into(),
            nullable: false,
            primary_key: false,
            unique: false,
            default: None,
            check: None,
            auto_increment: false,
            foreign_key: None,
            column_family: None,
            computed: None,
        },
        SqlColumnDef {
            name: "age".into(),
            data_type: "INTEGER".into(),
            nullable: true,
            primary_key: false,
            unique: false,
            default: Some(Expression::Literal(AstValue::Int(0))),
            check: None,
            auto_increment: false,
            foreign_key: None,
            column_family: None,
            computed: None,
        },
    ];

    adapter
        .create_table_with_defs("users", &["id".into(), "name".into(), "age".into()], &defs)
        .unwrap();

    let col_defs = adapter.get_column_defs("users").unwrap();
    assert_eq!(col_defs.len(), 3);
    assert_eq!(col_defs[0].name, "id");
    assert_eq!(col_defs[0].data_type, "INTEGER");
    assert!(!col_defs[0].nullable);
    assert!(col_defs[0].primary_key);
    assert!(col_defs[0].default_value.is_none());

    assert_eq!(col_defs[1].name, "name");
    assert!(!col_defs[1].nullable);

    assert_eq!(col_defs[2].name, "age");
    assert!(col_defs[2].nullable);
    assert!(col_defs[2].default_value.is_some());
}

#[test]
fn test_schema_backward_compat_no_defs() {
    let (_dir, adapter) = setup();
    // Old-style create_table without column defs
    adapter
        .create_table("legacy", &["a".into(), "b".into()])
        .unwrap();

    // get_column_defs should return empty vec (not an error)
    let defs = adapter.get_column_defs("legacy").unwrap();
    assert!(defs.is_empty());

    // columns() still works
    let cols = adapter.columns("legacy").unwrap();
    assert_eq!(cols, vec!["a", "b"]);
}

#[test]
fn test_schema_column_defs_preserved_on_alter() {
    let (_dir, adapter) = setup();
    let defs = vec![
        SqlColumnDef {
            name: "id".into(),
            data_type: "INTEGER".into(),
            nullable: false,
            primary_key: true,
            unique: false,
            default: None,
            check: None,
            auto_increment: false,
            foreign_key: None,
            column_family: None,
            computed: None,
        },
        SqlColumnDef {
            name: "name".into(),
            data_type: "TEXT".into(),
            nullable: true,
            primary_key: false,
            unique: false,
            default: None,
            check: None,
            auto_increment: false,
            foreign_key: None,
            column_family: None,
            computed: None,
        },
    ];

    adapter
        .create_table_with_defs("t", &["id".into(), "name".into()], &defs)
        .unwrap();

    // ALTER ADD COLUMN should preserve existing column_defs
    adapter.alter_add_column("t", "email").unwrap();

    let cols = adapter.columns("t").unwrap();
    assert_eq!(cols, vec!["id", "name", "email"]);

    // Column defs should still be retrievable (the original ones)
    let col_defs = adapter.get_column_defs("t").unwrap();
    assert_eq!(col_defs.len(), 2); // Only original defs preserved
    assert_eq!(col_defs[0].name, "id");
    assert_eq!(col_defs[1].name, "name");
}

// ==================== Schema inference tests ====================

#[test]
fn test_schema_inference_types() {
    // Verify infer_column_type returns correct SQL types
    assert_eq!(infer_column_type(&serde_json::json!("hello")), "TEXT");
    assert_eq!(infer_column_type(&serde_json::json!(42)), "INT");
    assert_eq!(infer_column_type(&serde_json::json!(3.14)), "FLOAT");
    assert_eq!(infer_column_type(&serde_json::json!(true)), "BOOLEAN");
    assert_eq!(infer_column_type(&serde_json::json!([1, 2, 3])), "TEXT");
    assert_eq!(infer_column_type(&serde_json::json!({"a": 1})), "TEXT");
    assert_eq!(infer_column_type(&serde_json::Value::Null), "TEXT");
}

#[test]
fn test_schema_inference_creates_table() {
    let (_dir, adapter) = setup();

    // Table shouldn't exist yet
    assert!(!adapter.table_exists("__default__").unwrap());

    // Ingest with schema should auto-create table
    let (id, coll) = adapter
        .ingest_with_schema(r#"{"name":"Alice","age":30}"#, None)
        .unwrap();

    assert!(id > 0);
    assert_eq!(coll, "__default__");

    // Schema should exist now with inferred columns
    let cols = adapter.columns("__default__").unwrap();
    assert!(cols.contains(&"name".to_string()));
    assert!(cols.contains(&"age".to_string()));
}

#[test]
fn test_schema_inference_merges_columns() {
    let (_dir, adapter) = setup();

    // First ingest creates schema with {name, age}
    adapter
        .ingest_with_schema(r#"{"name":"Alice","age":30}"#, None)
        .unwrap();

    let cols_before = adapter.columns("__default__").unwrap();

    // Second ingest with new field {name, dept} should merge -> {name, age, dept}
    adapter
        .ingest_with_schema(r#"{"name":"Bob","dept":"Engineering"}"#, None)
        .unwrap();

    let cols_after = adapter.columns("__default__").unwrap();
    assert!(cols_after.len() > cols_before.len());
    assert!(cols_after.contains(&"name".to_string()));
    assert!(cols_after.contains(&"age".to_string()));
    assert!(cols_after.contains(&"dept".to_string()));
}

#[test]
fn test_schema_inference_preserves_existing() {
    let (_dir, adapter) = setup();

    // Ingest with {a, b}
    adapter
        .ingest_with_schema(r#"{"a":1,"b":"hello"}"#, None)
        .unwrap();

    // Ingest with same fields -- schema should remain unchanged
    adapter
        .ingest_with_schema(r#"{"a":2,"b":"world"}"#, None)
        .unwrap();

    let cols = adapter.columns("__default__").unwrap();
    assert!(cols.contains(&"a".to_string()));
    assert!(cols.contains(&"b".to_string()));
    assert_eq!(cols.iter().filter(|c| c.as_str() != "__table__").count(), 2);
}

#[test]
fn test_ingest_with_collection_field() {
    let (_dir, adapter) = setup();

    // _collection field should determine the table name and be stripped from data
    let (id, coll) = adapter
        .ingest_with_schema(r#"{"_collection":"users","name":"Alice","age":30}"#, None)
        .unwrap();

    assert!(id > 0);
    assert_eq!(coll, "users");

    // Schema should be under "users" table
    let cols = adapter.columns("users").unwrap();
    assert!(cols.contains(&"name".to_string()));
    assert!(cols.contains(&"age".to_string()));
    // _collection should NOT appear as a column
    assert!(!cols.contains(&"_collection".to_string()));
}

#[test]
fn test_ingest_with_explicit_collection() {
    let (_dir, adapter) = setup();

    // Explicit collection param overrides _collection field
    let (id, coll) = adapter
        .ingest_with_schema(
            r#"{"_collection":"ignored","name":"Alice"}"#,
            Some("employees"),
        )
        .unwrap();

    assert!(id > 0);
    assert_eq!(coll, "employees");
    assert!(adapter.table_exists("employees").unwrap());
}

#[test]
fn test_batch_ingest_with_schema() {
    let (_dir, adapter) = setup();

    let records = vec![
        serde_json::json!({"name": "Alice", "age": 30}),
        serde_json::json!({"name": "Bob", "age": 25}),
        serde_json::json!({"name": "Carol", "age": 35}),
    ];

    let (ids, coll) = adapter
        .batch_ingest_with_schema(&records, Some("people"))
        .unwrap();

    assert_eq!(ids.len(), 3);
    assert_eq!(coll, "people");

    // All records should be scannable
    let rows = adapter.scan("people").unwrap();
    assert_eq!(rows.len(), 3);
}

#[test]
fn test_ingest_with_schema_then_sql_scan() {
    let (_dir, adapter) = setup();

    // Ingest via schema inference
    adapter
        .ingest_with_schema(r#"{"name":"Alice","age":30}"#, Some("users"))
        .unwrap();
    adapter
        .ingest_with_schema(r#"{"name":"Bob","age":25}"#, Some("users"))
        .unwrap();

    // Records should be visible via SQL scan
    let rows = adapter.scan("users").unwrap();
    assert_eq!(rows.len(), 2);

    // Values should be accessible
    let names: Vec<String> = rows
        .iter()
        .filter_map(|r| match r.get("name") {
            Some(AstValue::String(s)) => Some(s.clone()),
            _ => None,
        })
        .collect();
    assert!(names.contains(&"Alice".to_string()));
    assert!(names.contains(&"Bob".to_string()));
}

// ==================== Constraint enforcement tests ====================

#[test]
fn test_check_constraint_insert_valid() {
    let (_dir, adapter) = setup();
    let defs = vec![
        SqlColumnDef {
            name: "id".into(),
            data_type: "INTEGER".into(),
            nullable: false,
            primary_key: true,
            unique: false,
            default: None,
            check: None,
            auto_increment: false,
            foreign_key: None,
            column_family: None,
            computed: None,
        },
        SqlColumnDef {
            name: "age".into(),
            data_type: "INTEGER".into(),
            nullable: false,
            primary_key: false,
            unique: false,
            default: None,
            check: Some(Expression::Binary {
                left: Box::new(Expression::Column("age".into())),
                op: Operator::Ge,
                right: Box::new(Expression::Literal(AstValue::Int(0))),
            }),
            auto_increment: false,
            foreign_key: None,
            column_family: None,
            computed: None,
        },
    ];
    adapter
        .create_table_with_defs("t", &["id".into(), "age".into()], &defs)
        .unwrap();
    // Valid insert
    let row = RowData::new(
        vec!["id".into(), "age".into()],
        vec![AstValue::Int(1), AstValue::Int(25)],
    );
    adapter.insert("t", &row).unwrap();
    let rows = adapter.scan("t").unwrap();
    assert_eq!(rows.len(), 1);
}

#[test]
fn test_unique_constraint_insert_duplicate() {
    let (_dir, adapter) = setup();
    let defs = vec![
        SqlColumnDef {
            name: "id".into(),
            data_type: "INTEGER".into(),
            nullable: false,
            primary_key: false,
            unique: true,
            default: None,
            check: None,
            auto_increment: false,
            foreign_key: None,
            column_family: None,
            computed: None,
        },
        SqlColumnDef {
            name: "name".into(),
            data_type: "TEXT".into(),
            nullable: true,
            primary_key: false,
            unique: false,
            default: None,
            check: None,
            auto_increment: false,
            foreign_key: None,
            column_family: None,
            computed: None,
        },
    ];
    adapter
        .create_table_with_defs("t", &["id".into(), "name".into()], &defs)
        .unwrap();
    let row1 = RowData::new(
        vec!["id".into(), "name".into()],
        vec![AstValue::Int(1), AstValue::String("Alice".into())],
    );
    adapter.insert("t", &row1).unwrap();
    // Duplicate unique value should fail
    let row2 = RowData::new(
        vec!["id".into(), "name".into()],
        vec![AstValue::Int(1), AstValue::String("Bob".into())],
    );
    let result = adapter.check_unique_constraints("t", &row2);
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("UNIQUE constraint failed")
    );
}

#[test]
fn test_unique_constraint_null_allowed() {
    let (_dir, adapter) = setup();
    let defs = vec![
        SqlColumnDef {
            name: "id".into(),
            data_type: "INTEGER".into(),
            nullable: false,
            primary_key: true,
            unique: false,
            default: None,
            check: None,
            auto_increment: false,
            foreign_key: None,
            column_family: None,
            computed: None,
        },
        SqlColumnDef {
            name: "email".into(),
            data_type: "TEXT".into(),
            nullable: true,
            primary_key: false,
            unique: true,
            default: None,
            check: None,
            auto_increment: false,
            foreign_key: None,
            column_family: None,
            computed: None,
        },
    ];
    adapter
        .create_table_with_defs("t", &["id".into(), "email".into()], &defs)
        .unwrap();
    // Multiple NULLs in UNIQUE column are allowed (SQL standard)
    let row1 = RowData::new(
        vec!["id".into(), "email".into()],
        vec![AstValue::Int(1), AstValue::Null],
    );
    adapter.insert("t", &row1).unwrap();
    let row2 = RowData::new(
        vec!["id".into(), "email".into()],
        vec![AstValue::Int(2), AstValue::Null],
    );
    let result = adapter.check_unique_constraints("t", &row2);
    assert!(result.is_ok());
}

#[test]
fn test_auto_increment_sequence() {
    let (_dir, adapter) = setup();
    let defs = vec![
        SqlColumnDef {
            name: "id".into(),
            data_type: "INTEGER".into(),
            nullable: false,
            primary_key: true,
            unique: false,
            default: None,
            check: None,
            auto_increment: true,
            foreign_key: None,
            column_family: None,
            computed: None,
        },
        SqlColumnDef {
            name: "name".into(),
            data_type: "TEXT".into(),
            nullable: true,
            primary_key: false,
            unique: false,
            default: None,
            check: None,
            auto_increment: false,
            foreign_key: None,
            column_family: None,
            computed: None,
        },
    ];
    adapter
        .create_table_with_defs("t", &["id".into(), "name".into()], &defs)
        .unwrap();

    // Get sequential auto-increment values
    let v1 = adapter.next_sequence_value("t", "id").unwrap();
    let v2 = adapter.next_sequence_value("t", "id").unwrap();
    let v3 = adapter.next_sequence_value("t", "id").unwrap();
    assert_eq!(v1, 1);
    assert_eq!(v2, 2);
    assert_eq!(v3, 3);
}

#[test]
fn test_update_check_constraint_violation() {
    let (_dir, adapter) = setup();
    let defs = vec![
        SqlColumnDef {
            name: "id".into(),
            data_type: "INTEGER".into(),
            nullable: false,
            primary_key: true,
            unique: false,
            default: None,
            check: None,
            auto_increment: false,
            foreign_key: None,
            column_family: None,
            computed: None,
        },
        SqlColumnDef {
            name: "age".into(),
            data_type: "INTEGER".into(),
            nullable: false,
            primary_key: false,
            unique: false,
            default: None,
            check: Some(Expression::Binary {
                left: Box::new(Expression::Column("age".into())),
                op: Operator::Ge,
                right: Box::new(Expression::Literal(AstValue::Int(0))),
            }),
            auto_increment: false,
            foreign_key: None,
            column_family: None,
            computed: None,
        },
    ];
    adapter
        .create_table_with_defs("t", &["id".into(), "age".into()], &defs)
        .unwrap();
    let row = RowData::new(
        vec!["id".into(), "age".into()],
        vec![AstValue::Int(1), AstValue::Int(25)],
    );
    adapter.insert("t", &row).unwrap();

    // Update to a negative age should fail CHECK constraint
    let result = adapter.update_with_expressions(
        "t",
        &[("age".to_string(), Expression::Literal(AstValue::Int(-5)))],
        None,
        &[],
    );
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("CHECK constraint failed")
    );
}

#[test]
fn test_update_unique_constraint_violation() {
    let (_dir, adapter) = setup();
    let defs = vec![
        SqlColumnDef {
            name: "id".into(),
            data_type: "INTEGER".into(),
            nullable: false,
            primary_key: true,
            unique: false,
            default: None,
            check: None,
            auto_increment: false,
            foreign_key: None,
            column_family: None,
            computed: None,
        },
        SqlColumnDef {
            name: "email".into(),
            data_type: "TEXT".into(),
            nullable: false,
            primary_key: false,
            unique: true,
            default: None,
            check: None,
            auto_increment: false,
            foreign_key: None,
            column_family: None,
            computed: None,
        },
    ];
    adapter
        .create_table_with_defs("t", &["id".into(), "email".into()], &defs)
        .unwrap();
    let row1 = RowData::new(
        vec!["id".into(), "email".into()],
        vec![AstValue::Int(1), AstValue::String("a@b.com".into())],
    );
    let row2 = RowData::new(
        vec!["id".into(), "email".into()],
        vec![AstValue::Int(2), AstValue::String("c@d.com".into())],
    );
    adapter.insert("t", &row1).unwrap();
    adapter.insert("t", &row2).unwrap();

    // Update row 2 to have the same email as row 1 should fail
    let pk_predicate = Expression::Binary {
        left: Box::new(Expression::Column("id".into())),
        op: Operator::Eq,
        right: Box::new(Expression::Literal(AstValue::Int(2))),
    };
    let result = adapter.update_with_expressions(
        "t",
        &[(
            "email".to_string(),
            Expression::Literal(AstValue::String("a@b.com".into())),
        )],
        Some(&pk_predicate),
        &[],
    );
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("UNIQUE constraint failed")
    );
}

#[test]
fn test_update_same_row_unique_ok() {
    let (_dir, adapter) = setup();
    let defs = vec![
        SqlColumnDef {
            name: "id".into(),
            data_type: "INTEGER".into(),
            nullable: false,
            primary_key: true,
            unique: false,
            default: None,
            check: None,
            auto_increment: false,
            foreign_key: None,
            column_family: None,
            computed: None,
        },
        SqlColumnDef {
            name: "email".into(),
            data_type: "TEXT".into(),
            nullable: false,
            primary_key: false,
            unique: true,
            default: None,
            check: None,
            auto_increment: false,
            foreign_key: None,
            column_family: None,
            computed: None,
        },
    ];
    adapter
        .create_table_with_defs("t", &["id".into(), "email".into()], &defs)
        .unwrap();
    let row = RowData::new(
        vec!["id".into(), "email".into()],
        vec![AstValue::Int(1), AstValue::String("a@b.com".into())],
    );
    adapter.insert("t", &row).unwrap();

    // Updating a row to keep its same unique value should succeed
    let pk_predicate = Expression::Binary {
        left: Box::new(Expression::Column("id".into())),
        op: Operator::Eq,
        right: Box::new(Expression::Literal(AstValue::Int(1))),
    };
    let result = adapter.update_with_expressions(
        "t",
        &[(
            "email".to_string(),
            Expression::Literal(AstValue::String("a@b.com".into())),
        )],
        Some(&pk_predicate),
        &[],
    );
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 1);
}

#[test]
fn test_update_not_null_violation() {
    let (_dir, adapter) = setup();
    let defs = vec![
        SqlColumnDef {
            name: "id".into(),
            data_type: "INTEGER".into(),
            nullable: false,
            primary_key: true,
            unique: false,
            default: None,
            check: None,
            auto_increment: false,
            foreign_key: None,
            column_family: None,
            computed: None,
        },
        SqlColumnDef {
            name: "name".into(),
            data_type: "TEXT".into(),
            nullable: false,
            primary_key: false,
            unique: false,
            default: None,
            check: None,
            auto_increment: false,
            foreign_key: None,
            column_family: None,
            computed: None,
        },
    ];
    adapter
        .create_table_with_defs("t", &["id".into(), "name".into()], &defs)
        .unwrap();
    let row = RowData::new(
        vec!["id".into(), "name".into()],
        vec![AstValue::Int(1), AstValue::String("Alice".into())],
    );
    adapter.insert("t", &row).unwrap();

    // Setting a NOT NULL column to NULL should fail
    let result = adapter.update_with_expressions(
        "t",
        &[("name".to_string(), Expression::Literal(AstValue::Null))],
        None,
        &[],
    );
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("NOT NULL constraint failed")
    );
}

#[test]
fn test_type_coercion_int_to_float() {
    // coerce_row_types should convert Int to Float when column type is FLOAT
    let columns = vec!["val".to_string()];
    let mut values = vec![AstValue::Int(42)];
    let defs = vec![ColumnDefInfo {
        name: "val".into(),
        data_type: "FLOAT".into(),
        nullable: true,
        primary_key: false,
        unique: false,
        default_value: None,
        check_expr: None,
        auto_increment: false,
        foreign_key: None,
        column_family: None,
        computed_expr: None,
    }];
    coerce_row_types(&columns, &mut values, &defs);
    assert_eq!(values[0], AstValue::Float(42.0));
}

#[test]
fn test_type_coercion_string_to_int() {
    let columns = vec!["val".to_string()];
    let mut values = vec![AstValue::String("123".into())];
    let defs = vec![ColumnDefInfo {
        name: "val".into(),
        data_type: "INTEGER".into(),
        nullable: true,
        primary_key: false,
        unique: false,
        default_value: None,
        check_expr: None,
        auto_increment: false,
        foreign_key: None,
        column_family: None,
        computed_expr: None,
    }];
    coerce_row_types(&columns, &mut values, &defs);
    assert_eq!(values[0], AstValue::Int(123));
}

#[test]
fn test_type_coercion_int_to_bool() {
    let columns = vec!["val".to_string()];
    let mut values = vec![AstValue::Int(1)];
    let defs = vec![ColumnDefInfo {
        name: "val".into(),
        data_type: "BOOLEAN".into(),
        nullable: true,
        primary_key: false,
        unique: false,
        default_value: None,
        check_expr: None,
        auto_increment: false,
        foreign_key: None,
        column_family: None,
        computed_expr: None,
    }];
    coerce_row_types(&columns, &mut values, &defs);
    assert_eq!(values[0], AstValue::Bool(true));
}

#[test]
fn test_type_coercion_int_to_text() {
    let columns = vec!["val".to_string()];
    let mut values = vec![AstValue::Int(42)];
    let defs = vec![ColumnDefInfo {
        name: "val".into(),
        data_type: "TEXT".into(),
        nullable: true,
        primary_key: false,
        unique: false,
        default_value: None,
        check_expr: None,
        auto_increment: false,
        foreign_key: None,
        column_family: None,
        computed_expr: None,
    }];
    coerce_row_types(&columns, &mut values, &defs);
    assert_eq!(values[0], AstValue::String("42".into()));
}
