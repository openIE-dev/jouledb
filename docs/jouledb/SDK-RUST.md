# JouleDB Rust SDK Quickstart

**Version 1.0 — 2026-05-18**
**Crate:** [`joule-db-client`](../../crates/joule-db-client/) (51 tests)
**Wire:** JWP over TCP / WebSocket / QUIC
**Sister docs:** [`MGAI-JWP-PROTOCOL.md`](../MGAI-JWP-PROTOCOL.md), [`SDK-C.md`](SDK-C.md), [`SDK-WASM.md`](SDK-WASM.md)

## 1. Install

Add to your `Cargo.toml`:

```toml
[dependencies]
joule-db-client = { path = "../joule-db-client" }   # or workspace dep
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

## 2. Connect

```rust
use joule_db_client::Client;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Local: jouledb://localhost:9090
    // Cloud: jouledb://my-cluster.jouledb.cloud:9000
    let client = Client::connect("jouledb://localhost:9090").await?;
    Ok(())
}
```

The URL scheme `jouledb://` selects the JWP wire. Alternatives: `pgwire://` for PostgreSQL-compatible mode (port 5432 by default), `https://` for the HTTP REST surface.

## 3. Query

```rust
let rows = client.query("SELECT name, age FROM users WHERE active = true").await?;

for row in rows {
    let name: String = row.get("name")?;
    let age: i64 = row.get("age")?;
    println!("{name} {age}");
}

// Energy receipt
println!("total: {} µJ", rows.energy_uwh());
```

Prepared statements:

```rust
let stmt = client.prepare("SELECT * FROM users WHERE id = $1").await?;
let rows = stmt.execute(&[&user_id]).await?;
```

Transactions:

```rust
let tx = client.begin().await?;
tx.execute("UPDATE accounts SET balance = balance - 100 WHERE id = $1", &[&from]).await?;
tx.execute("UPDATE accounts SET balance = balance + 100 WHERE id = $1", &[&to]).await?;
tx.commit().await?;
```

## 4. Connection pool

```rust
use joule_db_client::Pool;

let pool = Pool::builder("jouledb://localhost:9090")
    .max_connections(20)
    .min_idle(2)
    .build()
    .await?;

let conn = pool.get().await?;
let rows = conn.query("SELECT 1").await?;
```

## 5. Energy budget

Set a per-query budget; the server rejects queries it projects will exceed it:

```rust
use joule_db_client::ClientOptions;

let client = Client::connect_with_options(
    "jouledb://localhost:9090",
    ClientOptions {
        max_energy_uj: Some(50_000_000),  // 50 J cap
        ..Default::default()
    },
).await?;

// Server returns a structured error if projected_energy_uj > 50_000_000
match client.query("SELECT very_expensive_aggregate()").await {
    Err(joule_db_client::Error::EnergyBudgetExceeded { required_uj, budget_uj, suggested_tier }) => {
        eprintln!("query needs {required_uj} µJ; budget is {budget_uj} µJ; try tier {suggested_tier}");
    }
    Ok(rows) => { /* ... */ }
    Err(e) => return Err(e.into()),
}
```

## 6. Subscriptions (real-time)

```rust
let mut sub = client.subscribe("SELECT * FROM events WHERE topic = 'orders'").await?;

while let Some(change) = sub.next().await {
    let change = change?;
    println!("{} {}: {:?}", change.operation, change.table, change.row);
}
```

## 7. Multi-language queries

Same `query()` call works for all 7 languages — the parser picks the right one from the first keyword:

```rust
// SQL
client.query("SELECT * FROM users").await?;

// Cypher
client.query("MATCH (u:User)-[:KNOWS]->(f:User) RETURN u.name, f.name").await?;

// Datalog
client.query("?- ancestor(\"Alice\", X).").await?;

// SPARQL
client.query("PREFIX foaf: <http://xmlns.com/foaf/0.1/> SELECT ?name WHERE { ?p foaf:name ?name }").await?;
```

For explicit dispatch:

```rust
client.query_in("cypher", "MATCH (n) RETURN n LIMIT 5").await?;
```

## 8. See also

- [`crates/joule-db-client/README.md`](../../crates/joule-db-client/README.md)
- [`MGAI-JWP-PROTOCOL.md`](../MGAI-JWP-PROTOCOL.md) — wire-level detail
- [`QUERY-SQL.md`](QUERY-SQL.md) and sibling per-language refs
