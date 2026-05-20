# JouleDB GraphQL Reference

**Version 1.0 — 2026-05-18**
**Source:** [`crates/joule-db-query/src/graphql.rs`](../../crates/joule-db-query/src/graphql.rs) (878 LOC)
**Executor:** [`crates/joule-db-server/src/graphql_executor.rs`](../../crates/joule-db-server/src/graphql_executor.rs)

JouleDB exposes a GraphQL endpoint backed directly by the database catalog. Schema is auto-derived from tables; nested fields traverse foreign keys.

## 1. Supported

| Surface | Status |
|---|---|
| `query { … }` | yes |
| `mutation { … }` | yes |
| `subscription { … }` | yes (over JWP / WebSocket) |
| Fragments (`fragment X on Y`) | yes |
| Inline fragments (`... on Y`) | yes |
| Variables (`$varname: Type`) | yes |
| Directives (`@include(if: $x)`, `@skip(if: $x)`) | yes |
| Aliases (`alias: field`) | yes |
| Field arguments (filters, pagination) | yes |

## 2. Auto-derived schema

For each JouleDB table, GraphQL exposes:

- A field `tableName(filter: …, orderBy: …, limit: Int, offset: Int): [Type!]!` on `Query`
- A `Type` matching the row schema
- Mutations `createTableName(input: …)`, `updateTableName(id: …, input: …)`, `deleteTableName(id: …)`
- Subscriptions `tableNameChanged(filter: …): Type!`

Foreign keys auto-resolve as nested fields — if `orders.user_id` references `users.id`, then `Order.user: User` is available.

## 3. Examples

```graphql
query {
  users(filter: { active: true }, limit: 10) {
    id
    name
    orders(limit: 5) {
      id
      total
    }
  }
}

mutation {
  createOrder(input: { userId: "u1", total: 99.99 }) {
    id
    createdAt
  }
}

subscription {
  orderChanged(filter: { userId: "u1" }) {
    id
    status
    total
  }
}
```

## 4. Energy receipts

Each response includes an `extensions.energy` block:

```json
{
  "data": { ... },
  "extensions": {
    "energy": {
      "total_uwh": 1245,
      "tier": "Extract",
      "elapsed_ms": 7
    }
  }
}
```

## 5. See also

- [`crates/joule-db-query/README.md`](../../crates/joule-db-query/README.md)
- [`QUERY-SQL.md`](QUERY-SQL.md) — what GraphQL compiles down to
