#!/usr/bin/env bash
# psql compatibility: simple protocol, multi-statement, \copy-free basics.
set -euo pipefail
P=${JOULEDB_PG_PORT:-5433}
C="host=127.0.0.1 port=$P user=postgres dbname=postgres sslmode=disable"
psql "$C" -v ON_ERROR_STOP=1 -qAt <<'SQL'
DROP TABLE IF EXISTS psql_t;
CREATE TABLE psql_t (id INT, name TEXT, x FLOAT);
INSERT INTO psql_t VALUES (1,'a',1.5),(2,'b',2.5);
SELECT count(*), sum(x) FROM psql_t;
SELECT name FROM psql_t WHERE id = 2;
SELECT version();
SQL
