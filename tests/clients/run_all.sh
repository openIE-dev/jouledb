#!/usr/bin/env bash
# Run every client against a jouledb on 127.0.0.1:$JOULEDB_PG_PORT. Prints a matrix line per check.
set -u
cd "$(dirname "$0")"; rc=0
PY=${PYTHON:-python3}
bash psql.sh && echo "psql: ok" || { echo "psql: FAIL"; rc=1; }
$PY psycopg3.py || rc=1
$PY sqlalchemy_test.py || rc=1
node node_pg.js || rc=1
if [ -n "${PGJDBC_JAR:-}" ]; then java -cp "$PGJDBC_JAR" Jdbc.java || rc=1; fi
exit $rc
