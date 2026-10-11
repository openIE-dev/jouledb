#!/usr/bin/env bash
# Run the harness on this host in ~/jouledb-bench: venv (duckdb, psycopg), a
# private jouledb on port 28080 (pgwire 28082), results in bench-<host>.{json,md}.
set -u
B=${BENCH_DIR:-$HOME/jouledb-bench}; cd "$B"
PY=$(command -v python3)
[ -d venv ] || "$PY" -m venv venv
./venv/bin/pip -q install duckdb "psycopg[binary]" >/dev/null
JDB=${JOULEDB_BIN:-$B/inst/bin/jouledb}
rm -rf "$B/bdata"; "$JDB" -p 28080 -d "$B/bdata" > "$B/bsrv.log" 2>&1 & SRV=$!
for i in $(seq 60); do (echo > /dev/tcp/127.0.0.1/28082) 2>/dev/null && break; sleep 1; done
H=$(hostname -s)
uptime; ./venv/bin/python "$B/harness.py" --pg-port 28082 --rows "${ROWS:-10000}" --min-seconds "${MINSEC:-2}" --out "bench-$H.json" > "bench-$H.md" 2>&1
echo "harness exit $?"
kill -9 $SRV
