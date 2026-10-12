#!/usr/bin/env bash
# usage: fleet_r12.sh REV TAG  (run on a fleet host from ~/jouledb-bench)
# Installs REV, runs the harness at 10k and 1M rows twice each (just after
# load, and after a 20 s settle), then a 1M Parquet load. Every output file
# carries TAG, host and settle, so runs never overwrite each other.
set -u
export PATH=$HOME/.cargo/bin:$PATH
B=$HOME/jouledb-bench; cd "$B"; REV=$1; TAG=$2; H=$(hostname -s)
cargo install -q --locked --force --git https://github.com/openIE-dev/jouledb --rev "$REV" --root "$B/inst-$TAG" joule-db-server > "install-$TAG.log" 2>&1 || { echo INSTALL_FAIL; tail -5 "install-$TAG.log"; exit 1; }
for f in harness.py parquet_load.py; do curl -fsSL "https://raw.githubusercontent.com/openIE-dev/jouledb/$REV/bench/$f" -o "$f"; done
if [ -x venv/bin/python ] && venv/bin/python -c "import duckdb, psycopg" 2>/dev/null; then PY=venv/bin/python; else PY=python3; export PYTHONPATH=$B/pylib; fi
J="$B/inst-$TAG/bin/jouledb"; mkdir -p "$B/copy"
start() { pkill -9 -x jouledb; sleep 1; rm -rf "bdata-$TAG"; JOULEDB_COPY_DIR="$B/copy" "$J" -p 28080 -d "bdata-$TAG" > "$1" 2>&1 &
  for i in $(seq 60); do (echo > /dev/tcp/127.0.0.1/28082) 2>/dev/null && break; sleep 1; done; }
for ROWS in 10000 1000000; do for S in 0 20; do
  O="$TAG-$H-$ROWS-s$S"; start "srv-$O.log"
  $PY harness.py --pg-port 28082 --rows $ROWS --min-seconds 2 --settle $S --out "$O.json" > "$O.md" 2>&1; echo "harness $O exit $?"
done; done
rm -f "$B/copy"/*; start "srv-$TAG-$H-pq.log"
$PY parquet_load.py --pg-port 28082 --rows 1000000 --dir "$B/copy" --repeat 3 > "$TAG-$H-pq.md" 2>&1; echo "parquet exit $?"
pkill -9 -x jouledb; rm -rf "bdata-$TAG"; echo FLEET_DONE "$(date)"
