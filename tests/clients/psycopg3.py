"""psycopg 3: simple + extended protocol (server-side params), executemany, types."""
import os, psycopg
port = int(os.environ.get("JOULEDB_PG_PORT", "5433"))
results = {}
def check(name, fn):
    try:
        fn(); results[name] = "ok"
    except Exception as e:  # report every case, not just the first failure
        results[name] = f"FAIL {type(e).__name__}: {str(e)[:120]}"
with psycopg.connect(host="127.0.0.1", port=port, user="postgres", dbname="postgres", sslmode="disable", autocommit=True) as conn:
    cur = conn.cursor()
    def ddl():
        cur.execute("DROP TABLE IF EXISTS pg3_t")
        cur.execute("CREATE TABLE pg3_t (id INT, name TEXT, x FLOAT)")
    check("ddl", ddl)
    check("insert_params", lambda: cur.execute("INSERT INTO pg3_t VALUES (%s, %s, %s)", (1, "a", 1.5)))
    check("executemany", lambda: cur.executemany("INSERT INTO pg3_t VALUES (%s, %s, %s)", [(2, "b", 2.5), (3, "c", 3.5)]))
    def sel():
        cur.execute("SELECT id, name, x FROM pg3_t WHERE id >= %s ORDER BY id", (1,))
        rows = cur.fetchall(); assert len(rows) == 3, rows
        assert rows[0][0] == 1 and rows[0][1] == "a", rows
    check("select_params", sel)
    def agg():
        cur.execute("SELECT count(*), sum(x) FROM pg3_t"); r = cur.fetchone()
        assert int(r[0]) == 3 and abs(float(r[1]) - 7.5) < 1e-9, r
    check("aggregate", agg)
    def prepared():
        for i in range(6):  # psycopg prepares server-side after 5 executions
            cur.execute("SELECT name FROM pg3_t WHERE id = %s", (2,)); assert cur.fetchone()[0] == "b"
    check("prepared_reuse", prepared)
    def binary():
        c2 = conn.cursor(binary=True); c2.execute("SELECT id FROM pg3_t WHERE id = %s", (3,)); assert c2.fetchone()[0] == 3
    check("binary_results", binary)
    def tx():
        with conn.transaction():
            cur.execute("INSERT INTO pg3_t VALUES (%s, %s, %s)", (4, "d", 4.5))
    check("transaction", tx)
    def copy_in():
        with cur.copy("COPY pg3_t (id, name, x) FROM STDIN") as cp:
            cp.write_row((10, "z", 10.5))
        cur.execute("SELECT count(*) FROM pg3_t WHERE id = 10"); assert int(cur.fetchone()[0]) == 1
    check("copy_from_stdin", copy_in)
    def copy_out():
        with cur.copy("COPY (SELECT id FROM pg3_t ORDER BY id) TO STDOUT") as cp:
            data = b"".join(bytes(b) for b in cp)
        assert data.startswith(b"1\n"), data[:40]
    check("copy_to_stdout", copy_out)
    def pgvector_hnsw():
        cur.execute("DROP TABLE IF EXISTS pg3_vec")
        cur.execute("CREATE TABLE pg3_vec (id INT PRIMARY KEY, emb VECTOR(3))")
        cur.executemany(
            "INSERT INTO pg3_vec VALUES (%s, %s)",
            [(1, "[1.0, 0.0, 0.0]"), (2, "[0.0, 1.0, 0.0]"), (3, "[0.9, 0.1, 0.0]")],
        )
        cur.execute("CREATE INDEX ON pg3_vec USING hnsw (emb vector_cosine_ops)")
        cur.execute(
            "SELECT id, emb <=> %s::vector AS dist FROM pg3_vec ORDER BY emb <=> %s::vector LIMIT 2",
            ("[1.0, 0.0, 0.0]", "[1.0, 0.0, 0.0]"),
        )
        vrows = cur.fetchall()
        assert [int(r[0]) for r in vrows] == [1, 3], vrows
    check("pgvector_hnsw", pgvector_hnsw)
    def ledger_sql_fn():
        cur.execute("LEDGER CREATE ACCOUNT pg3_vault LEDGER 1 CODE 1000 SCALE 2")
        cur.execute("LEDGER CREATE ACCOUNT pg3_w1 LEDGER 1 CODE 2000 SCALE 2 NO_OVERDRAFT")
        cur.execute("LEDGER CREATE ACCOUNT pg3_w2 LEDGER 1 CODE 2000 SCALE 2")
        seq = cur.execute(
            "SELECT ledger_transfer(%s, %s, %s, %s)",
            ("pg3-seed-1", "pg3_vault", "pg3_w1", 1000),
        ).fetchone()[0]
        assert int(seq) >= 1
        committed = cur.execute(
            "SELECT ledger_transfer_batch(%s::text[], %s::text[], %s::text[], %s::bigint[])",
            (["pg3-b1", "pg3-b2"], ["pg3_w1", "pg3_w1"], ["pg3_w2", "pg3_w2"], [300, 400]),
        ).fetchone()[0]
        assert int(committed) == 2
    check("ledger_sql_fn", ledger_sql_fn)
for k, v in results.items(): print(f"psycopg3 {k}: {v}")
raise SystemExit(0 if all(v == "ok" for v in results.values()) else 1)
