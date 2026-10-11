"""SQLAlchemy 2 (psycopg 3 dialect): engine connect (runs version/introspection queries), Core, pandas."""
import os, sqlalchemy as sa
port = os.environ.get("JOULEDB_PG_PORT", "5433")
eng = sa.create_engine(f"postgresql+psycopg://postgres@127.0.0.1:{port}/postgres?sslmode=disable")
res = {}
def check(n, f):
    try: f(); res[n] = "ok"
    except Exception as e: res[n] = f"FAIL {type(e).__name__}: {str(e).splitlines()[0][:160]}"
with eng.connect() as c:
    def core():
        c.execute(sa.text("DROP TABLE IF EXISTS sa_t")); c.execute(sa.text("CREATE TABLE sa_t (id INT, v FLOAT)"))
        c.execute(sa.text("INSERT INTO sa_t VALUES (:i, :v)"), [{"i": 1, "v": 0.5}, {"i": 2, "v": 1.5}])
        r = c.execute(sa.text("SELECT sum(v) FROM sa_t WHERE id > :m"), {"m": 0}).scalar(); assert abs(float(r) - 2.0) < 1e-9, r
    check("core_text", core)
    def pandas_read():
        import pandas as pd
        df = pd.read_sql("SELECT id, v FROM sa_t ORDER BY id", c); assert len(df) == 2, df
    check("pandas_read_sql", pandas_read)
    def pandas_write():
        import pandas as pd
        pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}).to_sql("sa_df", c, if_exists="replace", index=False)
        assert int(c.execute(sa.text("SELECT count(*) FROM sa_df")).scalar()) == 3
    check("pandas_to_sql", pandas_write)
    def inspect():
        assert "sa_t" in sa.inspect(c).get_table_names()
    check("inspector_tables", inspect)
# Known gaps, tracked in the compat matrix: SQLAlchemy reflection (and so
# DataFrame.to_sql, which reflects first) sends pg_catalog queries with
# `::"char"[]` array casts and `= ANY(ARRAY[...])` that the SQL engine does not
# parse yet. They are reported as xfail; an unexpected pass is also reported.
KNOWN = {"pandas_to_sql", "inspector_tables"}
bad = False
for k, v in res.items():
    if k in KNOWN:
        print(f"sqlalchemy {k}: {'XPASS' if v == 'ok' else 'xfail (' + v + ')'}")
    else:
        print(f"sqlalchemy {k}: {v}"); bad |= v != "ok"
raise SystemExit(1 if bad else 0)
