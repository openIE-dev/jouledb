# Postgres client compatibility

`run_all.sh` drives a running `jouledb` (pgwire on `$JOULEDB_PG_PORT`, which is the HTTP port + 2) with real drivers:
psql, psycopg 3, SQLAlchemy 2 + pandas, node-postgres, and pgJDBC (set `PGJDBC_JAR`). CI: `.github/workflows/pg-clients.yml`.

    jouledb -p 18080 -d /tmp/jdb &
    JOULEDB_PG_PORT=18082 bash tests/clients/run_all.sh

Known gaps print `xfail`: SQLAlchemy reflection (`inspect().get_table_names()`, and `DataFrame.to_sql`, which reflects first) uses `::"char"[]` casts and `= ANY(ARRAY[...])` that the SQL engine does not parse yet.
