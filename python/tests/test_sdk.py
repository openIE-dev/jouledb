"""Integration tests: need a running server at $JOULEDB_URL (default :8080)."""
import os
import uuid

import pyarrow as pa
import pytest

import jouledb

URL = os.environ.get("JOULEDB_URL", "http://localhost:8080")


@pytest.fixture()
def db():
    return jouledb.connect(URL)


@pytest.fixture()
def table(db):
    name = "py_" + uuid.uuid4().hex[:8]
    db.query(f"CREATE TABLE {name} (id INT, name TEXT, v FLOAT, ok BOOLEAN)")
    r = db.query(f"INSERT INTO {name} VALUES (1, 'a', 1.5, true), (2, NULL, 2.5, false), (3, 'c', NULL, true)")
    assert r.affected_rows in (3, None)
    yield name
    db.query(f"DROP TABLE {name}")


def test_arrow_types(db, table):
    r = db.query(f"SELECT id, name, v FROM {table} ORDER BY id")
    t = r.arrow
    assert isinstance(t, pa.Table)
    assert t.column_names == ["id", "name", "v"]
    assert t.schema.field("id").type == pa.int64()
    assert t.schema.field("v").type == pa.float64()
    assert t.column("name").to_pylist() == ["a", None, "c"]
    assert t.column("v").null_count == 1


def test_receipt(db, table):
    r = db.query(f"SELECT SUM(v) FROM {table}")
    assert r.arrow.column(0).to_pylist() == [4.0]
    assert isinstance(r.receipt, dict)
    assert "energy_joules" in r.receipt and r.receipt["energy_joules"] >= 0
    assert r.receipt.get("device_target")


def test_pandas_polars(db, table):
    r = db.query(f"SELECT id, v FROM {table} ORDER BY id")
    df = r.to_pandas()
    assert list(df["id"]) == [1, 2, 3]
    pl_df = r.to_polars()
    assert pl_df.shape == (3, 2)
    assert pl_df["id"].to_list() == [1, 2, 3]


def test_error(db):
    with pytest.raises(jouledb.JouleDBError):
        db.query("SELECT * FROM no_such_table_xyz")


def test_empty_result(db, table):
    r = db.query(f"SELECT id FROM {table} WHERE id > 100")
    assert r.num_rows == 0
