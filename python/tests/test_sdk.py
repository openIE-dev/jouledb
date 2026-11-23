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


def test_synapse_ward_embed_chat_and_messages_sdk(db):
    status = db.synapse_status()
    assert status.get("status") == "ok"

    emb = db.embed("thermodynamic database energy")
    assert emb.get("object") == "list"
    assert len(emb["data"][0]["embedding"]) >= 8
    assert emb["x_jouledb_energy"]["aggregate_rails"]["wall_duration_ns"] > 0

    chat = db.chat("Hello JouleSynapse")
    assert chat.get("object") == "chat.completion"
    assert len(chat["choices"][0]["message"]["content"]) > 0

    msg = db.messages("Verify 45 nJ JoulePulse guardrail")
    assert msg.get("type") == "message"

    minted = db.ward_mint_token(
        "py-sdk-tok-1",
        "pytest-agent",
        [{"kind": "max_total_nanojoules", "value": 50_000_000}],
    )
    assert minted.get("ok") is True

    ledger = db.ward_ledger()
    assert ledger.get("ok") is True
    assert ledger["ledger"]["total_events"] >= 2

    chunks = db.chat_stream("Stream metered SSE chunks")
    assert len(chunks) >= 2
    assert any("x_jouledb_token_nj" in c for c in chunks)

    schema = {
        "name": "plan",
        "schema": {
            "type": "object",
            "properties": {
                "tier": {"type": "string", "enum": ["mcu", "uma", "cloud"]},
                "budget_mj": {"type": "integer"},
            },
            "required": ["tier", "budget_mj"],
        },
    }
    structured = db.chat_structured("Run on uma with 320 mJ", schema)
    import json as _json

    parsed = _json.loads(structured["choices"][0]["message"]["content"])
    assert parsed["tier"] == "uma"
    assert parsed["budget_mj"] == 320

    tools = [
        {
            "type": "function",
            "function": {
                "name": "query_ledger",
                "description": "Query energy ledger by tier",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tier": {"type": "string", "enum": ["mcu", "uma", "cloud"]},
                        "limit": {"type": "integer"},
                    },
                    "required": ["tier", "limit"],
                },
            },
        }
    ]
    tool_resp = db.chat_tools("Query energy ledger on uma with limit 5", tools)
    tcall = tool_resp["choices"][0]["message"]["tool_calls"][0]
    assert tcall["function"]["name"] == "query_ledger"
    targs = _json.loads(tcall["function"]["arguments"])
    assert targs["tier"] == "uma"
    assert targs["limit"] == 5

    mcp_catalog = db.mcp_list_tools()
    assert mcp_catalog["ok"] is True
    assert mcp_catalog["total_tools"] >= 4

    tbl = "agent_" + uuid.uuid4().hex[:6]
    db.query(f"CREATE TABLE {tbl} (node TEXT, mw FLOAT)")
    db.query(f"INSERT INTO {tbl} VALUES ('m5-uma', 24500.0)")
    ag = db.agent_run(f"SELECT node, mw FROM {tbl}; summarize", max_turns=2)
    assert ag["ok"] is True
    assert ag["turns_executed"] == 2
    assert ag["steps"][0]["selected_tool"] == "jouledb.sql_query"
    assert ag["steps"][0]["output"]["row_count"] == 1
    db.query(f"DROP TABLE {tbl}")
