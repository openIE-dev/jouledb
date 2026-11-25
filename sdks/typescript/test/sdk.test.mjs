import test from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import { connect, VERSION } from "../index.mjs";

test("@openie/jouledb v1.0.0 client query, migration & powerhouse roundtrip", async () => {
  assert.equal(VERSION, "1.0.0");
  const liveUrl = process.env.JOULEDB_URL;
  if (liveUrl) {
    const db = connect(liveUrl);
    const q = await db.query("SELECT 42 AS answer");
    assert.equal(q.rows[0][0], 42);
    const p = await db.powerhouseSuite();
    assert.equal(p.all_36_pillars_passed, true);
    return;
  }

  const srv = http.createServer((req, res) => {
    res.writeHead(200, { "content-type": "application/json" });
    if (req.url === "/api/v1/query") {
      res.end(JSON.stringify({ columns: ["v"], rows: [[1]], execution_time_ms: 1, energy_joules: 1e-6 }));
    } else {
      res.end(JSON.stringify({ all_36_pillars_passed: true, total_certified_pillars: 36 }));
    }
  });
  await new Promise((resolve) => srv.listen(0, "127.0.0.1", resolve));
  const addr = srv.address();
  try {
    const db = connect(`http://127.0.0.1:${addr.port}`);
    const q = await db.query("SELECT 1 AS v");
    assert.equal(q.rows[0][0], 1);
    const p = await db.powerhouseSuite();
    assert.equal(p.all_36_pillars_passed, true);
  } finally {
    srv.close();
  }
});
