//! Self-contained MCP App UI (`SEP-1865` / `@modelcontextprotocol/ext-apps` wire protocol)
//! for **Claude Desktop**, **claude.ai**, **Claude Cowork**, **Google Antigravity**, **VS Code**,
//! and **ChatGPT Apps**.
//!
//! Served as `ui://jouledb/studio.html` with MIME `text/html;profile=mcp-app`.
//! Strictly obeys default MCP App CSP (`default-src 'none'`; inline CSS/JS/SVG only,
//! zero external network requests; all database operations go through auditable
//! `tools/call` JSON-RPC messages over `window.parent.postMessage`).

#![deny(clippy::unwrap_used)]

/// Canonical `ui://` resource URI advertised on interactive MCP tools.
pub const STUDIO_APP_URI: &str = "ui://jouledb/studio.html";

/// Standard MCP Apps profile MIME type (`SEP-1865`).
pub const MCP_APP_MIME: &str = "text/html;profile=mcp-app";

/// Return `_meta` attaching the interactive Studio UI (plus Antigravity side-pane/sidebar entrypoints).
pub fn studio_entrypoint_meta() -> serde_json::Value {
    serde_json::json!({
        "ui": {
            "resourceUri": STUDIO_APP_URI,
            "visibility": ["model", "app"]
        },
        "com.google.antigravity/ui": {
            "entrypoints": [
                { "type": "aux_pane", "title": "JouleDB Studio" },
                { "type": "global", "title": "JouleDB Studio" }
            ]
        }
    })
}

/// Return `_meta` attaching the inline interactive Studio view to data-returning tools.
pub fn inline_ui_meta() -> serde_json::Value {
    serde_json::json!({
        "ui": {
            "resourceUri": STUDIO_APP_URI,
            "visibility": ["model", "app"]
        }
    })
}

/// Self-contained HTML/CSS/JS application implementing the `SEP-1865` MCP Apps bridge.
pub fn studio_app_html() -> &'static str {
    STUDIO_HTML
}

const STUDIO_HTML: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:;" />
<title>JouleDB Studio — MCP App</title>
<style>
  :root {
    --bg-primary: var(--color-background-primary, #0d1117);
    --bg-secondary: var(--color-background-secondary, #161b22);
    --bg-tertiary: var(--color-background-tertiary, #21262d);
    --text-primary: var(--color-text-primary, #e6edf3);
    --text-secondary: var(--color-text-secondary, #8b949e);
    --border-primary: var(--color-border-primary, #30363d);
    --accent: var(--color-text-info, #22d3ee);
    --emerald: var(--color-text-success, #34d399);
    --amber: var(--color-text-warning, #fbbf24);
    --danger: var(--color-text-danger, #f87171);
    --font-ui: var(--font-sans, -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif);
    --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  }
  [data-theme="light"] {
    --bg-primary: var(--color-background-primary, #faf9f5);
    --bg-secondary: var(--color-background-secondary, #f0eee6);
    --bg-tertiary: var(--color-background-tertiary, #e5e2d8);
    --text-primary: var(--color-text-primary, #191918);
    --text-secondary: var(--color-text-secondary, #5c5b57);
    --border-primary: var(--color-border-primary, #d6d3c9);
    --accent: var(--color-text-info, #0284c7);
    --emerald: var(--color-text-success, #059669);
    --amber: var(--color-text-warning, #d97706);
    --danger: var(--color-text-danger, #dc2626);
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg-primary);
    color: var(--text-primary);
    font-family: var(--font-ui);
    font-size: 13px;
    line-height: 1.45;
    padding: 12px;
  }
  .shell {
    border: 1px solid var(--border-primary);
    border-radius: 10px;
    background: var(--bg-secondary);
    overflow: hidden;
  }
  .topbar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    padding: 10px 12px;
    border-bottom: 1px solid var(--border-primary);
    background: var(--bg-tertiary);
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 700;
    font-size: 13.5px;
    letter-spacing: -0.01em;
  }
  .brand-dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: var(--emerald);
    box-shadow: 0 0 8px var(--emerald);
  }
  .badges {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    font-family: var(--font-mono);
    font-size: 11px;
  }
  .badge {
    padding: 2px 7px;
    border-radius: 999px;
    border: 1px solid var(--border-primary);
    background: var(--bg-primary);
    color: var(--text-secondary);
  }
  .badge.joules { color: var(--emerald); border-color: var(--emerald); }
  .badge.est { color: var(--amber); border-color: var(--amber); }
  .tabs {
    display: flex;
    gap: 2px;
    padding: 6px 10px 0;
    border-bottom: 1px solid var(--border-primary);
    background: var(--bg-secondary);
  }
  .tab {
    padding: 6px 11px;
    font-size: 12px;
    font-weight: 600;
    color: var(--text-secondary);
    background: transparent;
    border: 1px solid transparent;
    border-bottom: none;
    border-radius: 6px 6px 0 0;
    cursor: pointer;
  }
  .tab.active {
    color: var(--accent);
    background: var(--bg-primary);
    border-color: var(--border-primary);
  }
  .panel { display: none; padding: 12px; background: var(--bg-primary); }
  .panel.active { display: block; }
  .sql-bar {
    display: flex;
    gap: 8px;
    margin-bottom: 10px;
  }
  textarea#sql-input {
    flex: 1;
    min-height: 52px;
    max-height: 150px;
    resize: vertical;
    padding: 8px 10px;
    border-radius: 7px;
    border: 1px solid var(--border-primary);
    background: var(--bg-secondary);
    color: var(--text-primary);
    font-family: var(--font-mono);
    font-size: 12px;
  }
  .btn-col { display: flex; flex-direction: column; gap: 6px; }
  button.btn {
    padding: 6px 11px;
    border-radius: 6px;
    border: 1px solid var(--border-primary);
    background: var(--bg-tertiary);
    color: var(--text-primary);
    font-size: 11.5px;
    font-weight: 600;
    cursor: pointer;
    white-space: nowrap;
  }
  button.btn.primary {
    background: var(--accent);
    color: #06121a;
    border-color: transparent;
  }
  button.btn:hover { filter: brightness(1.08); }
  button.btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .toolbar-actions {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    margin-bottom: 8px;
  }
  .filter-input {
    padding: 4px 8px;
    border-radius: 6px;
    border: 1px solid var(--border-primary);
    background: var(--bg-secondary);
    color: var(--text-primary);
    font-size: 12px;
    width: 190px;
  }
  .table-wrap {
    overflow-x: auto;
    max-height: 310px;
    border: 1px solid var(--border-primary);
    border-radius: 7px;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--font-mono);
    font-size: 11.5px;
  }
  th, td {
    padding: 6px 9px;
    border-bottom: 1px solid var(--border-primary);
    text-align: left;
    max-width: 280px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  th {
    position: sticky;
    top: 0;
    background: var(--bg-tertiary);
    color: var(--accent);
    cursor: pointer;
    user-select: none;
  }
  tr:hover td { background: var(--bg-secondary); }
  .chart-box {
    margin-top: 10px;
    padding: 10px;
    border: 1px solid var(--border-primary);
    border-radius: 7px;
    background: var(--bg-secondary);
  }
  .chart-title {
    font-size: 11px;
    font-family: var(--font-mono);
    color: var(--text-secondary);
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .bar-row {
    display: grid;
    grid-template-columns: 120px 1fr 78px;
    align-items: center;
    gap: 8px;
    margin: 4px 0;
    font-family: var(--font-mono);
    font-size: 11px;
  }
  .bar-label { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text-primary); }
  .bar-track { height: 10px; border-radius: 5px; background: var(--bg-tertiary); overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 5px; background: var(--accent); }
  .bar-val { text-align: right; color: var(--emerald); }
  .grid-cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(175px, 1fr));
    gap: 8px;
    margin-bottom: 10px;
  }
  .card {
    padding: 10px;
    border: 1px solid var(--border-primary);
    border-radius: 7px;
    background: var(--bg-secondary);
  }
  .card-k { font-family: var(--font-mono); font-size: 10.5px; color: var(--text-secondary); text-transform: uppercase; }
  .card-v { font-family: var(--font-mono); font-size: 16px; font-weight: 700; margin-top: 3px; color: var(--text-primary); }
  .status-msg {
    margin-top: 8px;
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-secondary);
  }
  .status-msg.err { color: var(--danger); }
  .list-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 7px 10px;
    border: 1px solid var(--border-primary);
    border-radius: 6px;
    margin-bottom: 6px;
    background: var(--bg-secondary);
    font-family: var(--font-mono);
    font-size: 12px;
  }
</style>
</head>
<body>
<div class="shell" id="app-root">
  <div class="topbar">
    <div class="brand">
      <span class="brand-dot"></span>
      <span>JouleDB Studio</span>
      <span style="font-family:var(--font-mono);font-size:11px;color:var(--text-secondary)">MCP App &middot; v0.10.0</span>
    </div>
    <div class="badges">
      <span class="badge" id="badge-rows">0 rows</span>
      <span class="badge" id="badge-ms">0 ms</span>
      <span class="badge est" id="badge-joules">est_j: 0.0000 J</span>
    </div>
  </div>

  <div class="tabs" role="tablist">
    <button class="tab active" data-tab="sql" role="tab">SQL &amp; Chart</button>
    <button class="tab" data-tab="schema" role="tab">Tables &amp; Schema</button>
    <button class="tab" data-tab="branches" role="tab">CoW Branches</button>
    <button class="tab" data-tab="energy" role="tab">Joule Telemetry</button>
  </div>

  <!-- TAB 1: SQL & VISUAL GRID -->
  <div class="panel active" id="panel-sql">
    <div class="sql-bar">
      <textarea id="sql-input" spellcheck="false" placeholder="SELECT * FROM information_schema.tables LIMIT 25;">SELECT table_name FROM information_schema.tables;</textarea>
      <div class="btn-col">
        <button class="btn primary" id="btn-run-sql">Run SQL</button>
        <button class="btn" id="btn-explain">Explain J</button>
      </div>
    </div>
    <div class="toolbar-actions">
      <input class="filter-input" id="row-filter" type="text" placeholder="Filter rows..." />
      <div style="display:flex;gap:6px;flex-wrap:wrap">
        <button class="btn" id="btn-pin-ctx" style="display:none" title="Attach current query &amp; summary to next chat turn">Pin Context</button>
        <button class="btn" id="btn-send-msg" style="display:none" title="Send result summary into conversation">Send to Chat</button>
      </div>
    </div>
    <div class="table-wrap" id="table-container">
      <table><thead id="res-thead"></thead><tbody id="res-tbody"></tbody></table>
    </div>
    <div class="chart-box" id="chart-container" style="display:none">
      <div class="chart-title" id="chart-title">Distribution</div>
      <div id="chart-bars"></div>
    </div>
  </div>

  <!-- TAB 2: SCHEMA -->
  <div class="panel" id="panel-schema">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
      <span style="font-weight:600">Database Tables</span>
      <button class="btn" id="btn-refresh-schema">Refresh Catalog</button>
    </div>
    <div id="schema-list"></div>
  </div>

  <!-- TAB 3: COW BRANCHES -->
  <div class="panel" id="panel-branches">
    <div style="display:flex;gap:8px;margin-bottom:10px;flex-wrap:wrap">
      <input class="filter-input" id="branch-name" type="text" placeholder="new-experiment-branch" />
      <input class="filter-input" id="branch-budget" type="number" placeholder="Budget uJ (optional)" style="width:155px" />
      <button class="btn primary" id="btn-create-branch">Fork O(1) Branch</button>
      <button class="btn" id="btn-refresh-branches">Refresh</button>
    </div>
    <div id="branches-list"></div>
  </div>

  <!-- TAB 4: ENERGY -->
  <div class="panel" id="panel-energy">
    <div class="grid-cards">
      <div class="card">
        <div class="card-k">Last Operation Energy</div>
        <div class="card-v" id="e-last">0.000000 J</div>
      </div>
      <div class="card">
        <div class="card-k">Evidence Class</div>
        <div class="card-v" id="e-prov">est_j</div>
      </div>
      <div class="card">
        <div class="card-k">Execution Path</div>
        <div class="card-v" id="e-algo">columnar / btree</div>
      </div>
      <div class="card">
        <div class="card-k">Session Budget (uJ)</div>
        <div class="card-v" id="e-budget">Unlimited</div>
      </div>
    </div>
    <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
      <input class="filter-input" id="budget-input" type="number" placeholder="Set session budget (uJ)" />
      <button class="btn" id="btn-set-budget">Set Budget</button>
      <span style="font-size:11.5px;color:var(--text-secondary)">Hardware RAPL/SMC counters emit <code>reported_j</code>; sub-ms paths emit analytical <code>est_j</code>.</span>
    </div>
  </div>

  <div style="padding:0 12px 10px"><div class="status-msg" id="status-line">Connecting to MCP host...</div></div>
</div>

<script>
(function() {
  "use strict";

  var nextId = 1;
  var pending = new Map();
  var hostCaps = {};
  var state = {
    sql: "SELECT table_name FROM information_schema.tables;",
    columns: [],
    rows: [],
    sortCol: -1,
    sortAsc: true,
    filter: "",
    joules: 0,
    ms: 0,
    provenance: "est_j",
    algo: "ready",
    tables: [],
    branches: []
  };

  function postRpc(msg) {
    if (window.parent && window.parent !== window) {
      window.parent.postMessage(msg, "*");
    }
  }

  function sendRequest(method, params) {
    return new Promise(function(resolve, reject) {
      var id = nextId++;
      pending.set(id, { resolve: resolve, reject: reject });
      postRpc({ jsonrpc: "2.0", id: id, method: method, params: params || {} });
      setTimeout(function() {
        if (pending.has(id)) {
          pending.delete(id);
          reject(new Error("RPC timeout: " + method));
        }
      }, 25000);
    });
  }

  function sendNotification(method, params) {
    postRpc({ jsonrpc: "2.0", method: method, params: params || {} });
  }

  function setStatus(msg, isErr) {
    var el = document.getElementById("status-line");
    el.textContent = msg;
    el.className = isErr ? "status-msg err" : "status-msg";
  }

  function applyHostContext(ctx) {
    if (!ctx) return;
    if (ctx.theme === "light" || ctx.theme === "dark") {
      document.documentElement.setAttribute("data-theme", ctx.theme);
      document.documentElement.style.colorScheme = ctx.theme;
    }
    if (ctx.styles && ctx.styles.variables) {
      var vars = ctx.styles.variables;
      Object.keys(vars).forEach(function(k) {
        if (k.indexOf("--") === 0 && typeof vars[k] === "string") {
          document.documentElement.style.setProperty(k, vars[k]);
        }
      });
    }
  }

  function updateCapButtons() {
    document.getElementById("btn-send-msg").style.display = hostCaps && hostCaps.message ? "inline-block" : "none";
    document.getElementById("btn-pin-ctx").style.display = hostCaps && hostCaps.updateModelContext ? "inline-block" : "none";
  }

  function callTool(name, args) {
    return sendRequest("tools/call", { name: name, arguments: args || {} });
  }

  function normalizePayload(payload, rawResult) {
    if (!payload) return;
    if (typeof payload.energy_joules === "number") state.joules = payload.energy_joules;
    if (rawResult && typeof rawResult.energy_joules === "number" && rawResult.energy_joules > 0) {
      state.joules = rawResult.energy_joules;
    }
    if (typeof payload.execution_time_ms === "number") state.ms = payload.execution_time_ms;
    if (payload.energy_evidence) state.provenance = payload.energy_evidence;
    else state.provenance = state.ms >= 5 ? "reported_j" : "est_j";
    if (payload.algorithm_type) state.algo = payload.algorithm_type;

    if (Array.isArray(payload.tables)) {
      state.tables = payload.tables;
      renderSchema();
    }
    if (Array.isArray(payload.branches)) {
      state.branches = payload.branches;
      renderBranches();
    }
    var q = payload.query_result || payload;
    if (Array.isArray(q.columns) && Array.isArray(q.rows)) {
      state.columns = q.columns.map(String);
      state.rows = q.rows.map(function(r) {
        if (Array.isArray(r)) return r;
        if (r && typeof r === "object") {
          return state.columns.map(function(c) { return r[c]; });
        }
        return [r];
      });
      state.sortCol = -1;
      renderTableAndChart();
    } else if (Array.isArray(payload)) {
      if (payload.length > 0 && typeof payload[0] === "object" && !Array.isArray(payload[0])) {
        state.columns = Object.keys(payload[0]);
        state.rows = payload.map(function(o) { return state.columns.map(function(c) { return o[c]; }); });
      } else {
        state.columns = ["value"];
        state.rows = payload.map(function(v) { return [v]; });
      }
      renderTableAndChart();
    }
    renderBadges();
  }

  function renderBadges() {
    document.getElementById("badge-rows").textContent = state.rows.length + " rows";
    document.getElementById("badge-ms").textContent = state.ms + " ms";
    var bj = document.getElementById("badge-joules");
    bj.textContent = state.provenance + ": " + Number(state.joules || 0).toFixed(6) + " J";
    bj.className = state.provenance === "reported_j" ? "badge joules" : "badge est";
    document.getElementById("e-last").textContent = Number(state.joules || 0).toFixed(6) + " J";
    document.getElementById("e-prov").textContent = state.provenance;
    document.getElementById("e-algo").textContent = state.algo || "columnar";
  }

  function formatCell(v) {
    if (v === null || v === undefined) return "NULL";
    if (typeof v === "object") return JSON.stringify(v);
    return String(v);
  }

  function renderTableAndChart() {
    var thead = document.getElementById("res-thead");
    var tbody = document.getElementById("res-tbody");
    thead.innerHTML = "";
    tbody.innerHTML = "";

    if (!state.columns.length) {
      var trEmpty = document.createElement("tr");
      var tdEmpty = document.createElement("td");
      tdEmpty.textContent = "No tabular rows returned.";
      trEmpty.appendChild(tdEmpty);
      tbody.appendChild(trEmpty);
      document.getElementById("chart-container").style.display = "none";
      return;
    }

    var trh = document.createElement("tr");
    state.columns.forEach(function(col, idx) {
      var th = document.createElement("th");
      var arrow = state.sortCol === idx ? (state.sortAsc ? " ▲" : " ▼") : "";
      th.textContent = col + arrow;
      th.addEventListener("click", function() {
        if (state.sortCol === idx) state.sortAsc = !state.sortAsc;
        else { state.sortCol = idx; state.sortAsc = true; }
        renderTableAndChart();
      });
      trh.appendChild(th);
    });
    thead.appendChild(trh);

    var qf = state.filter.toLowerCase();
    var filtered = state.rows.filter(function(r) {
      if (!qf) return true;
      for (var i = 0; i < r.length; i++) {
        if (formatCell(r[i]).toLowerCase().indexOf(qf) !== -1) return true;
      }
      return false;
    });

    if (state.sortCol >= 0) {
      var sc = state.sortCol;
      var dir = state.sortAsc ? 1 : -1;
      filtered = filtered.slice().sort(function(a, b) {
        var va = a[sc], vb = b[sc];
        if (typeof va === "number" && typeof vb === "number") return (va - vb) * dir;
        return String(va ?? "").localeCompare(String(vb ?? "")) * dir;
      });
    }

    filtered.slice(0, 250).forEach(function(r) {
      var tr = document.createElement("tr");
      state.columns.forEach(function(_, ci) {
        var td = document.createElement("td");
        td.textContent = formatCell(r[ci]);
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });

    // Auto-render horizontal bar chart if there is at least one numeric column
    var numCol = -1;
    var labelCol = 0;
    for (var c = 0; c < state.columns.length; c++) {
      var hasNum = filtered.some(function(r) { return typeof r[c] === "number" && isFinite(r[c]); });
      if (hasNum) {
        if (c > 0) { numCol = c; break; }
        if (numCol === -1) numCol = c;
      }
    }
    var chartBox = document.getElementById("chart-container");
    var chartBars = document.getElementById("chart-bars");
    chartBars.innerHTML = "";
    if (numCol !== -1 && filtered.length >= 1) {
      if (numCol === 0 && state.columns.length > 1) labelCol = 1;
      var sample = filtered.slice(0, 8);
      var maxVal = 0;
      sample.forEach(function(r) {
        var v = Math.abs(Number(r[numCol]) || 0);
        if (v > maxVal) maxVal = v;
      });
      if (maxVal > 0) {
        document.getElementById("chart-title").textContent =
          state.columns[numCol] + " by " + state.columns[labelCol] + " (top " + sample.length + ")";
        sample.forEach(function(r) {
          var raw = Number(r[numCol]) || 0;
          var pct = Math.max(2, Math.round((Math.abs(raw) / maxVal) * 100));
          var rowEl = document.createElement("div");
          rowEl.className = "bar-row";
          var lbl = document.createElement("div");
          lbl.className = "bar-label";
          lbl.textContent = numCol === labelCol ? ("Row") : formatCell(r[labelCol]);
          var track = document.createElement("div");
          track.className = "bar-track";
          var fill = document.createElement("div");
          fill.className = "bar-fill";
          fill.style.width = pct + "%";
          track.appendChild(fill);
          var val = document.createElement("div");
          val.className = "bar-val";
          val.textContent = raw.toLocaleString();
          rowEl.appendChild(lbl);
          rowEl.appendChild(track);
          rowEl.appendChild(val);
          chartBars.appendChild(rowEl);
        });
        chartBox.style.display = "block";
      } else {
        chartBox.style.display = "none";
      }
    } else {
      chartBox.style.display = "none";
    }
  }

  function renderSchema() {
    var box = document.getElementById("schema-list");
    box.innerHTML = "";
    if (!state.tables.length) {
      box.innerHTML = '<div class="status-msg">No user tables yet. Run CREATE TABLE in the SQL tab.</div>';
      return;
    }
    state.tables.forEach(function(t) {
      var name = typeof t === "string" ? t : (t.name || t.table_name || JSON.stringify(t));
      var item = document.createElement("div");
      item.className = "list-item";
      var span = document.createElement("span");
      span.textContent = name;
      var btn = document.createElement("button");
      btn.className = "btn";
      btn.textContent = "Preview 25 Rows";
      btn.addEventListener("click", function() {
        var q = 'SELECT * FROM "' + name.replace(/"/g, '""') + '" LIMIT 25;';
        document.getElementById("sql-input").value = q;
        switchTab("sql");
        executeSql(q);
      });
      item.appendChild(span);
      item.appendChild(btn);
      box.appendChild(item);
    });
  }

  function renderBranches() {
    var box = document.getElementById("branches-list");
    box.innerHTML = "";
    if (!state.branches.length) {
      box.innerHTML = '<div class="status-msg">Main trunk active. Create an O(1) CoW branch above to isolate experiments.</div>';
      return;
    }
    state.branches.forEach(function(b) {
      var name = typeof b === "string" ? b : (b.name || b.id || JSON.stringify(b));
      var item = document.createElement("div");
      item.className = "list-item";
      var span = document.createElement("span");
      span.textContent = name;
      var btn = document.createElement("button");
      btn.className = "btn";
      btn.textContent = "Merge";
      btn.addEventListener("click", function() {
        callTool("db.branch_merge", { name: name }).then(function() {
          setStatus("Merged branch: " + name, false);
          refreshBranches();
        }).catch(function(e) { setStatus(e.message, true); });
      });
      item.appendChild(span);
      item.appendChild(btn);
      box.appendChild(item);
    });
  }

  function switchTab(tabId) {
    document.querySelectorAll(".tab").forEach(function(b) {
      b.classList.toggle("active", b.getAttribute("data-tab") === tabId);
    });
    document.querySelectorAll(".panel").forEach(function(p) {
      p.classList.toggle("active", p.id === "panel-" + tabId);
    });
  }

  function executeSql(sql) {
    state.sql = sql;
    setStatus("Executing SQL via MCP tools/call...", false);
    callTool("db.query", { sql: sql }).then(function(res) {
      if (res && res.structuredContent) {
        normalizePayload(res.structuredContent, res);
      }
      setStatus("Executed OK (" + state.rows.length + " rows, " + state.ms + " ms).", false);
    }).catch(function(err) {
      setStatus("SQL error: " + err.message, true);
    });
  }

  function refreshSchema() {
    callTool("db.list_tables", {}).then(function(res) {
      if (res && res.structuredContent) normalizePayload(res.structuredContent, res);
      setStatus("Refreshed table catalog.", false);
    }).catch(function(e) { setStatus(e.message, true); });
  }

  function refreshBranches() {
    callTool("db.branch_list", {}).then(function(res) {
      if (res && res.structuredContent) normalizePayload(res.structuredContent, res);
      setStatus("Refreshed CoW branch list.", false);
    }).catch(function(e) { setStatus(e.message, true); });
  }

  // Wire UI controls
  document.querySelectorAll(".tab").forEach(function(btn) {
    btn.addEventListener("click", function() { switchTab(btn.getAttribute("data-tab")); });
  });
  document.getElementById("btn-run-sql").addEventListener("click", function() {
    executeSql(document.getElementById("sql-input").value);
  });
  document.getElementById("btn-explain").addEventListener("click", function() {
    var sql = document.getElementById("sql-input").value;
    callTool("db.explain_energy", { sql: sql }).then(function(res) {
      if (res && res.structuredContent) normalizePayload(res.structuredContent, res);
      switchTab("energy");
      setStatus("Computed pre-execution analytical energy estimate (est_j).", false);
    }).catch(function(e) { setStatus(e.message, true); });
  });
  document.getElementById("row-filter").addEventListener("input", function(ev) {
    state.filter = ev.target.value || "";
    renderTableAndChart();
  });
  document.getElementById("btn-refresh-schema").addEventListener("click", refreshSchema);
  document.getElementById("btn-refresh-branches").addEventListener("click", refreshBranches);
  document.getElementById("btn-create-branch").addEventListener("click", function() {
    var name = (document.getElementById("branch-name").value || "").trim();
    if (!name) { setStatus("Enter a branch name first.", true); return; }
    var args = { name: name };
    var b = parseInt(document.getElementById("branch-budget").value, 10);
    if (!isNaN(b) && b > 0) args.energy_budget_uj = b;
    callTool("db.branch_create", args).then(function() {
      document.getElementById("branch-name").value = "";
      setStatus("Created O(1) CoW branch: " + name, false);
      refreshBranches();
    }).catch(function(e) { setStatus(e.message, true); });
  });
  document.getElementById("btn-set-budget").addEventListener("click", function() {
    var b = parseInt(document.getElementById("budget-input").value, 10);
    if (isNaN(b) || b < 0) { setStatus("Enter a valid microjoule budget.", true); return; }
    callTool("db.energy_budget", { budget_uj: b }).then(function() {
      document.getElementById("e-budget").textContent = b.toLocaleString() + " uJ";
      setStatus("Updated session energy budget.", false);
    }).catch(function(e) { setStatus(e.message, true); });
  });
  document.getElementById("btn-send-msg").addEventListener("click", function() {
    var summary = "JouleDB Studio query (`" + state.sql + "`) returned " +
      state.rows.length + " rows in " + state.ms + " ms (" + state.provenance + ": " +
      Number(state.joules || 0).toFixed(6) + " J).";
    sendRequest("ui/message", {
      role: "user",
      content: [{ type: "text", text: summary }]
    }).then(function() { setStatus("Sent summary to conversation.", false); })
      .catch(function(e) { setStatus(e.message, true); });
  });
  document.getElementById("btn-pin-ctx").addEventListener("click", function() {
    var ctxText = "Active JouleDB SQL: " + state.sql + "\nColumns: " + state.columns.join(", ") +
      "\nRow count: " + state.rows.length;
    sendRequest("ui/update-model-context", {
      content: [{ type: "text", text: ctxText }]
    }).then(function() { setStatus("Pinned query context to next message.", false); })
      .catch(function(e) { setStatus(e.message, true); });
  });

  // Listen for JSON-RPC responses & host notifications BEFORE calling ui/initialize
  window.addEventListener("message", function(event) {
    var msg = event.data;
    if (!msg || typeof msg !== "object" || msg.jsonrpc !== "2.0") return;

    if (msg.id !== undefined && msg.method === undefined) {
      var entry = pending.get(msg.id);
      if (entry) {
        pending.delete(msg.id);
        if (msg.error) entry.reject(new Error(msg.error.message || "MCP RPC error"));
        else entry.resolve(msg.result);
      }
      return;
    }

    if (msg.method === "ping") {
      postRpc({ jsonrpc: "2.0", id: msg.id, result: {} });
      return;
    }
    if (msg.method === "ui/notifications/tool-input") {
      var args = (msg.params && msg.params.arguments) || {};
      if (args.sql) {
        state.sql = args.sql;
        document.getElementById("sql-input").value = args.sql;
      }
      if (args.tab) switchTab(args.tab);
      setStatus("Running tool with input...", false);
    } else if (msg.method === "ui/notifications/tool-result") {
      var res = msg.params || {};
      if (res.structuredContent) {
        normalizePayload(res.structuredContent, res);
      }
      setStatus("Ready (" + state.rows.length + " rows).", false);
    } else if (msg.method === "ui/notifications/host-context-changed") {
      applyHostContext(msg.params);
    }
  });

  // Auto-resize notification (`SEP-1865`)
  if (typeof ResizeObserver !== "undefined") {
    var ro = new ResizeObserver(function() {
      var rect = document.getElementById("app-root").getBoundingClientRect();
      sendNotification("ui/notifications/size-changed", {
        width: Math.ceil(rect.width),
        height: Math.ceil(rect.height + 24)
      });
    });
    ro.observe(document.getElementById("app-root"));
  }

  // Perform `ui/initialize` handshake
  sendRequest("ui/initialize", {
    protocolVersion: "2026-01-26",
    appInfo: { name: "JouleDB Studio", version: "0.10.0" },
    appCapabilities: {}
  }).then(function(initRes) {
    if (initRes) {
      hostCaps = initRes.hostCapabilities || {};
      applyHostContext(initRes.hostContext);
      updateCapButtons();
    }
    sendNotification("ui/notifications/initialized", {});
    setStatus("Connected to MCP host. Ready.", false);
  }).catch(function() {
    setStatus("Standalone preview mode (no parent MCP host frame).", false);
  });
})();
</script>
</body>
</html>
"#;
