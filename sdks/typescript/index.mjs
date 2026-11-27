/**
 * @openie/jouledb v0.13.0
 * Universal Fetch-native client for Node.js (>=18), Deno, Bun, Cloudflare Workers & Browsers.
 */

export const VERSION = "0.13.0";

export class JouleDBError extends Error {
  constructor(message, status = 0) {
    super(message);
    this.name = "JouleDBError";
    this.status = status;
  }
}

export class JouleDBClient {
  constructor(url = "http://127.0.0.1:8080", options = {}) {
    this.url = url.replace(/\/+$/, "");
    this.token = options.token ?? null;
    this.fetchImpl = options.fetch ?? globalThis.fetch;
  }

  async _request(method, path, body = undefined) {
    const headers = { "accept": "application/json" };
    if (body !== undefined) {
      headers["content-type"] = "application/json";
    }
    if (this.token) {
      headers["authorization"] = `Bearer ${this.token}`;
    }
    const resp = await this.fetchImpl(`${this.url}${path}`, {
      method,
      headers,
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
    const text = await resp.text();
    let parsed;
    try {
      parsed = text ? JSON.parse(text) : {};
    } catch {
      parsed = { raw: text };
    }
    if (!resp.ok) {
      throw new JouleDBError(parsed?.error || `HTTP ${resp.status}`, resp.status);
    }
    return parsed;
  }

  /** Execute SQL and return rows, columns, and the 6-rail nanojoule receipt. */
  async query(sql, params = {}) {
    return this._request("POST", "/api/v1/query", { sql, params });
  }

  /** Dissolve heterogeneous JSON/CSV/XML/Prose/Triples into the 10,240-bit Amorphic Continuum. */
  async unsilo(fragments = [], intent = "BalancedOverview") {
    return this._request("POST", "/api/v1/amorphic/unsilo", { fragments, intent });
  }

  /** Apply programmable MorphRule DSL + Custom Lens compilation. */
  async morphRules(fragments, morphDsl, customLensDsl = null) {
    return this._request("POST", "/api/v1/amorphic/morph_rules", {
      fragments,
      morph_dsl: morphDsl,
      custom_lens_dsl: customLensDsl,
    });
  }

  /** Run zero-downtime migration with multiset BLAKE3 reconciliation & ML-DSA-65 cutover certificate. */
  async migrateRun(sourceFormat, payload, collectionOrTable = "migrated", cdcEvents = []) {
    return this._request("POST", "/api/v1/migrate/run", {
      source_format: sourceFormat,
      collection_or_table: collectionOrTable,
      payload,
      cdc_events: cdcEvents,
    });
  }

  /** Certify zero-downtime migration across PostgreSQL, MongoDB, Elasticsearch & CSV. */
  async migrateCertify() {
    return this._request("POST", "/api/v1/migrate/certify", {});
  }

  /** Execute continuous SOC 2 Type II, HIPAA, GDPR Art.17, SEC 17a-4(f) & FedRAMP/MLS trust audit. */
  async trustAudit() {
    return this._request("POST", "/api/v1/trust/compliance_audit", {});
  }

  /** Fetch CycloneDX v1.6 SBOM + .jsyn AI-BOM. */
  async sbom() {
    return this._request("GET", "/api/v1/trust/sbom");
  }

  /** Inspect the 10-DB + 7-AI Universal Ecosystem Bridge matrix. */
  async bridgeMatrix() {
    return this._request("GET", "/api/v1/bridge/matrix");
  }

  /** Run OpenAI-compatible 4-tier thermodynamic chat completion over .jsyn bitplanes. */
  async chat(prompt, model = "jouledb-cascade-v1") {
    return this._request("POST", "/v1/chat/completions", {
      model,
      messages: [{ role: "user", content: prompt }],
    });
  }

  /** Run the 6 fast in-process subsystem smoke checks (`jouledb self-check`). */
  async powerhouseSuite() {
    return this._request("POST", "/api/v1/bench/v1_powerhouse", {});
  }
}

export function connect(url = "http://127.0.0.1:8080", options = {}) {
  return new JouleDBClient(url, options);
}
