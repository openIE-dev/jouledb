export declare const VERSION: string;

export interface JouleDBClientOptions {
  token?: string;
  fetch?: typeof fetch;
}

export interface QueryResponse {
  columns: string[];
  rows: unknown[][];
  affected_rows?: number | null;
  execution_time_ms: number;
  energy_joules?: number | null;
  device_target?: string | null;
  energy_receipt?: Record<string, unknown> | null;
}

export interface MigrationCutoverCertificate {
  migration_id: string;
  source_format: string;
  tables_or_collections: string[];
  snapshot_rows_ingested: number;
  cdc_events_applied: number;
  final_reconciled_rows: number;
  source_multiset_blake3: string;
  target_multiset_blake3: string;
  zero_row_drift: boolean;
  shadow_queries_checked: number;
  shadow_parity_rate: number;
  cutover_write_freeze_ms: number;
  pq_signature_verified: boolean;
}

export declare class JouleDBError extends Error {
  status: number;
  constructor(message: string, status?: number);
}

export declare class JouleDBClient {
  url: string;
  token: string | null;
  constructor(url?: string, options?: JouleDBClientOptions);
  query(sql: string, params?: Record<string, unknown>): Promise<QueryResponse>;
  unsilo(fragments?: unknown[], intent?: string): Promise<Record<string, unknown>>;
  morphRules(
    fragments: unknown[],
    morphDsl: string,
    customLensDsl?: string | null
  ): Promise<Record<string, unknown>>;
  migrateRun(
    sourceFormat: string,
    payload: string,
    collectionOrTable?: string,
    cdcEvents?: unknown[]
  ): Promise<MigrationCutoverCertificate>;
  migrateCertify(): Promise<Record<string, unknown>>;
  trustAudit(): Promise<Record<string, unknown>>;
  sbom(): Promise<Record<string, unknown>>;
  bridgeMatrix(): Promise<Record<string, unknown>>;
  chat(prompt: string, model?: string): Promise<Record<string, unknown>>;
  powerhouseSuite(): Promise<Record<string, unknown>>;
}

export declare function connect(
  url?: string,
  options?: JouleDBClientOptions
): JouleDBClient;
