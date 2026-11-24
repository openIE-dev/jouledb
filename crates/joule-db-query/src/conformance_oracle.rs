//! Portable `sqllogictest` (`.slt`) Conformance Runner & `SQLancer` (`TLP` / `NoREC`) Oracles.
//!
//! Closes the external SQL dialect conformance & logic-bug oracle gap inside Scope-A (`joule-db-query`)
//! so every CI push and HTTP verification call can validate:
//! 1. **Standard SQLite / DuckDB `.slt` scripts (`run_sqllogictest_script`):** Supports `statement ok`,
//!    `statement error`, `query <I|R|T...> <nosort|rowsort|valuesort>`, `----` literal expectations,
//!    and `N values hashing to <hex>` deterministic digest expectations.
//! 2. **`SQLancer` `TLP` (Ternary Logic Partitioning) & `NoREC` (Non-Optimizing Reference Engine):**
//!    Detects silent wrong-result SQL optimizer/null-handling bugs without an external reference DB.

use serde::{Deserialize, Serialize};

/// Sort mode specified on a `sqllogictest` `query` directive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SltSortMode {
    NoSort,
    RowSort,
    ValueSort,
}

/// Failure witness from a `.slt` record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SltFailure {
    pub record_index: usize,
    pub sql: String,
    pub reason: String,
}

/// Summary receipt from executing a `.slt` script.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SltConformanceReceipt {
    pub total_records: usize,
    pub statements_passed: usize,
    pub queries_passed: usize,
    pub failed_records: usize,
    pub pass_rate: f64,
    pub failures: Vec<SltFailure>,
}

/// Normalize a single cell according to `sqllogictest` `I` / `R` / `T` type conventions.
pub fn normalize_slt_cell(raw: &str, col_type: char) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("null") {
        return "NULL".to_string();
    }
    match col_type {
        'I' => {
            if let Ok(i) = trimmed.parse::<i64>() {
                i.to_string()
            } else if let Ok(f) = trimmed.parse::<f64>() {
                (f.trunc() as i64).to_string()
            } else if trimmed.eq_ignore_ascii_case("true") {
                "1".to_string()
            } else if trimmed.eq_ignore_ascii_case("false") {
                "0".to_string()
            } else {
                trimmed.to_string()
            }
        }
        'R' => {
            if let Ok(f) = trimmed.parse::<f64>() {
                let clean = if f == 0.0 { 0.0 } else { f };
                format!("{clean:.3}")
            } else {
                trimmed.to_string()
            }
        }
        _ => trimmed.to_string(),
    }
}

/// Pure-Rust RFC-1321 MD5 hex digest used exclusively for `sqllogictest` `"N values hashing to <md5>"` compatibility.
fn slt_md5_hex(input: &[u8]) -> String {
    const S: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5,
        9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10,
        15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];
    const K: [u32; 64] = [
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613,
        0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193,
        0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d,
        0x02441453, 0xd8a1e681, 0xe7d3fbc8, 0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
        0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122,
        0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa,
        0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665, 0xf4292244,
        0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
        0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb,
        0xeb86d391,
    ];

    let mut msg = input.to_vec();
    let bit_len = (msg.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_le_bytes());

    let mut a0: u32 = 0x67452301;
    let mut b0: u32 = 0xefcdab89;
    let mut c0: u32 = 0x98badcfe;
    let mut d0: u32 = 0x10325476;

    for chunk in msg.chunks_exact(64) {
        let mut m = [0u32; 16];
        for (i, word) in m.iter_mut().enumerate() {
            let mut b = [0u8; 4];
            b.copy_from_slice(&chunk[i * 4..(i + 1) * 4]);
            *word = u32::from_le_bytes(b);
        }
        let (mut a, mut b, mut c, mut d) = (a0, b0, c0, d0);
        for i in 0..64 {
            let (f, g) = if i < 16 {
                ((b & c) | ((!b) & d), i)
            } else if i < 32 {
                ((d & b) | ((!d) & c), (5 * i + 1) % 16)
            } else if i < 48 {
                (b ^ c ^ d, (3 * i + 5) % 16)
            } else {
                (c ^ (b | (!d)), (7 * i) % 16)
            };
            let temp = d;
            d = c;
            c = b;
            let sum = a.wrapping_add(f).wrapping_add(K[i]).wrapping_add(m[g]);
            b = b.wrapping_add(sum.rotate_left(S[i]));
            a = temp;
        }
        a0 = a0.wrapping_add(a);
        b0 = b0.wrapping_add(b);
        c0 = c0.wrapping_add(c);
        d0 = d0.wrapping_add(d);
    }

    let mut out = String::with_capacity(32);
    for v in [a0, b0, c0, d0] {
        for byte in v.to_le_bytes() {
            out.push_str(&format!("{byte:02x}"));
        }
    }
    out
}

/// Execute a portable `.slt` (`sqllogictest`) script against any SQL engine callback returning
/// `Result<Vec<Vec<String>>, String>`.
pub fn run_sqllogictest_script<F>(script: &str, mut exec_sql: F) -> SltConformanceReceipt
where
    F: FnMut(&str) -> Result<Vec<Vec<String>>, String>,
{
    let lines: Vec<&str> = script.lines().collect();
    let mut idx = 0usize;
    let mut total_records = 0usize;
    let mut statements_passed = 0usize;
    let mut queries_passed = 0usize;
    let mut failures = Vec::new();

    while idx < lines.len() {
        let header = lines[idx].trim();
        if header.is_empty()
            || header.starts_with('#')
            || header.starts_with("hash-threshold")
            || header.starts_with("halt")
        {
            if header.starts_with("halt") {
                break;
            }
            idx += 1;
            continue;
        }
        if header.starts_with("onlyif ") || header.starts_with("skipif ") {
            let is_onlyif = header.starts_with("onlyif ");
            let target = header.split_whitespace().nth(1).unwrap_or("");
            let matches_us = target.eq_ignore_ascii_case("jouledb")
                || target.eq_ignore_ascii_case("postgresql")
                || target.eq_ignore_ascii_case("sqlite");
            idx += 1;
            if (is_onlyif && !matches_us) || (!is_onlyif && matches_us) {
                while idx < lines.len() && !lines[idx].trim().is_empty() {
                    idx += 1;
                }
                continue;
            }
            continue;
        }

        if header.starts_with("statement ") {
            total_records += 1;
            let expect_ok = header.contains("statement ok");
            idx += 1;
            let mut sql_lines = Vec::new();
            while idx < lines.len() && !lines[idx].trim().is_empty() {
                sql_lines.push(lines[idx]);
                idx += 1;
            }
            let sql = sql_lines.join("\n");
            let res = exec_sql(&sql);
            match (expect_ok, res) {
                (true, Ok(_)) | (false, Err(_)) => statements_passed += 1,
                (true, Err(e)) => failures.push(SltFailure {
                    record_index: total_records,
                    sql,
                    reason: format!("expected statement ok, got error: {e}"),
                }),
                (false, Ok(_)) => failures.push(SltFailure {
                    record_index: total_records,
                    sql,
                    reason: "expected statement error, but statement succeeded".to_string(),
                }),
            }
        } else if header.starts_with("query ") {
            total_records += 1;
            let parts: Vec<&str> = header.split_whitespace().collect();
            let col_spec: Vec<char> = parts.get(1).copied().unwrap_or("T").chars().collect();
            let sort_mode = match parts.get(2).copied().unwrap_or("nosort") {
                "rowsort" => SltSortMode::RowSort,
                "valuesort" => SltSortMode::ValueSort,
                _ => SltSortMode::NoSort,
            };
            idx += 1;
            let mut sql_lines = Vec::new();
            while idx < lines.len() && lines[idx].trim() != "----" && !lines[idx].trim().is_empty()
            {
                sql_lines.push(lines[idx]);
                idx += 1;
            }
            if idx < lines.len() && lines[idx].trim() == "----" {
                idx += 1;
            }
            let mut expected_lines = Vec::new();
            while idx < lines.len() && !lines[idx].trim().is_empty() {
                expected_lines.push(lines[idx].trim().to_string());
                idx += 1;
            }
            let sql = sql_lines.join("\n");
            match exec_sql(&sql) {
                Err(e) => failures.push(SltFailure {
                    record_index: total_records,
                    sql,
                    reason: format!("query error: {e}"),
                }),
                Ok(rows) => {
                    let mut norm_rows: Vec<Vec<String>> = rows
                        .into_iter()
                        .map(|r| {
                            r.into_iter()
                                .enumerate()
                                .map(|(ci, cell)| {
                                    let ct = col_spec.get(ci).copied().unwrap_or('T');
                                    normalize_slt_cell(&cell, ct)
                                })
                                .collect()
                        })
                        .collect();
                    if sort_mode == SltSortMode::RowSort {
                        norm_rows.sort_by(|a, b| a.join("\t").cmp(&b.join("\t")));
                    }
                    let mut flat_vals: Vec<String> = norm_rows.into_iter().flatten().collect();
                    if sort_mode == SltSortMode::ValueSort {
                        flat_vals.sort();
                    }

                    let matched = if expected_lines.len() == 1
                        && expected_lines[0].contains("values hashing to ")
                    {
                        let wp: Vec<&str> = expected_lines[0].split_whitespace().collect();
                        let exp_cnt = wp
                            .first()
                            .and_then(|s| s.parse::<usize>().ok())
                            .unwrap_or(0);
                        let exp_hash = wp.last().copied().unwrap_or("");
                        let mut joined = flat_vals.join("\n");
                        if !joined.is_empty() {
                            joined.push('\n');
                        }
                        let got_hash = slt_md5_hex(joined.as_bytes());
                        flat_vals.len() == exp_cnt && got_hash == exp_hash
                    } else {
                        // Support both value-per-line AND tab/whitespace-per-row `.slt` expectations
                        let exp_flat: Vec<String> = if expected_lines.len() == flat_vals.len() {
                            expected_lines.clone()
                        } else {
                            expected_lines
                                .iter()
                                .flat_map(|l| l.split('\t').map(|s| s.trim().to_string()))
                                .collect()
                        };
                        flat_vals == exp_flat
                    };

                    if matched {
                        queries_passed += 1;
                    } else {
                        failures.push(SltFailure {
                            record_index: total_records,
                            sql,
                            reason: format!(
                                "result mismatch: got {:?}, expected {:?}",
                                flat_vals, expected_lines
                            ),
                        });
                    }
                }
            }
        } else {
            idx += 1;
        }
    }

    let passed = statements_passed + queries_passed;
    let pass_rate = if total_records == 0 {
        1.0
    } else {
        (passed as f64) / (total_records as f64)
    };
    SltConformanceReceipt {
        total_records,
        statements_passed,
        queries_passed,
        failed_records: failures.len(),
        pass_rate,
        failures,
    }
}

/// Receipt from running `SQLancer` `TLP` (Ternary Logic Partitioning) and `NoREC` differential checks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SqlancerOracleReceipt {
    pub predicates_tested: usize,
    pub tlp_passed: usize,
    pub norec_passed: usize,
    pub violations: Vec<String>,
}

/// Run `SQLancer` `TLP` (`Q(P) + Q(NOT P) + Q(P IS NULL) == Q(ALL)`) and `NoREC`
/// (`SELECT COUNT(*) FROM t WHERE P == SELECT COALESCE(SUM(CASE WHEN P THEN 1 ELSE 0 END), 0) FROM t`)
/// across a suite of predicates on `table`.
pub fn run_sqlancer_tlp_and_norec_suite<F>(
    table: &str,
    predicates: &[&str],
    mut exec_scalar_i64: F,
) -> SqlancerOracleReceipt
where
    F: FnMut(&str) -> Result<i64, String>,
{
    let mut tlp_passed = 0usize;
    let mut norec_passed = 0usize;
    let mut violations = Vec::new();

    let total_all = match exec_scalar_i64(&format!("SELECT COUNT(*) FROM {table}")) {
        Ok(c) => c,
        Err(e) => {
            return SqlancerOracleReceipt {
                predicates_tested: predicates.len(),
                tlp_passed: 0,
                norec_passed: 0,
                violations: vec![format!("base count failed: {e}")],
            };
        }
    };

    for &pred in predicates {
        // 1. TLP Oracle: 3-valued partition must sum identically to unpredicated `COUNT(*)`
        let q_true = format!("SELECT COUNT(*) FROM {table} WHERE ({pred})");
        let q_false = format!("SELECT COUNT(*) FROM {table} WHERE NOT ({pred})");
        let q_null = format!("SELECT COUNT(*) FROM {table} WHERE ({pred}) IS NULL");

        match (
            exec_scalar_i64(&q_true),
            exec_scalar_i64(&q_false),
            exec_scalar_i64(&q_null),
        ) {
            (Ok(ct), Ok(cf), Ok(cn)) => {
                if ct + cf + cn == total_all {
                    tlp_passed += 1;
                } else {
                    violations.push(format!(
                        "TLP violation on '{pred}': true({ct}) + false({cf}) + null({cn}) != total({total_all})"
                    ));
                }

                // 2. NoREC Oracle: optimized `WHERE` vs unoptimized full-scan `CASE WHEN`
                let q_norec = format!(
                    "SELECT COALESCE(SUM(CASE WHEN ({pred}) THEN 1 ELSE 0 END), 0) FROM {table}"
                );
                match exec_scalar_i64(&q_norec) {
                    Ok(cnorec) if cnorec == ct => norec_passed += 1,
                    Ok(cnorec) => violations.push(format!(
                        "NoREC violation on '{pred}': WHERE count={ct} != CASE scan={cnorec}"
                    )),
                    Err(e) => violations.push(format!("NoREC query error on '{pred}': {e}")),
                }
            }
            (r1, r2, r3) => {
                violations.push(format!("TLP query error on '{pred}': {r1:?} {r2:?} {r3:?}"));
            }
        }
    }

    SqlancerOracleReceipt {
        predicates_tested: predicates.len(),
        tlp_passed,
        norec_passed,
        violations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slt_md5_and_runner_conformance() {
        // RFC-1321 test vector: MD5("abc") == "900150983cd24fb0d6963f7d28e17f72"
        assert_eq!(slt_md5_hex(b"abc"), "900150983cd24fb0d6963f7d28e17f72");

        let script = "\
statement ok
CREATE TABLE t1(a INT, b TEXT)

query IT rowsort
SELECT a, b FROM t1
----
1
alpha
2
beta
";
        let rec = run_sqllogictest_script(script, |sql| {
            if sql.starts_with("CREATE") {
                Ok(Vec::new())
            } else {
                Ok(vec![
                    vec!["2".to_string(), "beta".to_string()],
                    vec!["1".to_string(), "alpha".to_string()],
                ])
            }
        });
        assert_eq!(rec.total_records, 2);
        assert_eq!(rec.failed_records, 0);
        assert!((rec.pass_rate - 1.0).abs() < 1e-9);
    }
}
