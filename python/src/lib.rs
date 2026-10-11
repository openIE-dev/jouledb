//! Native decode for the jouledb SDK: parse a JouleDB HTTP query response
//! and transpose its rows into columns with the GIL released, so Python
//! only hands finished column lists to pyarrow.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde_json::Value;

/// Scalar cell, decided off the GIL.
enum Cell {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Json(String),
}

fn cell(v: &Value) -> Cell {
    match v {
        Value::Null => Cell::Null,
        Value::Bool(b) => Cell::Bool(*b),
        Value::Number(n) => match n.as_i64() {
            Some(i) => Cell::Int(i),
            None => Cell::Float(n.as_f64().unwrap_or(f64::NAN)),
        },
        Value::String(s) => Cell::Str(s.clone()),
        other => Cell::Json(other.to_string()),
    }
}

fn to_py(py: Python<'_>, v: &Value) -> PyObject {
    match v {
        Value::Null => py.None(),
        Value::Bool(b) => b.into_py(py),
        Value::Number(n) => match n.as_i64() {
            Some(i) => i.into_py(py),
            None => n.as_f64().unwrap_or(f64::NAN).into_py(py),
        },
        Value::String(s) => s.into_py(py),
        Value::Array(a) => PyList::new_bound(py, a.iter().map(|x| to_py(py, x))).into_py(py),
        Value::Object(m) => {
            let d = PyDict::new_bound(py);
            for (k, x) in m {
                let _ = d.set_item(k, to_py(py, x));
            }
            d.into_py(py)
        }
    }
}

/// decode(body: bytes) -> (columns: list[str], data: list[list], receipt: dict, affected: int|None)
///
/// Raises RuntimeError with the server's message on an error response.
#[pyfunction]
fn decode(py: Python<'_>, body: &[u8]) -> PyResult<(Vec<String>, PyObject, PyObject, Option<u64>)> {
    let owned = body.to_vec();
    let parsed: Result<(Value, Vec<String>, Vec<Vec<Cell>>), String> = py.allow_threads(move || {
        let v: Value = serde_json::from_slice(&owned).map_err(|e| format!("bad response: {e}"))?;
        if let Some(err) = v.get("error").and_then(Value::as_str) {
            return Err(format!("server: {err}"));
        }
        let cols: Vec<String> = v
            .get("columns")
            .and_then(Value::as_array)
            .map(|a| a.iter().map(|c| c.as_str().unwrap_or("").to_string()).collect())
            .unwrap_or_default();
        let rows = v
            .get("rows")
            .or_else(|| v.get("data"))
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let mut out: Vec<Vec<Cell>> = (0..cols.len()).map(|_| Vec::with_capacity(rows.len())).collect();
        for r in &rows {
            let r = r.as_array().ok_or("row is not an array")?;
            for (c, col) in out.iter_mut().enumerate() {
                col.push(r.get(c).map(cell).unwrap_or(Cell::Null));
            }
        }
        Ok((v, cols, out))
    });
    let (v, cols, out) = parsed.map_err(|e| {
        if e.starts_with("server: ") {
            PyRuntimeError::new_err(e[8..].to_string())
        } else {
            PyValueError::new_err(e)
        }
    })?;
    let data = PyList::empty_bound(py);
    for col in out {
        let items = col.into_iter().map(|c| match c {
            Cell::Null => py.None(),
            Cell::Bool(b) => b.into_py(py),
            Cell::Int(i) => i.into_py(py),
            Cell::Float(f) => f.into_py(py),
            Cell::Str(s) | Cell::Json(s) => s.into_py(py),
        });
        data.append(PyList::new_bound(py, items))?;
    }
    let receipt = PyDict::new_bound(py);
    for k in [
        "energy_joules",
        "power_watts",
        "device_target",
        "algorithm_type",
        "execution_time_ms",
        "energy_receipt",
    ] {
        if let Some(x) = v.get(k) {
            receipt.set_item(k, to_py(py, x))?;
        }
    }
    let affected = v.get("affected_rows").and_then(Value::as_u64);
    Ok((cols, data.into_py(py), receipt.into_py(py), affected))
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(decode, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
