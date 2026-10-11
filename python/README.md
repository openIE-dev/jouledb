# jouledb (Python SDK)

Query JouleDB from Python and get Arrow results back, plus the energy receipt.

```python
import jouledb
db = jouledb.connect("http://localhost:8080")
r = db.query("SELECT region, SUM(v) AS s FROM t GROUP BY region")
r.arrow        # pyarrow.Table
r.to_pandas()  # pandas.DataFrame
r.to_polars()  # polars.DataFrame (zero-copy from Arrow)
r.receipt      # {'energy_joules': 0.04, 'power_watts': ..., 'device_target': 'cpu', 'algorithm_type': 'columnar', ...}
```

The package talks to the server's HTTP API (`POST /api/v1/query`). A small
Rust extension (PyO3, abi3, Python 3.9+) parses the response and turns rows
into columns with the GIL released.

Build and test from source:

```sh
pip install maturin
maturin develop --release -m python/Cargo.toml
jouledb -p 8080 &
JOULEDB_URL=http://localhost:8080 pytest python/tests
```

Not on PyPI yet. CI builds the wheels as artifacts only.
