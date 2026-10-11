# Benchmarks: time and joules per query, JouleDB vs DuckDB

`harness.py` loads one deterministic dataset into both engines (JouleDB through pgwire `COPY`, DuckDB through `COPY`), checks that every answer matches, and runs each query back-to-back for at least `--min-seconds`. It reports ms/query, whole-machine J/query, and J/query above the idle baseline. The meter is `powermetrics` (macOS, needs `sudo -n`), RAPL package plus NVIDIA GPU energy (Linux), or `none`. Time is always reported.

The dataset is a synthetic, seeded, ClickBench-style `hits` table. It is not the official ClickBench file, and TPC-H and ann-benchmarks are not wired yet.

    bash bench/fleet.sh           # on each host; ROWS=10000 by default
