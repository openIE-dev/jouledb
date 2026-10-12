#!/usr/bin/env python3
"""Reproducible time + joules per query: JouleDB (pgwire) vs DuckDB, same machine.

Dataset: a deterministic ClickBench-style `hits` table (synthetic, seeded; NOT the
official 100M-row ClickBench file). Queries are standard SQL aggregates that both
engines run unchanged; results are cross-checked.

Energy (whole machine, both engines measured the same way):
  macOS : `sudo -n powermetrics` Combined Power (CPU+GPU+ANE), ~100 ms samples
  Linux : RAPL package energy counters (/sys/class/powercap/intel-rapl:*), plus
          NVML GPU energy when an NVIDIA GPU is present (nvidia-smi query)
  else  : energy_source=none (time is still reported; never a refusal)
Each query runs back-to-back for >= --min-seconds; J/query = (window joules) / runs.
incremental_J subtracts an idle baseline measured before the run.
"""
import argparse, csv, json, os, platform, random, re, shutil, statistics, subprocess, sys, threading, time

QUERIES = {
    "Q1_count": "SELECT count(*) FROM hits",
    "Q2_filter_count": "SELECT count(*) FROM hits WHERE region = 7",
    "Q3_sum_avg": "SELECT sum(duration), avg(duration) FROM hits",
    "Q4_group_region": "SELECT region, count(*) AS c FROM hits GROUP BY region ORDER BY c DESC, region LIMIT 10",
    "Q5_distinct_users": "SELECT count(DISTINCT user_id) FROM hits",
    "Q6_top_users": "SELECT user_id, sum(duration) AS s FROM hits GROUP BY user_id ORDER BY s DESC, user_id LIMIT 10",
    "Q7_like_group": "SELECT url, count(*) AS c FROM hits WHERE url LIKE '%/search%' GROUP BY url ORDER BY c DESC, url LIMIT 5",
    "Q8_min_max": "SELECT min(ts), max(ts) FROM hits",
}

# ---------------------------------------------------------------- energy meters
class Meter:
    source = "none"
    def start(self): pass
    def joules(self, t0, t1): return None
    def stop(self): pass

class Rapl(Meter):
    def __init__(self):
        base = "/sys/class/powercap"
        self.zones = []
        for d in sorted(os.listdir(base)) if os.path.isdir(base) else []:
            if re.fullmatch(r"intel-rapl:\d+", d):
                p = os.path.join(base, d)
                try:
                    open(os.path.join(p, "energy_uj")).read()
                    self.zones.append((p, int(open(os.path.join(p, "max_energy_range_uj")).read())))
                except OSError:
                    pass
        self.nvsmi = shutil.which("nvidia-smi")
        self.source = "rapl-package" + ("+nvml" if self.nvsmi and self._gpu_mj() is not None else "") if self.zones else "none"
    def _gpu_mj(self):
        if not self.nvsmi: return None
        try:
            out = subprocess.run([self.nvsmi, "--query-gpu=total_energy_consumption", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
            return sum(float(x) for x in out) if out else None
        except Exception:
            return None
    def read(self):
        cpu = sum(int(open(os.path.join(p, "energy_uj")).read()) for p, _ in self.zones)
        return cpu, self._gpu_mj() if "nvml" in self.source else None
    def delta(self, a, b):
        (c0, g0), (c1, g1) = a, b
        dc = c1 - c0
        if dc < 0: dc += sum(m for _, m in self.zones)
        j = dc / 1e6
        if g0 is not None and g1 is not None: j += (g1 - g0) / 1e3
        return j

class PowerMetrics(Meter):
    source = "powermetrics-combined"
    def __init__(self):
        self.samples, self.proc, self.ok = [], None, False
        try:
            self.ok = subprocess.run(["sudo", "-n", "true"], capture_output=True, timeout=5).returncode == 0
        except Exception:
            pass
        if not self.ok: self.source = "none (sudo -n powermetrics needs a password)"
    def start(self):
        if not self.ok: return
        self.proc = subprocess.Popen(["sudo", "-n", "powermetrics", "-i", "100", "--samplers", "cpu_power,gpu_power,ane_power"],
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        def pump():
            for line in self.proc.stdout:
                m = re.search(r"Combined Power \(CPU \+ GPU \+ ANE\):\s*([\d.]+)\s*mW", line)
                if m: self.samples.append((time.monotonic(), float(m.group(1)) / 1000.0))
        threading.Thread(target=pump, daemon=True).start()
        t = time.monotonic()
        while not self.samples and time.monotonic() - t < 5: time.sleep(0.05)
    def joules(self, t0, t1):
        # each sample covers the 100 ms before its arrival
        time.sleep(0.25)
        e = 0.0
        for ts, w in list(self.samples):
            a, b = max(ts - 0.1, t0), min(ts, t1)
            if b > a: e += w * (b - a)
        return e if self.ok else None
    def stop(self):
        if self.proc:
            subprocess.run(["sudo", "-n", "kill", str(self.proc.pid)], capture_output=True)

def make_meter():
    if sys.platform == "darwin": return PowerMetrics()
    r = Rapl()
    return r if r.zones else Meter()

class Window:
    """Energy over a wall-clock window with whichever meter is active."""
    def __init__(self, m): self.m = m
    def __enter__(self):
        self.t0 = time.monotonic()
        self.c0 = self.m.read() if isinstance(self.m, Rapl) else None
        return self
    def __exit__(self, *a):
        self.t1 = time.monotonic()
        if isinstance(self.m, Rapl): self.j = self.m.delta(self.c0, self.m.read())
        else: self.j = self.m.joules(self.t0, self.t1)

# ---------------------------------------------------------------- data
def gen_csv(path, rows, seed=42):
    rnd = random.Random(seed)
    paths = ["/", "/search", "/search?q=joule", "/news", "/cart", "/item/1", "/item/2", "/search?q=db", "/about", "/help"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for i in range(rows):
            w.writerow([i, rnd.randrange(rows // 20 + 1), rnd.randrange(50), "https://example.com" + rnd.choice(paths),
                        round(rnd.random() * 100, 3), 1_700_000_000 + rnd.randrange(10_000_000)])

DDL = "CREATE TABLE hits (id INT, user_id INT, region INT, url TEXT, duration FLOAT, ts INT)"

def norm(rows):
    out = []
    for r in rows:
        out.append(tuple(round(float(v), 3) if isinstance(v, (int, float)) or (isinstance(v, str) and re.fullmatch(r"-?\d+(\.\d+)?", v or "")) else v for v in r))
    return out

# ---------------------------------------------------------------- engines
class DuckEngine:
    name = "duckdb"
    def __init__(self, csvp):
        import duckdb
        self.version = duckdb.__version__
        self.c = duckdb.connect()
        self.c.execute(DDL)
        self.c.execute(f"COPY hits FROM '{csvp}' (FORMAT csv)")
    def run(self, q): return self.c.execute(q).fetchall()

class JouleEngine:
    name = "jouledb"
    def __init__(self, csvp, port):
        import psycopg
        self.c = psycopg.connect(host="127.0.0.1", port=port, user="postgres", dbname="postgres", sslmode="disable", autocommit=True)
        self.version = self.c.execute("SELECT version()").fetchone()[0]
        self.c.execute("DROP TABLE IF EXISTS hits"); self.c.execute(DDL)
        with self.c.cursor().copy("COPY hits FROM STDIN WITH (FORMAT csv)") as cp:
            with open(csvp, "rb") as f:
                while chunk := f.read(1 << 20): cp.write(chunk)
    def run(self, q): return self.c.execute(q).fetchall()

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1_000_000)
    ap.add_argument("--pg-port", type=int, default=18082)
    ap.add_argument("--min-seconds", type=float, default=2.0)
    ap.add_argument("--engines", default="duckdb,jouledb")
    ap.add_argument("--out", default="bench-result.json")
    ap.add_argument("--settle", type=float, default=0.0,
                    help="seconds to wait after loading before querying (lets background indexing finish)")
    a = ap.parse_args()
    host = platform.node(); csvp = os.path.abspath(f"hits-{a.rows}.csv")
    if not os.path.exists(csvp): gen_csv(csvp, a.rows)
    meter = make_meter(); meter.start()
    # idle baseline
    with Window(meter) as w: time.sleep(3)
    idle_w = (w.j / (w.t1 - w.t0)) if w.j is not None else None
    res = {"host": host, "platform": platform.platform(), "machine": platform.machine(), "rows": a.rows,
           "energy_source": meter.source, "idle_w": idle_w, "load_avg": os.getloadavg(), "engines": {}, "queries": {}}
    engines = []
    for e in a.engines.split(","):
        t = time.monotonic()
        eng = DuckEngine(csvp) if e == "duckdb" else JouleEngine(csvp, a.pg_port)
        res["engines"][e] = {"version": eng.version, "load_s": round(time.monotonic() - t, 3)}
        engines.append(eng)
    res["settle_s"] = a.settle
    if a.settle > 0:
        time.sleep(a.settle)
    for qn, q in QUERIES.items():
        res["queries"][qn] = {"sql": q}
        answers = {}
        for eng in engines:
            try:
                answers[eng.name] = norm(eng.run(q))  # warm-up + answer
                runs = 0
                with Window(meter) as w:
                    while True:
                        eng.run(q); runs += 1
                        if time.monotonic() - w.t0 >= a.min_seconds: break
                secs = (w.t1 - w.t0) / runs
                j = w.j / runs if w.j is not None else None
                inc = (j - idle_w * secs) if j is not None and idle_w is not None else None
                res["queries"][qn][eng.name] = {"runs": runs, "seconds": secs, "joules": j, "incremental_joules": inc}
            except Exception as ex:
                res["queries"][qn][eng.name] = {"error": f"{type(ex).__name__}: {str(ex)[:200]}"}
        if len(answers) == 2:
            v = list(answers.values()); res["queries"][qn]["results_match"] = v[0] == v[1]
            if v[0] != v[1]: res["queries"][qn]["answers"] = {k: str(x)[:300] for k, x in answers.items()}
    meter.stop()
    json.dump(res, open(a.out, "w"), indent=1)
    # markdown
    print(f"### {host} ({res['machine']}), {a.rows:,} rows, energy={res['energy_source']}, idle={idle_w and round(idle_w,1)} W")
    print("| query | engine | ms/query | J/query | incr J/query | match |"); print("|---|---|---|---|---|---|")
    for qn, d in res["queries"].items():
        for e in res["engines"]:
            r = d.get(e, {})
            if "error" in r: print(f"| {qn} | {e} | error: {r['error'][:60]} | | | |"); continue
            f = lambda x, s=1: "n/a" if x is None else f"{x*s:.4g}"
            print(f"| {qn} | {e} | {f(r['seconds'],1000)} | {f(r['joules'])} | {f(r['incremental_joules'])} | {d.get('results_match','')} |")

if __name__ == "__main__":
    main()
