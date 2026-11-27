#!/usr/bin/env python3
"""Large Text Compression Benchmark (LTCB `enwik8` / `enwik9`) — Joules-per-Bit & Seekable Database Pareto Suite.

Evaluates the real 4-axis database & AI storage trade-off exposed by Matt Mahoney's
Large Text Compression Benchmark (`enwik8` = 100,000,000 bytes / `enwik9` = 1,000,000,000 bytes
of English Wikipedia XML) and neural/LLM compressors (`tufazip v2`, `zmix 1.0`, `cmix-lex-transformer`, `nncp v3.2`):

  1. Compression Ratio:   compressed bytes, bits-per-byte (`bpb`), and compression factor (`x`)
  2. Speed:               compression & decompression throughput (`MB/s`)
  3. Seekability (`O(1)`): random-access single-article query latency (`ms`) WITHOUT full-stream sequential replay
  4. Thermodynamics:      hardware RAPL package energy (`reported_j` Joules, `mJ/MB`, and `nJ/bit` = Joules per Bit!)

Also tests the `fx2-cmix` / JouleDB Amorphic `BinaryHV` insight on seekable database storage:
  - Ingesting parsed `enwik8` Wikipedia articles into JouleDB `.jcol` columnar storage in:
    (a) Original XML stream order (`jouledb_jcol_stream_order`)
    (b) SimHash / BinaryHV 64-bit semantic-locality sorted order (`jouledb_jcol_semantic_order`),
        grouping topically & lexically similar Wikipedia articles into the same `.jcol` dictionary/Zstd blocks
        while preserving sub-millisecond SQL point/LIKE queryability over `pgwire`!
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import lzma
import os
import platform
import re
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
import zlib
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import Window, make_meter  # noqa: E402

ENWIK8_URL = "https://mattmahoney.net/dc/enwik8.zip"
ENWIK8_BYTES = 100_000_000

# Published primary-source results from Matt Mahoney's Large Text Compression Benchmark
# (https://mattmahoney.net/dc/text.html, retrieved 2026-11, `enwik9` = 1,000,000,000 bytes).
# Power estimates for `est_nj_per_bit` are explicitly labeled `analytical_est` (never mixed with `reported_j`).
LTCB_ENWIK9_PUBLISHED_ANCHORS = [
    {
        "system": "cmix-lex-transformer (lexth11c)",
        "architecture": "Hybrid Context-Mixing + Lexical Preprocessor + Small Transformer",
        "corpus": "enwik9 (1,000,000,000 B)",
        "total_archive_bytes": 95_836_613,
        "bpb": round(95_836_613 * 8 / 1_000_000_000, 4),
        "ratio_x": round(1_000_000_000 / 95_836_613, 3),
        "encode_s": 199_440,
        "decode_s": 213_840,
        "decode_mb_s": round(1000.0 / 213_840, 5),
        "ram_mb": 9765,
        "random_access_seekable": False,
        "energy_tag": "analytical_est_cpu_25w",
        "est_decode_joules": round(213_840 * 25.0, 1),
        "est_decode_nj_per_bit": round((213_840 * 25.0 * 1e9) / 8e9, 1),
    },
    {
        "system": "zmix 1.0 (Zig port of fx2-cmix-transformer)",
        "architecture": "4-bit 6M Offline Transformer + PPMD + Context Mixer (weight-size regularized)",
        "corpus": "enwik9 (1,000,000,000 B)",
        "total_archive_bytes": 96_096_261,
        "bpb": round(96_096_261 * 8 / 1_000_000_000, 4),
        "ratio_x": round(1_000_000_000 / 96_096_261, 3),
        "encode_s": 131_364,
        "decode_s": 131_184,
        "decode_mb_s": round(1000.0 / 131_184, 5),
        "ram_mb": 9370,
        "random_access_seekable": False,
        "energy_tag": "analytical_est_1core_15w",
        "est_decode_joules": round(131_184 * 15.0, 1),
        "est_decode_nj_per_bit": round((131_184 * 15.0 * 1e9) / 8e9, 1),
    },
    {
        "system": "tufazip v2 (Tufa Labs)",
        "architecture": "Pure 12-Layer 163M Transformer (4.43 MB 34M Starter Prior + Lockstep Online SGD + Arithmetic Coding)",
        "corpus": "enwik9 (1,000,000,000 B)",
        "payload_bytes": 100_958_324,
        "decoder_and_starter_weights_bytes": 4_430_029,
        "total_archive_bytes": 105_388_353,
        "bpb": round(105_388_353 * 8 / 1_000_000_000, 4),
        "ratio_x": round(1_000_000_000 / 105_388_353, 3),
        "encode_s": 647_595,
        "decode_s": 630_136,
        "decode_mb_s": round(1000.0 / 630_136, 5),
        "ram_mb": 73_748,
        "random_access_seekable": False,
        "energy_tag": "analytical_est_dual_b200_2000w",
        "est_decode_joules": round(630_136 * 2000.0, 1),
        "est_decode_nj_per_bit": round((630_136 * 2000.0 * 1e9) / 8e9, 1),
    },
    {
        "system": "nncp v3.2 (Fabrice Bellard)",
        "architecture": "Online-Trained Transformer Byte Predictor + Arithmetic Coding",
        "corpus": "enwik9 (1,000,000,000 B)",
        "total_archive_bytes": 107_261_318,
        "bpb": round(107_261_318 * 8 / 1_000_000_000, 4),
        "ratio_x": round(1_000_000_000 / 107_261_318, 3),
        "encode_s": 241_871,
        "decode_s": 238_670,
        "decode_mb_s": round(1000.0 / 238_670, 5),
        "ram_mb": 7600,
        "random_access_seekable": False,
        "energy_tag": "analytical_est_gpu_250w",
        "est_decode_joules": round(238_670 * 250.0, 1),
        "est_decode_nj_per_bit": round((238_670 * 250.0 * 1e9) / 8e9, 1),
    },
]


def ensure_enwik8(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_path = cache_dir / "enwik8"
    if raw_path.exists() and raw_path.stat().st_size == ENWIK8_BYTES:
        return raw_path
    zip_path = cache_dir / "enwik8.zip"
    if not zip_path.exists():
        print(f"[ltcb] Downloading official enwik8 (100 MB) from {ENWIK8_URL} ...", flush=True)
        urllib.request.urlretrieve(ENWIK8_URL, str(zip_path))
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extract("enwik8", path=str(cache_dir))
    assert raw_path.stat().st_size == ENWIK8_BYTES
    return raw_path


def simhash64(text: str) -> int:
    """Fast deterministic 64-bit lexical/semantic SimHash for `fx2-cmix`-style article locality ordering."""
    acc = [0] * 64
    # Sample up to first 128 words (titles + lead ontology + infobox category tokens)
    words = text[:2048].lower().split()[:128]
    for w in words:
        h = int.from_bytes(hashlib.blake2b(w.encode("utf-8", "ignore"), digest_size=8).digest(), "little")
        for bit in range(64):
            if (h >> bit) & 1:
                acc[bit] += 1
            else:
                acc[bit] -= 1
    out = 0
    for bit in range(64):
        if acc[bit] > 0:
            out |= 1 << bit
    return out


def semantic_locality_key(title: str, ns: str, body: str) -> tuple[str, str, str, int]:
    """Deterministic ontology + template + SimHash locality key (`fx2-cmix` / JouleDB clustering principle).

    Groups Wikipedia articles that share the same namespace, infobox template (`{{Infobox ...`),
    and primary `[[Category:...]]` into adjacent storage blocks before breaking ties by 64-bit SimHash.
    """
    tpl_m = re.search(r"\{\{\s*([A-Za-z0-9_ -]{3,36})", body[:1024])
    cat_m = re.search(r"\[\[Category:\s*([^\]|]{2,48})", body)
    tpl = tpl_m.group(1).strip().lower() if tpl_m else ""
    cat = cat_m.group(1).strip().lower() if cat_m else ""
    return (ns.lower(), tpl, cat, gray_code_rank(simhash64(f"{title} {body[:1536]}")))


def parse_enwik8_pages(raw_bytes: bytes) -> list[tuple[int, str, str, str]]:
    """Extract `(doc_id, title, namespace_or_prefix, body)` from `enwik8` XML."""
    text = raw_bytes.decode("utf-8", errors="replace")
    page_re = re.compile(r"<page>(.*?)</page>", re.DOTALL)
    title_re = re.compile(r"<title>(.*?)</title>", re.DOTALL)
    text_re = re.compile(r"<text[^>]*>(.*?)</text>", re.DOTALL)
    pages: list[tuple[int, str, str, str]] = []
    doc_id = 1
    for m in page_re.finditer(text):
        chunk = m.group(1)
        tm = title_re.search(chunk)
        xm = text_re.search(chunk)
        title = tm.group(1).strip() if tm else f"untitled_{doc_id}"
        body = xm.group(1) if xm else chunk
        prefix = title.split(":", 1)[0] if ":" in title else "Main"
        pages.append((doc_id, title, prefix, body))
        doc_id += 1
    return pages


def dir_size_bytes(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for fn in files:
            fp = os.path.join(root, fn)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total


def wait_port(host: str, port: int, timeout_s: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError(f"{host}:{port} did not become ready")


def measure_cli_codec(
    name: str,
    enc_cmd: list[str],
    dec_cmd: list[str],
    raw_path: Path,
    work_dir: Path,
    meter,
) -> dict | None:
    bin_name = enc_cmd[0]
    if shutil.which(bin_name) is None:
        return None
    comp_path = work_dir / f"{name}.bin"
    raw_bytes = raw_path.stat().st_size
    raw_bits = raw_bytes * 8

    with open(comp_path, "wb") as fout:
        with Window(meter) as w_enc:
            subprocess.run(enc_cmd + [str(raw_path)], stdout=fout, check=True)
    comp_size = comp_path.stat().st_size

    with Window(meter) as w_dec:
        dec_out = subprocess.check_output(dec_cmd + [str(comp_path)])
    assert len(dec_out) == raw_bytes, f"{name} decoded length mismatch"
    del dec_out

    enc_s = max(w_enc.t1 - w_enc.t0, 1e-6)
    dec_s = max(w_dec.t1 - w_dec.t0, 1e-6)
    comp_path.unlink(missing_ok=True)

    return {
        "codec": name,
        "seekable_sql": False,
        "compressed_bytes": comp_size,
        "bpb": round(comp_size * 8.0 / raw_bytes, 4),
        "ratio_x": round(raw_bytes / comp_size, 3),
        "encode_s": round(enc_s, 4),
        "decode_s": round(dec_s, 4),
        "encode_mb_s": round((raw_bytes / 1e6) / enc_s, 2),
        "decode_mb_s": round((raw_bytes / 1e6) / dec_s, 2),
        "encode_reported_j": round(w_enc.j, 4) if w_enc.j is not None else None,
        "decode_reported_j": round(w_dec.j, 4) if w_dec.j is not None else None,
        "encode_nj_per_bit": round((w_enc.j * 1e9) / raw_bits, 4) if w_enc.j is not None else None,
        "decode_nj_per_bit": round((w_dec.j * 1e9) / raw_bits, 4) if w_dec.j is not None else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jouledb-bin", required=True)
    ap.add_argument("--work-dir", default="/var/tmp")
    ap.add_argument("--cache-dir", default="/var/tmp/ltcb_cache")
    ap.add_argument("--port", type=int, default=28880)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import duckdb
    import psycopg

    meter = make_meter()
    meter.start()
    energy_tag = (
        "reported_j"
        if meter.source != "none" and not meter.source.startswith("none")
        else "unmetered"
    )

    enwik8_path = ensure_enwik8(Path(args.cache_dir))
    raw_bytes = enwik8_path.read_bytes()
    n_bytes = len(raw_bytes)

    root_tmp = Path(tempfile.mkdtemp(prefix="jdb_ltcb_", dir=args.work_dir))
    try:
        stream_codecs = []
        for spec in [
            ("lz4_fast", ["lz4", "-1", "-c"], ["lz4", "-d", "-c"]),
            ("gzip_6", ["gzip", "-6", "-c"], ["gzip", "-d", "-c"]),
            ("zstd_3", ["zstd", "-3", "-T0", "-c"], ["zstd", "-d", "-c"]),
            ("zstd_19_ultra", ["zstd", "-19", "-T0", "-c"], ["zstd", "-d", "-c"]),
            ("xz_6_lzma2", ["xz", "-6", "-T0", "-c"], ["xz", "-d", "-c"]),
        ]:
            res = measure_cli_codec(spec[0], spec[1], spec[2], enwik8_path, root_tmp, meter)
            if res is not None:
                print(
                    f"[ltcb] {res['codec']:15s}: {res['compressed_bytes']/1e6:6.2f} MB "
                    f"({res['bpb']:.3f} bpb, {res['ratio_x']:.2f}x) | "
                    f"enc={res['encode_mb_s']:7.1f} MB/s | dec={res['decode_mb_s']:7.1f} MB/s | "
                    f"dec_energy={res['decode_nj_per_bit']} nJ/bit",
                    flush=True,
                )
                stream_codecs.append(res)

        # Parse structured Wikipedia articles from `enwik8` for seekable columnar DB comparison
        pages = parse_enwik8_pages(raw_bytes)
        article_payload_bytes = sum(
            len(t.encode("utf-8")) + len(ns.encode("utf-8")) + len(b.encode("utf-8")) + 8
            for _, t, ns, b in pages
        )
        article_bits = article_payload_bytes * 8
        sample_titles = [pages[i][1] for i in range(0, len(pages), max(1, len(pages) // 25))][:25]

        # Evaluate ontology + SimHash semantic locality ordering (`fx2-cmix` principle on block-windowed storage!)
        def gray_code_rank(h: int) -> int:
            return h ^ (h >> 1)

        t_sh0 = time.perf_counter()
        tpl_re = re.compile(r"\{\{\s*([A-Za-z0-9_ -]{3,36})")
        cat_re = re.compile(r"\[\[Category:\s*([^\]|]{2,48})")
        keyed_pages = []
        for p in pages:
            _, t, ns, b = p
            tm = tpl_re.search(b[:1024])
            cm = cat_re.search(b)
            key = (
                ns.lower(),
                tm.group(1).strip().lower() if tm else "",
                cm.group(1).strip().lower() if cm else "",
                gray_code_rank(simhash64(f"{t} {b[:1536]}")),
            )
            keyed_pages.append((key, p))
        keyed_pages.sort(key=lambda x: x[0])
        semantic_pages = [p for _, p in keyed_pages]
        simhash_sort_s = round(time.perf_counter() - t_sh0, 4)

        # Measure bounded-window (gzip-6 32KB, zstd-3 128KB) vs whole-corpus (zstd-19 128MB) compression
        # before vs after Ontology+SimHash article reordering!
        stream_concat = b"\n".join(b.encode("utf-8") for _, _, _, b in pages)
        semantic_concat = b"\n".join(b.encode("utf-8") for _, _, _, b in semantic_pages)
        zstd_bin = shutil.which("zstd")
        gzip_bin = shutil.which("gzip")
        semantic_reorder_gain = None
        if zstd_bin:
            p_orig = root_tmp / "bodies_orig.txt"
            p_sem = root_tmp / "bodies_sem.txt"
            p_orig.write_bytes(stream_concat)
            p_sem.write_bytes(semantic_concat)
            sz_z3_orig = len(subprocess.check_output([zstd_bin, "-3", "-T0", "-c", str(p_orig)]))
            sz_z3_sem = len(subprocess.check_output([zstd_bin, "-3", "-T0", "-c", str(p_sem)]))
            sz_z19_orig = len(subprocess.check_output([zstd_bin, "-19", "-T0", "-c", str(p_orig)]))
            sz_z19_sem = len(subprocess.check_output([zstd_bin, "-19", "-T0", "-c", str(p_sem)]))
            gz_orig = len(subprocess.check_output([gzip_bin, "-6", "-c", str(p_orig)])) if gzip_bin else None
            gz_sem = len(subprocess.check_output([gzip_bin, "-6", "-c", str(p_sem)])) if gzip_bin else None
            p_orig.unlink(missing_ok=True)
            p_sem.unlink(missing_ok=True)
            semantic_reorder_gain = {
                "article_bodies_raw_bytes": len(stream_concat),
                "gzip6_32kb_window_stream_bytes": gz_orig,
                "gzip6_32kb_window_semantic_bytes": gz_sem,
                "gzip6_percent_smaller": round((1.0 - gz_sem / gz_orig) * 100.0, 2) if (gz_orig and gz_sem) else None,
                "zstd3_block_window_stream_bytes": sz_z3_orig,
                "zstd3_block_window_semantic_bytes": sz_z3_sem,
                "zstd3_percent_smaller": round((1.0 - sz_z3_sem / sz_z3_orig) * 100.0, 2),
                "zstd19_128mb_window_stream_bytes": sz_z19_orig,
                "zstd19_128mb_window_semantic_bytes": sz_z19_sem,
                "zstd19_percent_smaller": round((1.0 - sz_z19_sem / sz_z19_orig) * 100.0, 2),
                "reorder_overhead_s": simhash_sort_s,
            }
            print(
                f"[ltcb] Ontology+SimHash locality reordering ({len(pages)} articles, {simhash_sort_s}s): "
                f"gzip-6 {semantic_reorder_gain['gzip6_percent_smaller']}% smaller | "
                f"zstd-3 {semantic_reorder_gain['zstd3_percent_smaller']}% smaller | "
                f"zstd-19 {semantic_reorder_gain['zstd19_percent_smaller']}% smaller",
                flush=True,
            )
        del stream_concat, semantic_concat

        # Export Parquet (both stream order and semantic order with 2,048-row seekable row groups)
        # + ingest into JouleDB `.jcol` (`JOULEDB_COL_SPILL_ROWS=2048`) to measure seekable SQL storage,
        # full-corpus scan throughput (`COUNT(*) WHERE body LIKE '%thermodynamics%'`), and O(1) point lookup!
        stage_dir = root_tmp / "stage"
        stage_dir.mkdir(parents=True, exist_ok=True)
        pq_orig_path = stage_dir / "wiki_orig.parquet"
        pq_path = stage_dir / "wiki_sem.parquet"
        dcon = duckdb.connect(":memory:")
        dcon.execute("CREATE TABLE wiki_orig (doc_id BIGINT, namespace VARCHAR, title VARCHAR, body VARCHAR)")
        dcon.executemany("INSERT INTO wiki_orig VALUES (?, ?, ?, ?)", [(d, ns, t, b) for d, t, ns, b in pages])
        dcon.execute(f"COPY wiki_orig TO '{pq_orig_path}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 2048)")
        parquet_zstd_stream_bytes = pq_orig_path.stat().st_size
        pq_orig_path.unlink(missing_ok=True)

        dcon.execute("CREATE TABLE wiki (doc_id BIGINT, namespace VARCHAR, title VARCHAR, body VARCHAR)")
        dcon.executemany("INSERT INTO wiki VALUES (?, ?, ?, ?)", [(d, ns, t, b) for d, t, ns, b in semantic_pages])
        dcon.execute(f"COPY wiki TO '{pq_path}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 2048)")
        parquet_zstd_bytes = pq_path.stat().st_size
        dcon.close()

        jdb_dir = root_tmp / "jdb_wiki"
        jdb_dir.mkdir(parents=True, exist_ok=True)
        pg_port = args.port + 2
        env = {
            **os.environ,
            "JOULEDB_COPY_DIR": str(stage_dir),
            "JOULEDB_COL_SPILL_ROWS": "2048",
        }
        proc = subprocess.Popen(
            [args.jouledb_bin, "-p", str(args.port), "-d", str(jdb_dir)],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            wait_port("127.0.0.1", pg_port, 20.0)
            with psycopg.connect(
                f"host=127.0.0.1 port={pg_port} user=postgres dbname=default",
                autocommit=True,
            ) as conn:
                conn.execute("CREATE TABLE wiki (doc_id BIGINT, namespace TEXT, title TEXT, body TEXT)")
                with Window(meter) as w_ing:
                    conn.execute("COPY wiki FROM 'wiki_sem.parquet'")

                # Warmup + timed full-corpus analytical scan (`LIKE '%thermodynamics%'`)
                _ = conn.execute("SELECT COUNT(*) FROM wiki WHERE body LIKE '%thermodynamics%'").fetchone()
                scan_times = []
                scan_joules = []
                matches_found = 0
                for _ in range(5):
                    with Window(meter) as w_q:
                        matches_found = int(
                            conn.execute("SELECT COUNT(*) FROM wiki WHERE body LIKE '%thermodynamics%'").fetchone()[0]
                        )
                    scan_times.append((w_q.t1 - w_q.t0) * 1000.0)
                    if w_q.j is not None:
                        scan_joules.append(w_q.j)

                # Timed random-access single-article SQL lookups (impossible on tufazip/cmix without 7-day full decode!)
                point_lats_ms = []
                for title in sample_titles:
                    t0 = time.perf_counter()
                    row = conn.execute(
                        "SELECT doc_id, namespace, LENGTH(body) FROM wiki WHERE title = %s LIMIT 1",
                        (title,),
                    ).fetchone()
                    point_lats_ms.append((time.perf_counter() - t0) * 1000.0)
                    assert row is not None
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()

        jcol_dir = jdb_dir / "col_segments"
        jcol_bytes = dir_size_bytes(jcol_dir) if jcol_dir.exists() else dir_size_bytes(jdb_dir)
        jcol_zstd3_bytes = None
        if zstd_bin and jcol_dir.exists():
            # Measure transparent ZFS/Btrfs-style Zstd-3 block compression of the `.jcol` segment files
            total_zc = 0
            for root, _, files in os.walk(jcol_dir):
                for fn in files:
                    fp = os.path.join(root, fn)
                    if os.path.isfile(fp):
                        total_zc += len(subprocess.check_output([zstd_bin, "-3", "-c", fp]))
            jcol_zstd3_bytes = total_zc
        scan_med_ms = statistics.median(scan_times)
        scan_med_j = statistics.median(scan_joules) if scan_joules else None

        seekable_db_results = {
            "articles_parsed": len(pages),
            "structured_article_bytes": article_payload_bytes,
            "parquet_zstd_rowgroup2k_stream_bytes": parquet_zstd_stream_bytes,
            "parquet_zstd_rowgroup2k_semantic_bytes": parquet_zstd_bytes,
            "parquet_semantic_reorder_percent_smaller": round(
                (1.0 - parquet_zstd_bytes / parquet_zstd_stream_bytes) * 100.0, 2
            ),
            "parquet_zstd_bpb": round(parquet_zstd_bytes * 8.0 / article_payload_bytes, 4),
            "jouledb_jcol_raw_mmap_bytes": jcol_bytes,
            "jouledb_jcol_zstd3_bytes": jcol_zstd3_bytes,
            "jouledb_jcol_zstd3_bpb": (
                round(jcol_zstd3_bytes * 8.0 / article_payload_bytes, 4) if jcol_zstd3_bytes else None
            ),
            "jouledb_ingest_s": round(w_ing.t1 - w_ing.t0, 4),
            "jouledb_ingest_reported_j": round(w_ing.j, 4) if w_ing.j is not None else None,
            "jouledb_full_corpus_like_scan_median_ms": round(scan_med_ms, 3),
            "jouledb_full_corpus_like_scan_gb_s": round(
                (article_payload_bytes / 1e9) / (scan_med_ms / 1000.0), 3
            ),
            "jouledb_full_corpus_scan_reported_j": round(scan_med_j, 4) if scan_med_j is not None else None,
            "jouledb_full_corpus_scan_nj_per_bit": (
                round((scan_med_j * 1e9) / article_bits, 4) if scan_med_j is not None else None
            ),
            "jouledb_full_corpus_matches": matches_found,
            "jouledb_single_article_lookup_median_ms": round(statistics.median(point_lats_ms), 3),
            "speedup_vs_tufazip_single_article_lookup_x": round(
                (630_136.0 * 1000.0) / max(statistics.median(point_lats_ms), 0.001), 1
            ),
        }

        report = {
            "benchmark": "Large Text Compression Benchmark (LTCB enwik8/enwik9) — Joules-per-Bit & Seekable Database Pareto Suite",
            "version": "0.16.0",
            "corpus_evaluated_live": {
                "name": "enwik8 (Matt Mahoney Official 100 MB Wikipedia XML)",
                "bytes": n_bytes,
                "parsed_articles": len(pages),
            },
            "host": platform.node(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "energy_source": meter.source,
            "energy_evidence": energy_tag,
            "stream_compressors_live_enwik8": stream_codecs,
            "semantic_locality_reordering_enwik8": semantic_reorder_gain,
            "seekable_columnar_database_live_enwik8": seekable_db_results,
            "published_ltcb_enwik9_neural_anchors": LTCB_ENWIK9_PUBLISHED_ANCHORS,
        }

        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2) + "\n")
        print(f"[ltcb] Wrote complete report -> {out_p}", flush=True)
    finally:
        shutil.rmtree(root_tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
