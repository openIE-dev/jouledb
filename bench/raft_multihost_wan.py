#!/usr/bin/env python3
"""Multi-Host Cross-Machine WAN/Tailscale 5-Node TCP Raft Fault-Injection + Clojure `elle` Harness.

Deploys 5 real OS `raft_wan_node` processes across two distinct physical Linux hosts over
encrypted Tailscale WAN (default: 3 nodes on bare-metal `100.91.48.80` + 2 nodes on cloud
VM `100.64.236.125`, with `--local-only` fallback for single-machine reproduction):
  - Phase 0: Cross-host TCP RTT measurement & initial 5-node Raft leader election
  - Phase 1: Concurrent cross-WAN linearizable `propose_and_commit` writes & §6.4 `ReadIndex` reads
  - Phase 2: Bidirectional WAN partition isolating `{old_leader, remote_peer}` (2/5 minority)
             from the remaining 3/5 cross-host majority — verifying minority rejection (`0`
             split-brain commits / `0` stale linearizable reads) and majority failover election
  - Phase 3: Real `SIGKILL` (`kill -9`) of a remote cloud node while commits continue on the
             surviving quorum, followed by WAN link heal + durable `raft.wal` process restart
  - Phase 4: Full 5-node state convergence audit across both physical hosts + EDN history export
             for official Clojure `elle/elle 0.2.8` (`:strict-serializable` `rw-register`)
"""
import argparse
import json
import os
import platform
import shlex
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class NodeSpec:
    node_id: str
    host_ip: str
    raft_port: int
    ctrl_port: int
    ssh_target: Optional[str]  # None => local host
    bin_path: str
    data_dir: str
    host_label: str


class NodeHandle:
    def __init__(self, spec: NodeSpec):
        self.spec = spec
        self.local_proc: Optional[subprocess.Popen] = None
        self.remote_pid: Optional[int] = None

    @property
    def raft_addr(self) -> str:
        return f"{self.spec.host_ip}:{self.spec.raft_port}"

    @property
    def ctrl_addr(self) -> tuple[str, int]:
        return (self.spec.host_ip, self.spec.ctrl_port)

    def start(self, all_specs: list[NodeSpec], wipe_data: bool = False):
        peer_args = []
        for s in all_specs:
            if s.node_id != self.spec.node_id:
                peer_args.extend(["--peer", f"{s.node_id}={s.host_ip}:{s.raft_port}"])

        if self.spec.ssh_target is None:
            if wipe_data and os.path.exists(self.spec.data_dir):
                shutil.rmtree(self.spec.data_dir, ignore_errors=True)
            os.makedirs(self.spec.data_dir, exist_ok=True)
            cmd = [
                self.spec.bin_path,
                "--id",
                self.spec.node_id,
                "--raft-bind",
                self.raft_addr,
                "--ctrl-bind",
                f"{self.spec.host_ip}:{self.spec.ctrl_port}",
                "--data",
                self.spec.data_dir,
                *peer_args,
            ]
            log_f = open(os.path.join(self.spec.data_dir, "node.log"), "ab")
            self.local_proc = subprocess.Popen(
                cmd, stdout=log_f, stderr=subprocess.STDOUT
            )
        else:
            cmd_list = [
                self.spec.bin_path,
                "--id",
                self.spec.node_id,
                "--raft-bind",
                self.raft_addr,
                "--ctrl-bind",
                f"{self.spec.host_ip}:{self.spec.ctrl_port}",
                "--data",
                self.spec.data_dir,
                *peer_args,
            ]
            py_launch = (
                "import os, shutil, subprocess; "
                f"d={repr(self.spec.data_dir)}; "
                f"{'shutil.rmtree(d, ignore_errors=True); ' if wipe_data else ''}"
                "os.makedirs(d, exist_ok=True); "
                "lf=open(os.path.join(d,'node.log'),'ab'); "
                f"p=subprocess.Popen({repr(cmd_list)}, stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True); "
                "print(p.pid)"
            )
            out = subprocess.check_output(
                [
                    "ssh",
                    "-n",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=8",
                    self.spec.ssh_target,
                    f"python3 -c {shlex.quote(py_launch)}",
                ],
                text=True,
                timeout=15,
            ).strip()
            self.remote_pid = int(out.splitlines()[-1].strip())

    def sigkill(self):
        if self.spec.ssh_target is None:
            if self.local_proc is not None:
                try:
                    os.kill(self.local_proc.pid, signal.SIGKILL)
                    self.local_proc.wait(timeout=3)
                except OSError:
                    pass
                self.local_proc = None
        else:
            if self.remote_pid is not None:
                subprocess.run(
                    [
                        "ssh",
                        "-o",
                        "BatchMode=yes",
                        self.spec.ssh_target,
                        f"kill -9 {self.remote_pid} 2>/dev/null || true",
                    ],
                    timeout=10,
                    check=False,
                )
                self.remote_pid = None

    def ctrl_rpc(self, line: str, timeout_s: float = 3.5) -> dict:
        with socket.create_connection(self.ctrl_addr, timeout=timeout_s) as s:
            s.settimeout(timeout_s)
            s.sendall((line.strip() + "\n").encode("utf-8"))
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = s.recv(4096)
                if not chunk:
                    break
                buf += chunk
        if not buf:
            return {"ok": False, "err": "eof"}
        return json.loads(buf.decode("utf-8").strip())


class ElleHistoryRecorder:
    def __init__(self):
        self._lock = threading.Lock()
        self._t0 = time.monotonic_ns()
        self._idx = 0
        self._next_proc = 0
        self.events: list[dict] = []

    def alloc_proc(self) -> int:
        with self._lock:
            p = self._next_proc
            self._next_proc += 1
            return p

    def record(self, ev_type: str, proc: int, op_kind: str, key_int: int, val_opt: Optional[int]) -> int:
        with self._lock:
            idx = self._idx
            self._idx += 1
            ts = time.monotonic_ns() - self._t0
            ev = {
                "index": idx,
                "type": ev_type,
                "f": "txn",
                "op": op_kind,
                "k": key_int,
                "v": val_opt,
                "process": proc,
                "time": ts,
            }
            self.events.append(ev)
            return idx

    def to_edn(self) -> str:
        lines = ["["]
        for e in self.events:
            v_str = "nil" if e["v"] is None else str(e["v"])
            lines.append(
                f'  {{:index {e["index"]} :type :{e["type"]} :f :txn '
                f':value [[:{e["op"]} {e["k"]} {v_str}]] :process {e["process"]} :time {e["time"]}}}'
            )
        lines.append("]\n")
        return "\n".join(lines)


def measure_tcp_rtt_ms(host: str, port: int, samples: int = 24) -> dict:
    rtts = []
    for _ in range(samples):
        t0 = time.perf_counter()
        with socket.create_connection((host, port), timeout=3.0):
            rtts.append((time.perf_counter() - t0) * 1000.0)
        time.sleep(0.015)
    rtts.sort()
    p95_idx = min(int(len(rtts) * 0.95), len(rtts) - 1)
    return {
        "samples": len(rtts),
        "min_ms": round(rtts[0], 3),
        "median_ms": round(statistics.median(rtts), 3),
        "p95_ms": round(rtts[p95_idx], 3),
        "max_ms": round(rtts[-1], 3),
    }


def wait_ctrl_ready(nodes: list[NodeHandle], timeout_s: float = 15.0):
    deadline = time.monotonic() + timeout_s
    for n in nodes:
        while time.monotonic() < deadline:
            try:
                st = n.ctrl_rpc("STATUS", timeout_s=1.5)
                if st.get("ok"):
                    break
            except Exception:
                time.sleep(0.08)
        else:
            raise RuntimeError(f"Node {n.spec.node_id} ({n.spec.host_ip}) never became reachable")


def find_leader(nodes: list[NodeHandle], allowed_ids: Optional[set[str]] = None, timeout_s: float = 10.0) -> tuple[NodeHandle, dict]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        best: Optional[tuple[int, NodeHandle, dict]] = None
        for n in nodes:
            if allowed_ids is not None and n.spec.node_id not in allowed_ids:
                continue
            try:
                st = n.ctrl_rpc("STATUS", timeout_s=1.5)
                if st.get("ok") and st.get("role") == "Leader":
                    term = int(st.get("term", 0))
                    if best is None or term > best[0]:
                        best = (term, n, st)
            except Exception:
                pass
        if best is not None:
            return best[1], best[2]
        time.sleep(0.10)
    raise RuntimeError(f"No Raft leader elected within {timeout_s}s among {allowed_ids}")


def parse_val(v) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, list):
        # Raw byte array fallback if STATUS returns Vec<u8>
        try:
            return int(bytes(v).decode("utf-8"))
        except Exception:
            return None
    s = str(v).strip()
    return int(s) if s != "" else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-only", action="store_true")
    ap.add_argument("--host-a", default="100.91.48.80")
    ap.add_argument("--host-a-ssh", default="", help="Optional SSH target for Host A when coordinator runs off-box")
    ap.add_argument("--host-a-label", default="framework-128gb-0000 (Bare-Metal AMD Ryzen AI Max+ 395, Linux 7.0)")
    ap.add_argument("--host-b", default="100.64.236.125")
    ap.add_argument("--host-b-ssh", default="dcharlot@100.64.236.125")
    ap.add_argument("--host-b-label", default="openie-core-1 (Cloud AMD64 VM, Debian 12 Linux 6.1)")
    ap.add_argument("--local-bin", default="./target/release/examples/raft_wan_node")
    ap.add_argument("--remote-bin", default="/tmp/jdb_wan_raft/raft_wan_node")
    ap.add_argument("--base-dir", default="/tmp/jdb_wan_raft_data")
    ap.add_argument("--out-edn", default="bench/results/raft_wan_multihost_elle_history.edn")
    ap.add_argument("--out-json", default="bench/results/raft_wan_multihost_summary.json")
    args = ap.parse_args()

    host_a_ssh = args.host_a_ssh.strip() or None
    if args.local_only:
        specs = [
            NodeSpec(
                node_id=f"n{i}",
                host_ip="127.0.0.1",
                raft_port=19400 + i,
                ctrl_port=19500 + i,
                ssh_target=None,
                bin_path=args.local_bin,
                data_dir=f"{args.base_dir}/n{i}",
                host_label=platform.node(),
            )
            for i in range(1, 6)
        ]
    else:
        specs = [
            NodeSpec("n1", args.host_a, 19401, 19501, host_a_ssh, args.local_bin, f"{args.base_dir}/n1", args.host_a_label),
            NodeSpec("n2", args.host_a, 19402, 19502, host_a_ssh, args.local_bin, f"{args.base_dir}/n2", args.host_a_label),
            NodeSpec("n3", args.host_a, 19403, 19503, host_a_ssh, args.local_bin, f"{args.base_dir}/n3", args.host_a_label),
            NodeSpec("n4", args.host_b, 19404, 19504, args.host_b_ssh, args.remote_bin, f"{args.base_dir}/n4", args.host_b_label),
            NodeSpec("n5", args.host_b, 19405, 19505, args.host_b_ssh, args.remote_bin, f"{args.base_dir}/n5", args.host_b_label),
        ]

    nodes = [NodeHandle(s) for s in specs]
    by_id = {n.spec.node_id: n for n in nodes}
    rec = ElleHistoryRecorder()
    val_seq = 0
    val_lock = threading.Lock()

    def next_val() -> int:
        nonlocal val_seq
        with val_lock:
            val_seq += 1
            return val_seq

    def do_write(target: NodeHandle, proc: int, key_idx: int) -> tuple[bool, int]:
        key_name = "x" if key_idx == 0 else "y"
        v = next_val()
        rec.record("invoke", proc, "w", key_idx, v)
        try:
            res = target.ctrl_rpc(f"WRITE {key_name} {v}", timeout_s=3.5)
            if res.get("ok"):
                rec.record("ok", proc, "w", key_idx, v)
                return True, proc
            if res.get("definite_fail"):
                rec.record("fail", proc, "w", key_idx, v)
                return False, proc
            rec.record("info", proc, "w", key_idx, v)
            return False, rec.alloc_proc()
        except Exception:
            rec.record("info", proc, "w", key_idx, v)
            return False, rec.alloc_proc()

    def do_read(target: NodeHandle, proc: int, key_idx: int) -> tuple[bool, Optional[int], int]:
        key_name = "x" if key_idx == 0 else "y"
        rec.record("invoke", proc, "r", key_idx, None)
        try:
            res = target.ctrl_rpc(f"READ {key_name}", timeout_s=2.8)
            if res.get("ok"):
                rv = parse_val(res.get("value"))
                rec.record("ok", proc, "r", key_idx, rv)
                return True, rv, proc
            rec.record("fail", proc, "r", key_idx, None)
            return False, None, proc
        except Exception:
            rec.record("fail", proc, "r", key_idx, None)
            return False, None, proc

    t_suite_start = time.monotonic()
    try:
        # Clean any leftover processes on ports first
        if args.local_only:
            subprocess.run(["pkill", "-9", "-f", "raft_wan_node"], check=False)
        else:
            if host_a_ssh:
                subprocess.run(
                    ["ssh", "-o", "BatchMode=yes", host_a_ssh, "pkill -9 -f raft_wan_node || true"],
                    check=False,
                    timeout=10,
                )
            else:
                subprocess.run(["pkill", "-9", "-f", "raft_wan_node"], check=False)
            subprocess.run(
                ["ssh", "-o", "BatchMode=yes", args.host_b_ssh, "pkill -9 -f raft_wan_node || true"],
                check=False,
                timeout=10,
            )
            time.sleep(0.2)

        # Phase 0: Start all 5 OS processes & measure cross-host WAN TCP RTT
        for n in nodes:
            n.start(specs, wipe_data=True)
        wait_ctrl_ready(nodes, timeout_s=18.0)

        if host_a_ssh and not args.local_only:
            rtt_py = (
                "import json, socket, statistics, time; r=[]; "
                f"[((t:=time.perf_counter()), (s:=socket.create_connection(('{args.host_b}', 19404), 3.0)), r.append((time.perf_counter()-t)*1000.0), s.close(), time.sleep(0.01)) for _ in range(24)]; "
                "r.sort(); print(json.dumps({'samples':len(r),'min_ms':round(r[0],3),'median_ms':round(statistics.median(r),3),'p95_ms':round(r[int(len(r)*0.95)],3),'max_ms':round(r[-1],3)}))"
            )
            wan_rtt = json.loads(
                subprocess.check_output(
                    ["ssh", "-o", "BatchMode=yes", host_a_ssh, f"python3 -c {shlex.quote(rtt_py)}"],
                    text=True,
                    timeout=15,
                ).strip()
            )
        else:
            wan_rtt = measure_tcp_rtt_ms(nodes[-1].spec.host_ip, nodes[-1].spec.ctrl_port, samples=24)
        leader0, leader0_st = find_leader(nodes, timeout_s=12.0)
        initial_leader_id = leader0.spec.node_id
        initial_term = int(leader0_st["term"])

        # Phase 1: Concurrent cross-WAN linearizable writes & ReadIndex reads (4 client threads, 80 ops)
        p1_lat_ms = []
        p1_lock = threading.Lock()

        def p1_worker(worker_idx: int):
            proc = rec.alloc_proc()
            cur_leader = leader0
            for step in range(20):
                key_idx = (worker_idx + step) & 1
                t0 = time.perf_counter()
                if step % 2 == 0:
                    ok, proc = do_write(cur_leader, proc, key_idx)
                    if not ok:
                        try:
                            cur_leader, _ = find_leader(nodes, timeout_s=4.0)
                        except Exception:
                            pass
                else:
                    ok, _, proc = do_read(cur_leader, proc, key_idx)
                    if not ok:
                        try:
                            cur_leader, _ = find_leader(nodes, timeout_s=4.0)
                        except Exception:
                            pass
                dt = (time.perf_counter() - t0) * 1000.0
                if ok:
                    with p1_lock:
                        p1_lat_ms.append(dt)

        threads = [threading.Thread(target=p1_worker, args=(w,)) for w in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Also verify follower rejects direct client writes cleanly (`definite_fail`)
        cur_leader, _ = find_leader(nodes, timeout_s=6.0)
        follower_probe = next(n for n in nodes if n.spec.node_id != cur_leader.spec.node_id)
        f_proc = rec.alloc_proc()
        follower_write_ok, f_proc = do_write(follower_probe, f_proc, 0)

        # Phase 2: Bidirectional WAN Partition!
        # Isolate `{cur_leader, remote_partner}` (2/5 minority spanning WAN) from `{3/5 majority}`!
        if cur_leader.spec.node_id in ("n1", "n2", "n3"):
            partner_id = "n4"
        else:
            partner_id = "n1"
        minority_ids = {cur_leader.spec.node_id, partner_id}
        majority_ids = {n.spec.node_id for n in nodes} - minority_ids

        for a_id in minority_ids:
            for b_id in majority_ids:
                by_id[a_id].ctrl_rpc(f"BLOCK {b_id}")
                by_id[b_id].ctrl_rpc(f"BLOCK {a_id}")

        # Wait for majority to elect a new leader AND old leader's heartbeat quorum to expire
        time.sleep(1.1)
        maj_leader, maj_st = find_leader(nodes, allowed_ids=majority_ids, timeout_s=12.0)

        # Probe isolated old leader: BOTH write and linearizable ReadIndex read MUST fail closed!
        iso_proc = rec.alloc_proc()
        minority_write_ok, iso_proc = do_write(cur_leader, iso_proc, 0)
        minority_read_ok, _, iso_proc = do_read(cur_leader, iso_proc, 0)

        # Meanwhile, commit 24 linearizable writes + ReadIndex reads on the 3/5 cross-WAN majority!
        maj_proc = rec.alloc_proc()
        maj_ok_ops = 0
        for step in range(24):
            k_idx = step & 1
            if step % 2 == 0:
                ok, maj_proc = do_write(maj_leader, maj_proc, k_idx)
            else:
                ok, _, maj_proc = do_read(maj_leader, maj_proc, k_idx)
            if not ok:
                maj_leader, maj_st = find_leader(nodes, allowed_ids=majority_ids, timeout_s=6.0)
            else:
                maj_ok_ops += 1

        # Phase 3: Real `SIGKILL` (`kill -9`) on remote Cloud node `n5` (or `n4` if `n5` is leader)
        kill_id = "n5" if maj_leader.spec.node_id != "n5" and "n5" in majority_ids else (
            "n4" if maj_leader.spec.node_id != "n4" else "n3"
        )
        # First heal WAN partition so 4/5 nodes form quorum while we SIGKILL `kill_id`!
        for a_id in minority_ids:
            for b_id in majority_ids:
                by_id[a_id].ctrl_rpc(f"ALLOW {b_id}")
                by_id[b_id].ctrl_rpc(f"ALLOW {a_id}")
        time.sleep(0.5)

        killed_node = by_id[kill_id]
        killed_node.sigkill()

        surviving_ids = {n.spec.node_id for n in nodes if n.spec.node_id != kill_id}
        surv_leader, _ = find_leader(nodes, allowed_ids=surviving_ids, timeout_s=10.0)
        crash_proc = rec.alloc_proc()
        post_kill_commits = 0
        for step in range(16):
            k_idx = step & 1
            if step % 2 == 0:
                ok, crash_proc = do_write(surv_leader, crash_proc, k_idx)
            else:
                ok, _, crash_proc = do_read(surv_leader, crash_proc, k_idx)
            if ok:
                post_kill_commits += 1
            else:
                surv_leader, _ = find_leader(nodes, allowed_ids=surviving_ids, timeout_s=6.0)

        # Restart SIGKILLed node from durable `raft.wal` (`wipe_data=False`!)
        killed_node.start(specs, wipe_data=False)
        wait_ctrl_ready([killed_node], timeout_s=12.0)

        # Phase 4: Final writes & full 5-node cross-host convergence check
        final_leader, _ = find_leader(nodes, timeout_s=10.0)
        fin_proc = rec.alloc_proc()
        for step in range(12):
            k_idx = step & 1
            if step % 2 == 0:
                _, fin_proc = do_write(final_leader, fin_proc, k_idx)
            else:
                _, _, fin_proc = do_read(final_leader, fin_proc, k_idx)

        # Wait up to 5s for all 5 nodes across both hosts to catch up to identical commit & state
        converged = False
        final_statuses = []
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline:
            statuses = []
            for n in nodes:
                st = n.ctrl_rpc("STATUS", timeout_s=2.0)
                st["host_ip"] = n.spec.host_ip
                st["host_label"] = n.spec.host_label
                st["x_int"] = parse_val(st.get("local_x"))
                st["y_int"] = parse_val(st.get("local_y"))
                statuses.append(st)
            xs = {s["x_int"] for s in statuses}
            ys = {s["y_int"] for s in statuses}
            commits = {s["commit"] for s in statuses}
            if len(xs) == 1 and None not in xs and len(ys) == 1 and None not in ys and len(commits) == 1:
                converged = True
                final_statuses = statuses
                break
            final_statuses = statuses
            time.sleep(0.15)

        os.makedirs(os.path.dirname(os.path.abspath(args.out_edn)), exist_ok=True)
        with open(args.out_edn, "w", encoding="utf-8") as f:
            f.write(rec.to_edn())

        p1_lat_ms.sort()
        ok_events = sum(1 for e in rec.events if e["type"] == "ok")
        fail_events = sum(1 for e in rec.events if e["type"] == "fail")
        info_events = sum(1 for e in rec.events if e["type"] == "info")

        summary = {
            "suite": "raft_multihost_wan_5node_elle_v0.9.0",
            "mode": "local_5proc" if args.local_only else "multihost_tailscale_wan_2hosts_5proc",
            "duration_s": round(time.monotonic() - t_suite_start, 3),
            "topology": {
                "total_nodes": 5,
                "quorum_size": 3,
                "distinct_hosts": 1 if args.local_only else 2,
                "nodes": [
                    {
                        "id": s.node_id,
                        "host_ip": s.host_ip,
                        "raft_port": s.raft_port,
                        "host_label": s.host_label,
                    }
                    for s in specs
                ],
                "cross_host_tcp_rtt_ms": wan_rtt,
            },
            "fault_injection_checks": {
                "initial_leader": initial_leader_id,
                "initial_term": initial_term,
                "follower_direct_write_rejected": (not follower_write_ok),
                "isolated_minority_nodes": sorted(minority_ids),
                "majority_quorum_nodes": sorted(majority_ids),
                "majority_failover_leader": maj_leader.spec.node_id,
                "majority_failover_term": int(maj_st["term"]),
                "isolated_old_leader_write_rejected": (not minority_write_ok),
                "isolated_old_leader_linearizable_read_rejected": (not minority_read_ok),
                "majority_partition_committed_ops": maj_ok_ops,
                "sigkill_crashed_node": kill_id,
                "sigkill_host": killed_node.spec.host_ip,
                "ops_committed_during_sigkill": post_kill_commits,
                "wal_rejoin_all_5_converged": converged,
            },
            "client_latency_ms": {
                "samples": len(p1_lat_ms),
                "p50_ms": round(statistics.median(p1_lat_ms), 3) if p1_lat_ms else None,
                "p95_ms": round(p1_lat_ms[min(int(len(p1_lat_ms) * 0.95), len(p1_lat_ms) - 1)], 3)
                if p1_lat_ms
                else None,
            },
            "elle_history": {
                "edn_path": args.out_edn,
                "total_events": len(rec.events),
                "ok_completions": ok_events,
                "definite_fails": fail_events,
                "indeterminate_infos": info_events,
            },
            "final_node_states": final_statuses,
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(json.dumps(summary, indent=2))
        if (
            not converged
            or minority_write_ok
            or minority_read_ok
            or follower_write_ok
            or maj_ok_ops == 0
        ):
            sys.exit(1)
    finally:
        for n in nodes:
            n.sigkill()


if __name__ == "__main__":
    main()
