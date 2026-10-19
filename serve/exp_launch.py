#!/usr/bin/env python3
"""Serving experiments: start the two ranks as exact clones of the production containers, with explicit edits.

`snapshot` saves `docker inspect` of the running production ranks (st5-r0 on the head, st5-r1 on the worker); `start`
re-creates both containers from that snapshot with the given edits to the vLLM arguments and the environment, so an
A/B differs from production only in what the command line names. The snapshot holds the fabric addresses (vLLM
arguments, VLLM_HOST_IP), so it is written 0600 and nothing here prints it; `show` and the failure log tails redact
IPv4 addresses. Run on the head from ~/step5 while `spark` is stopped (desired none).

    python3 serve/exp_launch.py snapshot
    python3 serve/exp_launch.py show
    python3 serve/exp_launch.py start NAME [--arg FLAG=VALUE | --arg FLAG] [--drop FLAG] [--env K=V] [--unset K]
                                           [--mount SRC:DST]
    python3 serve/exp_launch.py bench NAME [--no-concurrent]
    python3 serve/exp_launch.py stop

Each start mirrors `spark`'s pre-start steps (page cache of the checkpoint evicted, drop_caches + compact_memory,
MemAvailable settled on both nodes), starts memguard on both nodes, then the worker rank, then the head rank, and waits
for /v1/models plus a one-token completion. Results go to runs/exp/NAME*.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request

HOME = os.path.expanduser("~")
EXP = os.path.join(HOME, "step5", "runs", "exp")
SNAP = {0: os.path.join(EXP, "inspect-r0.json"), 1: os.path.join(EXP, "inspect-r1.json")}
NAMES = {0: "st5-r0", 1: "st5-r1"}
WORKER = os.environ.get("WORKERS", "spark-worker")
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(HOME, "models", "step5-exl3-b566"))
MEMGUARD = os.path.join(HOME, ".step5-sparks", "memguard.sh")
PAGECACHE = os.path.join(HOME, "spark-ctl", "pagecache.py")
DESIRED = os.path.join(HOME, ".cache", "spark-ctl", "desired")
KIT = os.path.join(HOME, "dsv41-tensorfold", "engine-tp2", "tools", "dsv41", "kit_bench.py")
API = "http://127.0.0.1:8888"
IPV4 = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
MIN_AVAIL_GIB = 100
MOUNT_ROOTS = (os.path.join(HOME, "step5", "serve", "patch"), os.path.join(HOME, "step5", "runs", "patch"))


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def redact(s: str) -> str:
    return IPV4.sub("<ip>", s)


def sh(rank: int, cmd: str, check: bool = True, timeout: float = 600) -> str:
    """Run a shell command on the head (rank 0) or the worker (rank 1); return stdout."""
    argv = ["bash", "-c", cmd] if rank == 0 else ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                                                  WORKER, cmd]
    r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise SystemExit(f"rank {rank}: command failed ({r.returncode}): {redact(cmd)[:200]}\n{redact(r.stderr)[-2000:]}")
    return r.stdout


# ------------------------------------------------------------------------------------------------ argument editing
def parse_args_list(cmd: list[str]) -> tuple[list[str], list[tuple[str, str | None]]]:
    """vLLM argv after `serve` -> (positionals, [(flag, value or None)]) in order."""
    pos, flags, i = [], [], 0
    while i < len(cmd):
        a = cmd[i]
        if a.startswith("--"):
            if i + 1 < len(cmd) and not cmd[i + 1].startswith("--"):
                flags.append((a, cmd[i + 1])); i += 2
            else:
                flags.append((a, None)); i += 1
        else:
            pos.append(a); i += 1
    return pos, flags


def edit_flags(flags: list[tuple[str, str | None]], sets: list[str], drops: list[str]) -> list[tuple[str, str | None]]:
    out = [f for f in flags if f[0] not in drops]
    for s in sets:
        k, v = (s.split("=", 1) + [None])[:2] if "=" in s else (s, None)
        for j, (fk, _) in enumerate(out):
            if fk == k:
                out[j] = (k, v)
                break
        else:
            out.append((k, v))
    return out


def join_flags(pos: list[str], flags: list[tuple[str, str | None]]) -> list[str]:
    out = list(pos)
    for k, v in flags:
        out += [k] if v is None else [k, v]
    return out


def env_dict(ins: dict) -> dict[str, str]:
    return dict(e.split("=", 1) for e in ins["Config"]["Env"])


def sync_to_worker(path: str) -> None:
    """Copy a head file or directory to the same path on the worker and check that the bytes match."""
    if not os.path.isabs(path) or not os.path.exists(path):
        raise SystemExit(f"--mount source {path!r}: need an existing absolute path on the head")
    real = os.path.realpath(path)
    if not any(real.startswith(root + os.sep) for root in MOUNT_ROOTS):
        raise SystemExit(f"--mount source {path!r}: only files under {MOUNT_ROOTS} are mounted (and replaced on the "
                         "worker)")
    parent = os.path.dirname(path.rstrip("/"))
    sh(1, f"mkdir -p {shlex.quote(parent)} && rm -rf {shlex.quote(path)}")
    argv = ["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"] + (["-r"] if os.path.isdir(path) else [])
    r = subprocess.run(argv + [path, f"{WORKER}:{parent}/"], capture_output=True, text=True, timeout=600)
    if r.returncode:
        raise SystemExit(f"copy {path} to the worker failed: {redact(r.stderr)[-500:]}")
    digest = f"cd {shlex.quote(parent)} && find {shlex.quote(os.path.basename(path.rstrip('/')))} -type f -print0 " \
             f"| sort -z | xargs -0 md5sum"
    if sh(0, digest) != sh(1, digest):
        raise SystemExit(f"{path}: worker copy differs from the head")


def run_argv(ins: dict, rank: int, cmd: list[str], env: dict[str, str], mounts: list[str] = ()) -> list[str]:
    hc, cfg = ins["HostConfig"], ins["Config"]
    argv = ["docker", "run", "-d", "--name", NAMES[rank]]
    if hc.get("NetworkMode") == "host":
        argv += ["--network", "host"]
    if hc.get("IpcMode") == "host":
        argv += ["--ipc", "host"]
    if hc.get("Privileged"):
        argv += ["--privileged"]
    for s in hc.get("SecurityOpt") or []:
        argv += ["--security-opt", s]
    if hc.get("ShmSize"):
        argv += ["--shm-size", str(hc["ShmSize"])]
    for dr in hc.get("DeviceRequests") or []:
        if dr.get("Count") == -1 and ["gpu"] in (dr.get("Capabilities") or []):
            argv += ["--gpus", "all"]
    for u in hc.get("Ulimits") or []:
        argv += ["--ulimit", f"{u['Name']}={u['Soft']}:{u['Hard']}"]
    for b in hc.get("Binds") or []:
        argv += ["-v", b]
    for m in mounts:                      # SRC:DST, read-only
        argv += ["-v", f"{m}:ro"]
    for k, v in env.items():
        argv += ["-e", f"{k}={v}"]
    if cfg.get("WorkingDir"):
        argv += ["-w", cfg["WorkingDir"]]
    ep = cfg.get("Entrypoint") or []
    if ep:
        argv += ["--entrypoint", ep[0]]
    return argv + [cfg["Image"]] + ep[1:] + cmd


# ------------------------------------------------------------------------------------------------------- commands
def cmd_snapshot(_a) -> int:
    os.makedirs(EXP, exist_ok=True)
    old = os.umask(0o077)
    try:
        for rank in (0, 1):
            out = sh(rank, f"docker inspect {NAMES[rank]}")
            ins = json.loads(out)[0]
            if not ins["State"]["Running"]:
                raise SystemExit(f"{NAMES[rank]} is not running; snapshot the live production ranks")
            with open(SNAP[rank], "w") as f:
                json.dump(ins, f)
            os.chmod(SNAP[rank], 0o600)
            log(f"snapshot {NAMES[rank]}: image {ins['Config']['Image'][-20:]}, {len(ins['Config']['Cmd'])} args")
    finally:
        os.umask(old)
    return 0


def load_snap(rank: int) -> dict:
    with open(SNAP[rank]) as f:
        return json.load(f)


def cmd_show(_a) -> int:
    for rank in (0, 1):
        ins = load_snap(rank)
        pos, flags = parse_args_list(ins["Config"]["Cmd"])
        print(f"rank {rank} ({NAMES[rank]}): {redact(' '.join(pos))}")
        for k, v in flags:
            print("   ", k, redact(v) if v is not None else "")
        env = env_dict(ins)
        keys = sorted(k for k in env if re.match(r"(VLLM|NCCL|GLOO|STEP5|ST_|EXL3|B12X|PYTORCH|OMP|MALLOC)", k))
        print("    env:", " ".join(f"{k}={redact(env[k])}" for k in keys))
    return 0


def present() -> list[str]:
    out = []
    for rank in (0, 1):
        if sh(rank, f"docker ps -aq --filter name=^{NAMES[rank]}$").strip():
            out.append(NAMES[rank])
    return out


def mem_avail(rank: int) -> float:
    return int(sh(rank, "awk '/MemAvailable/{print $2}' /proc/meminfo")) / 2**20


def reclaim_and_settle(timeout: float = 240) -> None:
    for rank in (0, 1):
        sh(rank, f"python3 {PAGECACHE} evict {shlex.quote(MODEL_DIR)}", check=False)
        sh(rank, "sync; echo 3 | sudo -n /usr/bin/tee /proc/sys/vm/drop_caches >/dev/null; "
                 "echo 1 | sudo -n /usr/bin/tee /proc/sys/vm/compact_memory >/dev/null", check=False)
    hist, t0 = [], time.time()
    while True:
        h, w = mem_avail(0), mem_avail(1)
        hist.append((h, w))
        if len(hist) > 3:
            h0, w0 = hist[-4]
            if h - h0 < 0.5 and w - w0 < 0.5 and h >= MIN_AVAIL_GIB and w >= MIN_AVAIL_GIB:
                log(f"memory settled: head {h:.1f} GiB, worker {w:.1f} GiB available")
                return
        if time.time() - t0 > timeout:
            raise SystemExit(f"memory not settled: head {h:.1f} GiB, worker {w:.1f} GiB available")
        time.sleep(5)


def http_ok(path: str, body: dict | None = None, timeout: float = 30) -> bool:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(API + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def tail_logs(rank: int, n: int = 40) -> str:
    return redact(sh(rank, f"docker logs --tail {n} {NAMES[rank]} 2>&1", check=False))


def cmd_start(a) -> int:
    desired = open(DESIRED).read().strip() if os.path.exists(DESIRED) else "?"
    if desired != "none":
        raise SystemExit(f"spark desired is {desired!r}: run `~/spark-ctl/spark stop` first")
    left = present()
    if left:
        raise SystemExit(f"containers present: {left}; run `exp_launch.py stop` first")
    record = {"name": a.name, "arg": a.arg, "drop": a.drop, "env": a.env, "unset": a.unset, "mount": a.mount,
              "model_dir": MODEL_DIR, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    for m in a.mount:
        src, _, dst = m.partition(":")
        if not dst.startswith("/"):
            raise SystemExit(f"--mount {m!r}: want SRC:DST with absolute paths")
        sync_to_worker(src)
    plans = {}
    for rank in (0, 1):
        ins = load_snap(rank)
        pos, flags = parse_args_list(ins["Config"]["Cmd"])
        cmd = join_flags(pos, edit_flags(flags, a.arg, a.drop))
        env = env_dict(ins)
        for kv in a.env:
            k, v = kv.split("=", 1)
            env[k] = v
        for k in a.unset:
            env.pop(k, None)
        plans[rank] = run_argv(ins, rank, cmd, env, a.mount)
    reclaim_and_settle()
    t0 = time.time()
    for rank in (0, 1):
        sh(rank, f"nohup setsid bash {MEMGUARD} 1 > /dev/null 2>&1 < /dev/null &")
    for rank in (1, 0):   # worker first, then the head (rank 0 hosts the rendezvous and the API)
        sh(rank, shlex.join(plans[rank]))
        log(f"started {NAMES[rank]}")
    deadline = t0 + a.timeout
    phase = "models"
    while time.time() < deadline:
        time.sleep(10)
        for rank in (0, 1):
            st = sh(rank, f"docker inspect -f '{{{{.State.Running}}}}' {NAMES[rank]} 2>/dev/null || echo gone",
                    check=False).strip()
            if st != "true":
                record.update(ok=False, failed=f"{NAMES[rank]} {st}", after_s=round(time.time() - t0))
                json.dump(record, open(os.path.join(EXP, f"{a.name}.json"), "w"), indent=1)
                log(f"{NAMES[rank]} not running ({st}) after {time.time() - t0:.0f}s; log tails:")
                for r in (0, 1):
                    print(f"--- {NAMES[r]}\n{tail_logs(r)}", flush=True)
                return 1
        if phase == "models" and http_ok("/v1/models"):
            phase = "probe"
        if phase == "probe" and http_ok("/v1/completions", {"model": "step5", "prompt": "Hello", "max_tokens": 1,
                                                             "temperature": 0}, timeout=600):
            record.update(ok=True, ready_after_s=round(time.time() - t0),
                          mem_after_gib={"head": round(mem_avail(0), 1), "worker": round(mem_avail(1), 1)})
            json.dump(record, open(os.path.join(EXP, f"{a.name}.json"), "w"), indent=1)
            log(f"ready after {record['ready_after_s']}s; MemAvailable {record['mem_after_gib']}")
            return 0
    record.update(ok=False, failed="timeout", after_s=round(time.time() - t0))
    json.dump(record, open(os.path.join(EXP, f"{a.name}.json"), "w"), indent=1)
    log("timeout; log tails:")
    for r in (0, 1):
        print(f"--- {NAMES[r]}\n{tail_logs(r)}", flush=True)
    return 1


def cmd_stop(_a) -> int:
    for rank in (0, 1):
        sh(rank, f"docker rm -f {NAMES[rank]} > /dev/null 2>&1; bash {MEMGUARD} stop", check=False)
    log("stopped both ranks and both memguards")
    return 0


def cmd_bench(a) -> int:
    py = sys.executable
    here = os.path.dirname(os.path.abspath(__file__))
    db = os.path.join(here, "decode_bench.py")
    out = lambda tag: os.path.join(EXP, f"{a.name}-{tag}.json")
    steps = [[py, db, "--kit", KIT, "--set", "b", "--tokens", "384", "--reps", "2", "--out", out("decode-b")],
             [py, db, "--kit", KIT, "--set", "a", "--tokens", "384", "--reps", "2", "--out", out("decode-a")]]
    if not a.no_concurrent:
        steps.append([py, db, "--kit", KIT, "--cmd", "concurrent", "--streams", "2,4", "--set", "b", "--tokens", "384",
                      "--reps", "2", "--out", out("concurrent")])
    steps.append([py, KIT, "--base", API, "--model", "step5", "prefill", "--lengths", "8192,32768", "--out",
                  out("prefill")])
    for argv in steps:
        r = subprocess.run(argv, capture_output=True, text=True)
        lines = (r.stdout.strip().splitlines() or [""])
        log(f"{os.path.basename(argv[1])} {' '.join(argv[2:])[:60]}... rc={r.returncode}: {lines[-1][:300]}")
        if r.returncode != 0:
            print(redact(r.stderr[-1500:]), flush=True)
            return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("snapshot")
    sub.add_parser("show")
    s = sub.add_parser("start")
    s.add_argument("name")
    s.add_argument("--arg", action="append", default=[], help="FLAG=VALUE (set or append) or FLAG (boolean)")
    s.add_argument("--drop", action="append", default=[], help="remove FLAG")
    s.add_argument("--env", action="append", default=[], help="K=V on both ranks")
    s.add_argument("--unset", action="append", default=[], help="remove K on both ranks")
    s.add_argument("--mount", action="append", default=[],
                   help="SRC:DST bind-mounted read-only on both ranks (SRC on the head, copied to the worker first)")
    s.add_argument("--timeout", type=float, default=2700)
    b = sub.add_parser("bench")
    b.add_argument("name")
    b.add_argument("--no-concurrent", action="store_true")
    sub.add_parser("stop")
    a = ap.parse_args()
    return {"snapshot": cmd_snapshot, "show": cmd_show, "start": cmd_start, "stop": cmd_stop,
            "bench": cmd_bench}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
