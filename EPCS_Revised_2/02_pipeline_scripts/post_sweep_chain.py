"""
Post-sweep chain: waits for the GPU sweep to finish and the GPU to go idle, then runs the
analysis/build steps in priority order. Continues past failures, recording each.

  python 07_revision3/06_scripts/post_sweep_chain.py --dry-run
  python 07_revision3/06_scripts/post_sweep_chain.py                 # wait, then run all
  python 07_revision3/06_scripts/post_sweep_chain.py --skip-wait --only stats
  --only accepts step names or group letters a..e (repeatable / comma separated).
  --log-dir overrides the log directory (used for dry-run tests).
Never passes --force to latency. Does not touch the sweep.
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys, time
from datetime import datetime
from pathlib import Path

R3 = Path(__file__).resolve().parents[1]
CODE = R3 / "03_code"
SCRIPTS = R3 / "06_scripts"
LOG_DEFAULT = R3 / "04_results" / "logs" / "post_sweep"
sys.path.insert(0, str(CODE / "runners"))
sys.path.insert(0, str(CODE / "analysis"))


def _helpers():
    import run_all_analysis as raa
    return raa


def build_plan() -> list[dict]:
    A = CODE / "analysis"

    def gradcam(ds):
        def f():
            rd = _helpers().newest_epcs_run(ds)
            return None if rd is None else ["--run", str(rd)]
        return f

    def nu_pred(extra):
        def f():
            raa = _helpers()
            rd = raa.newest_epcs_run("nuinsseg")
            if rd is None:
                return None
            return ["--predictions", str(rd / "test_predictions"),
                    "--split", str(raa.manifest_for(rd, "nuinsseg"))] + extra
        return f

    P = [dict(name="stats", group="a", script=A / "stats_r3.py", args=[], cwd=A),
         dict(name="latency", group="b", script=A / "latency_r3.py", args=[], cwd=A,
              needs_idle_gpu=True)]
    for ds in ("cryonuseg", "nuinsseg"):
        P.append(dict(name=f"gradcam_{ds}", group="c", script=A / "gradcam_r3.py",
                      args=gradcam(ds), cwd=A))
    P.append(dict(name="dpnp_nuinsseg", group="d", script=A / "dpnp_r3.py",
                  args=nu_pred(["--only-split", "test", "--tag", "r3_nuinsseg"]), cwd=A))
    P.append(dict(name="instances_nuinsseg", group="d", script=A / "instances_r3.py",
                  args=nu_pred(["--tag", "r3"]), cwd=A))
    builders = [("build_metrics_workbook", SCRIPTS), ("build_epoch_workbooks", SCRIPTS),
                ("make_result_figures", SCRIPTS), ("shrink_large_figures", SCRIPTS),
                ("build_paper_tables", CODE / "tables"), ("build_comparison_workbooks", SCRIPTS),
                ("build_master_workbook", SCRIPTS), ("build_comment_coverage", SCRIPTS),
                ("build_response_pdf", SCRIPTS), ("build_overleaf_package", SCRIPTS), ("build_overleaf_project", SCRIPTS), ("build_lite_zip", SCRIPTS),
                ("build_task_tracker", SCRIPTS), ("build_results_view", SCRIPTS)]
    for b, d in builders:
        P.append(dict(name=b, group="e", script=d / f"{b}.py", args=[], cwd=d))
    return P


def sweep_pending() -> tuple[int | None, str]:
    try:
        r = subprocess.run([sys.executable, str(CODE / "runners" / "run_sweep.py"), "--status"],
                           capture_output=True, text=True, timeout=300,
                           cwd=str(CODE / "runners"))
        m = re.search(r"pending:\s*(\d+)", r.stdout)
        return (int(m.group(1)) if m else None), r.stdout.strip()[-80:]
    except Exception as e:
        return None, f"status failed: {e}"


def other_procs() -> list[str]:
    ps = ("Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
          "ForEach-Object { '' + $_.ProcessId + [char]9 + $_.CommandLine }")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True,
                             text=True, timeout=60).stdout
    except Exception:
        return ["<could not list processes>"]
    me = str(os.getpid())
    found = []
    for line in out.splitlines():
        pid, _, cmd = line.partition("\t")
        if pid.strip() == me or not cmd:
            continue
        if "post_sweep_chain" in cmd or "--status" in cmd:
            continue
        if "run_sweep" in cmd or "protocol_r3" in cmd:
            found.append(f"{pid} {cmd[-120:]}")
    return found


def wait_for_sweep():
    from latency_r3 import gpu_is_busy
    while True:
        pend, _ = sweep_pending()
        procs = other_procs()
        if pend == 0 and not procs:
            busy, gnote = gpu_is_busy()
        else:
            busy, gnote = True, "not checked (sweep still active)"
        print(f"[{datetime.now():%H:%M:%S}] pending={pend} procs={len(procs)} "
              f"gpu_busy={busy} ({gnote})", flush=True)
        if pend == 0 and not procs and not busy:
            return
        time.sleep(60)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-wait", action="store_true")
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("--log-dir", default=str(LOG_DEFAULT))
    a = ap.parse_args()
    only = {x.strip() for v in a.only for x in v.split(",") if x.strip()}
    full = build_plan()
    unknown = only - {s["name"] for s in full} - set("abcde")
    if unknown:
        print("unknown --only:", unknown)
        return 2
    plan = [s for s in full if not only or s["name"] in only or s["group"] in only]

    if a.dry_run:
        print("PLAN (dry-run, nothing executed):")
        for s in plan:
            args = s["args"]() if callable(s["args"]) else s["args"]
            argtxt = "<SKIP: no completed run yet>" if args is None else " ".join(args)
            print(f"  [{s['group']}] {s['name']:28s} exists={s['script'].is_file()!s:5} "
                  f"{s['script'].name} {argtxt}"
                  + ("   (requires idle GPU, no --force)" if s.get("needs_idle_gpu") else ""))
        print("wait:", "skipped" if a.skip_wait else
              "poll 60 s until sweep 0 pending, no sweep procs, GPU idle")
        print("logs:", a.log_dir)
        return 0

    logd = Path(a.log_dir)
    logd.mkdir(parents=True, exist_ok=True)
    status = {"started": datetime.now().isoformat(timespec="seconds"), "steps": {}}

    def save():
        (logd / "chain_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    save()
    if not a.skip_wait:
        wait_for_sweep()
    status["sweep_done"] = datetime.now().isoformat(timespec="seconds")
    save()

    from latency_r3 import gpu_is_busy
    for s in plan:
        name = s["name"]
        rec = {"script": str(s["script"])}
        status["steps"][name] = rec
        if not s["script"].is_file():
            rec.update(result="skipped", note="script does not exist")
            save(); print(f"SKIP {name}: script missing"); continue
        args = s["args"]() if callable(s["args"]) else list(s["args"])
        if args is None:
            rec.update(result="skipped", note="no completed run")
            save(); print(f"SKIP {name}: no completed run"); continue
        if s.get("needs_idle_gpu"):
            busy, gnote = gpu_is_busy()
            if busy:
                rec.update(result="skipped", note=f"GPU busy: {gnote}")
                save(); print(f"SKIP {name}: GPU busy"); continue
        cmd = [sys.executable, str(s["script"])] + args
        rec["cmd"] = cmd
        rec["start"] = datetime.now().isoformat(timespec="seconds")
        save()
        print(f"RUN {name}: {' '.join(cmd)}", flush=True)
        t0 = time.time()
        try:
            p = subprocess.run(cmd, cwd=str(s["cwd"]), capture_output=True, text=True,
                               errors="replace")
            code, out, err = p.returncode, p.stdout, p.stderr
        except Exception as e:
            code, out, err = -1, "", f"launcher exception: {e}"
        (logd / f"{name}.log").write_text(
            f"# {' '.join(cmd)}\n# exit code: {code}\n# seconds: {time.time()-t0:.1f}\n\n"
            f"===== STDOUT =====\n{out}\n===== STDERR =====\n{err}\n", encoding="utf-8")
        rec.update(result="ok" if code == 0 else "FAILED", exit_code=code,
                   seconds=round(time.time() - t0, 1), log=str(logd / f"{name}.log"))
        save()
        print(f"  -> {rec['result']} (exit {code}, {rec['seconds']} s)", flush=True)
    status["finished"] = datetime.now().isoformat(timespec="seconds")
    save()
    bad = [n for n, r in status["steps"].items() if r["result"] != "ok"]
    print("not ok:", bad if bad else "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
