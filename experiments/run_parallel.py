"""Run a jobs JSON on several Modal containers at once (resumable: finished edits are skipped).

    PYTHONIOENCODING=utf-8 python experiments/run_parallel.py outputs/reshapebench_eval/reshapebench_xb.json --containers 10 --profile fys2

Jobs are dealt round-robin to the containers; results land in outputs/modal/<name> (default: the jobs file stem),
downloaded one run group (the part of "run" before the first slash) at a time. The FLUX weights
must already be on the volume (run one job with `modal run modal_app.py::batch` first on a fresh volume).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jobs")
    parser.add_argument("--containers", type=int, default=10)
    parser.add_argument("--profile")
    parser.add_argument("--name", help="batch folder on the volume (runs/<name>/<run>)")
    args = parser.parse_args()
    if args.profile:
        os.environ["MODAL_PROFILE"] = args.profile  # must be set before modal reads its config
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.path.insert(0, str(ROOT))
    import modal
    import modal_app

    name = args.name or Path(args.jobs).stem
    jobs = json.loads(Path(args.jobs).read_text(encoding="utf-8"))
    chunks = [jobs[i::args.containers] for i in range(args.containers)]
    status = {}
    with modal.enable_output(), modal_app.app.run():
        calls = [modal_app.run_batch.spawn(name, chunk, skip_done=True) for chunk in chunks if chunk]
        for call in calls:
            status.update(call.get())
    counts = {s: sum(v == s for v in status.values()) for s in ("ok", "skipped", "failed")}
    print(counts)
    for run, result in status.items():
        if result == "failed":
            print("failed", run)
    destination = ROOT / "outputs/modal" / name
    destination.mkdir(parents=True, exist_ok=True)
    for group in sorted({job["run"].split("/")[0] for job in jobs}):  # one big `volume get` drops the connection
        command = [sys.executable, "-m", "modal", "volume", "get", "--force", modal_app.VOLUME_NAME, f"runs/{name}/{group}", str(destination)]
        for attempt in range(4):
            if subprocess.run(command).returncode == 0:
                break
            print(f"download of {group} failed (attempt {attempt + 1}), retrying")
            time.sleep(10)
        else:
            raise SystemExit(f"could not download {group}")
    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
