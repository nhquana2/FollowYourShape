"""ReShapeBench: original FYS vs Attention Difference, scored with AS, CLIP, CLIP-dir, CLIP-I and DINO.

Run from the repository root:

    python experiments/reshapebench.py prepare                             # copy images, write jobs
    python experiments/reshapebench.py run --profile fys2 --containers 8   # edit on Modal, download
    python experiments/run_parallel.py <jobs.json> --name reshapebench      # or: run one jobs file, resumable
    python experiments/reshapebench.py metrics --profile fys2              # AS, CLIP, CLIP-dir, CLIP-I, DINO

FYS uses its paper config (Table 3: 15 steps, guidance 2, k_front 2, k_tail 3); Attention Difference uses
the same steps, guidance and k_tail with k_front 0. Neither uses ControlNet. Editing starts from the inverted source latent, so edit.py draws no random noise and
results depend only on the arguments and the code. `run` records the code hash in the run manifest,
and is resumable: finished edits are skipped. Results land in outputs/modal/reshapebench/<method>/<id>.
"""
import argparse
import csv
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "outputs/ReShapeBench"
IMAGES = ROOT / "src/examples/reshapebench"  # uploaded to Modal with src/
EVAL = ROOT / "outputs/reshapebench_eval"
RESULTS = ROOT / "outputs/modal/reshapebench"
BATCH = "reshapebench"  # volume folder: runs/reshapebench/<method>/<id>
SUBSETS = {"single": "single_object", "multi": "multi_object"}

COMMON = "--name flux-dev --num_steps 15 --controlnet_type none --offload"
METHODS = {
    "fys": COMMON + " --guidance 2 --front 2 --inject 3",  # original FYS path: no attention-difference flags
    # Per-step soft masks (P50-P98 range, sigmoid centre 0.3, steepness 15), guidance 2, front 0, inject 3: the defaults,
    # with the mask frozen after step 9 as when these runs were made.
    "attn_q_f0": COMMON + " --attn_diff --attn_diff_freeze",
}
# Upper anchor at P90: weaker signal reaches the mask (larger masks).
METHODS["attn_q_f0_p90"] = METHODS["attn_q_f0"] + " --attn_diff_percentiles 50,90"
# CLIP-I and DINO compare the edit with the source image (no ground-truth edits exist): higher = more preserved.
# Background metrics use the ReShapeBench mask: bg PSNR over background pixels; bg LPIPS (x1e3, SqueezeNet) and
# bg PSNR (PIE) on both images with the mask region zeroed, as in PIE-Bench. Values: (label, format, scale).
METRICS = {"aesthetic": ("AS", ".3f", 1), "clip": ("CLIP", ".2f", 1), "clip_dir": ("CLIP-dir", ".4f", 1),
           "clip_i": ("CLIP-I", ".4f", 1), "dino": ("DINO", ".4f", 1), "bg_psnr": ("bg PSNR", ".2f", 1),
           "bg_lpips": ("bg LPIPS x1e3", ".2f", 1000), "bg_psnr_pie": ("bg PSNR (PIE)", ".2f", 1)}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_samples() -> list[dict]:
    samples = []
    for subset, folder in SUBSETS.items():
        for line in (BENCH / folder / "metadata.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                samples.append({"subset": subset, **json.loads(line)})
    return samples


def code_state() -> dict:
    """Git commit plus a hash of every code file the edits and metrics depend on (the tree may be dirty)."""
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.rstrip()
    files = sorted([ROOT / "modal_app.py", *(ROOT / "src").glob("edit.py"), *(ROOT / "src/flux").rglob("*.py")])
    files = [f for f in files if "__pycache__" not in f.parts]
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty_files": git("status", "--porcelain", "--", "src", "modal_app.py").splitlines(),
        "code_sha256": {str(f.relative_to(ROOT)).replace("\\", "/"): sha256(f.read_bytes()) for f in files},
    }


def copy_inputs(samples: list[dict]) -> None:
    """Source images and evaluation masks into src/examples/reshapebench (uploaded to Modal with src/)."""
    (IMAGES / "masks").mkdir(parents=True, exist_ok=True)
    for sample in samples:
        folder = BENCH / SUBSETS[sample["subset"]]
        shutil.copy2(folder / sample["file_name"], IMAGES / Path(sample["file_name"]).name)
        shutil.copy2(folder / sample["mask"], IMAGES / "masks" / Path(sample["mask"]).name)


def prepare(_args) -> None:
    samples = load_samples()
    copy_inputs(samples)
    jobs = []
    for method, settings in METHODS.items():
        for sample in samples:
            image = Path(sample["file_name"]).name
            args = (f"--source_prompt {shlex.quote(sample['source_prompt'])} --target_prompt {shlex.quote(sample['target_prompt'])}"
                    f" --source_img_dir examples/reshapebench/{image} {settings}")
            jobs.append({"run": f"{method}/{sample['id']}", "args": args})
    EVAL.mkdir(parents=True, exist_ok=True)
    (EVAL / "jobs.json").write_text(json.dumps(jobs, indent=1), encoding="utf-8")
    (EVAL / "samples.json").write_text(json.dumps(samples, indent=1), encoding="utf-8")
    manifest = {
        "methods": METHODS,
        "num_samples": {s: sum(x["subset"] == s for x in samples) for s in SUBSETS},
        "metadata_sha256": {f: sha256((BENCH / f / "metadata.jsonl").read_bytes()) for f in SUBSETS.values()},
        "jobs_sha256": sha256((EVAL / "jobs.json").read_bytes()),
        "prepared_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        **code_state(),
    }
    (EVAL / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"{len(samples)} samples, {len(jobs)} jobs -> {EVAL / 'jobs.json'}")


def load_modal(profile: str | None):
    if profile:
        os.environ["MODAL_PROFILE"] = profile  # must be set before modal reads its config
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.path.insert(0, str(ROOT))
    import modal
    import modal_app
    return modal, modal_app


def run(args) -> None:
    jobs = json.loads((EVAL / "jobs.json").read_text(encoding="utf-8"))
    if args.limit:
        jobs = jobs[:args.limit]
    modal, modal_app = load_modal(args.profile)
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    with modal.enable_output(), modal_app.app.run():
        # One job per method first: a fresh volume downloads the FLUX weights once, not in every
        # container, and both argument sets are checked before fanning out.
        warmup = list({job["run"].split("/")[0]: job for job in reversed(jobs)}.values())
        status = modal_app.run_batch.remote(BATCH, warmup, skip_done=True)
        if "failed" in status.values():
            raise SystemExit(f"warm-up failed: {status}")
        chunks = [jobs[i::args.containers] for i in range(args.containers)]  # round-robin mixes both methods
        calls = [modal_app.run_batch.spawn(BATCH, chunk, skip_done=True) for chunk in chunks if chunk]
        for call in calls:
            status.update(call.get())
    counts = {s: sum(v == s for v in status.values()) for s in ("ok", "skipped", "failed")}
    runs = {"started_at": started, "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"), "profile": args.profile,
            "containers": args.containers, "counts": counts, "status": status, **code_state()}
    (EVAL / f"run_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(json.dumps(runs, indent=1), encoding="utf-8")
    print(counts)
    download(modal_app.VOLUME_NAME, sorted({run.split("/")[0] for run, s in status.items() if s == "ok"}))
    if counts["failed"]:
        raise SystemExit(f"{counts['failed']} jobs failed: " + ", ".join(k for k, v in status.items() if v == "failed"))


def download(volume: str, methods: list[str], attempts: int = 4) -> None:
    """Fetch results one method at a time with retries (one big `volume get` drops the connection)."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    for method in methods:
        command = [sys.executable, "-m", "modal", "volume", "get", "--force", volume, f"runs/{BATCH}/{method}", str(RESULTS)]
        for attempt in range(attempts):
            if subprocess.run(command).returncode == 0:
                break
            print(f"download of {method} failed (attempt {attempt + 1}), retrying")
            time.sleep(10)
        else:
            raise SystemExit(f"could not download {method}")


def edited_image(method: str, sample_id: str) -> Path:
    found = sorted((RESULTS / method / sample_id / "result").glob("img_*.jpg"))
    if len(found) != 1:
        raise SystemExit(f"expected one image for {method}/{sample_id}, found {[f.name for f in found]}")
    return found[0]


def metrics(args) -> None:
    samples = json.loads((EVAL / "samples.json").read_text(encoding="utf-8"))
    copy_inputs(samples)
    items = []
    for method in METHODS:
        for sample in samples:
            image = edited_image(method, sample["id"])
            items.append({
                "key": f"{method}/{sample['id']}",
                "source_image": f"examples/reshapebench/{Path(sample['file_name']).name}",
                "edited_image": f"runs/{BATCH}/{method}/{sample['id']}/result/{image.name}",
                "mask_image": f"examples/reshapebench/masks/{Path(sample['mask']).name}",
                "source_prompt": sample["source_prompt"],
                "target_prompt": sample["target_prompt"],
            })
    modal, modal_app = load_modal(args.profile)
    with modal.enable_output(), modal_app.app.run():
        results = modal_app.image_metrics.remote(items)

    by_key = {r["key"]: r for r in results}
    rows = []
    for method in METHODS:
        for sample in samples:
            r = by_key[f"{method}/{sample['id']}"]
            rows.append({"method": method, "subset": sample["subset"], "id": sample["id"],
                         "instruction": sample["instruction"], **{m: r[m] for m in METRICS}})
    with open(EVAL / "metrics_per_edit.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for method in METHODS:
        for subset in (*SUBSETS, "all"):
            picked = [r for r in rows if r["method"] == method and subset in ("all", r["subset"])]
            with_bg = [r for r in picked if r["bg_psnr"] is not None]  # a whole-image mask leaves no background
            entry = summary[f"{method}/{subset}"] = {"n": len(picked), "n_bg": len(with_bg)}
            for m in METRICS:
                pool = with_bg if m.startswith("bg_") else picked
                entry[m] = sum(r[m] for r in pool) / len(pool)
    (EVAL / "metrics_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    lines = ["| method | subset | n | n bg | " + " | ".join(label for label, _, _ in METRICS.values()) + " |",
             "|---|---|---|---|" + "---|" * len(METRICS)]
    for key, s in summary.items():
        method, subset = key.split("/")
        lines.append(f"| {method} | {subset} | {s['n']} | {s['n_bg']} | " + " | ".join(
            format(s[m] * scale, fmt) for m, (_, fmt, scale) in METRICS.items()) + " |")
    (EVAL / "metrics_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare").set_defaults(func=prepare)
    p = sub.add_parser("run")
    p.add_argument("--profile", help="Modal profile (default: the active one)")
    p.add_argument("--containers", type=int, default=8, help="parallel GPU containers after the warm-up job")
    p.add_argument("--limit", type=int, default=0, help="only the first N jobs (smoke test)")
    p.set_defaults(func=run)
    p = sub.add_parser("metrics")
    p.add_argument("--profile", help="Modal profile (default: the active one)")
    p.set_defaults(func=metrics)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
