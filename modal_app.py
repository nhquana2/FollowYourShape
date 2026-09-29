"""Run src/edit.py on a Modal GPU using the local working tree.

Mirrors the Modal notebook setup: the shared volume is mounted at
/mnt/follow-your-shape and doubles as HF_HOME, so cached FLUX weights are reused.
Local code is uploaded on every run, so no git push/pull is needed.

Usage (from the repository root):

    modal run modal_app.py --run parrot_attn --args "--source_img_dir examples/source/parrot.png ... --attn_diff"
    modal run modal_app.py::batch --jobs jobs.json   # [{"run": ..., "args": ...}, ...], models load once

Paths inside --args are relative to src/, as in the notebook. --output_dir and
--vis_path are set automatically to runs/<run>/result and runs/<run>/maps on the
volume, and the run folder is downloaded to outputs/modal/<run> afterwards.

Configuration via environment variables: FYS_VOLUME, FYS_HF_SECRET, FYS_GPU.
"""

import os
import shlex
import subprocess
import sys
from pathlib import Path

import modal

VOLUME_NAME = os.environ.get("FYS_VOLUME", "follow-your-shape")
HF_SECRET_NAME = os.environ.get("FYS_HF_SECRET", "huggingface-secret")  # provides HF_TOKEN
GPU = os.environ.get("FYS_GPU", "L40S")

VOLUME_ROOT = "/mnt/follow-your-shape"
SRC = "/root/FollowYourShape/src"
LOCAL_ROOT = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")  # OpenCV runtime libraries
    # Core packages pinned to the Modal notebook versions for reproducible results.
    .uv_pip_install("torch==2.8.0", "torchvision==0.23.0", index_url="https://download.pytorch.org/whl/cu129")
    .uv_pip_install(
        "torch==2.8.0", "torchvision==0.23.0",  # keep the CUDA build above
        "transformers==4.56.0", "tokenizers==0.22.0", "diffusers==0.35.1", "huggingface-hub==0.34.4",
        "einops", "fire", "safetensors", "sentencepiece", "protobuf", "requests",
        "invisible-watermark", "matplotlib", "seaborn", "scipy", "scikit-image",
        "opencv-python", "datasets",
    )
    .env({
        "HF_HOME": VOLUME_ROOT,
        "HF_HUB_DOWNLOAD_TIMEOUT": "120",
        "PYTHONPATH": SRC,
        "MPLBACKEND": "Agg",
    })
    .uv_pip_install("torch==2.8.0", "torchvision==0.23.0", "lpips==0.1.4")  # background LPIPS metric
    # Upload the local source last so code edits do not rebuild the dependency layers.
    .add_local_dir(
        LOCAL_ROOT / "src", SRC,
        ignore=["**/__pycache__/**", "examples/edit-result/**", "examples/edit-map-visualization/**"],
    )
)

app = modal.App("follow-your-shape", image=image)
volume = modal.Volume.from_name(VOLUME_NAME)


FUNCTION_OPTIONS = dict(
    gpu=GPU,
    volumes={VOLUME_ROOT: volume},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    memory=65536,  # --offload keeps T5/FLUX on CPU; dynamic masks cache source K/V on CPU.
    timeout=3600,
)


def with_run_outputs(argv: list[str], run_dir: Path) -> list[str]:
    if "--output_dir" not in argv:
        argv = argv + ["--output_dir", str(run_dir / "result")]
    if "--vis_path" not in argv:
        argv = argv + ["--vis_path", str(run_dir / "maps")]
    return argv


@app.function(**FUNCTION_OPTIONS)
def run_edit(run: str, args: str, script: str = "edit.py") -> int:
    run_dir = Path(VOLUME_ROOT) / "runs" / run
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, script, *with_run_outputs(shlex.split(args), run_dir)]
    (run_dir / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")

    with open(run_dir / "log.txt", "w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=SRC, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        returncode = process.wait()
    volume.commit()
    return returncode


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def cache_model_loaders(edit) -> None:
    """Memoize edit.py's model loaders so batch jobs share one copy of each model."""
    import functools

    from diffusers import FluxControlNetModel

    cached = functools.cache
    for name in ("load_t5", "load_clip", "load_flow_model", "load_ae", "pipeline"):
        setattr(edit, name, cached(getattr(edit, name)))
    # Keyword arguments (torch_dtype) are hashable, so the classmethods can be cached too.
    edit.FluxControlNetModel = type("CachedFluxControlNetModel", (), {
        "from_pretrained": staticmethod(cached(FluxControlNetModel.from_pretrained)),
    })


@app.function(**{**FUNCTION_OPTIONS, "timeout": 4 * 3600})
def run_batch(batch: str, jobs: list[dict], skip_done: bool = False) -> dict[str, str]:
    """Run several edit.py jobs in one process, loading each model only once.

    With skip_done, jobs whose result/ folder already holds an image are skipped (resumable runs).
    """
    import contextlib
    import traceback

    os.chdir(SRC)
    sys.argv = ["edit.py"]
    import edit

    cache_model_loaders(edit)
    status = {}
    for job in jobs:
        run_dir = Path(VOLUME_ROOT) / "runs" / batch / job["run"]
        if skip_done and any((run_dir / "result").glob("img_*.jpg")):
            status[job["run"]] = "skipped"
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        argv = with_run_outputs(shlex.split(job["args"]), run_dir)
        (run_dir / "command.txt").write_text(shlex.join(["edit.py", *argv]) + "\n", encoding="utf-8")
        print(f"=== [{len(status) + 1}/{len(jobs)}] {job['run']}", flush=True)
        with open(run_dir / "log.txt", "w", encoding="utf-8") as log:
            with contextlib.redirect_stdout(Tee(sys.__stdout__, log)), contextlib.redirect_stderr(Tee(sys.__stderr__, log)):
                try:
                    edit.main(edit.parse_args(argv))
                    status[job["run"]] = "ok"
                except BaseException:  # argparse raises SystemExit on bad arguments
                    traceback.print_exc()
                    status[job["run"]] = "failed"
        volume.commit()
    return status


AESTHETIC_URL = "https://github.com/LAION-AI/aesthetic-predictor/raw/main/sa_0_4_vit_l_14_linear.pth"


@app.function(**FUNCTION_OPTIONS)
def image_metrics(items: list[dict]) -> list[dict]:
    """LAION aesthetic score (V1 linear head, as in the FYS paper), CLIP score x100, CLIP directional similarity,
    and source-edit similarity: CLIP-I (CLIP image cosine) and DINO (DINO ViT-S/16 CLS cosine).

    Each item: {"key", "source_image" (relative to src/), "edited_image" (relative to the volume, or src/ if
    "edited_in_src"), "source_prompt", "target_prompt"}. CLIP metrics use OpenAI CLIP ViT-L/14; text is truncated
    to 77 tokens. DINO follows the DreamBooth/MagicBrush preprocessing (resize 256 bicubic, center crop 224).
    With an optional "mask_image" (relative to src/, white = edit region) background PSNR and LPIPS are added.
    """
    import urllib.request

    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms
    from transformers import CLIPModel, CLIPProcessor, ViTModel

    weights = Path(VOLUME_ROOT) / "metrics" / Path(AESTHETIC_URL).name
    if not weights.exists():
        weights.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(AESTHETIC_URL, weights)
        volume.commit()
    head = torch.nn.Linear(768, 1)
    head.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    head = head.cuda().eval()
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float32).cuda().eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    @torch.no_grad()
    def embed_images(paths):
        pixels = processor(images=[Image.open(p).convert("RGB") for p in paths], return_tensors="pt")["pixel_values"]
        features = model.get_image_features(pixel_values=pixels.cuda())
        return features / features.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def embed_texts(texts):
        tokens = processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77)
        features = model.get_text_features(input_ids=tokens["input_ids"].cuda(), attention_mask=tokens["attention_mask"].cuda())
        return features / features.norm(dim=-1, keepdim=True)

    # No pooling layer: the DINO checkpoint has no pooler weights; its embedding is the final-norm CLS token.
    dino = ViTModel.from_pretrained("facebook/dino-vits16", add_pooling_layer=False).cuda().eval()
    dino_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    @torch.no_grad()
    def embed_dino(paths):
        pixels = torch.stack([dino_transform(Image.open(p).convert("RGB")) for p in paths]).cuda()
        features = dino(pixel_values=pixels).last_hidden_state[:, 0]
        return features / features.norm(dim=-1, keepdim=True)

    lpips_model = None

    @torch.no_grad()
    def background_metrics(paths, mask_path):
        """Outside the edit mask (white = edited). PSNR over background pixels only; PIE-Bench style PSNR and
        LPIPS (SqueezeNet) on both images with the edit region zeroed."""
        nonlocal lpips_model
        if lpips_model is None:
            import lpips
            lpips_model = lpips.LPIPS(net="squeeze", verbose=False).cuda().eval()
        src, edit = (torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255).permute(2, 0, 1)
                     for p in paths)
        mask = Image.open(mask_path).convert("L").resize((src.shape[2], src.shape[1]), Image.NEAREST)
        keep = torch.from_numpy(np.asarray(mask) < 128)[None].float()  # 1 = background
        if keep.sum() == 0:  # a mask covering the whole image leaves no background to measure
            return {"bg_psnr": None, "bg_psnr_pie": None, "bg_lpips": None}
        mse = ((src - edit) ** 2 * keep).sum() / (keep.sum() * 3)
        mse_pie = ((src * keep - edit * keep) ** 2).mean()
        distance = lpips_model((src * keep)[None].cuda() * 2 - 1, (edit * keep)[None].cuda() * 2 - 1)
        return {"bg_psnr": (10 * torch.log10(1 / mse.clamp_min(1e-10))).item(),
                "bg_psnr_pie": (10 * torch.log10(1 / mse_pie.clamp_min(1e-10))).item(),
                "bg_lpips": distance.item()}

    results = []
    for item in items:
        edited_root = Path(SRC) if item.get("edited_in_src") else Path(VOLUME_ROOT)
        paths = [Path(SRC) / item["source_image"], edited_root / item["edited_image"]]
        img_src, img_edit = embed_images(paths)
        dino_src, dino_edit = embed_dino(paths)
        txt_src, txt_tgt = embed_texts([item["source_prompt"], item["target_prompt"]])
        direction = torch.nn.functional.cosine_similarity(img_edit - img_src, txt_tgt - txt_src, dim=0)
        with torch.no_grad():
            aesthetic = head(img_edit[None]).item()
        results.append({
            "key": item["key"],
            "aesthetic": aesthetic,
            "clip": 100 * max((img_edit @ txt_tgt).item(), 0.0),
            "clip_dir": direction.item(),
            "clip_i": (img_src @ img_edit).item(),
            "dino": (dino_src @ dino_edit).item(),
            **(background_metrics(paths, Path(SRC) / item["mask_image"]) if item.get("mask_image") else {}),
        })
    return results


@app.local_entrypoint()
def metrics(items: str, out: str):
    """modal run modal_app.py::metrics --items items.json --out metrics.json"""
    import json

    results = image_metrics.remote(json.loads(Path(items).read_text(encoding="utf-8")))
    Path(out).write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"wrote {len(results)} rows to {out}")


def download(remote: str) -> Path:
    destination = LOCAL_ROOT / "outputs" / "modal"
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "modal", "volume", "get", "--force", VOLUME_NAME, f"runs/{remote}", str(destination)],
        check=True,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},  # the CLI prints non-cp1252 symbols on Windows
    )
    print(f"Downloaded to {destination / remote}")
    return destination / remote


@app.local_entrypoint()
def main(run: str, args: str, script: str = "edit.py", download_results: bool = True):
    returncode = run_edit.remote(run, args, script)
    print(f"{script} exited with code {returncode}; results in volume {VOLUME_NAME}:/runs/{run}")
    if download_results:
        download(run)
    if returncode:
        raise SystemExit(returncode)


@app.local_entrypoint()
def batch(jobs: str, name: str = "", download_results: bool = True):
    """Run a JSON list of {"run": ..., "args": ...} jobs: modal run modal_app.py::batch --jobs jobs.json"""
    import json

    name = name or Path(jobs).stem
    status = run_batch.remote(name, json.loads(Path(jobs).read_text(encoding="utf-8")))
    for run, result in status.items():
        print(f"{result:>6}  {run}")
    if download_results:
        download(name)
    if any(result != "ok" for result in status.values()):
        raise SystemExit(1)
