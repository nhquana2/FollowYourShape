"""
Batch inference over the ReShapeBench benchmark (3087richard/ReShapeBench).

This mirrors the per-sample editing logic in `edit.py`, but:
  * loads all heavy models (T5, CLIP, FLUX, VAE, ControlNet, DPT, NSFW) ONCE,
  * iterates over all 240 editing cases from the HuggingFace dataset,
  * gives every case its own feature directory (so KV / TDM features from one
    case never leak into the next inversion),
  * writes results into a flat, id-keyed directory tree for later evaluation.

Output layout:
    <output_dir>/
        000101_1/
            edited.jpg      # the edited result
            source.jpg      # original source image (for side-by-side eval)
            meta.json       # prompts, instruction, foreground/background, hyperparams, nsfw score
        000101_2/
        ...

Run from inside `src/` (same import root as edit.py):
    cd src
    HF_HUB_DOWNLOAD_TIMEOUT=120 python run_reshapebench.py \
        --controlnet_type none \
        --output_dir ../output/reshapebench
"""

import os
import re
import json
import time
import argparse

import numpy as np
import torch
import cv2
from einops import rearrange
from PIL import ExifTags, Image

from flux.sampling import (denoise, get_schedule, prepare, unpack,
                           denoise_with_TDM, build_inject_list)
from flux.util import (configs, embed_watermark, load_ae, load_clip,
                       load_flow_model, load_t5)
from transformers import pipeline, DPTForDepthEstimation, DPTImageProcessor
from diffusers import FluxControlNetModel, FluxMultiControlNetModel
from datasets import load_dataset
from torch import tensor

NSFW_THRESHOLD = 0.85
DATASET_NAME = "3087richard/ReShapeBench"


@torch.inference_mode()
def encode(init_image, torch_device, ae):
    """Identical to edit.py: numpy HWC uint8 -> bf16 latent."""
    init_image = torch.from_numpy(init_image).permute(2, 0, 1).float() / 127.5 - 1
    init_image = init_image.unsqueeze(0)
    init_image = init_image.to(torch_device)
    init_image = ae.encode(init_image.to()).to(torch.bfloat16)
    return init_image


def generate_depth_control_image(source_img, dpt_model, dpt_processor, device="cuda"):
    """Same as edit.py, but the DPT model/processor are passed in (loaded once)."""
    W, H = source_img.size
    inputs = dpt_processor(images=source_img, return_tensors="pt").to(device)
    with torch.no_grad():
        depth_map = dpt_model(**inputs).predicted_depth[0].cpu().numpy()
    depth_resized = cv2.resize(depth_map, (W, H), interpolation=cv2.INTER_CUBIC)
    depth_norm = cv2.normalize(depth_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_rgb = cv2.cvtColor(depth_norm, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(depth_rgb)


def generate_canny_control_image(source_img, device="cuda"):
    """Identical to edit.py."""
    image_gray = np.array(source_img.convert("L"))
    image_edges = cv2.Canny(image_gray, 100, 200)
    edge_rgb = cv2.cvtColor(image_edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edge_rgb)


def build_control_patch(raw_pil_image, new_w, new_h, args, ae, controlnet,
                        dpt_model, dpt_processor, torch_device):
    """
    Reproduces the ControlNet preprocessing block of edit.py for one image.
    Returns control_patch (or list of patches for 'multi'), or None.
    """
    if args.controlnet_type == 'single':
        if args.controlnet_kind == 'depth':
            control_pil = generate_depth_control_image(raw_pil_image, dpt_model, dpt_processor, device=torch_device)
        else:  # canny
            control_pil = generate_canny_control_image(raw_pil_image, device=torch_device)
        control_pil = control_pil.crop((0, 0, new_w, new_h))
        control_tensor = (torch.from_numpy(np.array(control_pil)).permute(2, 0, 1).float().unsqueeze(0) / 255.0).to(torch_device).to(torch.bfloat16)
        control_tensor = control_tensor * 2 - 1
        control_latent = ae.encode(control_tensor.float())
        return rearrange(control_latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2).to(torch.bfloat16)

    elif args.controlnet_type == 'multi':
        control_pils = [
            generate_depth_control_image(raw_pil_image, dpt_model, dpt_processor, device=torch_device),
            generate_canny_control_image(raw_pil_image, device=torch_device),
        ]
        control_patches = []
        for control_pil in control_pils:
            control_pil = control_pil.crop((0, 0, new_w, new_h))
            control_tensor = (torch.from_numpy(np.array(control_pil)).permute(2, 0, 1).float().unsqueeze(0) / 255.0).to(torch_device).to(torch.bfloat16)
            control_tensor = control_tensor * 2 - 1
            control_latent = ae.encode(control_tensor.float())
            control_patches.append(rearrange(control_latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2).to(torch.bfloat16))
        return control_patches

    return None


@torch.inference_mode()
def edit_one(sample, args, models, torch_device):
    """
    Run a single editing case. Mirrors the body of edit.py's while-loop for one
    (source image, source prompt, target prompt). Returns (PIL image, nsfw_score).
    """
    t5, clip, model, ae, controlnet, dpt_model, dpt_processor = (
        models['t5'], models['clip'], models['model'], models['ae'],
        models['controlnet'], models['dpt_model'], models['dpt_processor'])

    name = args.name
    sample_id = sample['id']

    # --- preprocess image (ReShapeBench images are already 512x512 RGB) ---
    raw_pil_image = sample['image'].convert('RGB')
    init_image = np.array(raw_pil_image)
    shape = init_image.shape
    new_h = shape[0] if shape[0] % 16 == 0 else shape[0] - shape[0] % 16
    new_w = shape[1] if shape[1] % 16 == 0 else shape[1] - shape[1] % 16
    init_image = init_image[:new_h, :new_w, :]
    # NOTE: edit.py assigns width<-H, height<-W (square images, so harmless). Kept identical.
    width, height = init_image.shape[0], init_image.shape[1]

    control_patch = build_control_patch(raw_pil_image, new_w, new_h, args, ae,
                                         controlnet, dpt_model, dpt_processor, torch_device)

    init_image = encode(init_image, torch_device, ae)

    # --- per-sample feature directory (isolation between cases) ---
    feature_path = os.path.join(args.feature_root, sample_id)
    os.makedirs(feature_path, exist_ok=True)

    info = {}
    info['feature_path'] = feature_path
    info['feature'] = {}
    info['inject_step'] = args.inject

    inp = prepare(t5, clip, init_image, prompt=sample['source_prompt'])
    inp_target = prepare(t5, clip, init_image, prompt=sample['target_prompt'])
    timesteps = get_schedule(args.num_steps, inp["img"].shape[1], shift=(name != "flux-schnell"))

    inject_list = build_inject_list(num_inference_steps=len(timesteps),
                                    inject_step=info['inject_step'], tail_pad=1, front_pad=args.front)

    # ControlNet schedule (identical values to edit.py)
    if args.controlnet_type == 'single':
        control_mode = 0
        controlnet_scale = 0.5
        guidance_start, guidance_end = 0.0, 0.4
    elif args.controlnet_type == 'multi':
        control_mode = [tensor([0], dtype=torch.long).to(torch_device),
                        tensor([0], dtype=torch.long).to(torch_device)]
        controlnet_scale = [2.5, 3.35]
        guidance_start, guidance_end = 0.1, 0.7
    else:
        control_mode = None
        controlnet_scale = None
        guidance_start, guidance_end = 0.0, 0.0

    # inversion -> initial noise
    z, info = denoise(model, **inp, timesteps=timesteps, guidance=1, inverse=True, info=info,
                      inject_list=inject_list, controlnet=controlnet, control_patch=control_patch,
                      controlnet_scale=controlnet_scale, controlnet_mode=control_mode,
                      guidance_start=guidance_start, guidance_end=guidance_end)

    inp_target["img"] = z
    timesteps = get_schedule(args.num_steps, inp_target["img"].shape[1], shift=(name != "flux-schnell"))

    # denoise with trajectory-guided region control
    x, _ = denoise_with_TDM(model, **inp_target, timesteps=timesteps, guidance=args.guidance,
                            inverse=False, info=info, width=width, height=height,
                            inject_list=inject_list, tail_pad=1, front_pad=args.front,
                            controlnet=controlnet, control_patch=control_patch,
                            controlnet_scale=controlnet_scale, controlnet_mode=control_mode,
                            guidance_start=guidance_start, guidance_end=guidance_end)

    # decode latent -> pixels
    batch_x = unpack(x.float(), width, height)
    x = batch_x[0].unsqueeze(0)
    with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
        x = ae.decode(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    x = x.clamp(-1, 1)
    x = embed_watermark(x.float())
    x = rearrange(x[0], "c h w -> h w c")
    img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

    nsfw_score = [c["score"] for c in models['nsfw_classifier'](img) if c["label"] == "nsfw"][0]
    return img, raw_pil_image, float(nsfw_score)


def main():
    parser = argparse.ArgumentParser(description='Batch inference over ReShapeBench')
    parser.add_argument('--name', default='flux-dev', type=str, help='flux model')
    parser.add_argument('--output_dir', default='../output/reshapebench', type=str,
                        help='root dir for edited images + metadata')
    parser.add_argument('--feature_root', default='/tmp/reshapebench_features', type=str,
                        help='root dir for per-sample feature caches')
    # paper hyperparameters (Table 3)
    parser.add_argument('--guidance', type=float, default=2.0, help='guidance scale (paper: 2)')
    parser.add_argument('--num_steps', type=int, default=15, help='inversion/denoising steps (paper: 15)')
    parser.add_argument('--front', type=int, default=2, help='k_front trajectory stabilization steps (paper: 2)')
    parser.add_argument('--inject', type=int, default=4, help='feature-sharing steps')
    parser.add_argument('--controlnet_type', type=str, default='none', choices=['none', 'single', 'multi'],
                        help="'none' = Ours w/o ControlNet, 'multi' = full model (depth+canny), 'single' = depth or canny")
    parser.add_argument('--controlnet_kind', type=str, default='depth', choices=['depth', 'canny'],
                        help="used only when --controlnet_type single")
    parser.add_argument('--offload', action='store_true', help='offload modules to CPU for <40GB GPUs')
    parser.add_argument('--limit', type=int, default=None, help='process only the first N cases (debugging)')
    parser.add_argument('--overwrite', action='store_true', help='re-run cases even if edited.jpg exists')
    parser.add_argument('--no_save_source', action='store_true', help='do not also save the source image')
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    if args.name not in configs:
        raise ValueError(f"Unknown model name {args.name}; choose from {', '.join(configs.keys())}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.feature_root, exist_ok=True)

    # ----------------- load every model ONCE -----------------
    print("Loading models ...")
    t5 = load_t5(torch_device, max_length=256 if args.name == "flux-schnell" else 512)
    clip = load_clip(torch_device)
    model = load_flow_model(args.name, device="cpu" if args.offload else torch_device)
    ae = load_ae(args.name, device="cpu" if args.offload else torch_device)
    nsfw_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection", device=device)

    if args.offload:
        # offload keeps weights on CPU and is incompatible with the single-pass
        # batch flow below, which expects all modules resident. Warn loudly.
        print("WARNING: --offload moves modules to CPU but this batch script keeps them "
              "resident for speed. On <40GB GPUs prefer running fewer cases per process.")
        ae.encoder.to(torch_device)

    controlnet = None
    dpt_model = dpt_processor = None
    if args.controlnet_type in ('single', 'multi'):
        dpt_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to(torch_device).eval()
        dpt_processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas", use_fast=True)
    if args.controlnet_type == 'single':
        repo = ("Shakker-Labs/FLUX.1-dev-ControlNet-Depth" if args.controlnet_kind == 'depth'
                else "InstantX/FLUX.1-dev-Controlnet-Canny")
        controlnet = FluxControlNetModel.from_pretrained(repo, torch_dtype=torch.bfloat16).to(torch_device)
    elif args.controlnet_type == 'multi':
        controlnet_union = FluxControlNetModel.from_pretrained(
            'Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro', torch_dtype=torch.bfloat16).to(torch_device)
        controlnet = FluxMultiControlNetModel([controlnet_union])

    models = dict(t5=t5, clip=clip, model=model, ae=ae, controlnet=controlnet,
                  dpt_model=dpt_model, dpt_processor=dpt_processor, nsfw_classifier=nsfw_classifier)

    # ----------------- load dataset -----------------
    print(f"Loading dataset {DATASET_NAME} ...")
    ds = load_dataset(DATASET_NAME)['train']
    n_total = len(ds) if args.limit is None else min(args.limit, len(ds))
    print(f"Processing {n_total} / {len(ds)} cases with controlnet_type='{args.controlnet_type}'.")

    hyperparams = dict(name=args.name, guidance=args.guidance, num_steps=args.num_steps,
                       front=args.front, inject=args.inject, controlnet_type=args.controlnet_type,
                       controlnet_kind=args.controlnet_kind if args.controlnet_type == 'single' else None)

    done, skipped, failed = 0, 0, 0
    for i in range(n_total):
        sample = ds[i]
        sample_id = sample['id']
        out_dir = os.path.join(args.output_dir, sample_id)
        edited_path = os.path.join(out_dir, "edited.jpg")

        if os.path.exists(edited_path) and not args.overwrite:
            skipped += 1
            print(f"[{i+1}/{n_total}] {sample_id}: exists, skip")
            continue

        os.makedirs(out_dir, exist_ok=True)
        t0 = time.perf_counter()
        try:
            img, src_img, nsfw_score = edit_one(sample, args, models, torch_device)
        except Exception as e:
            failed += 1
            print(f"[{i+1}/{n_total}] {sample_id}: FAILED ({type(e).__name__}: {e})")
            continue

        # save edited image with the same EXIF metadata edit.py writes
        exif_data = Image.Exif()
        exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux"
        exif_data[ExifTags.Base.Make] = "Black Forest Labs"
        exif_data[ExifTags.Base.Model] = args.name
        exif_data[ExifTags.Base.ImageDescription] = sample['target_prompt']
        img.save(edited_path, exif=exif_data, quality=95, subsampling=0)

        if not args.no_save_source:
            src_img.save(os.path.join(out_dir, "source.jpg"), quality=95, subsampling=0)

        meta = dict(
            id=sample_id,
            source_prompt=sample['source_prompt'],
            target_prompt=sample['target_prompt'],
            instruction=sample.get('instruction'),
            foreground=sample.get('foreground'),
            foreground_target=sample.get('foreground_target'),
            background=sample.get('background'),
            num_objects=sample.get('num_objects'),
            mask=sample.get('mask'),
            nsfw_score=nsfw_score,
            nsfw_flagged=bool(nsfw_score >= NSFW_THRESHOLD),
            hyperparams=hyperparams,
        )
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        done += 1
        dt = time.perf_counter() - t0
        flag = "  [NSFW!]" if nsfw_score >= NSFW_THRESHOLD else ""
        print(f"[{i+1}/{n_total}] {sample_id}: done in {dt:.1f}s (nsfw={nsfw_score:.3f}){flag}")

    print(f"\nFinished. saved={done}, skipped={skipped}, failed={failed}, "
          f"output -> {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
