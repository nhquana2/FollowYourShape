<div align="center">
  
# ✂️ Follow-Your-Shape: Shape-Aware Image Editing via Trajectory-Guided Region Control

[Zeqian Long](https://github.com/Zeqian-Long)<sup>2*</sup>, [Mingzhe Zheng](https://scholar.google.com/citations?user=U6bikksAAAAJ&hl=en)<sup>1*</sup>, [Kunyu Feng](https://github.com/fkyyyy)<sup>1*</sup>, [Xinhua Zhang](https://github.com/mayuelala/FollowYourShape), [Hongyu Liu](https://scholar.google.com/citations?user=bLRjUzAAAAAJ&hl=en)<sup>1</sup>, [Harry Yang](https://leehomyc.github.io/)<sup>1</sup>, [Linfeng Zhang](http://www.zhanglinfeng.tech/)<sup>3</sup>, [Qifeng Chen](https://cqf.io/)<sup>1</sup>, [Yue Ma](https://github.com/mayuelala)<sup>1</sup>,

<sup>1</sup> HKUST,  <sup>2</sup> UIUC,  <sup>3</sup> Shanghai Jiao Tong University


<a href='https://follow-your-shape.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
[![arXiv](https://img.shields.io/badge/arXiv-2508.08134-b31b1b.svg)](https://arxiv.org/abs/2508.08134)
[![Huggingface space](https://img.shields.io/badge/🤗-Huggingface%20Space-orange.svg)](https://huggingface.co/papers/2508.08134) 
[![GitHub Stars](https://img.shields.io/github/stars/mayuelala/FollowYourShape)](https://github.com/mayuelala/FollowYourShape)

<!-- [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/kv-edit-training-free-image-editing-for/text-based-image-editing-on-pie-bench)](https://paperswithcode.com/sota/text-based-image-editing-on-pie-bench?p=kv-edit-training-free-image-editing-for)
[![Static Badge](https://img.shields.io/badge/comfyUI-KV_Edit-blue)](https://github.com/smthemex/ComfyUI_KV_Edit) -->

</div>


<p>
We propose <b>Follow-Your-Shape</b>, a training-free and mask-free framework that supports precise and controllable editing of object shapes while strictly preserving non-target content. Our method achieves superior editability and visual fidelity, particularly in tasks requiring large-scale shape replacement.
</p>



<p align="center">
<img src="resources/teaser.jpg" width="1080px"/>
</p>

# 🔥 News
<!-- - [2025.3.12] Thanks to @[smthemex](https://github.com/smthemex) for integrating KV-Edit into [ComfyUI](https://github.com/smthemex/ComfyUI_KV_Edit)!
- [2025.3.4] We update "attention scale" feature to reduce the discontinuity with the background.
- [2025.2.26] Our paper is featured in [huggingface Papers](https://huggingface.co/papers/2502.17363)! -->
- [2026.3.30] 🤗 The ReShapeBench Dataset can be found at our huggingface space!
- [2026.1.25] 🎉 Our paper is accepted to ICLR 2026!
- [2025.8.11] Code released!
- [2025.8.11] Paper released!
<!-- - [2025.2.25] More results can be found in our [project page](https://xilluill.github.io/projectpages/KV-Edit/)! -->

<!-- # 👨‍💻 ToDo
- ☑️ Release the gradio demo
- ☑️ Release the huggingface space for image editing
- ☑️ Release the paper -->


# 📖 Pipeline
<p>
<img src="resources/pipeline.jpg" width="1080px"/>


# 🛠️ Code Setup
The environment of our code is the same as FLUX, you can refer to the [official repo](https://github.com/black-forest-labs/flux/tree/main) of FLUX, or running the following command to construct the environment.
```
conda create --n FollowYourShape python=3.10
conda activate FollowYourShape
pip install -e ".[all]"
```

We recommend you to run the experiment on a single A100 GPU.

<!-- # 🚀 Test
We have provided several scripts to reproduce the results in the paper, mainly including 3 types of editing: Stylization, Adding, Replacing. We suggest to run the experiment on a single A100 GPU. -->

<!-- ## Stylization
<table class="center">
<tr>
  <td width=10% align="center">Ref Style</td>
  <td width=30% align="center"><img src="../assets/repo_figures/examples/source/nobel.jpg" raw=true></td>
	<td width=30% align="center"><img src="../assets/repo_figures/examples/source/art.jpg" raw=true></td>
  <td width=30% align="center"><img src="../assets/repo_figures/examples/source/cartoon.jpg" raw=true></td>
</tr>
<tr>
  <td width="10%" align="center">Editing Scripts</td>
  <td width="30%" align="center"><a href="src/run_nobel_trump.sh">Trump</a></td>
  <td width="30%" align="center"><a href="src/run_art_mari.sh"> Marilyn Monroe</a></td>
  <td width="30%" align="center"><a href="src/run_cartoon_ein.sh">Einstein</a></td>
</tr>
<tr>
  <td width=10% align="center">Edtied image</td>
  <td width=30% align="center"><img src="../assets/repo_figures/examples/edit/nobel_Trump.jpg" raw=true></td>
	<td width=30% align="center"><img src="../assets/repo_figures/examples/edit/art_mari.jpg" raw=true></td>
  <td width=30% align="center"><img src="../assets/repo_figures/examples/edit/cartoon_ein.jpg" raw=true></td>
</tr>

<tr>
  <td width="10%" align="center">Editing Scripts</td>
  <td width="30%" align="center"><a href="src/run_nobel_biden.sh">Biden</a></td>
  <td width="30%" align="center"><a href="src/run_art_batman.sh">Batman</a></td>
  <td width="30%" align="center"><a href="src/run_cartoon_herry.sh">Herry Potter</a></td>
</tr>
<tr>
  <td width=10% align="center">Edtied image</td>
  <td width=30% align="center"><img src="../assets/repo_figures/examples/edit/nobel_Biden.jpg" raw=true></td>
	<td width=30% align="center"><img src="../assets/repo_figures/examples/edit/art_batman.jpg" raw=true></td>
  <td width=30% align="center"><img src="../assets/repo_figures/examples/edit/cartoon_herry.jpg" raw=true></td>
</tr>
</table>

## Adding & Replacing
<table class="center">
<tr>
  <td width=10% align="center">Source image</td>
  <td width=30% align="center"><img src="../assets/repo_figures/examples/source/hiking.jpg" raw=true></td>
	<td width=30% align="center"><img src="../assets/repo_figures/examples/source/horse.jpg" raw=true></td>
  <td width=30% align="center"><img src="../assets/repo_figures/examples/source/boy.jpg" raw=true></td>
</tr>
<tr>
  <td width="10%" align="center">Editing Scripts</td>
  <td width="30%" align="center"><a href="src/run_hiking.sh">+ hiking stick</a></td>
  <td width="30%" align="center"><a href="src/run_horse.sh">horse -> camel</a></td>
  <td width="30%" align="center"><a href="src/run_boy.sh">+ dog</a></td>
</tr>
<tr>
  <td width=10% align="center">Edtied image</td>
  <td width=30% align="center"><img src="../assets/repo_figures/examples/edit/hiking.jpg" raw=true></td>
	<td width=30% align="center"><img src="../assets/repo_figures/examples/edit/horse.jpg" raw=true></td>
  <td width=30% align="center"><img src="../assets/repo_figures/examples/edit/boy.jpg" raw=true></td>
</tr>

</table> -->


# 🪄 Edit Your Own Image

## Toy tests
Gradio demo for image editing will be released soon. 

For now, we provide several toy test examples in `src/toy_test`.
You can either run the provided bash script directly or create your own custom bash scripts.


## Command Line
You can also run the following scripts in the terminal to edit your own image. 
```
cd src
python edit.py  --source_prompt [your source image prompt] \
                --target_prompt [your editing prompt] \
                --guidance 2 \
                --source_img_dir [the path of your source image] \
                --num_steps 15 --offload  \
                --front [typically set to 1 or 2] \
                --inject [typically set to 3 or 4] \
                --name 'flux-dev' --offload \
                --output_dir [output path] \
                --controlnet_type [specify your controlnet type] \
```

Please refer to the paper for the rationale and recommended values of the hyperparameters.

## Optional Attention Difference

Add `--attn_diff` to an existing editing command to use attention-output
divergence instead of the original velocity TDM. Without this flag, the original
velocity path is used. No training, gradient optimization, or additional concept
tokens are involved.

From the repository root, for example:

```bash
python src/edit.py \
    --source_img_dir src/examples/source/parrot.png \
    --source_prompt "A vibrant macaw perched on a tree branch in a tropical jungle." \
    --target_prompt "A brown hat resting on a tree branch in a tropical jungle." \
    --name flux-dev --num_steps 15 --guidance 2 --front 2 --inject 3 \
    --controlnet_type none --offload \
    --attn_diff --attn_diff_layers 13,14,15,16,17,18 \
    --output_dir outputs/parrot_attention \
    --vis_path outputs/parrot_attention/maps
```

`--attn_diff_layers` selects **zero-based double-stream block indices** (default:
13 through 18). FLUX.1-dev has 19 double-stream blocks, indexed 0 through 18.
The collector reads image-token attention outputs after attention heads are
concatenated, before the output projection, gate, and residual addition. It does
not extract attention-weight matrices or modify the captured activations.

For interval `k`, the attention variant computes:

```text
delta[k, patch] = mean_over_selected_blocks(
    L2_channels(source_midpoint_attention - target_midpoint_attention)
)
```

The source comes from the inversion midpoint evaluation. The target comes from
the existing uninjected target-probe midpoint evaluation. The intervals are
paired in reverse order, with an explicit midpoint-time check. These are different
latent trajectories at matching times, not same-latent prompt contrast. Original
FYS instead compares averages of start and midpoint velocities. Inversion guidance
(1), target guidance, solver updates, probe passes, ControlNet behavior, and KV
injection are retained. Start comparisons with ControlNet disabled to avoid its
additional influence on the inversion features.

The attention maps use the original accumulation window, Gaussian smoothing
(sigma 0.7), and Otsu thresholding. Temporal weights use `softmax(delta_stack)`
without a scaling multiplier (equivalent to scale 1); velocity TDM retains
`softmax(5 * delta_stack)`. A constant attention-divergence map is safely
normalized to zeros.

Visualizations are saved to `--vis_path`. In attention mode, omitting that option
automatically saves them under `<output_dir>/attn_diff_visualization`:

- `delta/delta_map_<step>.png`: one normalized divergence map per denoising step.
- `edit_map.png`: final binary edit-map plot, as in the original implementation.
- `soft_edit_map.png`: temporally aggregated, smoothed map before thresholding.
- `edit_map.npy`: the binary patch-grid mask, with 1 indicating editable patches.
- `attn_diff_config.json`: selected blocks, midpoint times, and accumulation settings.

The same final edit indices drive the original KV-injection code. Use a separate
output/visualization directory for each experiment; map filenames are reused on
reruns. Source attention is cached on CPU in its native dtype and released block
by block during comparison; distances are computed in FP32. With 1024x1024 input,
15 steps, six blocks, and 16-bit features, the extra source cache is about 2.1 GiB
of CPU RAM. Every midpoint is collected to provide the per-step visualizations;
only the original accumulation window contributes to the final mask.

For a baseline comparison, run the same command without `--attn_diff`, using
different output and visualization directories. No extra velocity mode is added.

CPU-only implementation checks (no model weights required), in a project
environment with the dependencies installed:

```bash
python -m pip install pytest
python -m pytest tests
```

### Per-step masks with immediate K/V injection

Add `--attn_diff_dynamic` alongside `--attn_diff` to use each mask immediately:

```bash
python src/edit.py \
    --source_img_dir src/examples/source/parrot.png \
    --source_prompt "A vibrant macaw perched on a tree branch in a tropical jungle." \
    --target_prompt "A brown hat resting on a tree branch in a tropical jungle." \
    --name flux-dev --num_steps 15 --guidance 2 --front 2 --inject 4 \
    --controlnet_type none --offload \
    --attn_diff --attn_diff_dynamic \
    --output_dir outputs/parrot_dynamic \
    --vis_path outputs/parrot_dynamic/maps
```

This mode uses the existing uninjected midpoint probe to compute attention
divergence. Each map is independently normalized, Gaussian-smoothed (sigma 0.7),
and Otsu-thresholded, without temporal aggregation or temporal softmax. The new
mask is applied to both evaluations of the actual solver update in that same
interval. Editable patches retain target K/V; other patches receive source K/V.
The selected attention blocks and original single-stream injection blocks are
unchanged. No extra model passes, training, or prompt parsing are introduced.

Mask updates stop automatically at the original TDM window end:
`cut = num_steps - inject - 3` (the CLI uses a one-step uninjected tail).
The latest mask is then frozen, including the gap before the original late
injection stage. For `num_steps=15`, `front=2`, and `inject=4`:

| Steps (zero-based) | K/V behavior |
| --- | --- |
| 0–1 | Original initial source injection (respecting an optional input mask) |
| 2–8 | Fresh per-step mask, applied immediately |
| 9–13 | Frozen mask from step 8 |
| 14 | Original uninjected final step |

There is no freeze-step parameter. Later delta maps remain available for
inspection but do not update the frozen mask. Omitting `--attn_diff_dynamic` keeps the
existing attention-aggregation behavior; omitting both flags keeps velocity TDM.

In addition to the existing delta maps, dynamic mode saves:

- `masks/edit_map_<step>.png` and `.npy`: the mask actually applied at that step,
  with 1 meaning target K/V and 0 meaning source K/V. The uninjected tail is all 1.
- `masks/soft_edit_map_<step>.png`: the smoothed map for each mask-update step.
- `edit_map.png`, `edit_map.npy`, `soft_edit_map.png`: the final frozen mask and
  its smoothed map, not the all-1 tail mask.
- `mask_diagnostics.json`: injection/update status, mask source step, editable
  area, changed patch count versus the previous applied mask, and raw divergence
  min/max/mean/std at each step.
- `attn_diff_config.json`: the actual update, freeze, and injection schedule. Its
  temporal softmax scale is `null` because no temporal softmax is used.

Dynamic mode also caches source K/V for the middle-stage intervals during
inversion, at both solver evaluations. This increases peak CPU memory usage;
features are released as denoising consumes them. At 1024x1024, with 16-bit K/V
and the settings in the table, the eight additional intervals require about
13.5 GiB of CPU RAM on top of the existing caches. `--offload` does not remove
this CPU-memory cost. Use separate output directories when comparing variants.


# 🖋️ Citation

If you find our work helpful, please **star 🌟** this repo and **cite 📑** our paper. Thanks for your support!

```
@article{long2025follow,
  title={Follow-your-shape: Shape-aware image editing via trajectory-guided region control},
  author={Long, Zeqian and Zheng, Mingzhe and Feng, Kunyu and Zhang, Xinhua and Liu, Hongyu and Yang, Harry and Zhang, Linfeng and Chen, Qifeng and Ma, Yue},
  journal={arXiv preprint arXiv:2508.08134},
  year={2025}
}
```

