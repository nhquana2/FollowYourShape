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

## Attention Difference

Add `--attn_diff` to an editing command to localize the edit with attention-output
differences instead of the original velocity TDM. Without this flag, the original
path is used unchanged. No training, gradient optimization, or additional concept
tokens are involved.

From the repository root, for example:

```bash
python src/edit.py \
    --source_img_dir src/examples/source/parrot.png \
    --source_prompt "A vibrant macaw perched on a tree branch in a tropical jungle." \
    --target_prompt "A brown hat resting on a tree branch in a tropical jungle." \
    --name flux-dev --num_steps 15 --controlnet_type none --offload \
    --attn_diff \
    --output_dir outputs/parrot_attention
```

With `--attn_diff`, the defaults of `--guidance`, `--front`, and `--inject` become
2, 0, and 3 (they stay 3, 2, and 4 otherwise). Values given on the command line
always take precedence.

**Signal.** For denoising interval `k`, the map is the distance between the source
and target image-token attention outputs, averaged over the selected double-stream
blocks:

```text
delta[k, patch] = mean_over_selected_blocks(
    L2_channels(source_midpoint_attention - target_midpoint_attention)
)
```

The source comes from the inversion midpoint evaluation and the target from the
uninjected target-probe midpoint evaluation of the same interval, with an explicit
midpoint-time check. These are different latent trajectories at matching times.
The collector reads attention outputs after the heads are concatenated, before the
output projection, gate, and residual addition, and does not modify them.
Inversion guidance (1), solver updates, probe passes, and ControlNet behavior are
unchanged. No extra model passes are introduced.

**Soft mask.** Each map is min-max normalized, Gaussian-smoothed, clipped to the
range between two of its percentiles, rescaled to `[0, 1]`, and passed through a
sigmoid:

```text
u    = clip((map - P_low) / (P_high - P_low), 0, 1)
mask = sigmoid(steepness * (u - center))
```

The mask is applied to both evaluations of the solver update in the same interval,
in the original single-stream injection blocks: V is blended as
`mask * V_target + (1 - mask) * V_source`, and K is taken from the target where
`mask > 0.5` (that is, `u > center`) and from the source elsewhere.

**Schedule.** A fresh soft mask is computed and applied at every injected step
(the CLI uses a one-step uninjected tail). For `num_steps=15` and the defaults:

| Steps (zero-based) | K/V behavior |
| --- | --- |
| 0–13 | Fresh soft mask at every step, applied immediately |
| 14 | Original uninjected final step |

With `--attn_diff_freeze`, masks are updated only up to the original TDM window end,
`cut = num_steps - inject - 3` (step 9 for the defaults), and that mask is reused for
the remaining injected steps (10–13).

With `--attn_diff_aggregate`, the original three-stage schedule is kept: the maps of
the TDM window (steps `front`–`cut`, 0–9 for the defaults) are collected without injection,
combined by the softmax-weighted sum of the original TDM, and turned into one soft mask
that is applied only in the last `--inject` steps (11–13).

With `--front N`, the first `N` steps instead inject the source everywhere
(respecting an optional input mask), as in the original method.

**Options.**

| Option | Default | Meaning |
| --- | --- | --- |
| `--attn_diff_layers` | `0,...,18` | zero-based double-stream blocks used for the signal (FLUX.1-dev has 19) |
| `--attn_diff_sigma` | `0.7` | Gaussian sigma in patches before the soft mask (0 disables smoothing) |
| `--attn_diff_percentiles` | `50,98` | lower and upper percentiles rescaled to 0 and 1 |
| `--attn_diff_center` | `0.3` | rescaled value where the mask is 0.5; lower values give larger masks |
| `--attn_diff_steepness` | `15` | sigmoid steepness; higher values give a harder mask edge |
| `--attn_diff_freeze` | off | stop updating the mask after the original TDM window and reuse it for the remaining injected steps |
| `--attn_diff_aggregate` | off | original schedule: no injection in the TDM window, one soft mask from the aggregated window maps, injection only in the last `--inject` steps |
| `--attn_diff_free_steps` | `0` | skip K/V injection for the first N mask-update steps (every patch keeps the target K/V); their masks are still computed |

**Outputs.** Visualizations are saved to `--vis_path`, or to
`<output_dir>/attn_diff_visualization` when that option is omitted:

- `delta/delta_map_<step>.png`: the normalized map of every denoising step.
- `masks/soft_edit_map_<step>.png`: the smoothed map of each mask-update step.
- `masks/edit_map_<step>.png` and `.npy`: the patches above 0.5 at that step, with
  1 meaning target K and 0 meaning source K. The uninjected tail is all 1.
- `edit_map.png`, `edit_map.npy`, `soft_edit_map.png`, `soft_mask.npy`: the mask
  of the last update step, its smoothed map, and its soft values.
- `mask_diagnostics.json`: injection/update status, editable area, changed patch
  count versus the previous step, and raw difference statistics at each step.
- `attn_diff_config.json`: selected blocks, midpoint times, schedule, and soft-mask
  settings.

Use a separate output directory for each run; map filenames are reused. Source
attention outputs and the source K/V of every injected step are cached on CPU
during inversion and released as denoising consumes them, so peak CPU memory is
higher than on the original path. `--offload` does not remove this cost.

CPU-only implementation checks (no model weights required), in a project
environment with the dependencies installed:

```bash
python -m pip install pytest
python -m pytest tests
```


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

