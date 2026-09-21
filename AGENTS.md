# Repository Guidelines

## Project Structure & Module Organization

This is a Python/PyTorch implementation of Follow-Your-Shape, built on FLUX.

- `src/edit.py` is the command-line image-editing entry point.
- `src/flux/` contains the model, sampling, attention layers, conditioning, and utilities. Keep algorithmic changes close to the relevant module; for example, TDM scheduling belongs in `src/flux/sampling.py` and KV injection in `src/flux/modules/layers.py`.
- `src/toy_test/` holds runnable example scripts. `src/examples/` contains their source images, results, and TDM visualizations.
- `resources/` contains README images; `paper/` holds the published paper; model cards and licenses are in `model_cards/` and `model_licenses/`.

## Setup, Development, and Verification

Use Python 3.10 or newer and a CUDA-capable PyTorch installation. From the repository root:

```bash
conda create -n FollowYourShape python=3.10
conda activate FollowYourShape
pip install -e ".[all]"
cd src
python edit.py --help
```

Run a documented smoke example with `bash toy_test/run_toy_test.sh` from `src/`, after reviewing paths and GPU settings. The project has no automated unit-test suite; validate changes with a small edit run and, when touching TDM logic, inspect the generated `edit_map.png` and per-step `delta/` maps.

## Coding Style & Naming Conventions

Follow the existing Python style: four-space indentation, `snake_case` functions/variables, `PascalCase` classes, and descriptive tensor names such as `delta_map` or `controlnet_scale`. Keep device, dtype, and tensor-shape handling explicit in model and sampler code. Ruff is configured for Python 3.10 with a 110-character line length and double quotes; run `ruff check src` and `ruff format --check src` when available. Pyright configuration covers `src/`; run `pyright` after changing type-sensitive interfaces.

## Commits & Pull Requests

Recent commits use short imperative subjects, e.g. `Add paper`, `update README`, and `update with the Otsu threhold implementation`. Prefer `Add TDM visualization` over vague messages. Keep each commit focused. Pull requests should explain the behavioral change, list the command/example used to verify it, link related issues when applicable, and include before/after output images for changes affecting edit quality, masks, or ControlNet behavior. Do not commit model weights, generated caches, or large ad-hoc artifacts.
