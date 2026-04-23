# Tie-Only Desktop Try-On

This branch turns RTV into a lightweight necktie try-on app. The supported product is a CPU-first desktop webcam experience:

- MediaPipe pose + face landmarks
- transparent PNG tie assets
- no training, no garment checkpoints, no GPU-only stack

The app auto-places a tie from shoulders and lower-face landmarks, smooths motion, and lets the user fine-tune position, scale, and rotation from the keyboard.

## Setup

```bash
git checkout ties-only
python -m pip install -r requirements.txt
```

The first run downloads official MediaPipe task bundles into `models/mediapipe/`.

## Run

```bash
python tie_demo.py
```

Optional flags:

```bash
python tie_demo.py --camera-id 1 --catalog assets/ties/catalog.json --fullscreen --debug
```

`demo.py` and `rtl_demo.py` now forward to the same tie app for convenience.

## Controls

- `0-9`: select the first ten list entries
- `Esc` or `Q`: quit
- Arrow keys: nudge tie position
- `[` and `]`: scale down or up
- `,` and `.`: rotate left or right
- `R`: reset the selected tie's manual adjustment

## Catalog Format

The public asset registry lives at `assets/ties/catalog.json`. Each entry must contain:

- `id`: stable tie identifier
- `name`: display label
- `asset_path`: transparent BGRA PNG path, relative to the catalog
- `thumbnail_path`: catalog thumbnail path, relative to the catalog
- `knot_anchor`: `[x, y]` pixel coordinate of the knot center in the asset canvas
- `knot_width_ref`: reference knot width in pixels inside the asset canvas
- `default_scale`: multiplier applied to the runtime knot width
- `default_offset_x`: horizontal screen-space offset in knot-width units
- `default_offset_y`: vertical screen-space offset in knot-width units
- `default_rotation_deg`: per-tie rotation trim in degrees

Assets should use a common transparent canvas with the knot near the top. Collar-covered parts should already be transparent in the PNG so runtime collar masking is unnecessary.

## Branch Notes

- This branch is ties-only. The legacy garment training and DensePose/SMPL/VITON code remains in the tree for reference, but it is not part of the supported runtime or documented workflow.
- The default install no longer includes Torch, Detectron2, ROMP/BEV, or OpenGL.
- MediaPipe Tasks supports Python 3.9-3.12, and this branch is intended to run on Python 3.12 as well as current 3.10 environments.
