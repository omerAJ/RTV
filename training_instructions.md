# Tie Asset Authoring Notes

This branch no longer trains per-garment checkpoints. A tie is just a transparent PNG plus one catalog entry in `assets/ties/catalog.json`.

## Asset Checklist

- Use a transparent BGRA PNG.
- Keep the tie centered on a common canvas with the knot near the top.
- Remove any collar-covered pixels from the PNG itself.
- Measure the knot center and record it as `knot_anchor`.
- Measure the on-image knot width and record it as `knot_width_ref`.
- Add optional default tuning with `default_scale`, `default_offset_x`, `default_offset_y`, and `default_rotation_deg`.

## Runtime Behavior

- Placement is automatic from MediaPipe pose and face landmarks.
- Fine tuning happens at runtime with the keyboard controls in `tie_demo.py`.
- No dataset capture, segmentation export, or model training step is required.
