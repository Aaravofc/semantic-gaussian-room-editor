# Synthetic Demo Data

`scan_demo/` is generated entirely by `scripts/generate_synthetic_demo.py`. It contains no
frames, geometry, or appearance data from a real room.

The Gaussian PLY follows Nerfstudio's exported field layout so it exercises the same browser
loading path as a real reconstruction. The scene is deliberately small enough for normal Git
history. Regenerate it with:

```bash
python scripts/generate_synthetic_demo.py
```

