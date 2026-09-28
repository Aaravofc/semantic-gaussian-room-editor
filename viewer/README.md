# Browser Editor

The viewer renders one canonical Gaussian Splat and overlays semantic object proxies in the
same coordinate system.

## Hosted demo

Open <https://aaravofc.github.io/semantic-gaussian-room-editor/>. The hosted version loads the
bundled synthetic room and keeps saved edit state in the visitor's browser. It does not upload
data or require a backend.

## Run the checked-in demo

From the repository root:

```bash
node viewer/server.js \
  --data-root "$PWD/demo/scan_demo" \
  --scan scan_demo \
  --output "$PWD/demo/scan_demo/scene_edit_state.json"
```

Open <http://127.0.0.1:5177/viewer/>.

The viewer imports pinned Three.js and Gaussian viewer modules from `unpkg.com`, so its first
load requires internet access.

## Expected inputs

The server accepts files directly in `--data-root` or in its scan subdirectories:

```text
splat_labeled.ply                  preferred canonical scene
splat.ply                          fallback Gaussian scene
semantic_scene_manifest.json      semantic metadata
object_bounding_boxes.json        editable proxies
combined_objects_semantic_colored.ply  optional diagnostic overlay
```

## Editing contract

- Scene visibility controls the canonical Gaussian Splat.
- Proxy, box, and label controls affect the semantic overlay.
- Move, rotate, and scale update either the selected proxy or the whole displayed scene.
- Remove/restore is non-destructive and changes the saved proxy state only.
- Save writes the configured `scene_edit_state.json` file.

Proxy transforms do not rewrite individual Gaussians. True object editing requires stable
per-Gaussian instance membership and group transforms.
