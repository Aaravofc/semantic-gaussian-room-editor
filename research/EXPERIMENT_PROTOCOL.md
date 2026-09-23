# Gaussian Semantic Experiment Protocol

## 1. Freeze Inputs

For every run, record checksums and versions for the source video, registered frames,
`transforms.json`, monolithic `splat.ply`, tracked detection JSON, accepted masks,
segmentation audit, notebook, dependencies, and configuration.

Do not compare methods that use different camera normalization, segmentation fingerprints, or
train/validation/test splits as if they were controlled.

## 2. Freeze Camera Splits

Write deterministic trajectory-aware lists under:

```text
scans/<scan>/research_splits/train_frames.txt
scans/<scan>/research_splits/validation_frames.txt
scans/<scan>/research_splits/test_frames.txt
```

Use validation views for thresholds and test views once for final reporting.

## 3. Curate Evaluation Masks

Manually inspect evaluated test masks. Record false positives, missing regions, occlusion,
identity switches, and ambiguous boundaries. Keep evaluation masks separate from training
masks and preserve both.

## 4. Register Every Run

Add a row to `results/experiment_registry.csv` with an immutable ID:

```text
<date>-<scan>-<method>-<unit>-<seed>
```

Never reuse an ID after changing code, configuration, or input fingerprints.

## 5. Preserve Outputs

Retain exact configuration, logs, traceback, camera split, checkpoint/model reference, raw and
selected PLYs, Gaussian count, runtime, memory, file size, and rendered RGB/alpha/semantic ID
for every test camera. A filtered result never replaces its raw source.

The archived semantic-surface output remains a negative-result baseline; it is not regenerated
as part of the production pipeline.

## 6. Render Quantitatively

Use scripted held-out cameras. Manual SuperSplat or viewer screenshots are qualitative only.
For each evaluated camera and method, save:

```text
rgb.png
alpha.png
semantic_id.png
semantic_confidence.npy
target_mask.png
```

For target render support `R` and verified mask `T`:

```text
IoU       = |R intersect T| / |R union T|
precision = |R intersect T| / |R|
recall    = |R intersect T| / |T|
leakage   = |R minus T| / |R|
```

Report confidence calibration and performance across an abstention threshold sweep.

## 7. Editing Test

For each persistent instance, execute fixed hide, translate, rotate, scale, reset, and remove
operations. Measure pixels outside the target that changed, world-placement stability, whether
group membership remains intact, and whether save/reload reproduces the edit.

Proxy-only viewer transforms must be reported as UI validation, not true Gaussian editing.

## 8. Failure Taxonomy

- `segmentation_false_positive`
- `segmentation_false_negative`
- `instance_identity_switch`
- `camera_contract_mismatch`
- `visibility_error`
- `semantic_boundary_leakage`
- `semantic_over_abstention`
- `object_splat_background_leakage`
- `cleanup_over_removal`
- `gaussian_group_fragmentation`
- `edit_cross_talk`
- `surface_conversion_failure`
- `export_or_render_failure`

Failed and abstained units remain in aggregate reporting.

## 9. Statistical Plan

Report per-instance values, category macro-averages, room averages, and paired bootstrap 95%
confidence intervals across room-instance units. Do not tune on test cameras. A one-room run
supports a case study, not a general indoor-reconstruction claim.
