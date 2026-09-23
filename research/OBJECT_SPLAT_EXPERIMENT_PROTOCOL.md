# Experiment Protocol

Use this protocol for every result intended for the paper.

## 1. Freeze the Input

For each scan, record:

- Scan ID and capture date.
- Source video checksum.
- Extracted frame count.
- `transforms.json` checksum.
- Segmentation model and checkpoint.
- Detection JSON and mask-directory checksums.
- Monolithic `splat.ply` checksum.
- Pipeline and notebook version strings.

Never compare methods that use different camera poses, frame resolutions, or
train/test splits without explicitly marking the comparison as non-controlled.

## 2. Create One Camera Split

Create deterministic lists:

```text
scans/<scan_id>/research_splits/train_frames.txt
scans/<scan_id>/research_splits/validation_frames.txt
scans/<scan_id>/research_splits/test_frames.txt
```

Recommended pilot split:

- 70% train.
- 10% validation.
- 20% test.

Select frames by camera trajectory rather than pure random sampling so test
views include genuinely different viewpoints. Every method must use the same
lists.

## 3. Verify Ground Truth

Before training, manually inspect all test-frame masks for the evaluated
components. Record:

- Missing object regions.
- Incorrect class assignments.
- Mask leakage.
- Severe motion blur.
- Occlusion.
- Whether multiple instances share one class mask.

Exclude a frame only by a written rule fixed before comparing methods.

## 4. Register the Run

Add one row to `results/experiment_registry.csv` before or immediately after
launching the run. Give every run an immutable ID:

```text
<date>-<scan>-<method>-<label>-<seed>
```

Example:

```text
20260729-scan001-m4-bed-s0
```

Do not reuse a run ID after changing configuration.

## 5. Run the Minimum Pilot

For `scan_001`, evaluate:

| Component | Reason |
|---|---|
| floor | Large structural surface and known cleanup challenge. |
| bed | Large target object with a successful prior experiment. |
| chair or desk | Smaller, more occluded furniture component. |

Run:

- M0 monolithic 3DGS.
- M1 post-hoc semantics.
- M3 object-first raw.
- M4 object-first cleaned.

M0 does not have an independent component render. For semantic metrics, render
its RGB/depth as the appearance baseline and use either no semantic prediction
or a clearly identified external segmentation baseline.

## 6. Preserve Outputs

Every run must retain:

- Exact configuration JSON.
- Console/training log.
- Checkpoint reference.
- Raw export.
- Cleaned export, if applicable.
- Gaussian count and file size.
- Train, validation, and test frame lists.
- Rendered RGB, alpha, depth, and semantic ID for every test camera.
- Metric CSV and summary JSON.
- Failure status and traceback when unsuccessful.

Never replace the raw export with the cleaned export.

## 7. Quantitative Rendering

Use a scripted renderer with the exact test camera intrinsics and transforms.
SuperSplat is suitable for qualitative inspection and editing demonstrations,
but manual screenshots are not reproducible quantitative evidence.

For each component and test view, save:

```text
rgb.png
alpha.png
depth.npy
semantic_id.png
target_mask.png
```

Use a fixed alpha threshold selected on validation views. Also report a
threshold sweep or area under the precision-recall curve so conclusions do not
depend on one threshold.

## 8. Metric Definitions

Given binary rendered support \(R\) and verified target mask \(T\):

```text
IoU = |R intersection T| / |R union T|
precision = |R intersection T| / |R|
recall = |R intersection T| / |T|
leakage = |R minus T| / |R|
```

For empty-mask views:

```text
empty_view_false_positive_rate =
rendered_component_pixels / image_pixels
```

For image metrics, calculate:

- Full-frame PSNR, SSIM, and LPIPS.
- Target-mask PSNR, SSIM, and LPIPS.
- Boundary-band metrics inside a fixed-width band around the verified mask.

## 9. Cleanup Evaluation

Evaluate raw and cleaned exports against the same test views.

A cleanup is useful only if it improves isolation without unacceptable loss of
completeness. Report both:

```text
delta_leakage = leakage_cleaned - leakage_raw
delta_recall = recall_cleaned - recall_raw
```

Do not select a candidate using test metrics. Candidate selection must use
training/validation evidence only.

## 10. Composition and Editing Test

Import separate component exports into SuperSplat without applying manual
transforms.

Record:

- Whether identity placement aligns each component.
- Any visible duplicate/background geometry.
- Whether each imported file remains independently selectable and hideable.
- A fixed hide-object and translate-object demonstration.

Save the editable `.ssproj` and keep the individual source `.ply` files.
Qualitative figures must use fixed viewpoints documented in the experiment
report.

## 11. Failure Taxonomy

Classify failures rather than only marking a run bad:

- `mask_error`
- `camera_or_normalization_error`
- `insufficient_views`
- `training_background_leakage`
- `cleanup_over_removal`
- `cleanup_under_removal`
- `same_class_instance_merge`
- `structural_surface_fragmentation`
- `export_failure`
- `render_failure`

Multiple tags are allowed.

## 12. Decision Gate After Pilot

Proceed to multiple rooms only if:

1. All four minimum methods render on the identical test split.
2. Metrics can distinguish raw leakage from cleanup over-removal.
3. Shared-coordinate identity placement is numerically and visually verified.
4. At least one object-first component improves semantic isolation.
5. Failure causes can be traced to masks, training, cleanup, or composition.

If these conditions fail, improve the measurement pipeline before adding
physical priors or training more rooms.

## 13. Bed Containment Pilot V8

`object_splat_batch_pipeline.ipynb` now runs one persistent bed instance through three arms
with identical cameras, RGB frames, tracked masks, and sparse initialization:

| Arm | Training constraint |
|---|---|
| `masked_baseline` | Stock Splatfacto mask-as-ignore behavior. |
| `alpha_constrained` | Rendered-alpha foreground, background, and Dice losses plus silhouette-gated densification. |
| `hull_constrained` | Alpha arm plus a conservative multi-view visual support grid, center projection, and Gaussian extent cap. |

Post-export deletion is disabled. This experiment tests whether leakage can be prevented during
optimization rather than repaired after export. Keep all three raw `splat.ply` files and inspect
them from identical cameras. The viewer manifest exposes only `hull_constrained`, while its
`ablation_variants` field preserves every arm.

The output roots are:

```text
scans/<scan>/object_splat_batches/bed_instance_containment_pilot_v8/bed/variants/<arm>/exports/splat.ply
scans/<scan>/reports/object_splat_batch/bed_instance_containment_pilot_v8/
scans/<scan>/object_splats/object_splat_manifest.json
```

Do not extend this method to all labels until the bed pilot demonstrates both lower leakage and
acceptable completeness. A smaller splat is not evidence of improvement by itself.

## 14. Bed Ellipsoid-Containment Pilot V9

V8 showed that center-only hull projection activated for very few Gaussians. V9 therefore replaces
the third arm with `ellipsoid_constrained` while preserving `masked_baseline` and
`alpha_constrained`. The completed V8 output remains a separate historical comparison.

The V9 finite extent is the learned, quaternion-rotated 2-sigma ellipsoid. A mathematical Gaussian
has infinite support, so the sigma multiplier is part of the experimental definition. The method
samples 42 deterministic Fibonacci-sphere directions plus six principal axes on radial shells at
0.5 and 1.0 of the 2-sigma contour. Training adds differentiable occupancy and grid-boundary losses
for those probes. Every 100 steps, it:

1. Projects any escaped center to the nearest occupied support cell.
2. Caps pathological principal scales using the existing voxel-relative cap.
3. Finds Gaussians with unsupported ellipsoid probes.
4. Binary-searches the largest uniform scale fraction whose sampled volume is supported.
5. Snaps only extreme cases to their occupied voxel center before repeating the scale search.

No Gaussian is deleted after export. Inspect the raw arms at identical cameras and report both
silhouette quality and the containment diagnostics:

```text
hull_center_inside_fraction
hull_ellipsoid_inside_fraction
hull_boundary_sample_inside_fraction
hull_ellipsoid_shrunk_last
hull_ellipsoid_snapped_last
```

The V9 output roots are:

```text
scans/<scan>/object_splat_batches/bed_instance_ellipsoid_containment_pilot_v9/bed/variants/<arm>/exports/splat.ply
scans/<scan>/reports/object_splat_batch/bed_instance_ellipsoid_containment_pilot_v9/
```

Do not infer success from bounding-box span or Gaussian count alone. V9 succeeds only if fixed-camera
renders reduce off-object alpha without materially reducing target-mask recall or visual completeness.

## 15. Bed Depth-Visible Containment Pilot V10

V9's lower wedge survived alpha, center-hull, and ellipsoid constraints. Its final ellipsoid boundary
inside fraction was `1.0`, which showed that the artifact was already inside the shared support
volume. V10 therefore changes the upstream evidence while preserving the same three training arms:

```text
masked_baseline
alpha_constrained
ellipsoid_constrained
```

For every positive tracked bed mask, V10 projects the complete room sparse cloud and compares each
point's camera-space depth with the rendered full-room depth. It tests both native camera-Z depth
and ray-distance converted to camera Z using the pixel viewing angle, then robustly calibrates the
better convention to the raw coordinate scale using overlapping nearest sparse surfaces. Rendered
depth is authoritative where valid; a room sparse-point z-buffer is used only where rendered depth
has no valid sample. A sparse point becomes an object seed only after at least two
visibility-consistent mask hits and a visible-hit ratio of at least `0.50`.

The remaining seeds are grouped in a 26-neighbor voxel graph. The highest-evidence component is
kept, while another component must pass point-count, frame-provenance, and relative-score gates.
The support grid then keeps near-surface mask-supported cells, rejects cells observed in free space
in front of the rendered surface, treats cells behind a surface as unknown rather than negative,
and removes unsupported 3D components. Only these cells can constrain the finite 2-sigma ellipsoids.

Preflight is deliberately strict: at least 75% of positive mask frames must have depth files and at
least 60% must calibrate reliably. Do not weaken these gates to force a run through a depth/camera
mismatch. Generate missing maps by setting `RUN_OPTIONAL_DEPTH_RENDERING=True` in
`reconstruction.ipynb` and running its depth-rendering and verification sections.

V10 writes the following evidence before GPU training:

```text
scans/<scan>/reports/object_splat_batch/bed_instance_depth_visible_containment_pilot_v10/bed/
  sparse_visibility_summary.json
  sparse_visibility_frame_report.csv
  sparse_visibility_component_report.csv
  sparse_visibility_evidence.npz
  seed_visibility_previews/*.jpg
  bed_sparse_mask_hits_only.ply
  bed_sparse_depth_visible_precomponent.ply
  bed_sparse_depth_visible_final.ply
  visual_support_grid_report.json
  visual_support_grid_frames.csv
  visual_support_grid_components.csv
```

Raw arm exports remain under:

```text
scans/<scan>/object_splat_batches/bed_instance_depth_visible_containment_pilot_v10/bed/variants/<arm>/exports/splat.ply
```

Green points in a seed preview passed visibility; red points landed inside the mask but disagreed
with the first observed surface. A successful V10 result should remove the persistent lower wedge
without deleting true bed surfaces. Post-export Gaussian deletion remains disabled.

### V10 Outcome And Rollback

The first V10 run removed approximately 25% of the true bed. This is a completeness failure: the
depth-consistency and connected-component gates rejected valid object evidence along with the
suspected lower artifact. On 2026-08-17, `object_splat_batch_pipeline.ipynb` was restored to V9 and
its immutable `bed_instance_ellipsoid_containment_pilot_v9` output root. The V10 updater and its
existing Drive outputs remain preserved for ablation and failure analysis.

Do not treat V10 as the active pipeline. Any future visibility-aware variant must first measure
object recall against V9 and should soften uncertain or occluded evidence instead of deleting it
before optimization.

## 16. Bounded Volumetric Object Batch V11

V11 scales the accepted V9 method from one bed instance to a small bounded-object batch. The
default labels are:

```text
bed
chair
desk
table
pillow
bag
```

For each label, the notebook reads only accepted `tracked_semantic_frame.v2` records, selects the
persistent instance with the largest visible-frame count, and requires at least five positive mask
frames. A missing or short-lived class is reported as `skipped`; dataset, sparse initializer,
support-grid, training, and export errors remain `failed` and are not silently converted to skips.

Each eligible object reuses the V9 coordinate and geometry contract: the complete registered camera
set, empty target masks in non-positive frames, at least two mask hits per sparse seed, the
conservative multi-view visual-support grid, alpha supervision, and oriented finite 2-sigma
ellipsoid containment. Post-export deletion remains disabled. Only `ellipsoid_constrained` runs by
default; the baseline and alpha-only definitions remain available for an intentional ablation.

V11 deliberately rejects `wall`, `floor`, `ceiling`, `door`, `window`, and `curtain`. These are
structural or thin surfaces for which a compact occupied volume and uniform ellipsoid shrinking are
the wrong prior. They require separate planar, boundary, or layered-surface experiments.

Outputs are immutable under:

```text
scans/<scan>/object_splat_batches/volumetric_objects_ellipsoid_batch_v11/<label>/variants/ellipsoid_constrained/exports/splat.ply
scans/<scan>/reports/object_splat_batch/volumetric_objects_ellipsoid_batch_v11/
scans/<scan>/object_splats/manifests/volumetric_objects_ellipsoid_batch_v11_object_splat_manifest.json
```

The reports include `batch_label_audit.csv`, `volumetric_variant_status.csv`, and
`volumetric_export_inspection.csv/json`. The latest viewer manifest contains one recommended export
for each successful label and records `success`, `success_with_skips`, or `partial_failure` for the
batch. Run segmentation v2.7 through Audit before V11 so its detection fingerprints and persistent
instances are current.

## 17. All-Instance Volumetric Object Batch V12

V12 changes the unit of reconstruction from one selected class to one persistent object instance.
It discovers accepted `tracked_semantic_frame.v2` records, intersects their labels with an explicit
compact-volume allowlist, and trains every instance with at least five positive mask frames. Same-
class objects therefore receive separate keys and exports, for example `chair_001` and `chair_003`.
`TARGET_LABELS` and `TARGET_INSTANCE_IDS` remain optional reproducibility filters; empty values mean
all qualifying allowlisted instances.

The geometry method is deliberately unchanged from the accepted V9 arm: full registered cameras,
empty masks outside positive views, two-view sparse initialization, conservative visual support,
alpha supervision, mask-gated densification, and oriented finite 2-sigma ellipsoid containment.
This isolates scale-up and instance handling from a simultaneous change to the reconstruction prior.
No post-export Gaussian deletion is performed.

The volumetric allowlist covers compact furniture, storage, electronics, portable objects, plants,
and per-track `miscellaneous` objects. Structural planes, architectural surfaces, thin wall-mounted
surfaces, hanging/deformable surfaces, and articulated fixtures are written to the discovery report
with exclusion reason codes. They are reserved for later class-specific reconstruction branches.

V12 outputs are immutable under:

```text
scans/<scan>/object_splat_batches/all_volumetric_instances_ellipsoid_batch_v12/<object_key>/variants/ellipsoid_constrained/exports/splat.ply
scans/<scan>/reports/object_splat_batch/all_volumetric_instances_ellipsoid_batch_v12/
scans/<scan>/object_splats/manifests/all_volumetric_instances_ellipsoid_batch_v12_object_splat_manifest.json
```

`volumetric_label_discovery.csv` records every accepted class and its inclusion decision;
`batch_object_audit.csv` records every selected instance and eligibility result. The viewer-facing
manifest contains exactly one recommended ellipsoid export per successful object key. The batch is
resumable because dataset, training, and export fingerprints are stored independently per instance.
