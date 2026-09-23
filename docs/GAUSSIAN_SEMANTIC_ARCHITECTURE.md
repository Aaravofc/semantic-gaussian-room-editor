# Gaussian-Native Semantic Architecture

## Production Principle

The room is reconstructed once as a coherent Gaussian scene. Semantics are evidence attached
to those Gaussians, not a second reconstruction made from noisy depth.

For Gaussian `i`, the reconstruction already contains:

```text
G_i = (mu_i, Sigma_i, alpha_i, appearance_i)
```

where `mu_i` is its world-space center, `Sigma_i` comes from scale and rotation, `alpha_i` is
opacity, and `appearance_i` contains the spherical-harmonic color coefficients. Semantic
processing must preserve all of those fields.

## Data Flow

```text
video
-> registered RGB frames and camera poses
-> one full-room 3D Gaussian Splat
-> tracked 2D object and layout masks
-> per-Gaussian multiview semantic evidence
-> semantic Gaussian groups and editable proxies
-> non-destructive edit state
```

The shared Gaussian scene prevents separately reconstructed objects from drifting into
different coordinate normalizations. Every semantic result refers to the same `mu_i` values.

## Mask Evidence on Gaussians

For camera `f`, project Gaussian center `mu_i` with the registered world-to-camera transform
and intrinsics:

```text
x_cam = T_world_to_camera[f] * [mu_i, 1]
u_i,f = project(K_f, x_cam)
```

The current `pipeline.ipynb` estimates the projected Gaussian footprint from its world-space
scale and samples multiple pixels in that footprint. A label receives evidence only when:

1. the Gaussian projects inside the image;
2. rendered depth or the self z-buffer considers it visible;
3. enough sampled footprint pixels overlap the accepted mask;
4. structural and foreground geometry gates allow the label.

A conceptual vote for label `c` is:

```text
V_i,c = sum_f visibility_i,f
              * mask_overlap_i,f,c
              * frame_quality_f
              * label_frequency_weight_c
              * mask_area_weight_f,c
              * role_weight_c
```

The winning label, confidence margin, and distinct supporting-view count are appended to the
Gaussian PLY. Unknown remains valid when the evidence is insufficient or contradictory.

## Output Contract

`splat_labeled.ply` contains the complete renderable Gaussian scene plus semantic fields such
as label ID, confidence, and support views. `splat_semantic_labels.json` records the label
table and per-Gaussian semantic package. The manifest and bounding-box files summarize
connected semantic groups for the editor.

The diagnostic `combined_objects_semantic_colored.ply` is not the visual scene. It is a normal
colored point cloud for inspecting labels in tools that cannot display Gaussian PLYs.

## Editing Contract

The current browser editor has two layers:

```text
canonical layer: full-room Gaussian scene
editable layer: semantic object proxy boxes and labels
```

Proxy transforms are saved separately in `scene_edit_state.json`. They do not modify Gaussian
centers yet. This preserves the original reconstruction while the semantic grouping method is
still being evaluated.

The next production stage should replace hard winner-only labels with a probability vector
`p_i(c)` or semantic logits per Gaussian. Object operations can then select a stable Gaussian
index set, transform its centers/covariances together, hide its opacity, or invoke a separate
inpainting operation for the revealed region.

## Why the Surface Branch Was Archived

Both independent-object TSDF fusion and one shared TSDF surface produced unusable geometry on
`scan_001`. The rendered Gaussian depth did not form reliable hard surfaces under the consumed
camera contract, causing distorted shells, impossible object bounds, and mostly unassigned or
misassigned triangles. Converting the successful room splat into that mesh discarded visual
quality without improving editability.

Those files remain under `research/archive/semantic_surface/` and
`scans/<scan>/research_baselines/semantic_surface/` as negative-result evidence. They are not
production inputs.

## Non-Negotiable Invariants

1. Never change Gaussian coordinates merely to attach a label.
2. Never mix raw and dataparser-transformed camera frames without an explicit transform.
3. Never delete uncertain Gaussians solely because one mask misses them.
4. Keep unknown semantics instead of forcing every Gaussian into a class.
5. Track segmentation, camera, splat, and configuration fingerprints for every experiment.
6. Evaluate appearance, semantic isolation, completeness, and edit leakage separately.
