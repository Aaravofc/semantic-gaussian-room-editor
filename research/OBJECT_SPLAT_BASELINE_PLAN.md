# Object-First Gaussian Reconstruction for Semantically Editable Indoor Scenes

Status: study design, before comparative evaluation

## 1. Research Aim

To determine the trade-offs introduced by reconstructing indoor semantic
components as independently trained but spatially aligned Gaussian splats,
compared with reconstructing the room as one undifferentiated Gaussian model.

The study will evaluate whether object-first reconstruction improves semantic
isolation, object completeness, spatial editability, and physical consistency,
and will quantify any cost in novel-view quality, runtime, and storage.

## 2. Primary Research Question

What is gained and lost when an indoor scene is reconstructed as composable
semantic Gaussian components rather than as one monolithic Gaussian-splat
model?

## 3. Secondary Questions

1. Does object-first training reduce semantic leakage outside an object's
   multi-view masks?
2. Does it reconstruct target objects more completely than post-hoc labeling?
3. Do independently trained components remain aligned when they share the full
   camera set and scene normalization?
4. How much photometric quality is lost at mask boundaries and occlusion
   boundaries?
5. How do results differ between structural surfaces, large furniture, and
   small or sparsely observed objects?
6. How much do post-export evidence cleanup and joint composition affect the
   isolation-completeness trade-off?

## 4. Falsifiable Hypotheses

H1: Object-first reconstruction improves held-out semantic IoU and reduces
semantic leakage compared with post-hoc labeling.

H2: Object-first reconstruction improves edit isolation: transforming or
hiding one component changes fewer pixels belonging to unrelated components.

H3: Reusing the complete registered camera set preserves object placement well
enough that separately trained exports can be composed with an identity
transform.

H4: Object-first reconstruction initially performs worse than monolithic 3DGS
on held-out PSNR, SSIM, or LPIPS near uncertain mask boundaries.

H5: Evidence-based cleanup reduces background contamination, but excessive
cleanup decreases object completeness. Therefore no single cleanup threshold is
expected to dominate every object category.

## 5. Intended Contributions

### Primary contribution

A reproducible shared-coordinate pipeline for training selectable semantic
Gaussian components from multi-view masks while retaining their positions in
the original room.

### Evaluation contribution

An evaluation protocol that measures semantic isolation, object completeness,
alignment, editability, physical consistency, rendering quality, runtime, and
storage together.

### Potential method contribution

An evidence-based Gaussian cleanup method using multi-view mask support, sparse
geometry support, opacity, and 3D component coherence. This should be claimed as
a method contribution only if an ablation shows that it consistently improves
the isolation-completeness trade-off.

## 6. Comparison Methods

| ID | Method | Description |
|---|---|---|
| M0 | Monolithic 3DGS | One room model trained from all RGB frames, without semantics. |
| M1 | Post-hoc semantics | Labels assigned to an already trained room splat using 2D-mask projection. |
| M2 | Semantic scene training | One room model trained with semantic supervision while preserving one shared Gaussian set. This is future work until implemented. |
| M3 | Object-first raw | One independently trained splat per semantic component using shared poses; no post-export cleanup. |
| M4 | Object-first cleaned | M3 followed by the evidence-based cleanup stage. |
| M5 | Object-first composed | Cleaned components jointly rendered in one scene while retaining component identity. |

M0, M1, M3, and M4 form the minimum viable paper. M2 and learned joint
fine-tuning must not be described as completed methods until implemented.
SuperSplat may be used to validate composition and editing, but quantitative
rendering must use a scripted, repeatable camera path.

## 7. Experimental Unit

The experimental unit is one semantic component in one captured room under one
fixed train/test camera split.

The current pipeline reconstructs one component per semantic label. It does not
yet reliably distinguish multiple instances of the same class. Until
multi-view instance tracking is implemented, claims must use "semantic
component" or "class-level component", not "object instance", where a room
contains multiple same-class objects.

## 8. Evaluation Categories

Report results separately for:

- Structural surfaces: wall, floor, ceiling.
- Large rigid furniture: bed, desk, wardrobe, sofa.
- Medium objects: chair, door, window.
- Small or sparsely observed objects: pillow, fan, lamp, and similar classes.

Macro-average across components so walls and floors cannot dominate the
reported result by size.

## 9. Metrics

### Novel-view appearance

- PSNR: higher is better.
- SSIM: higher is better.
- LPIPS: lower is better.

Metrics must be computed on held-out views. Report full-frame scores and
target-mask-only scores separately.

### Semantic isolation

- Mask IoU between rendered component alpha and held-out target masks.
- Boundary F-score.
- Leakage rate: rendered component pixels outside the target mask divided by
  all rendered component pixels.
- Background contamination rate measured on empty-mask views.

### Completeness

- Recall against verified held-out masks.
- Depth or surface recall when RGB-D or reference geometry is available.
- Multi-view coverage: fraction of positive held-out views in which the
  component renders in the expected region.

### Alignment

- Pixel reprojection displacement between separately trained components and the
  monolithic scene on shared held-out cameras.
- Identity-placement success rate in the composed scene.

### Editability

- Edit isolation score: fraction of pixels outside the edited component that
  remain unchanged after hide, translation, or rotation.
- Component selection success in the composed scene.

### Physical consistency

- Pairwise collision volume or sampled density overlap.
- Support violation rate for supported objects.
- Point-to-plane residual for structural surfaces.
- Floor-ceiling normal parallelism.

### Cost

- Training time.
- Peak GPU memory.
- Export size and Gaussian count.
- Total storage and total training time per room.

## 10. Minimum Dataset Plan

### Pilot

- One room (`scan_001`).
- At least one structural surface, one large furniture component, and one
  smaller component.
- Fixed camera split shared by every method.
- Purpose: validate metrics, identify failures, and estimate compute.

### Main study

- Target: at least 10 rooms with varied layout, clutter, lighting, and
  occlusion.
- Include repeated classes in some rooms to expose the limitation of
  class-level reconstruction.
- Keep raw videos, extracted frames, poses, masks, splits, configurations, and
  reports versioned.

A one-room study can support a technical report or case study, but not a broad
claim about indoor reconstruction.

## 11. Required Ablations

1. M3 raw export versus M4 cleaned export.
2. Full registered camera set versus positive-mask-only cameras.
3. Empty-mask supervision enabled versus disabled.
4. Random/full-scene initialization versus mask-supported sparse
   initialization.
5. Cleanup evidence terms individually removed:
   multi-view mask support, sparse support, component coherence, and opacity.
6. Structural cleanup rules versus generic object cleanup.

Only one factor should change within each ablation pair.

## 12. Statistical Plan

- Predefine the train/test split before method comparison.
- Use the same held-out cameras and masks for every method.
- Report per-component results, category macro-averages, room averages, and
  overall macro-averages.
- For the main study, report paired bootstrap 95% confidence intervals for
  method differences across room-component units.
- Report failures and missing outputs; do not silently omit them.
- Do not tune thresholds on test views. Use a validation subset or a separate
  development room.

## 13. Current Evidence and Limitations

Observed so far:

- The single-object masked training experiment produced a visually useful
  object splat.
- Batch object splats retained the correct room alignment.
- Raw object splats contained substantial background fragments.
- Generic opacity and bounding-box cleanup was insufficient.
- The current v4 notebook adds multi-view and geometric cleanup evidence, but
  its quantitative benefit has not yet been established.

These observations motivate the study. They are not yet comparative results.

## 14. Paper Outline

1. Abstract
2. Introduction
3. Related Work
4. Problem Formulation
5. Shared-Coordinate Object-First Reconstruction
6. Evidence-Based Component Cleanup
7. Experimental Protocol
8. Results
9. Ablation Studies
10. Failure Cases and Limitations
11. Discussion
12. Conclusion

## 15. Immediate Milestone

Complete a preregistered `scan_001` pilot comparing M0, M1, M3, and M4 for at
least `floor`, `bed`, and one smaller component. Do not add geometric priors or
joint fine-tuning until the pilot identifies whether the dominant error comes
from masks, training, cleanup, or composition.
