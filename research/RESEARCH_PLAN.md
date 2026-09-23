# Gaussian-Native Semantic Reconstruction for Editable Indoor Scenes

Status: restored production baseline and proposed semantic-aware training study

## Research Aim

Determine whether semantic variables learned on, or consistently attached to, one coherent
room Gaussian Splat can provide accurate instance isolation and editing while preserving the
appearance quality and placement of monolithic 3DGS.

The study must measure trade-offs rather than assume the proposed method is superior.

## Primary Question

Does joint or post-hoc Gaussian-native semantic reconstruction produce better object
completeness, lower cross-object leakage, and stronger edit isolation than independently
trained object splats or surface conversion, without an unacceptable loss in novel-view
quality?

## Hypotheses

- H1: One shared room splat preserves global placement better than independent object models.
- H2: Semantic logits optimized through Gaussian alpha compositing outperform center-only
  post-hoc voting at boundaries and under occlusion.
- H3: Confidence abstention lowers leakage but reduces completeness; both must be reported.
- H4: Persistent instance Gaussian groups improve edit isolation over class-level groups.
- H5: Independent object splats may isolate large objects but retain more background leakage
  and duplicate appearance than a shared Gaussian model.
- H6: TSDF conversion will underperform Gaussian-native methods when rendered depth and camera
  contracts are not sufficiently surface accurate.

## Compared Methods

| ID | Method | Status |
|---|---|---|
| G0 | Monolithic room 3DGS without semantics | Implemented baseline |
| G1 | Post-hoc footprint voting in `pipeline.ipynb` | Implemented baseline |
| G2 | Joint semantic logits in one Gaussian model | Proposed primary method |
| G3 | Raw independently trained object splats | Implemented baseline |
| G4 | Cleaned independently trained object splats | Implemented ablation |
| G5 | Archived shared/object TSDF semantic surfaces | Failed negative-result baseline |
| G6 | G2 plus persistent editable instance groups | Product research target |

The first controlled paper can compare G0, G1, G3, G4, and G5 while G2 is implemented. The
main systems paper should include G2 and G6 before claiming semantic-first reconstruction.

## Proposed Method

Each Gaussian retains its appearance parameters and gains semantic logits `s_i` over object
instances plus unknown. Cameras render both RGB and semantic alpha using the same ordering and
transmittance calculation. Accepted tracked masks provide 2D supervision. Temporal track IDs,
3D Gaussian neighborhoods, and uncertainty regularize membership without forcing every
Gaussian into a class.

At export, the model writes:

```text
full-room Gaussian parameters
per-Gaussian class and instance probabilities
confidence and support metadata
stable instance-to-Gaussian index groups
editable pivots and non-destructive transform state
```

## Metrics

- Appearance: PSNR, SSIM, LPIPS on fixed held-out cameras.
- Semantics: per-instance IoU, boundary F-score, calibration error, unknown rate.
- Completeness: target-mask recall and observed-region coverage.
- Leakage: pixels rendered outside the verified target and empty-view false positives.
- Placement: alignment to the monolithic room and cross-view instance consistency.
- Editing: non-target pixel change after hide, move, rotate, and remove operations.
- Cost: runtime, memory, file size, and Gaussian count.

Report category macro-averages so walls and floors cannot dominate furniture and small objects.

## Required Ablations

1. Center-only masks versus projected Gaussian footprint coverage.
2. Hard winner labels versus calibrated probability vectors and unknown.
3. Semantic alpha rendering versus independent point projection.
4. Temporal tracking loss on/off.
5. 3D neighborhood consistency on/off.
6. Class-level groups versus persistent instance groups.
7. Raw versus cleaned independent object splats.
8. Gaussian-native output versus archived surface conversion.

## Claim Gate

A paper may claim improved semantic editability only when all methods use identical frames,
poses, masks, and test cameras; G2 or G6 improves a predefined held-out semantic or edit metric;
appearance loss and abstained coverage are reported; failed instances remain in aggregates;
and results include multiple rooms or are explicitly a single-room case study.

## Immediate Milestone

Freeze a `scan_001` camera split and evaluate G0 and G1 first. The result should establish a
reproducible semantic-rendering harness and identify whether remaining error is dominated by
segmentation, visibility, compositing, or Gaussian membership. Implement G2 only after that
measurement path is trustworthy.
