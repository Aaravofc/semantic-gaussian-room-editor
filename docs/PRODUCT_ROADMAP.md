# Product Roadmap

## Direction

The production representation is one coherent full-room Gaussian Splat with Gaussian-native
semantic and instance metadata. The same Gaussians provide appearance, placement, and object
membership. Explicit meshes may be added later as validated collision proxies, but they do not
replace the visual reconstruction.

## V1: Restore the Gaussian Backbone

Implemented locally:

- monolithic room reconstruction and `splat.ply` export;
- tracked object masks and structural layout masks;
- footprint-aware multiview votes on existing Gaussian centers;
- `splat_labeled.ply` with semantic metadata;
- semantic manifests, point overlays, and editable bounding-box proxies;
- viewer routing that prefers the labeled splat and cannot auto-load archived TSDF scenes;
- non-destructive proxy and full-scene transform state.

V1 is a measurable baseline, not consumer-quality object editing. Proxy edits do not yet move
the member Gaussians.

## V1.1: Measurement and Calibration

Before another cleanup rule or model change:

1. freeze train, validation, and test cameras;
2. curate test masks for selected instances;
3. render Gaussian alpha and semantic IDs from the exact cameras;
4. measure mask IoU, boundary F-score, completeness, leakage, and empty-view false positives;
5. calibrate confidence and abstention thresholds on validation views only;
6. trace every failure to segmentation, visibility, semantic voting, or reconstruction.

## V2: Semantic-Aware Gaussian Training

Add learnable semantic logits `s_i` to each Gaussian and optimize them with the same rendering
and camera model used by appearance training. A target objective can combine:

```text
L = lambda_rgb      * L_rgb
  + lambda_mask     * L_mask
  + lambda_boundary * L_boundary
  + lambda_temporal * L_track_consistency
  + lambda_spatial  * L_neighbor_consistency
  + lambda_sparse   * L_semantic_entropy
```

The semantic renderer must account for Gaussian alpha compositing and occlusion instead of
treating each center as an isolated point. Unknown/uncertain membership remains available.

## V3: Instance Groups and Real Edits

Convert calibrated per-Gaussian probabilities into persistent instance groups. Each group has
a stable ID, Gaussian index set, pivot, confidence, support relations, and reversible transform.

- Hide: disable the selected group's opacity.
- Move/rotate/scale: transform both Gaussian centers and covariance orientations.
- Remove: preserve the original group and expose a separate background-completion task.
- Add: insert a registered Gaussian asset with provenance and collision bounds.

The editor must show uncertain boundary Gaussians and allow membership correction before an
operation is committed.

## V4: Composition and Completion

Add physical constraints only after instance groups are reliable:

- structural planarity for walls, floor, and ceiling;
- support and non-penetration constraints;
- object rigidity under transforms;
- room-aware collision proxies;
- AI completion of revealed or missed regions as a separate generated layer.

Generated content must never be merged invisibly with observed Gaussians. Users must be able
to hide it, and research metrics must report it separately.

## Consumer Release Gates

A room is ready for end-user editing only when:

- the full-room appearance remains stable after semantic processing;
- object identity and membership persist across views;
- semantic leakage and object completeness pass category-specific thresholds;
- moving or removing one object changes only its intended Gaussian group;
- save/reload and undo are deterministic;
- unknown and generated regions are clearly distinguishable;
- editing remains responsive at the target device budget.
