# Portfolio and Application Summary

## One-line description

A research prototype that reconstructs an indoor scene as one coherent 3D Gaussian Splat,
tracks objects across video frames, attaches multiview semantic evidence to the scene, and
exposes semantic proxies in an interactive browser editor.

## Resume bullet

Built a semantic 3D room reconstruction prototype combining Gaussian Splatting,
open-vocabulary detection, multi-frame segmentation, and a browser scene editor; implemented
persistent object tracking, multiview semantic fusion, and object-constrained Gaussian training.

## Short application description

I built an end-to-end computer-vision prototype for turning a room video into a semantically
organized 3D scene. The system uses camera registration and Gaussian Splatting for appearance,
Grounding DINO and SAM 2.1 for persistent object masks, multiview projection for semantic
evidence, and a Three.js interface for inspection and non-destructive proxy edits. I also
implemented a constrained Gaussian-training experiment and automated contract tests. The
current version is a research prototype: object transforms affect semantic proxies, while
Gaussian-level object editing remains future work.

## Accurate talking points

- Explain why all semantics share one room coordinate system.
- Discuss temporal tracking, occlusion, uncertain labels, and multiview voting.
- Show the synthetic demo and walk through object selection and saved edit state.
- Describe the containment-loss experiment as an implemented method under evaluation.
- Mention that the repository separates measurable results from proposed research.

## Claims to avoid until evaluation is complete

- Production-quality object removal or movement.
- Accuracy improvements over a baseline.
- Generalization beyond the evaluated rooms.
- Real-time reconstruction.

