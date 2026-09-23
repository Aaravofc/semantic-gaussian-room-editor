"""Generate a small, privacy-safe Gaussian room for the browser demo.

The asset is intentionally synthetic. It uses Nerfstudio's Gaussian PLY field
layout so the same viewer path exercised by real exports can load it.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import struct
from dataclasses import dataclass
from pathlib import Path


SH_C0 = 0.28209479177387814
GAUSSIAN_STRUCT = struct.Struct("<62f")
POINT_STRUCT = struct.Struct("<3f3B")


@dataclass(frozen=True)
class DemoObject:
    object_id: str
    label: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    color: tuple[int, int, int]
    point_count: int


OBJECTS = [
    DemoObject("floor_001", "floor", (0.0, -0.04, 0.0), (6.2, 0.08, 4.2), (112, 126, 140), 3200),
    DemoObject("wall_001", "wall", (0.0, 1.5, -2.04), (6.2, 3.0, 0.08), (185, 193, 203), 2600),
    DemoObject("couch_001", "couch", (-1.1, 0.52, -0.72), (1.9, 1.04, 0.82), (62, 126, 172), 3600),
    DemoObject("table_001", "table", (1.05, 0.48, 0.18), (1.25, 0.96, 1.0), (151, 101, 61), 3000),
    DemoObject("lamp_001", "lamp", (1.45, 1.3, -1.28), (0.55, 1.45, 0.55), (238, 194, 74), 1900),
    DemoObject("plant_001", "plant", (-2.05, 0.74, 0.82), (0.78, 1.48, 0.78), (64, 145, 98), 2200),
    DemoObject("rug_001", "rug", (0.0, 0.015, 0.62), (2.6, 0.03, 1.35), (169, 92, 126), 1600),
]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def color_to_sh(color: tuple[int, int, int]) -> tuple[float, float, float]:
    return tuple(((channel / 255.0) - 0.5) / SH_C0 for channel in color)


def random_on_box(
    rng: random.Random,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
) -> tuple[float, float, float]:
    axis = rng.randrange(3)
    side = -1.0 if rng.random() < 0.5 else 1.0
    point = [center[index] + rng.uniform(-0.5, 0.5) * size[index] for index in range(3)]
    point[axis] = center[axis] + side * size[axis] * 0.5
    return tuple(point)


def random_object_point(
    rng: random.Random, obj: DemoObject
) -> tuple[float, float, float]:
    x, y, z = obj.center
    sx, sy, sz = obj.size

    if obj.label in {"floor", "rug"}:
        return (rng.uniform(x - sx / 2, x + sx / 2), y, rng.uniform(z - sz / 2, z + sz / 2))
    if obj.label == "wall":
        return (rng.uniform(x - sx / 2, x + sx / 2), rng.uniform(y - sy / 2, y + sy / 2), z)
    if obj.label == "lamp":
        if rng.random() < 0.48:
            angle = rng.random() * math.tau
            radius = 0.045
            return (x + math.cos(angle) * radius, rng.uniform(y - sy / 2, y + sy * 0.12), z + math.sin(angle) * radius)
        shade_y = rng.uniform(y + sy * 0.05, y + sy / 2)
        shade_radius = 0.12 + (shade_y - (y + sy * 0.05)) / (sy * 0.45) * 0.15
        angle = rng.random() * math.tau
        return (x + math.cos(angle) * shade_radius, shade_y, z + math.sin(angle) * shade_radius)
    if obj.label == "plant":
        if rng.random() < 0.30:
            angle = rng.random() * math.tau
            radius = rng.uniform(0.20, 0.34)
            return (x + math.cos(angle) * radius, rng.uniform(y - sy / 2, y - sy * 0.08), z + math.sin(angle) * radius)
        angle = rng.random() * math.tau
        radius = rng.uniform(0.04, sx / 2)
        height_bias = rng.random() ** 0.7
        return (x + math.cos(angle) * radius, y - sy * 0.08 + height_bias * sy * 0.58, z + math.sin(angle) * radius)
    return random_on_box(rng, obj.center, obj.size)


def gaussian_record(
    position: tuple[float, float, float],
    color: tuple[int, int, int],
    rng: random.Random,
) -> bytes:
    jittered = tuple(value + rng.gauss(0.0, 0.006) for value in position)
    noisy_color = tuple(int(clamp(channel + rng.gauss(0.0, 5.0), 0, 255)) for channel in color)
    dc = color_to_sh(noisy_color)
    radius = rng.uniform(0.025, 0.045)
    values = [
        *jittered,
        0.0,
        0.0,
        0.0,
        *dc,
        *([0.0] * 45),
        4.2,
        math.log(radius),
        math.log(radius),
        math.log(radius),
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    assert len(values) == 62
    return GAUSSIAN_STRUCT.pack(*values)


def gaussian_header(count: int) -> bytes:
    properties = ["x", "y", "z", "nx", "ny", "nz"]
    properties += [f"f_dc_{index}" for index in range(3)]
    properties += [f"f_rest_{index}" for index in range(45)]
    properties += ["opacity", "scale_0", "scale_1", "scale_2"]
    properties += [f"rot_{index}" for index in range(4)]
    lines = [
        "ply",
        "format binary_little_endian 1.0",
        "comment Synthetic portfolio demo; no real room data",
        "comment Vertical Axis: y",
        f"element vertex {count}",
        *[f"property float {name}" for name in properties],
        "end_header",
    ]
    return ("\n".join(lines) + "\n").encode("ascii")


def point_header(count: int) -> bytes:
    return (
        "ply\nformat binary_little_endian 1.0\n"
        "comment Synthetic semantic point overlay\n"
        f"element vertex {count}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")


def proxy_record(obj: DemoObject) -> dict[str, object]:
    minimum = [obj.center[index] - obj.size[index] / 2 for index in range(3)]
    maximum = [obj.center[index] + obj.size[index] / 2 for index in range(3)]
    return {
        "object_id": obj.object_id,
        "label": obj.label,
        "role": "synthetic_demo",
        "center": list(obj.center),
        "size": list(obj.size),
        "min": minimum,
        "max": maximum,
        "confidence": 1.0,
        "support_views": 12,
        "point_count": obj.point_count,
        "color": list(obj.color),
    }


def generate(output_dir: Path, seed: int = 7) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    points: list[tuple[tuple[float, float, float], DemoObject]] = []
    for obj in OBJECTS:
        for _ in range(obj.point_count):
            points.append((random_object_point(rng, obj), obj))
    rng.shuffle(points)

    splat_path = output_dir / "splat.ply"
    with splat_path.open("wb") as handle:
        handle.write(gaussian_header(len(points)))
        for position, obj in points:
            handle.write(gaussian_record(position, obj.color, rng))

    semantic_path = output_dir / "combined_objects_semantic_colored.ply"
    with semantic_path.open("wb") as handle:
        handle.write(point_header(len(points)))
        for position, obj in points:
            handle.write(POINT_STRUCT.pack(*position, *obj.color))

    records = [proxy_record(obj) for obj in OBJECTS]
    boxes = {
        "schema": "object_bounding_boxes.synthetic_demo.v1",
        "created_at": "2026-09-22T00:00:00Z",
        "scan_id": "scan_demo",
        "synthetic": True,
        "boxes": records,
    }
    manifest = {
        "schema": "semantic_scene_manifest.synthetic_demo.v1",
        "created_at": "2026-09-22T00:00:00Z",
        "scan_id": "scan_demo",
        "synthetic": True,
        "world_up": [0, 1, 0],
        "objects": records,
    }
    (output_dir / "object_bounding_boxes.json").write_text(
        json.dumps(boxes, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "semantic_scene_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(points):,} synthetic Gaussians to {splat_path}")


def main() -> None:
    default_output = Path(__file__).resolve().parents[1] / "demo" / "scan_demo"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    generate(args.output.resolve(), seed=args.seed)


if __name__ == "__main__":
    main()
