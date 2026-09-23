"""Tracked open-vocabulary segmentation for object-first Gaussian training.

The pipeline deliberately separates four concerns:

1. Grounding DINO detects canonical object classes on anchor frames.
2. Anchor detections are associated into persistent instance tracks.
3. SAM 2.1 propagates those tracks through the registered video frames.
4. SegFormer owns wall/floor/ceiling masks and never emits object labels.

Only masks that pass the quality and overlap gates are published to the
canonical ``semantic_recon/detections`` directory. Rejected candidates and
tri-state visibility decisions are retained in reports for diagnosis.
"""

from __future__ import annotations

import argparse
import csv
import errno
import hashlib
import inspect
import json
import logging
import math
import os
import re
import shutil
import sys
import tempfile
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


PIPELINE_VERSION = "2026-08-18-grounded-sam2-tracked-segmentation-v2.7-room-wide-audit"


DEFAULT_CONFIG: dict[str, Any] = {
    "pipeline_version": PIPELINE_VERSION,
    "project_root": "/content/drive/MyDrive/room-model-project",
    "scan_id": "scan_001",
    "source_ns_data_dir": None,
    "require_gpu": True,
    "resume": True,
    "force_stage": False,
    "prune_stale_outputs_after_audit": True,
    "max_frames": None,
    "grounding_dino_model_id": "IDEA-Research/grounding-dino-base",
    "grounding_dino_text_threshold": 0.25,
    "detection_keyframe_stride": 8,
    "detector_nms_iou": 0.50,
    "minimum_box_area_fraction": 0.00015,
    "default_maximum_box_area_fraction": 0.70,
    "class_prompts": {
        "bed": "bed",
        "chair": "chair",
        "table": "table",
        "door": "door",
        "tv": "television",
        "wardrobe": "wardrobe",
        "cabinet": "cabinet",
        "drawer": "drawer",
        "shelf": "shelf",
        "mirror": "mirror",
        "desk": "desk",
        "sofa": "sofa",
        "lamp": "lamp",
        "fan": "ceiling fan",
        "light": "light fixture",
        "curtain": "fabric curtain hanging over a window",
        "window": "window",
        "rug": "rug",
        "pillow": "pillow",
        "blanket": "loose blanket on top of a bed",
        "bag": "bag",
        "shoe": "shoe",
        "bottle": "bottle",
        "book": "book",
        "plant": "indoor plant",
        "picture_frame": "picture frame",
        "miscellaneous": "miscellaneous room object",
    },
    # Aliases improve recall without creating additional semantic classes. Every detection from
    # these prompts is published under the canonical key on the left.
    "class_prompt_aliases": {
        "bed": ["mattress"],
        "chair": ["armchair", "stool"],
        "table": ["bedside table", "nightstand"],
        "door": ["room door"],
        "tv": ["computer monitor", "television screen"],
        "wardrobe": ["closet"],
        "cabinet": ["cupboard"],
        "desk": ["writing desk", "work desk"],
        "sofa": ["couch"],
        "lamp": ["table lamp", "floor lamp"],
        "fan": ["standing fan", "electric fan"],
        "light": ["ceiling light"],
        "curtain": ["window drape"],
        "window": ["window pane"],
        "pillow": ["cushion", "bed pillow"],
        "blanket": ["folded blanket", "folded duvet", "folded quilt"],
        "bag": ["backpack", "handbag", "suitcase", "duffel bag"],
        "rug": ["carpet", "floor rug", "mat"],
        "shoe": ["footwear", "slipper"],
        "bottle": ["water bottle"],
        "plant": ["potted plant"],
        "picture_frame": ["photo frame", "wall art"],
        "miscellaneous": [
            "headboard",
            "switchboard",
            "electrical panel",
            "power outlet",
            "wall socket",
            "wall switch",
            "extension board",
            "remote control",
            "wall clock",
            "smoke detector",
        ],
    },
    # The detector does not activate the complete vocabulary at once. Major editable objects are
    # probed first. Accepted core evidence activates only the detail profiles relevant to this
    # scene, while room fixtures remain active in every scan. This keeps aliases and uncommon
    # concepts available without creating one large, competing prompt list.
    "core_detector_labels": [
        "bed",
        "chair",
        "table",
        "door",
        "tv",
        "wardrobe",
        "cabinet",
        "desk",
        "sofa",
        "curtain",
        "window",
    ],
    "class_prompt_profiles": {
        "room_details": {
            "trigger_labels": [],
            "labels": [
                "drawer",
                "shelf",
                "mirror",
                "lamp",
                "fan",
                "light",
                "rug",
                "plant",
                "picture_frame",
            ],
        },
        "bedroom": {
            "trigger_labels": ["bed", "wardrobe"],
            "labels": ["pillow", "blanket", "bag", "shoe"],
        },
        "workspace": {
            "trigger_labels": ["desk", "table"],
            "labels": ["book", "bottle", "bag"],
        },
        "living_room": {
            "trigger_labels": ["sofa", "tv"],
            "labels": ["pillow", "blanket", "book", "bottle", "bag"],
        },
    },
    # The complete alias dictionary above is a vocabulary library. Only this small subset is
    # activated for each pass/profile; the remaining synonyms stay available for later profiles
    # without competing in every Grounding DINO query.
    "core_prompt_aliases": {
        "table": ["nightstand"],
        "tv": ["computer monitor"],
        "wardrobe": ["closet"],
        "sofa": ["couch"],
    },
    "profile_prompt_aliases": {
        "room_details": {
            "rug": ["carpet"],
            "plant": ["potted plant"],
            "picture_frame": ["wall art"],
        },
        "bedroom": {
            "pillow": ["cushion", "bed pillow"],
            "blanket": ["folded blanket", "folded duvet"],
            "bag": ["backpack", "suitcase"],
        },
        "workspace": {
            "bag": ["backpack"],
            "bottle": ["water bottle"],
        },
        "living_room": {
            "pillow": ["cushion"],
            "blanket": ["folded duvet"],
            "bag": ["backpack"],
        },
    },
    "miscellaneous_prompt_profiles": {
        "room_details": [
            "switchboard",
            "electrical panel",
            "power outlet",
            "wall socket",
            "wall switch",
            "extension board",
            "wall clock",
            "smoke detector",
        ],
        "bedroom": ["headboard", "remote control"],
        "workspace": ["remote control"],
        "living_room": ["remote control"],
    },
    "always_active_prompt_profiles": ["room_details"],
    "forced_prompt_profiles": [],
    "profile_activation_minimum_anchor_frames": 1,
    "profile_activation_minimum_detections": 1,
    "enable_miscellaneous_canonical_prompt": False,
    # These prompts provide limited open-set coverage. They are deliberately narrower and use a
    # higher threshold than named aliases. Generic fallback boxes overlapping a known-class box
    # are rejected before tracking, which prevents miscellaneous from duplicating normal objects.
    "miscellaneous_generic_prompts": [
        "small unidentified room object",
        "small household object",
        "wall-mounted object",
    ],
    "miscellaneous_generic_detection_threshold": 0.42,
    "miscellaneous_generic_maximum_box_area_fraction": 0.20,
    "miscellaneous_known_overlap_iou": 0.20,
    "miscellaneous_known_containment_fraction": 0.65,
    "class_detection_thresholds": {
        "bed": 0.40,
        "chair": 0.34,
        "table": 0.36,
        "door": 0.36,
        "tv": 0.34,
        "wardrobe": 0.38,
        "cabinet": 0.36,
        "drawer": 0.34,
        "shelf": 0.34,
        "mirror": 0.36,
        "desk": 0.38,
        "sofa": 0.38,
        "lamp": 0.32,
        "fan": 0.34,
        "light": 0.34,
        "curtain": 0.48,
        "window": 0.36,
        "rug": 0.34,
        "pillow": 0.28,
        "blanket": 0.46,
        "bag": 0.28,
        "shoe": 0.30,
        "bottle": 0.30,
        "book": 0.30,
        "plant": 0.32,
        "picture_frame": 0.34,
        "miscellaneous": 0.34,
    },
    "class_maximum_box_area_fraction": {
        "bed": 0.70,
        "curtain": 0.55,
        "door": 0.60,
        "window": 0.60,
        "rug": 0.50,
        "wardrobe": 0.65,
        "sofa": 0.65,
        "chair": 0.45,
        "desk": 0.50,
        "pillow": 0.30,
        "blanket": 0.35,
        "bag": 0.35,
        "miscellaneous": 0.60,
        "book": 0.18,
        "bottle": 0.15,
        "shoe": 0.18,
    },
    "track_minimum_anchor_count": 2,
    "track_single_anchor_strong_confidence": 0.52,
    "track_maximum_anchor_gap_multiplier": 3,
    "track_association_minimum_iou": 0.10,
    "track_association_maximum_center_distance": 0.28,
    "track_association_minimum_area_ratio": 0.25,
    "maximum_instances_by_label": {
        "bed": 2,
        "desk": 3,
        "table": 4,
        "door": 4,
        "tv": 3,
        "wardrobe": 3,
        "cabinet": 5,
        "sofa": 2,
        "chair": 8,
        "pillow": 10,
        "bag": 8,
        "miscellaneous": 12,
    },
    # Do not use /content/sam2 as the checkout directory. Because /content is on Colab's
    # sys.path, that directory can shadow the actual sam2 package as a namespace package.
    "sam2_repo": "/content/facebookresearch_sam2",
    "sam2_checkpoint": "/content/room_model_models/sam2.1_hiera_large.pt",
    "sam2_model_config": "configs/sam2.1/sam2.1_hiera_l.yaml",
    "sam2_use_cuda_postprocessing": False,
    "sam2_offload_video_to_cpu": True,
    "sam2_offload_state_to_cpu": False,
    "sam_mask_logit_threshold": 0.0,
    "sam_stability_offset": 1.0,
    "minimum_sam_stability": 0.80,
    "minimum_combined_mask_quality": 0.55,
    "minimum_bidirectional_iou": 0.20,
    "minimum_largest_component_ratio": 0.50,
    "minimum_component_area_ratio": 0.02,
    "maximum_components_to_keep": 2,
    "maximum_hole_area_pixels": 64,
    "minimum_mask_area_fraction": 0.00010,
    "default_maximum_mask_area_fraction": 0.70,
    "class_maximum_mask_area_fraction": {
        "bed": 0.70,
        "curtain": 0.60,
        "door": 0.60,
        "window": 0.60,
        "rug": 0.55,
        "wardrobe": 0.70,
        "sofa": 0.70,
        "chair": 0.45,
        "desk": 0.55,
        "pillow": 0.30,
        "blanket": 0.38,
        "bag": 0.40,
        "miscellaneous": 0.60,
        "book": 0.18,
        "bottle": 0.15,
        "shoe": 0.18,
    },
    "track_visibility_margin_multiplier": 1.5,
    "minimum_track_visible_frames": 3,
    "maximum_adjacent_area_ratio": 5.0,
    "minimum_remaining_after_overlap": 0.35,
    # Cross-label gates run on Grounding DINO anchors before SAM2 propagation. They target two
    # recurring whole-room confusions without changing the canonical output taxonomy.
    "curtain_requires_window_context": True,
    "curtain_window_minimum_intersection_over_smaller": 0.08,
    "curtain_window_maximum_center_distance": 0.20,
    "curtain_without_window_minimum_confidence": 0.86,
    "blanket_bed_minimum_blanket_overlap": 0.80,
    "blanket_bed_minimum_area_ratio": 0.55,
    "overlap_priority": {
        "pillow": 1.35,
        "blanket": 1.20,
        "bag": 1.35,
        "book": 1.25,
        "bottle": 1.25,
        "shoe": 1.20,
        "lamp": 1.15,
        "fan": 1.15,
        "light": 1.15,
        "picture_frame": 1.15,
        "miscellaneous": 1.15,
        "window": 1.05,
        "rug": 1.10,
        "bed": 1.00,
        "desk": 1.05,
        "chair": 1.05,
        "curtain": 0.90,
    },
    "confirmed_absence_requires_neighboring_anchors": True,
    "write_all_visual_checks": False,
    "visual_check_limit": 180,
    "visual_check_sampling_mode": "uniform_full_sequence",
    "rejected_mask_sample_limit": 400,
    "enable_sam3_pilot": False,
    "sam3_model_id": "facebook/sam3",
    "sam3_pilot_labels": ["bed", "desk", "chair", "door"],
    "sam3_pilot_start_frame": 0,
    "sam3_pilot_frame_count": 48,
    "sam3_pilot_score_threshold": 0.50,
    "structural_model_id": "nvidia/segformer-b2-finetuned-ade-512-512",
    "structural_labels": ["wall", "floor", "ceiling"],
    "structural_minimum_pixel_confidence": {
        "wall": 0.50,
        "floor": 0.45,
        "ceiling": 0.45,
    },
    "structural_minimum_area_fraction": {
        "wall": 0.01,
        "floor": 0.01,
        "ceiling": 0.003,
    },
    "structural_cutout_minimum_object_confidence": 0.55,
    "structural_close_size": 9,
    "structural_open_size": 3,
}


def utc_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def normalize_label(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def visual_check_frame_indexes(
    frames: list[dict[str, Any]], config: dict[str, Any]
) -> set[int]:
    """Choose review frames across the complete registered sequence."""

    frame_indexes = [
        int(frame.get("frame_index", position))
        for position, frame in enumerate(frames)
    ]
    if not frame_indexes:
        return set()
    if bool(config["write_all_visual_checks"]):
        return set(frame_indexes)
    if config.get("visual_check_sampling_mode") != "uniform_full_sequence":
        raise ValueError(
            "visual_check_sampling_mode must be 'uniform_full_sequence'."
        )
    limit = min(len(frame_indexes), max(0, int(config["visual_check_limit"])))
    if limit == 0:
        return set()
    if limit == 1:
        return {frame_indexes[0]}
    positions = {
        round(sample_index * (len(frame_indexes) - 1) / (limit - 1))
        for sample_index in range(limit)
    }
    return {frame_indexes[position] for position in positions}


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


_TRANSIENT_STORAGE_ERRNOS = {
    errno.EIO,
    errno.EBUSY,
    errno.EAGAIN,
    errno.ETIMEDOUT,
    errno.ENOTCONN,
    errno.ESTALE,
    getattr(errno, "EREMOTEIO", 121),
}


def _commit_staged_file(
    staged_path: Path, destination: Path, maximum_attempts: int = 5
) -> None:
    """Commit a local file to Drive with atomic replacement and bounded retries."""
    last_error: OSError | None = None
    for attempt in range(1, maximum_attempts + 1):
        remote_temporary = destination.with_name(
            f".{destination.name}.{os.getpid()}.{time.time_ns()}.{attempt}.tmp"
        )
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(staged_path, remote_temporary)
            remote_temporary.replace(destination)
            return
        except OSError as error:
            last_error = error
            try:
                remote_temporary.unlink(missing_ok=True)
            except OSError:
                pass
            if (
                error.errno not in _TRANSIENT_STORAGE_ERRNOS
                or attempt >= maximum_attempts
            ):
                break
            wait_seconds = min(8.0, float(2 ** (attempt - 1)))
            print(
                f"Transient storage error while writing {destination} "
                f"(attempt {attempt}/{maximum_attempts}): {error}. "
                f"Retrying in {wait_seconds:.0f}s."
            )
            time.sleep(wait_seconds)

    if last_error is None:
        raise RuntimeError(f"Could not write output file: {destination}")
    guidance = ""
    if str(destination).startswith("/content/drive/"):
        guidance = (
            " Google Drive may be temporarily unavailable or disconnected. "
            "Reconnect Drive, then rerun only the failed stage."
        )
    raise OSError(
        last_error.errno,
        f"Could not write {destination} after {maximum_attempts} attempts.{guidance}",
        str(destination),
    ) from last_error


def _staging_directory() -> Path:
    directory = Path(tempfile.gettempdir()) / "room-model-report-staging"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(path: Path, value: Any) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        prefix=f"{path.stem}-",
        dir=_staging_directory(),
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        staged_path = Path(handle.name)
    try:
        _commit_staged_file(staged_path, path)
    finally:
        staged_path.unlink(missing_ok=True)


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    records = list(rows)
    columns: list[str] = []
    for record in records:
        for key in record:
            if key not in columns:
                columns.append(key)
    with tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        encoding="utf-8",
        suffix=".csv",
        prefix=f"{path.stem}-",
        dir=_staging_directory(),
        delete=False,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in columns})
        staged_path = Path(handle.name)
    try:
        _commit_staged_file(staged_path, path)
    finally:
        staged_path.unlink(missing_ok=True)


def _delete_file_with_retries(path: Path, maximum_attempts: int = 5) -> None:
    """Delete a generated file while tolerating transient Google Drive failures."""
    last_error: OSError | None = None
    for attempt in range(1, maximum_attempts + 1):
        try:
            path.unlink(missing_ok=True)
            return
        except OSError as error:
            last_error = error
            if error.errno not in _TRANSIENT_STORAGE_ERRNOS or attempt >= maximum_attempts:
                break
            wait_seconds = min(8.0, float(2 ** (attempt - 1)))
            print(
                f"Transient storage error while deleting {path} "
                f"(attempt {attempt}/{maximum_attempts}): {error}. "
                f"Retrying in {wait_seconds:.0f}s."
            )
            time.sleep(wait_seconds)

    if last_error is not None:
        raise OSError(
            last_error.errno,
            f"Could not delete stale generated file {path} after {maximum_attempts} attempts.",
            str(path),
        ) from last_error


def _path_key(path: str | Path) -> str:
    return os.path.normcase(os.path.abspath(os.path.normpath(str(path))))


def _prune_managed_directory(
    area: str,
    root: Path,
    keep_paths: Iterable[str | Path],
    recursive: bool = True,
) -> list[dict[str, str]]:
    """Remove files not referenced by the current successful pipeline state."""
    if not root.exists():
        return []
    keep = {_path_key(path) for path in keep_paths}
    candidates = root.rglob("*") if recursive else root.glob("*")
    removed: list[dict[str, str]] = []
    for candidate in sorted(candidates):
        if not candidate.is_file() or _path_key(candidate) in keep:
            continue
        _delete_file_with_retries(candidate)
        removed.append({"area": area, "path": str(candidate)})
    return removed


def prune_stale_generated_outputs(
    config: dict[str, Any],
    paths: dict[str, Path],
    frames: list[dict[str, Any]],
    canonical_mask_paths: Iterable[str | Path],
    layout_mask_paths: Iterable[str | Path],
) -> dict[str, Any]:
    """Prune only pipeline-owned files that are absent from the audited state."""
    visual_indexes = visual_check_frame_indexes(frames, config)
    visual_frames = [
        frame
        for position, frame in enumerate(frames)
        if int(frame.get("frame_index", position)) in visual_indexes
    ]

    removed_rows: list[dict[str, str]] = []
    managed_areas = [
        (
            "frame_detections",
            paths["detection_dir"],
            [paths["detection_dir"] / f"{frame['frame_key']}.json" for frame in frames],
            True,
        ),
        ("canonical_masks", paths["mask_dir"], canonical_mask_paths, True),
        (
            "tracked_visual_checks",
            paths["visual_dir"],
            [paths["visual_dir"] / f"{frame['frame_key']}_tracked.jpg" for frame in visual_frames],
            True,
        ),
        ("layout_masks", paths["layout_mask_dir"], layout_mask_paths, True),
        (
            "layout_labels",
            paths["layout_label_dir"],
            [paths["layout_label_dir"] / f"{frame['frame_key']}.json" for frame in frames],
            True,
        ),
        (
            "layout_visual_checks",
            paths["layout_visual_dir"],
            [
                paths["layout_visual_dir"] / f"{frame['frame_key']}_structural.jpg"
                for frame in visual_frames
            ],
            True,
        ),
        (
            "tracking_rejections",
            paths["rejected_dir"],
            [paths["rejected_dir"] / f"{frame['frame_key']}.json" for frame in frames],
            False,
        ),
    ]

    anchor_manifest_path = paths["tracking_dir"] / "anchor_detection_manifest.json"
    anchor_paths: list[str] = []
    if anchor_manifest_path.exists():
        anchor_paths = read_json(anchor_manifest_path).get("anchor_json_paths", [])
    managed_areas.append(("anchor_detections", paths["anchor_dir"], anchor_paths, True))

    for area, root, keep_paths, recursive in managed_areas:
        removed_rows.extend(_prune_managed_directory(area, root, keep_paths, recursive))

    cleanup_report = paths["report_dir"] / "stale_output_cleanup.csv"
    write_csv(cleanup_report, removed_rows)
    return {
        "enabled": True,
        "removed_file_count": len(removed_rows),
        "removed_by_area": dict(Counter(row["area"] for row in removed_rows)),
        "report": str(cleanup_report),
    }


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path) -> dict[str, Any]:
    supplied = read_json(path)
    config = deep_merge(DEFAULT_CONFIG, supplied)
    config["config_path"] = str(path)
    if config.get("pipeline_version") != PIPELINE_VERSION:
        raise RuntimeError(
            f"Configuration pipeline_version={config.get('pipeline_version')!r} does not match "
            f"code version {PIPELINE_VERSION!r}. Rerun the notebook configuration cell."
        )
    build_scene_prompt_plan(config, [])
    return config


def project_paths(config: dict[str, Any]) -> dict[str, Path]:
    project_root = Path(config["project_root"])
    scan_dir = project_root / "scans" / config["scan_id"]
    semantic_dir = scan_dir / "semantic_recon"
    source_ns_data = (
        Path(config["source_ns_data_dir"])
        if config.get("source_ns_data_dir")
        else scan_dir / "nerfstudio-data"
    )
    report_dir = scan_dir / "reports" / "segmentation_v2"
    tracking_dir = semantic_dir / "tracking"
    return {
        "project_root": project_root,
        "scan_dir": scan_dir,
        "source_ns_data": source_ns_data,
        "transforms": source_ns_data / "transforms.json",
        "semantic_dir": semantic_dir,
        "detection_dir": semantic_dir / "detections",
        "mask_dir": semantic_dir / "segmentation" / "masks",
        "visual_dir": semantic_dir / "segmentation" / "visual_checks",
        "tracking_dir": tracking_dir,
        "anchor_dir": tracking_dir / "grounding_dino_anchors",
        "track_manifest": tracking_dir / "instance_tracks.json",
        "tracking_manifest": tracking_dir / "tracking_manifest.json",
        "rejected_dir": tracking_dir / "rejected",
        "layout_dir": semantic_dir / "layout",
        "layout_mask_dir": semantic_dir / "layout" / "masks",
        "layout_label_dir": semantic_dir / "layout" / "labels",
        "layout_visual_dir": semantic_dir / "layout" / "visual_checks",
        "report_dir": report_dir,
        "layout_report_dir": scan_dir / "reports" / "layout_masks_v2",
        "fatal_dir": scan_dir / "reports" / "fatal_errors",
        "local_cache": Path("/content/room-model-segmentation-cache") / config["scan_id"],
    }


def ensure_directories(paths: dict[str, Path]) -> None:
    for key in [
        "detection_dir",
        "mask_dir",
        "visual_dir",
        "tracking_dir",
        "anchor_dir",
        "rejected_dir",
        "layout_mask_dir",
        "layout_label_dir",
        "layout_visual_dir",
        "report_dir",
        "layout_report_dir",
        "fatal_dir",
    ]:
        paths[key].mkdir(parents=True, exist_ok=True)


def setup_logger(path: Path, name: str) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def record_fatal(paths: dict[str, Path], stage: str, error: BaseException) -> None:
    fatal_path = paths["fatal_dir"] / "fatal_error.log"
    fatal_path.parent.mkdir(parents=True, exist_ok=True)
    with fatal_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{utc_now()}] tracked_segmentation/{stage}: {error}\n")
        handle.write(traceback.format_exc() + "\n")


def resolve_frame_path(frame: dict[str, Any], paths: dict[str, Path]) -> Path:
    raw = frame.get("file_path")
    if not raw:
        raise RuntimeError("A transforms.json frame is missing file_path.")
    candidate = Path(raw)
    candidates = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.extend(
            [
                paths["source_ns_data"] / candidate,
                paths["source_ns_data"] / "images" / candidate.name,
                paths["scan_dir"] / candidate,
                paths["scan_dir"] / "frames" / candidate.name,
            ]
        )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Image referenced by transforms.json is missing: {raw}. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def load_registered_frames(
    config: dict[str, Any], paths: dict[str, Path]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not paths["transforms"].exists():
        raise FileNotFoundError(
            f"Missing transforms.json: {paths['transforms']}. Run reconstruction.ipynb first."
        )
    transforms = read_json(paths["transforms"])
    source_frames = transforms.get("frames", [])
    if not source_frames:
        raise RuntimeError(f"No frames found in {paths['transforms']}.")
    limit = config.get("max_frames")
    if limit is not None:
        source_frames = source_frames[: int(limit)]
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source_index, frame in enumerate(source_frames):
        image_path = resolve_frame_path(frame, paths)
        frame_key = Path(frame["file_path"]).stem
        if frame_key in seen:
            raise RuntimeError(f"Duplicate registered frame key in transforms.json: {frame_key}")
        seen.add(frame_key)
        records.append(
            {
                "frame_index": len(records),
                "source_index": source_index,
                "frame_key": frame_key,
                "image_path": str(image_path),
                "source_file_path": frame["file_path"],
            }
        )
    return transforms, records


def frame_inventory(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for frame in frames:
        path = Path(frame["image_path"])
        stat = path.stat()
        rows.append(
            {
                "frame_key": frame["frame_key"],
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return rows


def stage_fingerprint(
    config: dict[str, Any], paths: dict[str, Path], frames: list[dict[str, Any]], stage: str
) -> str:
    ignored = {
        "resume",
        "force_stage",
        "prune_stale_outputs_after_audit",
        "config_path",
    }
    relevant_config = {key: value for key, value in config.items() if key not in ignored}
    return stable_hash(
        {
            "stage": stage,
            "pipeline_version": PIPELINE_VERSION,
            "config": relevant_config,
            "transforms_sha256": file_sha256(paths["transforms"]),
            "frames": frame_inventory(frames),
        }
    )


def validate_environment(config: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    ensure_directories(paths)
    _, frames = load_registered_frames(config, paths)
    report = {
        "schema": "tracked_segmentation_validation.v2",
        "pipeline_version": PIPELINE_VERSION,
        "created_at": utc_now(),
        "scan_id": config["scan_id"],
        "project_root": str(paths["project_root"]),
        "transforms_path": str(paths["transforms"]),
        "registered_frame_count": len(frames),
        "first_frame": frames[0],
        "last_frame": frames[-1],
        "grounding_dino_model_id": config["grounding_dino_model_id"],
        "sam2_checkpoint": config["sam2_checkpoint"],
        "structural_model_id": config["structural_model_id"],
        "active_object_labels": sorted(config["class_prompts"]),
        "structural_labels": list(config["structural_labels"]),
    }
    write_json(paths["report_dir"] / "validation.json", report)
    return report


def box_iou(first: list[float], second: list[float]) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def box_center_distance(first: list[float], second: list[float], width: int, height: int) -> float:
    first_x = (float(first[0]) + float(first[2])) * 0.5
    first_y = (float(first[1]) + float(first[3])) * 0.5
    second_x = (float(second[0]) + float(second[2])) * 0.5
    second_y = (float(second[1]) + float(second[3])) * 0.5
    diagonal = max(1.0, math.hypot(width, height))
    return math.hypot(first_x - second_x, first_y - second_y) / diagonal


def box_area_ratio(first: list[float], second: list[float]) -> float:
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    if first_area <= 0 or second_area <= 0:
        return 0.0
    return min(first_area, second_area) / max(first_area, second_area)


def canonical_detector_label(
    value: Any, prompt_to_label: dict[str, str], ordered_labels: list[str]
) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) or (isinstance(value, str) and value.isdigit()):
        index = int(value)
        return ordered_labels[index] if 0 <= index < len(ordered_labels) else None
    normalized = normalize_label(value)
    normalized = re.sub(r"^(a|an|the)_", "", normalized)
    if normalized in prompt_to_label:
        return prompt_to_label[normalized]
    for prompt, label in prompt_to_label.items():
        if normalized == prompt or normalized.endswith("_" + prompt) or prompt.endswith("_" + normalized):
            return label
    return None


def detector_prompt_records(config: dict[str, Any]) -> list[dict[str, str]]:
    """Flatten canonical prompts and aliases while preserving one output label taxonomy."""

    class_prompts = config["class_prompts"]
    aliases = config.get("class_prompt_aliases", {})
    unknown_alias_labels = sorted(set(aliases) - set(class_prompts))
    if unknown_alias_labels:
        raise ValueError(
            "class_prompt_aliases contains unknown canonical labels: "
            + ", ".join(unknown_alias_labels)
        )
    generic_prompts = list(config.get("miscellaneous_generic_prompts", []))
    if generic_prompts and "miscellaneous" not in class_prompts:
        raise ValueError(
            "miscellaneous_generic_prompts requires a miscellaneous canonical class."
        )

    records: list[dict[str, str]] = []
    for label, prompt in class_prompts.items():
        records.append(
            {
                "label": label,
                "prompt": str(prompt),
                "prompt_kind": "generic_fallback"
                if label == "miscellaneous"
                else "canonical",
            }
        )
        records.extend(
            {
                "label": label,
                "prompt": str(alias),
                "prompt_kind": "named_fallback" if label == "miscellaneous" else "alias",
            }
            for alias in aliases.get(label, [])
        )
    records.extend(
        {
            "label": "miscellaneous",
            "prompt": str(prompt),
            "prompt_kind": "generic_fallback",
        }
        for prompt in generic_prompts
    )

    seen: dict[str, dict[str, str]] = {}
    for record in records:
        key = normalize_label(record["prompt"])
        if not key:
            raise ValueError(f"Empty detector prompt for label {record['label']!r}.")
        if key in seen:
            previous = seen[key]
            raise ValueError(
                f"Duplicate detector prompt {record['prompt']!r} for "
                f"{previous['label']!r} and {record['label']!r}."
            )
        seen[key] = record
    return records


def build_scene_prompt_plan(
    config: dict[str, Any], core_detections: list[dict[str, Any]]
) -> dict[str, Any]:
    """Select detail prompts from core object evidence without changing output labels."""

    records = detector_prompt_records(config)
    class_labels = set(config["class_prompts"])
    core_labels = set(config.get("core_detector_labels", []))
    if not core_labels:
        raise ValueError("core_detector_labels must contain at least one canonical label.")
    if "miscellaneous" in core_labels:
        raise ValueError("miscellaneous cannot be a core detector label.")
    unknown_core = sorted(core_labels - class_labels)
    if unknown_core:
        raise ValueError("Unknown core detector labels: " + ", ".join(unknown_core))

    profiles = config.get("class_prompt_profiles", {})
    profile_names = set(profiles)
    always_profiles = set(config.get("always_active_prompt_profiles", []))
    forced_profiles = set(config.get("forced_prompt_profiles", []))
    unknown_requested = sorted((always_profiles | forced_profiles) - profile_names)
    if unknown_requested:
        raise ValueError("Unknown requested prompt profiles: " + ", ".join(unknown_requested))

    profile_labels: set[str] = set()
    for profile_name, profile in profiles.items():
        labels = set(profile.get("labels", []))
        triggers = set(profile.get("trigger_labels", []))
        unknown_labels = sorted(labels - class_labels)
        if unknown_labels:
            raise ValueError(
                f"Prompt profile {profile_name!r} contains unknown labels: "
                + ", ".join(unknown_labels)
            )
        unavailable_triggers = sorted(triggers - core_labels)
        if unavailable_triggers:
            raise ValueError(
                f"Prompt profile {profile_name!r} has triggers outside core_detector_labels: "
                + ", ".join(unavailable_triggers)
            )
        profile_labels.update(labels)

    available_aliases = {
        label: {normalize_label(prompt): str(prompt) for prompt in prompts}
        for label, prompts in config.get("class_prompt_aliases", {}).items()
        if label != "miscellaneous"
    }

    def validate_alias_selection(
        owner: str,
        selection: dict[str, list[str]],
        allowed_labels: set[str],
    ) -> set[tuple[str, str]]:
        selected: set[tuple[str, str]] = set()
        unknown_labels = sorted(set(selection) - allowed_labels)
        if unknown_labels:
            raise ValueError(
                f"{owner} selects aliases for unavailable labels: "
                + ", ".join(unknown_labels)
            )
        for label, prompts in selection.items():
            for prompt in prompts:
                key = normalize_label(prompt)
                if key not in available_aliases.get(label, {}):
                    raise ValueError(
                        f"{owner} selects unknown alias {prompt!r} for label {label!r}."
                    )
                selected.add((label, key))
        return selected

    selected_core_aliases = validate_alias_selection(
        "core_prompt_aliases",
        config.get("core_prompt_aliases", {}),
        core_labels,
    )
    profile_alias_config = config.get("profile_prompt_aliases", {})
    unknown_alias_profiles = sorted(set(profile_alias_config) - profile_names)
    if unknown_alias_profiles:
        raise ValueError(
            "profile_prompt_aliases contains unknown profiles: "
            + ", ".join(unknown_alias_profiles)
        )
    selected_aliases_by_profile: dict[str, set[tuple[str, str]]] = {}
    for profile_name, selection in profile_alias_config.items():
        selected_aliases_by_profile[profile_name] = validate_alias_selection(
            f"profile_prompt_aliases[{profile_name!r}]",
            selection,
            set(profiles[profile_name].get("labels", [])),
        )

    unreachable = sorted(class_labels - core_labels - profile_labels - {"miscellaneous"})
    if unreachable:
        raise ValueError(
            "Canonical labels are unreachable from the hierarchical prompt plan: "
            + ", ".join(unreachable)
        )

    named_miscellaneous = {
        normalize_label(record["prompt"]): record["prompt"]
        for record in records
        if record["label"] == "miscellaneous"
        and record["prompt_kind"] == "named_fallback"
    }
    miscellaneous_profiles = config.get("miscellaneous_prompt_profiles", {})
    unknown_misc_profiles = sorted(set(miscellaneous_profiles) - profile_names)
    if unknown_misc_profiles:
        raise ValueError(
            "miscellaneous_prompt_profiles contains unknown profiles: "
            + ", ".join(unknown_misc_profiles)
        )
    profiled_miscellaneous: set[str] = set()
    for profile_name, prompts in miscellaneous_profiles.items():
        for prompt in prompts:
            key = normalize_label(prompt)
            if key not in named_miscellaneous:
                raise ValueError(
                    f"Profile {profile_name!r} references unknown miscellaneous prompt {prompt!r}."
                )
            profiled_miscellaneous.add(key)
    unprofiled_miscellaneous = sorted(set(named_miscellaneous) - profiled_miscellaneous)
    if unprofiled_miscellaneous:
        raise ValueError(
            "Named miscellaneous prompts are not assigned to a profile: "
            + ", ".join(named_miscellaneous[key] for key in unprofiled_miscellaneous)
        )

    evidence: dict[str, dict[str, Any]] = {}
    for label in sorted(core_labels):
        matching = [row for row in core_detections if row.get("label") == label]
        evidence[label] = {
            "detection_count": len(matching),
            "anchor_frame_count": len({row.get("frame_key") for row in matching}),
            "maximum_confidence": max(
                (float(row.get("confidence", 0.0)) for row in matching),
                default=0.0,
            ),
        }

    minimum_frames = max(1, int(config["profile_activation_minimum_anchor_frames"]))
    minimum_detections = max(1, int(config["profile_activation_minimum_detections"]))
    active_profiles = set(always_profiles | forced_profiles)
    profile_report: dict[str, dict[str, Any]] = {}
    for profile_name, profile in profiles.items():
        triggers = list(profile.get("trigger_labels", []))
        matched_triggers = [
            label
            for label in triggers
            if evidence[label]["anchor_frame_count"] >= minimum_frames
            and evidence[label]["detection_count"] >= minimum_detections
        ]
        if matched_triggers:
            active_profiles.add(profile_name)
        profile_report[profile_name] = {
            "active": profile_name in active_profiles,
            "always_active": profile_name in always_profiles,
            "forced": profile_name in forced_profiles,
            "trigger_labels": triggers,
            "matched_trigger_labels": matched_triggers,
            "labels": list(profile.get("labels", [])),
        }

    active_labels = set(core_labels)
    active_named_miscellaneous: set[str] = set()
    active_profile_aliases: set[tuple[str, str]] = set()
    for profile_name in active_profiles:
        active_labels.update(profiles[profile_name].get("labels", []))
        active_profile_aliases.update(selected_aliases_by_profile.get(profile_name, set()))
        active_named_miscellaneous.update(
            normalize_label(prompt)
            for prompt in miscellaneous_profiles.get(profile_name, [])
        )

    core_records = [
        record
        for record in records
        if record["label"] in core_labels
        and (
            record["prompt_kind"] == "canonical"
            or (
                record["prompt_kind"] == "alias"
                and (record["label"], normalize_label(record["prompt"]))
                in selected_core_aliases
            )
        )
    ]
    expanded_records = [
        record
        for record in records
        if (
            (
                record["prompt_kind"] == "generic_fallback"
                and (
                    bool(config.get("enable_miscellaneous_canonical_prompt", False))
                    or normalize_label(record["prompt"])
                    != normalize_label(config["class_prompts"]["miscellaneous"])
                )
            )
            or (
                record["label"] == "miscellaneous"
                and record["prompt_kind"] == "named_fallback"
                and normalize_label(record["prompt"]) in active_named_miscellaneous
            )
            or (
                record["label"] in active_labels
                and record["label"] not in core_labels
                and record["label"] != "miscellaneous"
                and (
                    record["prompt_kind"] == "canonical"
                    or (
                        record["prompt_kind"] == "alias"
                        and (record["label"], normalize_label(record["prompt"]))
                        in active_profile_aliases
                    )
                )
            )
        )
    ]
    active_records = core_records + expanded_records
    return {
        "selection_mode": "hierarchical_scene_profiles",
        "core_labels": sorted(core_labels),
        "core_evidence": evidence,
        "active_profiles": sorted(active_profiles),
        "profile_report": profile_report,
        "active_labels": sorted(active_labels | {"miscellaneous"}),
        "inactive_labels": sorted(class_labels - active_labels - {"miscellaneous"}),
        "active_named_miscellaneous_prompts": sorted(
            named_miscellaneous[key] for key in active_named_miscellaneous
        ),
        "active_alias_prompts": sorted(
            available_aliases[label][key]
            for label, key in selected_core_aliases | active_profile_aliases
        ),
        "library_prompt_count": len(records),
        "core_prompt_count": len(core_records),
        "expanded_prompt_count": len(expanded_records),
        "active_prompt_count": len(active_records),
        "core_prompt_records": core_records,
        "expanded_prompt_records": expanded_records,
        "active_prompt_records": active_records,
    }


def box_intersection_over_smaller(first: list[float], second: list[float]) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    smaller = min(first_area, second_area)
    return intersection / smaller if smaller > 0 else 0.0


def suppress_generic_miscellaneous_over_known(
    detections: list[dict[str, Any]],
    iou_threshold: float,
    containment_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reject only generic fallback boxes duplicated by a named-class detection."""

    known = [item for item in detections if item["label"] != "miscellaneous"]
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for item in detections:
        if item["label"] != "miscellaneous" or item.get("prompt_kind") != "generic_fallback":
            accepted.append(item)
            continue
        suppressor = next(
            (
                candidate
                for candidate in known
                if box_iou(item["box_xyxy"], candidate["box_xyxy"]) >= iou_threshold
                or box_intersection_over_smaller(
                    item["box_xyxy"], candidate["box_xyxy"]
                )
                >= containment_threshold
            ),
            None,
        )
        if suppressor is None:
            accepted.append(item)
            continue
        rejected.append(
            {
                **item,
                "accepted": False,
                "rejection_reason": "generic_miscellaneous_duplicates_known_class",
                "suppressed_by_detection_id": suppressor["detection_id"],
                "suppressed_by_label": suppressor["label"],
            }
        )
    return accepted, rejected


def suppress_semantic_conflicts(
    detections: list[dict[str, Any]],
    config: dict[str, Any],
    image_width: int,
    image_height: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reject high-impact class confusions before they become persistent SAM2 tracks."""

    def area(box: list[float]) -> float:
        return max(0.0, float(box[2]) - float(box[0])) * max(
            0.0, float(box[3]) - float(box[1])
        )

    def intersection(first: list[float], second: list[float]) -> float:
        width = max(
            0.0,
            min(float(first[2]), float(second[2]))
            - max(float(first[0]), float(second[0])),
        )
        height = max(
            0.0,
            min(float(first[3]), float(second[3]))
            - max(float(first[1]), float(second[1])),
        )
        return width * height

    windows = [item for item in detections if item["label"] == "window"]
    beds = [item for item in detections if item["label"] == "bed"]
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for item in detections:
        if item["label"] == "curtain" and bool(
            config["curtain_requires_window_context"]
        ):
            context = next(
                (
                    window
                    for window in windows
                    if box_intersection_over_smaller(
                        item["box_xyxy"], window["box_xyxy"]
                    )
                    >= float(
                        config[
                            "curtain_window_minimum_intersection_over_smaller"
                        ]
                    )
                    or box_center_distance(
                        item["box_xyxy"],
                        window["box_xyxy"],
                        image_width,
                        image_height,
                    )
                    <= float(config["curtain_window_maximum_center_distance"])
                ),
                None,
            )
            if context is None and float(item["confidence"]) < float(
                config["curtain_without_window_minimum_confidence"]
            ):
                rejected.append(
                    {
                        **item,
                        "accepted": False,
                        "rejection_reason": "curtain_without_window_context",
                        "required_confidence_without_window": float(
                            config["curtain_without_window_minimum_confidence"]
                        ),
                    }
                )
                continue

        if item["label"] == "blanket":
            blanket_area = area(item["box_xyxy"])
            suppressor = None
            for bed in beds:
                bed_area = area(bed["box_xyxy"])
                shared = intersection(item["box_xyxy"], bed["box_xyxy"])
                blanket_overlap = shared / max(1.0, blanket_area)
                area_ratio = blanket_area / max(1.0, bed_area)
                if blanket_overlap >= float(
                    config["blanket_bed_minimum_blanket_overlap"]
                ) and area_ratio >= float(config["blanket_bed_minimum_area_ratio"]):
                    suppressor = bed
                    break
            if suppressor is not None:
                rejected.append(
                    {
                        **item,
                        "accepted": False,
                        "rejection_reason": "blanket_duplicates_bed_extent",
                        "suppressed_by_detection_id": suppressor["detection_id"],
                        "suppressed_by_label": "bed",
                    }
                )
                continue

        accepted.append(item)
    return accepted, rejected


def nms_by_label(
    detections: list[dict[str, Any]], iou_threshold: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for label in sorted({item["label"] for item in detections}):
        label_items = sorted(
            [item for item in detections if item["label"] == label],
            key=lambda item: (
                {
                    "named_fallback": 3,
                    "canonical": 2,
                    "alias": 2,
                    "generic_fallback": 1,
                }.get(item.get("prompt_kind"), 0),
                float(item["confidence"]),
            ),
            reverse=True,
        )
        kept_for_label: list[dict[str, Any]] = []
        for item in label_items:
            suppressor = next(
                (
                    kept
                    for kept in kept_for_label
                    if box_iou(item["box_xyxy"], kept["box_xyxy"]) >= iou_threshold
                ),
                None,
            )
            if suppressor is None:
                kept_for_label.append(item)
                accepted.append(item)
            else:
                rejected.append(
                    {
                        **item,
                        "accepted": False,
                        "rejection_reason": "same_label_nms",
                        "suppressed_by_detection_id": suppressor["detection_id"],
                    }
                )
    return accepted, rejected


def detection_manifest_complete(manifest: dict[str, Any], paths: dict[str, Path]) -> bool:
    for value in manifest.get("anchor_json_paths", []):
        if not Path(value).exists():
            return False
    return bool(manifest.get("anchor_json_paths"))


def run_anchor_detection(config: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    import torch
    from PIL import Image
    from tqdm.auto import tqdm
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    ensure_directories(paths)
    _, frames = load_registered_frames(config, paths)
    fingerprint = stage_fingerprint(config, paths, frames, "anchor_detection")
    manifest_path = paths["tracking_dir"] / "anchor_detection_manifest.json"
    if config["resume"] and not config["force_stage"] and manifest_path.exists():
        previous = read_json(manifest_path)
        if previous.get("fingerprint") == fingerprint and detection_manifest_complete(previous, paths):
            print("Reusing Grounding DINO anchor detections:", manifest_path)
            return previous

    logger = setup_logger(paths["report_dir"] / "anchor_detection.log", "anchor_detection")
    if config["require_gpu"] and not torch.cuda.is_available():
        raise RuntimeError("A GPU runtime is required. In Colab choose Runtime > Change runtime type > GPU.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading %s on %s", config["grounding_dino_model_id"], device)
    processor = AutoProcessor.from_pretrained(config["grounding_dino_model_id"])
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        config["grounding_dino_model_id"]
    ).to(device)
    model.eval()

    thresholds = config["class_detection_thresholds"]
    stride = max(1, int(config["detection_keyframe_stride"]))
    anchor_indexes = list(range(0, len(frames), stride))
    if anchor_indexes[-1] != len(frames) - 1:
        anchor_indexes.append(len(frames) - 1)

    def detect_prompt_batch(
        frame: dict[str, Any],
        image: Any,
        prompt_records: list[dict[str, str]],
        detection_pass: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not prompt_records:
            return [], []
        width, height = image.size
        ordered_labels = [record["label"] for record in prompt_records]
        ordered_prompts = [record["prompt"] for record in prompt_records]
        ordered_prompt_kinds = [record["prompt_kind"] for record in prompt_records]
        prompt_to_label = {
            normalize_label(record["prompt"]): record["label"] for record in prompt_records
        }
        prompt_to_kind = {
            normalize_label(record["prompt"]): record["prompt_kind"]
            for record in prompt_records
        }
        prompt_to_prompt = {
            normalize_label(record["prompt"]): record["prompt"] for record in prompt_records
        }
        required_thresholds = []
        for record in prompt_records:
            threshold = float(thresholds[record["label"]])
            if record["prompt_kind"] == "generic_fallback":
                threshold = max(
                    threshold,
                    float(config["miscellaneous_generic_detection_threshold"]),
                )
            required_thresholds.append(threshold)
        global_threshold = min(required_thresholds)

        inputs = processor(images=image, text=[ordered_prompts], return_tensors="pt")
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            outputs = model(**inputs)
        processed = processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.get("input_ids"),
            threshold=global_threshold,
            text_threshold=float(config["grounding_dino_text_threshold"]),
            target_sizes=[(height, width)],
        )[0]
        label_values = processed.get("text_labels")
        if label_values is None:
            label_values = processed.get("labels", [])
        boxes = processed.get("boxes", [])
        scores = processed.get("scores", [])
        candidates: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for detection_index, (box_value, score_value, raw_label) in enumerate(
            zip(boxes, scores, label_values)
        ):
            box = [float(value) for value in box_value.detach().cpu().tolist()]
            score = float(score_value.detach().cpu().item())
            label = canonical_detector_label(raw_label, prompt_to_label, ordered_labels)
            prompt_kind = canonical_detector_label(
                raw_label, prompt_to_kind, ordered_prompt_kinds
            )
            matched_prompt = canonical_detector_label(
                raw_label, prompt_to_prompt, ordered_prompts
            )
            base = {
                "detection_id": (
                    f"{frame['frame_key']}_{detection_pass}_d{detection_index:04d}"
                ),
                "detection_pass": detection_pass,
                "frame_index": frame["frame_index"],
                "frame_key": frame["frame_key"],
                "image_path": frame["image_path"],
                "raw_label": str(raw_label),
                "label": label,
                "matched_prompt": matched_prompt,
                "prompt_kind": prompt_kind,
                "confidence": score,
                "box_xyxy": box,
                "detector_model": config["grounding_dino_model_id"],
            }
            if label is None:
                rejected.append(
                    {
                        **base,
                        "accepted": False,
                        "rejection_reason": "unknown_prompt_label",
                    }
                )
                continue
            box[0] = max(0.0, min(float(width - 1), box[0]))
            box[1] = max(0.0, min(float(height - 1), box[1]))
            box[2] = max(0.0, min(float(width - 1), box[2]))
            box[3] = max(0.0, min(float(height - 1), box[3]))
            area_fraction = (
                max(0.0, box[2] - box[0])
                * max(0.0, box[3] - box[1])
                / max(1.0, float(width * height))
            )
            base["box_xyxy"] = box
            base["box_area_fraction"] = area_fraction
            class_threshold = float(thresholds[label])
            maximum_area = float(
                config["class_maximum_box_area_fraction"].get(
                    label, config["default_maximum_box_area_fraction"]
                )
            )
            if label == "miscellaneous" and prompt_kind == "generic_fallback":
                class_threshold = max(
                    class_threshold,
                    float(config["miscellaneous_generic_detection_threshold"]),
                )
                maximum_area = min(
                    maximum_area,
                    float(config["miscellaneous_generic_maximum_box_area_fraction"]),
                )
            if score < class_threshold:
                rejected.append(
                    {
                        **base,
                        "accepted": False,
                        "rejection_reason": "below_class_threshold",
                        "required_confidence": class_threshold,
                    }
                )
            elif area_fraction < float(config["minimum_box_area_fraction"]):
                rejected.append(
                    {**base, "accepted": False, "rejection_reason": "box_too_small"}
                )
            elif area_fraction > maximum_area:
                rejected.append(
                    {**base, "accepted": False, "rejection_reason": "box_too_large"}
                )
            elif box[2] <= box[0] or box[3] <= box[1]:
                rejected.append(
                    {**base, "accepted": False, "rejection_reason": "invalid_box"}
                )
            else:
                candidates.append({**base, "accepted": True})
        return candidates, rejected

    initial_plan = build_scene_prompt_plan(config, [])
    core_prompt_records = initial_plan["core_prompt_records"]
    core_by_frame: dict[str, list[dict[str, Any]]] = {}
    core_rejected_by_frame: dict[str, list[dict[str, Any]]] = {}
    core_evidence_rows: list[dict[str, Any]] = []
    for frame_index in tqdm(anchor_indexes, desc="Grounding DINO core prompts"):
        frame = frames[frame_index]
        image = Image.open(frame["image_path"]).convert("RGB")
        core_candidates, core_rejected = detect_prompt_batch(
            frame, image, core_prompt_records, "core"
        )
        core_accepted, core_nms_rejected = nms_by_label(
            core_candidates, float(config["detector_nms_iou"])
        )
        core_by_frame[frame["frame_key"]] = core_accepted
        core_rejected_by_frame[frame["frame_key"]] = core_rejected + core_nms_rejected
        core_evidence_rows.extend(core_accepted)

    prompt_plan = build_scene_prompt_plan(config, core_evidence_rows)
    prompt_plan_path = paths["report_dir"] / "scene_prompt_plan.json"
    write_json(prompt_plan_path, prompt_plan)
    logger.info(
        "Prompt plan activated profiles=%s labels=%s prompts=%d/%d",
        prompt_plan["active_profiles"],
        prompt_plan["active_labels"],
        prompt_plan["active_prompt_count"],
        prompt_plan["library_prompt_count"],
    )
    expanded_prompt_records = prompt_plan["expanded_prompt_records"]

    accepted_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    anchor_paths: list[str] = []
    for frame_index in tqdm(anchor_indexes, desc="Grounding DINO expanded prompts"):
        frame = frames[frame_index]
        image = Image.open(frame["image_path"]).convert("RGB")
        width, height = image.size
        expanded_candidates, expanded_rejected = detect_prompt_batch(
            frame, image, expanded_prompt_records, "expanded"
        )
        candidates = core_by_frame[frame["frame_key"]] + expanded_candidates
        frame_rejected = (
            core_rejected_by_frame[frame["frame_key"]] + expanded_rejected
        )

        accepted, nms_rejected = nms_by_label(candidates, float(config["detector_nms_iou"]))
        accepted, fallback_rejected = suppress_generic_miscellaneous_over_known(
            accepted,
            float(config["miscellaneous_known_overlap_iou"]),
            float(config["miscellaneous_known_containment_fraction"]),
        )
        accepted, semantic_conflict_rejected = suppress_semantic_conflicts(
            accepted,
            config,
            width,
            height,
        )
        frame_rejected.extend(nms_rejected)
        frame_rejected.extend(fallback_rejected)
        frame_rejected.extend(semantic_conflict_rejected)
        anchor_path = paths["anchor_dir"] / f"{frame['frame_key']}.json"
        write_json(
            anchor_path,
            {
                "schema": "grounding_dino_anchor.v2",
                "pipeline_version": PIPELINE_VERSION,
                "fingerprint": fingerprint,
                "frame": frame,
                "image_size": [width, height],
                "active_profiles": prompt_plan["active_profiles"],
                "active_labels": prompt_plan["active_labels"],
                "prompt_records": prompt_plan["active_prompt_records"],
                "detections": accepted,
                "rejected_detections": frame_rejected,
            },
        )
        anchor_paths.append(str(anchor_path))
        accepted_rows.extend(accepted)
        rejected_rows.extend(frame_rejected)

    write_csv(paths["report_dir"] / "anchor_detections_accepted.csv", accepted_rows)
    write_csv(paths["report_dir"] / "anchor_detections_rejected.csv", rejected_rows)
    summary = {
        "schema": "grounding_dino_anchor_manifest.v2",
        "pipeline_version": PIPELINE_VERSION,
        "fingerprint": fingerprint,
        "created_at": utc_now(),
        "model_id": config["grounding_dino_model_id"],
        "frame_count": len(frames),
        "anchor_count": len(anchor_indexes),
        "anchor_indexes": anchor_indexes,
        "accepted_detection_count": len(accepted_rows),
        "rejected_detection_count": len(rejected_rows),
        "accepted_by_label": dict(Counter(row["label"] for row in accepted_rows)),
        "rejected_by_reason": dict(Counter(row["rejection_reason"] for row in rejected_rows)),
        "scene_prompt_plan_path": str(prompt_plan_path),
        "active_prompt_profiles": prompt_plan["active_profiles"],
        "active_object_labels": prompt_plan["active_labels"],
        "active_prompt_count": prompt_plan["active_prompt_count"],
        "library_prompt_count": prompt_plan["library_prompt_count"],
        "miscellaneous_accepted_by_prompt_kind": dict(
            Counter(
                row.get("prompt_kind", "unknown")
                for row in accepted_rows
                if row.get("label") == "miscellaneous"
            )
        ),
        "anchor_json_paths": anchor_paths,
    }
    write_json(manifest_path, summary)
    logger.info("Accepted %d anchor detections; rejected %d", len(accepted_rows), len(rejected_rows))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def associate_anchor_detections(
    config: dict[str, Any], paths: dict[str, Path]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, set[str]]]:
    manifest_path = paths["tracking_dir"] / "anchor_detection_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Anchor detections are missing. Run the detect stage first.")
    manifest = read_json(manifest_path)
    anchor_payloads = [read_json(Path(value)) for value in manifest["anchor_json_paths"]]
    anchor_payloads.sort(key=lambda item: int(item["frame"]["frame_index"]))
    stride = int(config["detection_keyframe_stride"])
    max_gap = max(1, int(round(stride * float(config["track_maximum_anchor_gap_multiplier"]))))
    temporary_tracks: list[dict[str, Any]] = []
    detector_labels_by_anchor: dict[int, set[str]] = {}

    for payload in anchor_payloads:
        frame_index = int(payload["frame"]["frame_index"])
        width, height = [int(value) for value in payload["image_size"]]
        detections = sorted(
            payload.get("detections", []),
            key=lambda item: float(item["confidence"]),
            reverse=True,
        )
        detector_labels_by_anchor[frame_index] = {item["label"] for item in detections}
        assigned_track_indexes: set[int] = set()
        for detection in detections:
            candidates: list[tuple[float, int]] = []
            for track_index, track in enumerate(temporary_tracks):
                if track_index in assigned_track_indexes or track["label"] != detection["label"]:
                    continue
                previous = track["anchors"][-1]
                gap = frame_index - int(previous["frame_index"])
                if gap <= 0 or gap > max_gap:
                    continue
                iou = box_iou(detection["box_xyxy"], previous["box_xyxy"])
                center_distance = box_center_distance(
                    detection["box_xyxy"], previous["box_xyxy"], width, height
                )
                area_ratio = box_area_ratio(detection["box_xyxy"], previous["box_xyxy"])
                passes = (
                    iou >= float(config["track_association_minimum_iou"])
                    or (
                        center_distance
                        <= float(config["track_association_maximum_center_distance"])
                        and area_ratio >= float(config["track_association_minimum_area_ratio"])
                    )
                )
                if not passes:
                    continue
                score = 0.60 * iou + 0.25 * max(0.0, 1.0 - center_distance) + 0.15 * area_ratio
                candidates.append((score, track_index))
            if candidates:
                _, selected_track_index = max(candidates)
                temporary_tracks[selected_track_index]["anchors"].append(dict(detection))
                assigned_track_indexes.add(selected_track_index)
            else:
                temporary_tracks.append(
                    {
                        "temporary_id": len(temporary_tracks) + 1,
                        "label": detection["label"],
                        "anchors": [dict(detection)],
                    }
                )
                assigned_track_indexes.add(len(temporary_tracks) - 1)

    accepted_tracks: list[dict[str, Any]] = []
    rejected_tracks: list[dict[str, Any]] = []
    for track in temporary_tracks:
        confidences = [float(anchor["confidence"]) for anchor in track["anchors"]]
        track.update(
            {
                "anchor_count": len(track["anchors"]),
                "first_anchor_frame": min(int(anchor["frame_index"]) for anchor in track["anchors"]),
                "last_anchor_frame": max(int(anchor["frame_index"]) for anchor in track["anchors"]),
                "mean_detector_confidence": sum(confidences) / len(confidences),
                "max_detector_confidence": max(confidences),
            }
        )
        persistent = len(track["anchors"]) >= int(config["track_minimum_anchor_count"])
        strong_single = max(confidences) >= float(config["track_single_anchor_strong_confidence"])
        if persistent or strong_single:
            accepted_tracks.append(track)
        else:
            rejected_tracks.append({**track, "rejection_reason": "insufficient_anchor_persistence"})

    limited_tracks: list[dict[str, Any]] = []
    for label in sorted({track["label"] for track in accepted_tracks}):
        label_tracks = [track for track in accepted_tracks if track["label"] == label]
        label_tracks.sort(
            key=lambda track: (
                int(track["anchor_count"]),
                float(track["mean_detector_confidence"]),
                int(track["last_anchor_frame"]) - int(track["first_anchor_frame"]),
            ),
            reverse=True,
        )
        maximum = int(config["maximum_instances_by_label"].get(label, 6))
        limited_tracks.extend(label_tracks[:maximum])
        for track in label_tracks[maximum:]:
            rejected_tracks.append({**track, "rejection_reason": "maximum_instances_for_label"})

    limited_tracks.sort(
        key=lambda track: (
            int(track["first_anchor_frame"]),
            track["label"],
            float(track["anchors"][0]["box_xyxy"][0]),
        )
    )
    label_counts: Counter[str] = Counter()
    for sam_object_id, track in enumerate(limited_tracks, start=1):
        label_counts[track["label"]] += 1
        track["sam_object_id"] = sam_object_id
        track["track_id"] = f"{track['label']}_{label_counts[track['label']]:03d}"
        for anchor in track["anchors"]:
            anchor["track_id"] = track["track_id"]
            anchor["sam_object_id"] = sam_object_id
    return limited_tracks, rejected_tracks, detector_labels_by_anchor


def call_with_supported_kwargs(function: Any, /, *args: Any, **kwargs: Any) -> Any:
    signature = inspect.signature(function)
    supported = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return function(*args, **supported)


def stage_video_frames(
    config: dict[str, Any], paths: dict[str, Path], frames: list[dict[str, Any]], fingerprint: str
) -> Path:
    from PIL import Image

    frame_dir = paths["local_cache"] / fingerprint[:16] / "video_frames"
    manifest_path = frame_dir.parent / "frame_cache_manifest.json"
    expected_names = [f"{index:06d}.jpg" for index in range(len(frames))]
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get("fingerprint") == fingerprint and all(
            (frame_dir / name).exists() for name in expected_names
        ):
            return frame_dir
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    expected_size: tuple[int, int] | None = None
    for index, frame in enumerate(frames):
        source = Path(frame["image_path"])
        output = frame_dir / expected_names[index]
        with Image.open(source) as image:
            rgb = image.convert("RGB")
            if expected_size is None:
                expected_size = rgb.size
            elif rgb.size != expected_size:
                raise RuntimeError(
                    f"SAM2 video frames must have one resolution. Expected {expected_size}, "
                    f"found {rgb.size} for {source}."
                )
            rgb.save(output, format="JPEG", quality=95, subsampling=0)
    write_json(
        manifest_path,
        {
            "schema": "sam2_local_frame_cache.v2",
            "fingerprint": fingerprint,
            "created_at": utc_now(),
            "frame_count": len(frames),
            "frame_size": list(expected_size or []),
            "frame_dir": str(frame_dir),
        },
    )
    return frame_dir


def mask_iou(first: Any, second: Any) -> float:
    import numpy as np

    first_bool = np.asarray(first, dtype=bool)
    second_bool = np.asarray(second, dtype=bool)
    union = np.logical_or(first_bool, second_bool).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(first_bool, second_bool).sum() / union)


def clean_mask_components(mask: Any, config: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    import cv2
    import numpy as np

    binary = np.asarray(mask, dtype=np.uint8)
    original_count = int(binary.sum())
    if original_count == 0:
        return binary.astype(bool), {
            "component_count": 0,
            "largest_component_ratio": 0.0,
            "kept_component_count": 0,
            "filled_hole_count": 0,
            "filled_hole_pixels": 0,
        }
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components = []
    for component_id in range(1, component_count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        components.append((area, component_id))
    components.sort(reverse=True)
    minimum_area = max(4, int(round(original_count * float(config["minimum_component_area_ratio"]))))
    selected = [
        component_id
        for area, component_id in components[: int(config["maximum_components_to_keep"])]
        if area >= minimum_area
    ]
    cleaned = np.isin(labels, selected) if selected else np.zeros_like(binary, dtype=bool)
    maximum_hole_area = max(0, int(config.get("maximum_hole_area_pixels", 0)))
    filled_hole_count = 0
    filled_hole_pixels = 0
    if maximum_hole_area > 0 and cleaned.any():
        inverse = np.asarray(~cleaned, dtype=np.uint8)
        hole_count, hole_labels, hole_stats, _ = cv2.connectedComponentsWithStats(
            inverse, connectivity=8
        )
        border_values = np.concatenate(
            [
                hole_labels[0, :],
                hole_labels[-1, :],
                hole_labels[:, 0],
                hole_labels[:, -1],
            ]
        )
        border_ids = set(int(value) for value in np.unique(border_values))
        fill_ids = [
            hole_id
            for hole_id in range(1, hole_count)
            if hole_id not in border_ids
            and int(hole_stats[hole_id, cv2.CC_STAT_AREA]) <= maximum_hole_area
        ]
        if fill_ids:
            fill_mask = np.isin(hole_labels, fill_ids)
            cleaned |= fill_mask
            filled_hole_count = len(fill_ids)
            filled_hole_pixels = int(fill_mask.sum())
    largest_ratio = float(components[0][0] / original_count) if components else 0.0
    return cleaned, {
        "component_count": len(components),
        "largest_component_ratio": largest_ratio,
        "kept_component_count": len(selected),
        "filled_hole_count": filled_hole_count,
        "filled_hole_pixels": filled_hole_pixels,
    }


def binary_mask_box(mask: Any) -> list[float]:
    import numpy as np

    y, x = np.nonzero(mask)
    if len(x) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(x.min()), float(y.min()), float(x.max() + 1), float(y.max() + 1)]


def mask_candidate_metrics(logits: Any, config: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    logits = np.asarray(logits, dtype=np.float32)
    threshold = float(config["sam_mask_logit_threshold"])
    raw_mask = logits > threshold
    cleaned, components = clean_mask_components(raw_mask, config)
    offset = float(config["sam_stability_offset"])
    high = logits > (threshold + offset)
    low = logits > (threshold - offset)
    stability_union = int(low.sum())
    stability = float(np.logical_and(high, low).sum() / stability_union) if stability_union else 0.0
    if cleaned.any():
        clipped = np.clip(logits[cleaned], -20.0, 20.0)
        positive_confidence = float((1.0 / (1.0 + np.exp(-clipped))).mean())
    else:
        positive_confidence = 0.0
    metrics = {
        **components,
        "stability": stability,
        "positive_logit_confidence": positive_confidence,
        "area_fraction": float(cleaned.mean()),
        "pixel_count": int(cleaned.sum()),
        "box_xyxy": binary_mask_box(cleaned),
    }
    return cleaned, metrics


def collect_sam2_direction(
    predictor: Any,
    inference_state: Any,
    direction: str,
    output_root: Path,
    frame_count: int,
    config: dict[str, Any],
    logger: logging.Logger,
    start_frame_idx: int,
) -> dict[tuple[int, int], dict[str, Any]]:
    import cv2
    import numpy as np
    from tqdm.auto import tqdm

    reverse = direction == "reverse"
    direction_dir = output_root / direction
    direction_dir.mkdir(parents=True, exist_ok=True)
    generator = call_with_supported_kwargs(
        predictor.propagate_in_video,
        inference_state,
        start_frame_idx=int(start_frame_idx),
        reverse=reverse,
    )
    candidates: dict[tuple[int, int], dict[str, Any]] = {}
    for frame_index, object_ids, mask_logits in tqdm(
        generator,
        total=frame_count,
        desc=f"SAM2 {direction}",
    ):
        frame_index = int(frame_index)
        for output_index, object_id in enumerate(object_ids):
            object_id = int(object_id)
            logits = mask_logits[output_index].detach().float().cpu().numpy().squeeze()
            mask, metrics = mask_candidate_metrics(logits, config)
            if metrics["pixel_count"] == 0:
                continue
            mask_path = direction_dir / f"{frame_index:06d}_{object_id:04d}.png"
            if not cv2.imwrite(str(mask_path), np.asarray(mask, dtype=np.uint8) * 255):
                raise RuntimeError(f"Failed to write temporary SAM2 mask: {mask_path}")
            candidates[(frame_index, object_id)] = {
                "frame_index": frame_index,
                "sam_object_id": object_id,
                "direction": direction,
                "mask_path": str(mask_path),
                **metrics,
            }
    logger.info("SAM2 %s produced %d non-empty candidates", direction, len(candidates))
    return candidates


def tracking_outputs_complete(
    manifest: dict[str, Any], fingerprint: str
) -> bool:
    if manifest.get("fingerprint") != fingerprint or manifest.get("status") != "success":
        return False
    output_json_paths = [Path(value) for value in manifest.get("output_json_paths", [])]
    if not output_json_paths or not all(path.exists() for path in output_json_paths):
        return False
    for path in output_json_paths:
        try:
            payload = read_json(path)
        except Exception:
            return False
        if payload.get("segmentation_fingerprint") != fingerprint:
            return False
        for detection in payload.get("detections", []):
            if detection.get("segmentation_role") == "tracked_object" and not Path(
                detection.get("mask_path", "")
            ).exists():
                return False
    return True


def nearest_anchor(track: dict[str, Any], frame_index: int) -> dict[str, Any]:
    return min(track["anchors"], key=lambda item: abs(int(item["frame_index"]) - frame_index))


def initial_mask_rejection_reasons(
    config: dict[str, Any], track: dict[str, Any], frame_index: int, candidate: dict[str, Any]
) -> list[str]:
    reasons: list[str] = []
    label = track["label"]
    maximum_area = float(
        config["class_maximum_mask_area_fraction"].get(
            label, config["default_maximum_mask_area_fraction"]
        )
    )
    if candidate["area_fraction"] < float(config["minimum_mask_area_fraction"]):
        reasons.append("mask_too_small")
    if candidate["area_fraction"] > maximum_area:
        reasons.append("mask_too_large")
    if candidate["stability"] < float(config["minimum_sam_stability"]):
        reasons.append("low_sam_stability")
    if candidate["quality_score"] < float(config["minimum_combined_mask_quality"]):
        reasons.append("low_combined_mask_quality")
    if candidate["largest_component_ratio"] < float(config["minimum_largest_component_ratio"]):
        reasons.append("fragmented_mask")
    bidirectional_iou = candidate.get("bidirectional_iou")
    if bidirectional_iou is not None and bidirectional_iou < float(
        config["minimum_bidirectional_iou"]
    ):
        reasons.append("low_bidirectional_agreement")
    margin = int(
        round(
            float(config["track_visibility_margin_multiplier"])
            * int(config["detection_keyframe_stride"])
        )
    )
    if frame_index < int(track["first_anchor_frame"]) - margin or frame_index > int(
        track["last_anchor_frame"]
    ) + margin:
        reasons.append("outside_anchor_visibility_window")
    exact_anchor = next(
        (anchor for anchor in track["anchors"] if int(anchor["frame_index"]) == frame_index), None
    )
    if exact_anchor is not None and box_iou(candidate["box_xyxy"], exact_anchor["box_xyxy"]) < 0.10:
        reasons.append("anchor_mask_box_mismatch")
    return reasons


def state_for_label(
    label: str,
    frame_index: int,
    visible_labels: set[str],
    anchor_indexes: list[int],
    detector_labels_by_anchor: dict[int, set[str]],
    config: dict[str, Any],
) -> dict[str, str]:
    if label in visible_labels:
        return {"state": "visible", "reason": "accepted_tracked_mask"}
    if frame_index not in detector_labels_by_anchor:
        return {"state": "unknown", "reason": "not_a_detector_anchor"}
    if label in detector_labels_by_anchor.get(frame_index, set()):
        return {"state": "unknown", "reason": "detected_but_mask_rejected"}
    if not config["confirmed_absence_requires_neighboring_anchors"]:
        return {"state": "absent", "reason": "no_anchor_detection_or_tracked_mask"}
    position = anchor_indexes.index(frame_index)
    if position == 0 or position == len(anchor_indexes) - 1:
        return {"state": "unknown", "reason": "boundary_anchor_without_two_sided_evidence"}
    previous_index = anchor_indexes[position - 1]
    next_index = anchor_indexes[position + 1]
    if (
        label not in detector_labels_by_anchor.get(previous_index, set())
        and label not in detector_labels_by_anchor.get(next_index, set())
    ):
        return {"state": "absent", "reason": "three_consecutive_anchor_misses"}
    return {"state": "unknown", "reason": "neighboring_anchor_evidence_disagrees"}


def color_for_track(track_id: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(track_id.encode("utf-8")).digest()
    return tuple(70 + int(value) % 170 for value in digest[:3])


def enforce_minimum_published_track_support(
    config: dict[str, Any],
    paths: dict[str, Path],
    output_json_paths: Iterable[str | Path],
    rejected_rows: Iterable[dict[str, Any]] | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Remove tracks that fall below minimum support after overlap arbitration."""
    payload_entries = [
        (Path(path), read_json(Path(path))) for path in output_json_paths
    ]
    minimum_frames = max(1, int(config["minimum_track_visible_frames"]))
    published_counts: Counter[str] = Counter()
    labels_by_track: dict[str, str] = {}
    for _, payload in payload_entries:
        for record in payload.get("detections", []):
            track_id = str(record.get("track_id") or "")
            if (
                record.get("segmentation_role") == "tracked_object"
                and record.get("accepted", False)
                and record.get("state") == "visible"
                and track_id
            ):
                published_counts[track_id] += 1
                labels_by_track[track_id] = normalize_label(record.get("label"))

    suppressed_counts = {
        track_id: count
        for track_id, count in published_counts.items()
        if count < minimum_frames
    }
    updated_rejected_rows = list(rejected_rows or [])
    removed_counts: Counter[str] = Counter()
    quarantine_dir = paths["rejected_dir"] / "under_supported_masks"
    mask_root = paths["mask_dir"].resolve()

    for json_path, payload in payload_entries:
        frame_index = int(payload.get("frame_index", -1))
        frame_key = str(payload.get("frame_key") or json_path.stem)
        retained_records = []
        frame_rejections = []
        removed_labels: set[str] = set()
        for record in payload.get("detections", []):
            track_id = str(record.get("track_id") or "")
            if track_id not in suppressed_counts:
                retained_records.append(record)
                continue

            removed_counts[track_id] += 1
            removed_labels.add(normalize_label(record.get("label")))
            rejection = {
                **record,
                "frame_index": frame_index,
                "frame_key": frame_key,
                "rejection_reasons": "insufficient_visible_track_frames_after_overlap",
                "published_visible_frames": suppressed_counts[track_id],
                "minimum_visible_frames": minimum_frames,
            }
            source_mask = Path(record.get("mask_path", ""))
            if source_mask.exists():
                try:
                    source_mask.resolve().relative_to(mask_root)
                except ValueError as error:
                    raise RuntimeError(
                        f"Refusing to move an under-supported mask outside {mask_root}: "
                        f"{source_mask}"
                    ) from error
                quarantine_dir.mkdir(parents=True, exist_ok=True)
                destination = quarantine_dir / source_mask.name
                if destination.exists() and destination.resolve() != source_mask.resolve():
                    destination.unlink()
                shutil.move(str(source_mask), str(destination))
                rejection["mask_path"] = str(destination)
            frame_rejections.append(rejection)
            updated_rejected_rows.append(rejection)

        if not frame_rejections:
            continue

        payload["detections"] = retained_records
        remaining_visible_labels = {
            normalize_label(record.get("label"))
            for record in retained_records
            if record.get("segmentation_role") == "tracked_object"
            and record.get("accepted", False)
            and record.get("state") == "visible"
        }
        label_states = payload.setdefault("label_states", {})
        for label in removed_labels - remaining_visible_labels:
            label_states[label] = {
                "state": "unknown",
                "reason": "track_removed_below_minimum_visible_frames_after_overlap",
            }
        write_json(json_path, payload)

        rejected_path = Path(
            payload.get("rejected_detection_report")
            or paths["rejected_dir"] / f"{frame_key}.json"
        )
        if rejected_path.exists():
            rejected_payload = read_json(rejected_path)
        else:
            rejected_payload = {
                "schema": "tracked_semantic_rejections.v2",
                "segmentation_fingerprint": payload.get("segmentation_fingerprint"),
                "image_path": payload.get("image_path"),
                "frame_index": frame_index,
                "rejected_candidates": [],
            }
        existing_rejections = [
            item
            for item in rejected_payload.get("rejected_candidates", [])
            if not (
                str(item.get("track_id") or "") in suppressed_counts
                and "insufficient_visible_track_frames_after_overlap"
                in str(item.get("rejection_reasons", ""))
            )
        ]
        rejected_payload["rejected_candidates"] = existing_rejections + frame_rejections
        write_json(rejected_path, rejected_payload)

    accepted_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    object_labels = set(config["class_prompts"])
    for json_path, payload in payload_entries:
        frame_index = int(payload.get("frame_index", -1))
        frame_key = str(payload.get("frame_key") or json_path.stem)
        for record in payload.get("detections", []):
            if (
                record.get("segmentation_role") == "tracked_object"
                and record.get("accepted", False)
                and record.get("state") == "visible"
            ):
                accepted_rows.append(
                    {"frame_index": frame_index, "frame_key": frame_key, **record}
                )
        for label, state in payload.get("label_states", {}).items():
            if label in object_labels:
                state_rows.append(
                    {
                        "frame_index": frame_index,
                        "frame_key": frame_key,
                        "label": label,
                        **state,
                    }
                )

    suppression_rows = [
        {
            "track_id": track_id,
            "label": labels_by_track.get(track_id, "unknown"),
            "published_visible_frames": visible_frames,
            "minimum_visible_frames": minimum_frames,
            "removed_mask_count": int(removed_counts[track_id]),
            "reason": "insufficient_visible_track_frames_after_overlap",
        }
        for track_id, visible_frames in sorted(suppressed_counts.items())
    ]
    write_csv(paths["report_dir"] / "tracks_suppressed_after_overlap.csv", suppression_rows)
    return accepted_rows, updated_rejected_rows, state_rows, suppression_rows


def repair_tracking_publication_support(
    config: dict[str, Any], paths: dict[str, Path]
) -> dict[str, Any]:
    """Repair completed tracking JSON without rerunning detector or SAM2 inference."""
    if not paths["tracking_manifest"].exists():
        raise FileNotFoundError("Tracking manifest is missing. Run the track stage first.")
    manifest = read_json(paths["tracking_manifest"])
    rejected_report = Path(
        manifest.get("rejected_report")
        or paths["report_dir"] / "tracked_masks_rejected.csv"
    )
    existing_rejections: list[dict[str, Any]] = []
    if rejected_report.exists():
        with rejected_report.open(newline="", encoding="utf-8") as handle:
            existing_rejections = list(csv.DictReader(handle))
    accepted_rows, rejected_rows, state_rows, suppression_rows = (
        enforce_minimum_published_track_support(
            config,
            paths,
            manifest.get("output_json_paths", []),
            existing_rejections,
        )
    )
    accepted_report = Path(
        manifest.get("accepted_report")
        or paths["report_dir"] / "tracked_masks_accepted.csv"
    )
    state_report = Path(
        manifest.get("state_report")
        or paths["report_dir"] / "frame_label_states.csv"
    )
    write_csv(accepted_report, accepted_rows)
    write_csv(rejected_report, rejected_rows)
    write_csv(state_report, state_rows)
    manifest.update(
        {
            "accepted_mask_count": len(accepted_rows),
            "rejected_mask_count": len(rejected_rows),
            "accepted_by_label": dict(Counter(row["label"] for row in accepted_rows)),
            "rejected_by_reason": dict(
                Counter(
                    reason
                    for row in rejected_rows
                    for reason in str(row.get("rejection_reasons", "unknown")).split(";")
                )
            ),
            "state_counts": dict(Counter(row["state"] for row in state_rows)),
            "published_track_count": len(
                {row["track_id"] for row in accepted_rows if row.get("track_id")}
            ),
            "suppressed_after_overlap_track_count": len(suppression_rows),
            "suppressed_after_overlap_tracks": suppression_rows,
            "publication_support_repaired_at": utc_now(),
            "publication_support_repair_version": PIPELINE_VERSION,
        }
    )
    write_json(paths["tracking_manifest"], manifest)
    return manifest


def run_sam2_tracking(config: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    import cv2
    import importlib
    import numpy as np
    import torch
    from contextlib import nullcontext

    sam2_repo = Path(config["sam2_repo"]).resolve()
    if not sam2_repo.exists():
        raise FileNotFoundError(
            f"Missing official SAM2 checkout: {sam2_repo}. Run the notebook install cell."
        )
    repo_text = str(sam2_repo)
    sys.path = [value for value in sys.path if value != repo_text]
    sys.path.insert(0, repo_text)
    loaded_sam2 = sys.modules.get("sam2")
    loaded_origin = getattr(loaded_sam2, "__file__", None)
    if loaded_sam2 is not None and (
        loaded_origin is None or sam2_repo not in Path(loaded_origin).resolve().parents
    ):
        for module_name in [
            name for name in sys.modules if name == "sam2" or name.startswith("sam2.")
        ]:
            del sys.modules[module_name]
    importlib.invalidate_caches()
    try:
        from sam2.build_sam import build_sam2_video_predictor
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            f"Could not import sam2.build_sam from the official checkout {sam2_repo}. "
            "Rerun the notebook install cell before tracking."
        ) from error

    ensure_directories(paths)
    _, frames = load_registered_frames(config, paths)
    anchor_manifest_path = paths["tracking_dir"] / "anchor_detection_manifest.json"
    if not anchor_manifest_path.exists():
        raise FileNotFoundError("Anchor detection manifest is missing. Run the detect stage first.")
    anchor_manifest = read_json(anchor_manifest_path)
    base_fingerprint = stage_fingerprint(config, paths, frames, "sam2_tracking")
    fingerprint = stable_hash(
        {"base": base_fingerprint, "anchor_fingerprint": anchor_manifest.get("fingerprint")}
    )
    if config["resume"] and not config["force_stage"] and paths["tracking_manifest"].exists():
        previous = read_json(paths["tracking_manifest"])
        if tracking_outputs_complete(previous, fingerprint):
            print("Reusing completed tracked segmentation:", paths["tracking_manifest"])
            return previous

    logger = setup_logger(paths["report_dir"] / "sam2_tracking.log", "sam2_tracking")
    if config["require_gpu"] and not torch.cuda.is_available():
        raise RuntimeError("SAM2 tracking requires a GPU runtime. Select a Colab GPU runtime.")
    checkpoint = Path(config["sam2_checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Missing SAM2 checkpoint: {checkpoint}. Run the notebook dependency cell."
        )
    tracks, rejected_tracks, detector_labels_by_anchor = associate_anchor_detections(config, paths)
    if not tracks:
        raise RuntimeError(
            "No persistent object tracks survived anchor association. Inspect "
            f"{paths['report_dir'] / 'anchor_detections_accepted.csv'} and lower only the affected "
            "class threshold after visual review."
        )
    write_json(
        paths["track_manifest"],
        {
            "schema": "semantic_instance_tracks.v2",
            "pipeline_version": PIPELINE_VERSION,
            "fingerprint": fingerprint,
            "created_at": utc_now(),
            "tracks": tracks,
            "rejected_tracks": rejected_tracks,
        },
    )
    write_csv(
        paths["report_dir"] / "instance_tracks.csv",
        [
            {
                "track_id": track["track_id"],
                "sam_object_id": track["sam_object_id"],
                "label": track["label"],
                "anchor_count": track["anchor_count"],
                "first_anchor_frame": track["first_anchor_frame"],
                "last_anchor_frame": track["last_anchor_frame"],
                "mean_detector_confidence": track["mean_detector_confidence"],
                "max_detector_confidence": track["max_detector_confidence"],
                "anchor_detection_ids": ";".join(
                    anchor["detection_id"] for anchor in track["anchors"]
                ),
            }
            for track in tracks
        ],
    )
    write_csv(
        paths["report_dir"] / "instance_tracks_rejected.csv",
        [
            {
                "temporary_id": track["temporary_id"],
                "label": track["label"],
                "anchor_count": track["anchor_count"],
                "max_detector_confidence": track["max_detector_confidence"],
                "rejection_reason": track["rejection_reason"],
            }
            for track in rejected_tracks
        ],
    )

    local_frame_dir = stage_video_frames(config, paths, frames, fingerprint)
    logger.info("Staged %d ordered video frames at %s", len(frames), local_frame_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_cuda_postprocessing = bool(config["sam2_use_cuda_postprocessing"])
    postprocessing_overrides = []
    if not use_cuda_postprocessing:
        postprocessing_overrides = [
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            "++model.fill_hole_area=0",
        ]
        logger.info(
            "Using portable SAM2 post-processing with OpenCV hole cleanup; CUDA _C is optional."
        )
    predictor = call_with_supported_kwargs(
        build_sam2_video_predictor,
        config["sam2_model_config"],
        str(checkpoint),
        device=device,
        apply_postprocessing=use_cuda_postprocessing,
        hydra_overrides_extra=postprocessing_overrides,
        vos_optimized=False,
    )
    inference_state = call_with_supported_kwargs(
        predictor.init_state,
        video_path=str(local_frame_dir),
        offload_video_to_cpu=bool(config["sam2_offload_video_to_cpu"]),
        offload_state_to_cpu=bool(config["sam2_offload_state_to_cpu"]),
        async_loading_frames=True,
    )
    predictor.reset_state(inference_state)
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        autocast_dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
        autocast_context = torch.autocast("cuda", dtype=autocast_dtype)
    else:
        autocast_context = nullcontext()

    candidate_root = paths["local_cache"] / fingerprint[:16] / "sam2_candidates"
    if candidate_root.exists():
        shutil.rmtree(candidate_root)
    candidate_root.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode(), autocast_context:
        for track in tracks:
            for anchor in track["anchors"]:
                box = np.asarray(anchor["box_xyxy"], dtype=np.float32)
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=int(anchor["frame_index"]),
                    obj_id=int(track["sam_object_id"]),
                    box=box,
                )
        forward = collect_sam2_direction(
            predictor,
            inference_state,
            "forward",
            candidate_root,
            len(frames),
            config,
            logger,
            min(int(anchor["frame_index"]) for track in tracks for anchor in track["anchors"]),
        )
        reverse = collect_sam2_direction(
            predictor,
            inference_state,
            "reverse",
            candidate_root,
            len(frames),
            config,
            logger,
            max(int(anchor["frame_index"]) for track in tracks for anchor in track["anchors"]),
        )

    tracks_by_id = {int(track["sam_object_id"]): track for track in tracks}
    candidates_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    rejected_rows: list[dict[str, Any]] = []
    all_keys = sorted(set(forward) | set(reverse))
    for frame_index, object_id in all_keys:
        track = tracks_by_id.get(object_id)
        if track is None:
            continue
        directional = [item for item in [forward.get((frame_index, object_id)), reverse.get((frame_index, object_id))] if item]
        loaded_masks = []
        for item in directional:
            mask = cv2.imread(item["mask_path"], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Could not read temporary SAM2 mask: {item['mask_path']}")
            loaded_masks.append(mask > 0)
        bidirectional_iou = mask_iou(*loaded_masks) if len(loaded_masks) == 2 else None
        selected_index = max(
            range(len(directional)),
            key=lambda index: 0.60 * float(directional[index]["stability"])
            + 0.40 * float(directional[index]["positive_logit_confidence"]),
        )
        selected = dict(directional[selected_index])
        selected["bidirectional_iou"] = bidirectional_iou
        agreement = bidirectional_iou if bidirectional_iou is not None else 0.55
        selected["quality_score"] = float(
            max(
                0.0,
                min(
                    1.0,
                    0.30 * float(track["mean_detector_confidence"])
                    + 0.25 * float(selected["stability"])
                    + 0.20 * float(selected["positive_logit_confidence"])
                    + 0.25 * float(agreement),
                ),
            )
        )
        selected.update(
            {
                "track_id": track["track_id"],
                "label": track["label"],
                "detector_confidence": float(track["mean_detector_confidence"]),
                "anchor_count": int(track["anchor_count"]),
                "nearest_anchor_distance": abs(
                    int(nearest_anchor(track, frame_index)["frame_index"]) - frame_index
                ),
            }
        )
        reasons = initial_mask_rejection_reasons(config, track, frame_index, selected)
        if reasons:
            rejected_rows.append({**selected, "rejection_reasons": ";".join(reasons)})
        else:
            candidates_by_frame[frame_index].append(selected)

    # Suppress isolated area explosions before cross-label overlap ownership is assigned.
    for track in tracks:
        track_candidates = sorted(
            [
                candidate
                for values in candidates_by_frame.values()
                for candidate in values
                if candidate["track_id"] == track["track_id"]
            ],
            key=lambda item: int(item["frame_index"]),
        )
        for first, second in zip(track_candidates, track_candidates[1:]):
            if int(second["frame_index"]) - int(first["frame_index"]) != 1:
                continue
            low = min(float(first["area_fraction"]), float(second["area_fraction"]))
            high = max(float(first["area_fraction"]), float(second["area_fraction"]))
            ratio = high / max(low, 1e-12)
            if ratio <= float(config["maximum_adjacent_area_ratio"]):
                continue
            rejected = first if float(first["quality_score"]) < float(second["quality_score"]) else second
            values = candidates_by_frame[int(rejected["frame_index"])]
            if rejected in values:
                values.remove(rejected)
                rejected_rows.append({**rejected, "rejection_reasons": "abrupt_temporal_area_jump"})

    # Tracks with too little accepted temporal support are not published.
    accepted_count_by_track = Counter(
        candidate["track_id"] for values in candidates_by_frame.values() for candidate in values
    )
    for frame_index in list(candidates_by_frame):
        retained = []
        for candidate in candidates_by_frame[frame_index]:
            if accepted_count_by_track[candidate["track_id"]] < int(config["minimum_track_visible_frames"]):
                rejected_rows.append({**candidate, "rejection_reasons": "insufficient_visible_track_frames"})
            else:
                retained.append(candidate)
        candidates_by_frame[frame_index] = retained

    accepted_rows: list[dict[str, Any]] = []
    output_json_paths: list[str] = []
    state_rows: list[dict[str, Any]] = []
    visual_frame_indexes = visual_check_frame_indexes(frames, config)
    visual_rows: list[dict[str, Any]] = []
    rejected_sample_count = 0
    anchor_indexes = sorted(detector_labels_by_anchor)
    all_object_labels = sorted(config["class_prompts"])
    for frame in frames:
        frame_index = int(frame["frame_index"])
        image = cv2.imread(frame["image_path"], cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read registered frame: {frame['image_path']}")
        height, width = image.shape[:2]
        ownership = np.zeros((height, width), dtype=bool)
        frame_records: list[dict[str, Any]] = []
        frame_rejected = [row for row in rejected_rows if int(row["frame_index"]) == frame_index]
        ordered_candidates = sorted(
            candidates_by_frame.get(frame_index, []),
            key=lambda item: float(item["quality_score"])
            * float(config["overlap_priority"].get(item["label"], 1.0)),
            reverse=True,
        )
        for candidate in ordered_candidates:
            mask = cv2.imread(candidate["mask_path"], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Could not read SAM2 candidate: {candidate['mask_path']}")
            mask = mask > 0
            original_pixels = int(mask.sum())
            owned_mask = np.logical_and(mask, ~ownership)
            remaining_ratio = float(owned_mask.sum() / max(1, original_pixels))
            if remaining_ratio < float(config["minimum_remaining_after_overlap"]):
                rejection = {
                    **candidate,
                    "remaining_after_overlap": remaining_ratio,
                    "rejection_reasons": "overlap_dominated",
                }
                frame_rejected.append(rejection)
                rejected_rows.append(rejection)
                continue
            owned_mask, component_info = clean_mask_components(owned_mask, config)
            if not owned_mask.any():
                rejection = {**candidate, "rejection_reasons": "empty_after_overlap"}
                frame_rejected.append(rejection)
                rejected_rows.append(rejection)
                continue
            ownership |= owned_mask
            mask_name = (
                f"{frame['frame_key']}_track_{int(candidate['sam_object_id']):04d}_"
                f"{candidate['label']}.png"
            )
            mask_path = paths["mask_dir"] / mask_name
            if not cv2.imwrite(str(mask_path), owned_mask.astype(np.uint8) * 255):
                raise RuntimeError(f"Failed to write accepted tracked mask: {mask_path}")
            box = binary_mask_box(owned_mask)
            area_fraction = float(owned_mask.mean())
            record = {
                "image_path": frame["image_path"],
                "mask_path": str(mask_path),
                "label": candidate["label"],
                "raw_label": candidate["label"],
                "confidence": float(candidate["quality_score"]),
                "detector_confidence": float(candidate["detector_confidence"]),
                "sam_score": float(candidate["stability"]),
                "sam_positive_logit_confidence": float(candidate["positive_logit_confidence"]),
                "bidirectional_iou": candidate.get("bidirectional_iou"),
                "mask_source": "grounding_dino_sam2_video",
                "segmentation_role": "tracked_object",
                "state": "visible",
                "accepted": True,
                "track_id": candidate["track_id"],
                "instance_id": candidate["track_id"],
                "object_id": candidate["track_id"],
                "sam_object_id": int(candidate["sam_object_id"]),
                "anchor_count": int(candidate["anchor_count"]),
                "nearest_anchor_distance": int(candidate["nearest_anchor_distance"]),
                "box_xyxy": box,
                "area_fraction": area_fraction,
                "remaining_after_overlap": remaining_ratio,
                "largest_component_ratio": float(component_info["largest_component_ratio"]),
                "segmentation_fingerprint": fingerprint,
            }
            frame_records.append(record)
            accepted_rows.append({"frame_index": frame_index, "frame_key": frame["frame_key"], **record})

        visible_labels = {record["label"] for record in frame_records}
        uncertain_labels = {
            row.get("label") for row in frame_rejected if row.get("label") in all_object_labels
        }
        label_states = {}
        for label in all_object_labels:
            if label in visible_labels:
                label_states[label] = {"state": "visible", "reason": "accepted_tracked_mask"}
            elif label in uncertain_labels:
                label_states[label] = {
                    "state": "unknown",
                    "reason": "tracked_candidate_rejected_by_quality_gate",
                }
            else:
                label_states[label] = state_for_label(
                    label,
                    frame_index,
                    visible_labels,
                    anchor_indexes,
                    detector_labels_by_anchor,
                    config,
                )
        for label, state in label_states.items():
            state_rows.append(
                {
                    "frame_index": frame_index,
                    "frame_key": frame["frame_key"],
                    "label": label,
                    **state,
                }
            )

        json_path = paths["detection_dir"] / f"{frame['frame_key']}.json"
        write_json(
            json_path,
            {
                "schema": "tracked_semantic_frame.v2",
                "pipeline_version": PIPELINE_VERSION,
                "segmentation_fingerprint": fingerprint,
                "image_path": frame["image_path"],
                "frame_index": frame_index,
                "detections": frame_records,
                "label_states": label_states,
                "rejected_detection_count": len(frame_rejected),
                "rejected_detection_report": str(
                    paths["rejected_dir"] / f"{frame['frame_key']}.json"
                ),
                "object_label_source": "grounding_dino_plus_sam2_video",
                "structural_label_source": "segformer_stage_not_run_yet",
            },
        )
        output_json_paths.append(str(json_path))
        write_json(
            paths["rejected_dir"] / f"{frame['frame_key']}.json",
            {
                "schema": "tracked_semantic_rejections.v2",
                "segmentation_fingerprint": fingerprint,
                "image_path": frame["image_path"],
                "frame_index": frame_index,
                "rejected_candidates": frame_rejected,
            },
        )

        should_write_visual = frame_index in visual_frame_indexes
        if should_write_visual:
            overlay = image.copy()
            for record in frame_records:
                mask = cv2.imread(record["mask_path"], cv2.IMREAD_GRAYSCALE) > 0
                color = color_for_track(record["track_id"])
                colored = np.zeros_like(overlay)
                colored[:, :] = color
                overlay[mask] = (0.45 * colored[mask] + 0.55 * overlay[mask]).astype(np.uint8)
                x1, y1, x2, y2 = [int(value) for value in record["box_xyxy"]]
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    overlay,
                    f"{record['track_id']} {record['confidence']:.2f}",
                    (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )
            visual_path = paths["visual_dir"] / f"{frame['frame_key']}_tracked.jpg"
            if not cv2.imwrite(str(visual_path), overlay):
                raise RuntimeError(f"Failed to write tracked visual check: {visual_path}")
            visual_rows.append(
                {
                    "frame_index": frame_index,
                    "frame_key": frame["frame_key"],
                    "visible_labels": ";".join(sorted(visible_labels)),
                    "visible_object_count": len(frame_records),
                    "visual_path": str(visual_path),
                }
            )

        for rejected in frame_rejected:
            if rejected_sample_count >= int(config["rejected_mask_sample_limit"]):
                break
            source_path = Path(rejected.get("mask_path", ""))
            if source_path.exists():
                sample_dir = paths["report_dir"] / "rejected_mask_samples"
                sample_dir.mkdir(parents=True, exist_ok=True)
                destination = sample_dir / (
                    f"{frame['frame_key']}_{rejected.get('track_id', 'unknown')}_"
                    f"{rejected_sample_count:04d}.png"
                )
                shutil.copy2(source_path, destination)
                rejected_sample_count += 1

    accepted_rows, rejected_rows, state_rows, suppressed_track_rows = (
        enforce_minimum_published_track_support(
            config, paths, output_json_paths, rejected_rows
        )
    )
    write_csv(paths["report_dir"] / "tracked_masks_accepted.csv", accepted_rows)
    write_csv(paths["report_dir"] / "tracked_masks_rejected.csv", rejected_rows)
    write_csv(paths["report_dir"] / "frame_label_states.csv", state_rows)
    write_csv(paths["report_dir"] / "tracked_visual_check_inventory.csv", visual_rows)
    summary = {
        "schema": "tracked_segmentation_manifest.v2",
        "pipeline_version": PIPELINE_VERSION,
        "fingerprint": fingerprint,
        "status": "success",
        "created_at": utc_now(),
        "sam2_cuda_postprocessing": use_cuda_postprocessing,
        "portable_hole_cleanup_max_pixels": int(config["maximum_hole_area_pixels"]),
        "frame_count": len(frames),
        "track_count": len(tracks),
        "rejected_track_count": len(rejected_tracks),
        "published_track_count": len(
            {row["track_id"] for row in accepted_rows if row.get("track_id")}
        ),
        "suppressed_after_overlap_track_count": len(suppressed_track_rows),
        "suppressed_after_overlap_tracks": suppressed_track_rows,
        "accepted_mask_count": len(accepted_rows),
        "rejected_mask_count": len(rejected_rows),
        "accepted_by_label": dict(Counter(row["label"] for row in accepted_rows)),
        "rejected_by_reason": dict(
            Counter(
                reason
                for row in rejected_rows
                for reason in str(row.get("rejection_reasons", "unknown")).split(";")
            )
        ),
        "state_counts": dict(Counter(row["state"] for row in state_rows)),
        "output_json_paths": output_json_paths,
        "track_manifest": str(paths["track_manifest"]),
        "accepted_report": str(paths["report_dir"] / "tracked_masks_accepted.csv"),
        "rejected_report": str(paths["report_dir"] / "tracked_masks_rejected.csv"),
        "state_report": str(paths["report_dir"] / "frame_label_states.csv"),
        "visual_check_count": len(visual_rows),
        "visual_check_sampling_mode": config["visual_check_sampling_mode"],
        "visual_check_inventory": str(
            paths["report_dir"] / "tracked_visual_check_inventory.csv"
        ),
    }
    write_json(paths["tracking_manifest"], summary)
    logger.info(
        "Published %d accepted masks from %d tracks; suppressed %d under-supported tracks; "
        "rejected %d candidates",
        len(accepted_rows),
        summary["published_track_count"],
        len(suppressed_track_rows),
        len(rejected_rows),
    )
    return summary


def class_ids_for_terms(id2label: dict[Any, Any], terms: list[str]) -> list[int]:
    normalized_terms = [normalize_label(term) for term in terms]
    matches = []
    for raw_id, raw_label in id2label.items():
        normalized = normalize_label(raw_label)
        parts = [normalize_label(value) for value in re.split(r"[,;/]", str(raw_label))]
        if any(
            normalized == term
            or term in parts
            or any(part == term or part.startswith(term + "_") for part in parts)
            for term in normalized_terms
        ):
            matches.append(int(raw_id))
    return sorted(set(matches))


def structural_class_map(id2label: dict[Any, Any]) -> dict[str, list[int]]:
    return {
        "wall": class_ids_for_terms(id2label, ["wall"]),
        "floor": class_ids_for_terms(id2label, ["floor", "flooring"]),
        "ceiling": class_ids_for_terms(id2label, ["ceiling"]),
    }


def clean_structural_mask(mask: Any, config: dict[str, Any]) -> Any:
    import cv2
    import numpy as np

    cleaned = np.asarray(mask, dtype=np.uint8)
    close_size = int(config["structural_close_size"])
    open_size = int(config["structural_open_size"])
    if close_size > 1:
        cleaned = cv2.morphologyEx(
            cleaned, cv2.MORPH_CLOSE, np.ones((close_size, close_size), dtype=np.uint8)
        )
    if open_size > 1:
        cleaned = cv2.morphologyEx(
            cleaned, cv2.MORPH_OPEN, np.ones((open_size, open_size), dtype=np.uint8)
        )
    return cleaned > 0


def is_structural_record(record: dict[str, Any]) -> bool:
    source = str(record.get("mask_source", ""))
    return bool(
        record.get("segmentation_role") == "semantic_room_structure"
        or source.startswith("semantic_layout_")
        or source.startswith("strict_2d_layout_")
        or source == "layout_from_floor_ceiling_object_masks"
        or source == "segformer_ade_structural"
    )


def layout_outputs_complete(manifest: dict[str, Any], fingerprint: str) -> bool:
    if manifest.get("fingerprint") != fingerprint or manifest.get("status") != "success":
        return False
    for value in manifest.get("output_json_paths", []):
        path = Path(value)
        if not path.exists():
            return False
        payload = read_json(path)
        if payload.get("layout_fingerprint") != fingerprint:
            return False
        for record in payload.get("detections", []):
            if is_structural_record(record) and not Path(record.get("mask_path", "")).exists():
                return False
    return bool(manifest.get("output_json_paths"))


def run_structural_layout(config: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as functional
    from PIL import Image
    from tqdm.auto import tqdm
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

    ensure_directories(paths)
    _, frames = load_registered_frames(config, paths)
    if not paths["tracking_manifest"].exists():
        raise FileNotFoundError("Tracked object masks are missing. Run the track stage first.")
    tracking_manifest = read_json(paths["tracking_manifest"])
    base_fingerprint = stage_fingerprint(config, paths, frames, "structural_layout")
    fingerprint = stable_hash(
        {"base": base_fingerprint, "tracking_fingerprint": tracking_manifest.get("fingerprint")}
    )
    manifest_path = paths["layout_report_dir"] / "layout_manifest.json"
    if config["resume"] and not config["force_stage"] and manifest_path.exists():
        previous = read_json(manifest_path)
        if layout_outputs_complete(previous, fingerprint):
            print("Reusing completed structural layout:", manifest_path)
            return previous

    logger = setup_logger(paths["layout_report_dir"] / "layout_masks.log", "layout_masks_v2")
    if config["require_gpu"] and not torch.cuda.is_available():
        raise RuntimeError("Structural segmentation requires a GPU runtime for this pipeline.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading structural model %s on %s", config["structural_model_id"], device)
    processor = AutoImageProcessor.from_pretrained(config["structural_model_id"])
    model = SegformerForSemanticSegmentation.from_pretrained(
        config["structural_model_id"]
    ).to(device)
    model.eval()
    class_map = structural_class_map(model.config.id2label)
    missing_classes = [label for label in config["structural_labels"] if not class_map.get(label)]
    if missing_classes:
        raise RuntimeError(
            f"Structural model {config['structural_model_id']} has no class IDs for: {missing_classes}"
        )

    report_rows: list[dict[str, Any]] = []
    output_json_paths: list[str] = []
    visual_frame_indexes = visual_check_frame_indexes(frames, config)
    visual_rows: list[dict[str, Any]] = []
    for frame in tqdm(frames, desc="SegFormer structure"):
        frame_index = int(frame["frame_index"])
        json_path = paths["detection_dir"] / f"{frame['frame_key']}.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Tracked frame JSON is missing: {json_path}")
        payload = read_json(json_path)
        if payload.get("segmentation_fingerprint") != tracking_manifest.get("fingerprint"):
            raise RuntimeError(
                f"Tracked frame fingerprint mismatch in {json_path}. Rerun the track stage."
            )
        image = Image.open(frame["image_path"]).convert("RGB")
        image_rgb = np.asarray(image)
        height, width = image_rgb.shape[:2]
        inputs = processor(images=image, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
            logits = functional.interpolate(
                outputs.logits,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            probabilities = torch.softmax(logits, dim=1)[0].float().cpu().numpy()
        prediction = probabilities.argmax(axis=0)

        object_cutout = np.zeros((height, width), dtype=bool)
        object_records = []
        for record in payload.get("detections", []):
            if record.get("segmentation_role") != "tracked_object" or not record.get("accepted", False):
                continue
            object_records.append(record)
            if float(record.get("confidence", 0.0)) < float(
                config["structural_cutout_minimum_object_confidence"]
            ):
                continue
            mask_path = Path(record.get("mask_path", ""))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Could not read accepted object mask: {mask_path}")
            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            object_cutout |= mask > 0

        preliminary_masks: dict[str, Any] = {}
        label_probability: dict[str, Any] = {}
        for label in config["structural_labels"]:
            class_ids = class_map[label]
            class_probability = probabilities[class_ids].max(axis=0)
            label_probability[label] = class_probability
            minimum_confidence = float(config["structural_minimum_pixel_confidence"][label])
            raw_mask = np.isin(prediction, class_ids) & (class_probability >= minimum_confidence)
            preliminary_masks[label] = clean_structural_mask(raw_mask, config)

        # Morphology can make structural masks overlap. Resolve every contested pixel using the
        # model's class probability, then remove only high-confidence accepted object masks.
        score_stack = np.stack(
            [
                np.where(preliminary_masks[label], label_probability[label], -1.0)
                for label in config["structural_labels"]
            ],
            axis=0,
        )
        winning_label = score_stack.argmax(axis=0)
        any_structure = score_stack.max(axis=0) >= 0.0
        structural_masks = {}
        for label_index, label in enumerate(config["structural_labels"]):
            mask = any_structure & (winning_label == label_index)
            mask &= ~object_cutout
            structural_masks[label] = mask

        retained_records = [
            record for record in payload.get("detections", []) if not is_structural_record(record)
        ]
        layout_records = []
        layout_states = {}
        for label in config["structural_labels"]:
            mask = structural_masks[label]
            area_fraction = float(mask.mean())
            minimum_area = float(config["structural_minimum_area_fraction"][label])
            if area_fraction < minimum_area:
                layout_states[label] = {
                    "state": "unknown",
                    "reason": "structural_mask_below_minimum_area",
                }
                report_rows.append(
                    {
                        "frame_index": frame["frame_index"],
                        "frame_key": frame["frame_key"],
                        "label": label,
                        "state": "rejected",
                        "area_fraction": area_fraction,
                        "minimum_area_fraction": minimum_area,
                        "mean_pixel_confidence": 0.0,
                        "reason": "below_minimum_area",
                    }
                )
                continue
            confidence = float(label_probability[label][mask].mean()) if mask.any() else 0.0
            layout_mask_path = paths["layout_mask_dir"] / f"{frame['frame_key']}_{label}_layout.png"
            canonical_mask_path = paths["mask_dir"] / f"{frame['frame_key']}_{label}_layout.png"
            encoded = mask.astype(np.uint8) * 255
            if not cv2.imwrite(str(layout_mask_path), encoded):
                raise RuntimeError(f"Failed to write layout mask: {layout_mask_path}")
            if not cv2.imwrite(str(canonical_mask_path), encoded):
                raise RuntimeError(f"Failed to write canonical layout mask: {canonical_mask_path}")
            record = {
                "image_path": frame["image_path"],
                "mask_path": str(canonical_mask_path),
                "layout_mask_path": str(layout_mask_path),
                "label": label,
                "raw_label": label,
                "confidence": confidence,
                "sam_score": None,
                "mask_source": "segformer_ade_structural",
                "segmentation_role": "semantic_room_structure",
                "state": "visible",
                "accepted": True,
                "box_xyxy": binary_mask_box(mask),
                "area_fraction": area_fraction,
                "semantic_class_ids": class_map[label],
                "minimum_pixel_confidence": float(
                    config["structural_minimum_pixel_confidence"][label]
                ),
                "layout_fingerprint": fingerprint,
                "segmentation_fingerprint": tracking_manifest["fingerprint"],
            }
            layout_records.append(record)
            layout_states[label] = {"state": "visible", "reason": "accepted_structural_mask"}
            report_rows.append(
                {
                    "frame_index": frame["frame_index"],
                    "frame_key": frame["frame_key"],
                    "label": label,
                    "state": "accepted",
                    "area_fraction": area_fraction,
                    "minimum_area_fraction": minimum_area,
                    "mean_pixel_confidence": confidence,
                    "reason": "",
                    "mask_path": str(canonical_mask_path),
                }
            )

        payload["detections"] = retained_records + layout_records
        payload.setdefault("label_states", {}).update(layout_states)
        payload["layout_fingerprint"] = fingerprint
        payload["structural_label_source"] = "segformer_ade_structural_only"
        payload["structural_model_id"] = config["structural_model_id"]
        payload["structural_context_labels_disabled"] = ["door", "curtain"]
        write_json(json_path, payload)
        output_json_paths.append(str(json_path))
        write_json(
            paths["layout_label_dir"] / f"{frame['frame_key']}.json",
            {
                "schema": "structural_layout_frame.v2",
                "pipeline_version": PIPELINE_VERSION,
                "layout_fingerprint": fingerprint,
                "image_path": frame["image_path"],
                "model_id": config["structural_model_id"],
                "layout_masks": layout_records,
                "label_states": layout_states,
                "object_cutout_record_count": len(object_records),
            },
        )

        should_write_visual = frame_index in visual_frame_indexes
        if should_write_visual:
            overlay = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            colors = {"wall": (0, 128, 255), "floor": (90, 220, 0), "ceiling": (255, 140, 80)}
            for label, mask in structural_masks.items():
                if not mask.any():
                    continue
                colored = np.zeros_like(overlay)
                colored[:, :] = colors[label]
                overlay[mask] = (0.42 * colored[mask] + 0.58 * overlay[mask]).astype(np.uint8)
            visual_path = (
                paths["layout_visual_dir"] / f"{frame['frame_key']}_structural.jpg"
            )
            if not cv2.imwrite(str(visual_path), overlay):
                raise RuntimeError(f"Failed to write structural visual check: {visual_path}")
            visual_rows.append(
                {
                    "frame_index": frame_index,
                    "frame_key": frame["frame_key"],
                    "accepted_structural_labels": ";".join(
                        sorted(label for label, mask in structural_masks.items() if mask.any())
                    ),
                    "visual_path": str(visual_path),
                }
            )

    write_csv(paths["layout_report_dir"] / "structural_mask_report.csv", report_rows)
    write_csv(
        paths["layout_report_dir"] / "structural_visual_check_inventory.csv",
        visual_rows,
    )
    summary = {
        "schema": "structural_layout_manifest.v2",
        "pipeline_version": PIPELINE_VERSION,
        "fingerprint": fingerprint,
        "tracking_fingerprint": tracking_manifest["fingerprint"],
        "status": "success",
        "created_at": utc_now(),
        "model_id": config["structural_model_id"],
        "class_ids": class_map,
        "frame_count": len(frames),
        "accepted_mask_count": sum(row["state"] == "accepted" for row in report_rows),
        "rejected_mask_count": sum(row["state"] == "rejected" for row in report_rows),
        "accepted_by_label": dict(
            Counter(row["label"] for row in report_rows if row["state"] == "accepted")
        ),
        "output_json_paths": output_json_paths,
        "report_csv": str(paths["layout_report_dir"] / "structural_mask_report.csv"),
        "visual_check_count": len(visual_rows),
        "visual_check_sampling_mode": config["visual_check_sampling_mode"],
        "visual_check_inventory": str(
            paths["layout_report_dir"] / "structural_visual_check_inventory.csv"
        ),
        "authoritative_labels": list(config["structural_labels"]),
        "disabled_context_outputs": ["door", "curtain"],
    }
    write_json(manifest_path, summary)
    logger.info(
        "Published %d structural masks; rejected %d low-coverage masks",
        summary["accepted_mask_count"],
        summary["rejected_mask_count"],
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def summarize_numbers(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(float(value) for value in values)
    count = len(ordered)

    def percentile(fraction: float) -> float:
        position = (count - 1) * fraction
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "count": count,
        "min": ordered[0],
        "p25": percentile(0.25),
        "median": percentile(0.50),
        "p75": percentile(0.75),
        "max": ordered[-1],
        "mean": sum(ordered) / count,
    }


def run_audit(config: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    import cv2
    import numpy as np

    ensure_directories(paths)
    _, frames = load_registered_frames(config, paths)
    if not paths["tracking_manifest"].exists():
        raise FileNotFoundError("Tracking manifest is missing. Run the track stage.")
    layout_manifest_path = paths["layout_report_dir"] / "layout_manifest.json"
    if not layout_manifest_path.exists():
        raise FileNotFoundError("Layout manifest is missing. Run the layout stage.")
    tracking_manifest = read_json(paths["tracking_manifest"])
    layout_manifest = read_json(layout_manifest_path)
    logger = setup_logger(paths["report_dir"] / "segmentation_audit.log", "segmentation_audit")
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    confidence_by_label: dict[str, list[float]] = defaultdict(list)
    area_by_label: dict[str, list[float]] = defaultdict(list)
    track_frames: dict[str, set[int]] = defaultdict(set)
    maximum_overlap_fraction = 0.0
    canonical_mask_paths: set[Path] = set()
    layout_mask_paths: set[Path] = set()

    for frame in frames:
        json_path = paths["detection_dir"] / f"{frame['frame_key']}.json"
        if not json_path.exists():
            errors.append({"frame_key": frame["frame_key"], "error": "missing_detection_json"})
            continue
        payload = read_json(json_path)
        if payload.get("segmentation_fingerprint") != tracking_manifest.get("fingerprint"):
            errors.append({"frame_key": frame["frame_key"], "error": "tracking_fingerprint_mismatch"})
        if payload.get("layout_fingerprint") != layout_manifest.get("fingerprint"):
            errors.append({"frame_key": frame["frame_key"], "error": "layout_fingerprint_mismatch"})

        image = cv2.imread(frame["image_path"], cv2.IMREAD_GRAYSCALE)
        if image is None:
            errors.append({"frame_key": frame["frame_key"], "error": "unreadable_source_image"})
            continue
        height, width = image.shape[:2]
        owner_count = np.zeros((height, width), dtype=np.uint16)
        seen_tracks: set[str] = set()
        for record in payload.get("detections", []):
            label = normalize_label(record.get("label"))
            role = record.get("segmentation_role", "unknown")
            source = record.get("mask_source", "unknown")
            mask_path = Path(record.get("mask_path", ""))
            if not record.get("accepted", False) or record.get("state") != "visible":
                errors.append(
                    {
                        "frame_key": frame["frame_key"],
                        "label": label,
                        "error": "nonaccepted_record_in_canonical_detections",
                    }
                )
                continue
            if role == "tracked_object" and source != "grounding_dino_sam2_video":
                errors.append(
                    {"frame_key": frame["frame_key"], "label": label, "error": "object_source_violation"}
                )
            if role == "semantic_room_structure" and source != "segformer_ade_structural":
                errors.append(
                    {
                        "frame_key": frame["frame_key"],
                        "label": label,
                        "error": "structural_source_violation",
                    }
                )
            if label in set(config["structural_labels"]) and role != "semantic_room_structure":
                errors.append(
                    {"frame_key": frame["frame_key"], "label": label, "error": "structural_role_violation"}
                )
            if label not in set(config["structural_labels"]) and role != "tracked_object":
                errors.append(
                    {"frame_key": frame["frame_key"], "label": label, "error": "object_role_violation"}
                )
            track_id = record.get("track_id")
            if track_id:
                if track_id in seen_tracks:
                    errors.append(
                        {"frame_key": frame["frame_key"], "track_id": track_id, "error": "duplicate_track"}
                    )
                seen_tracks.add(track_id)
                track_frames[track_id].add(int(frame["frame_index"]))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                errors.append(
                    {
                        "frame_key": frame["frame_key"],
                        "label": label,
                        "error": "missing_or_unreadable_mask",
                        "mask_path": str(mask_path),
                    }
                )
                continue
            if mask.shape[:2] != (height, width):
                errors.append(
                    {"frame_key": frame["frame_key"], "label": label, "error": "mask_shape_mismatch"}
                )
                continue
            canonical_mask_paths.add(mask_path)
            layout_mask_path = record.get("layout_mask_path")
            if layout_mask_path:
                layout_mask_paths.add(Path(layout_mask_path))
            binary = mask > 0
            owner_count += binary.astype(np.uint16)
            confidence = float(record.get("confidence", 0.0))
            area_fraction = float(binary.mean())
            label_counts[label] += 1
            source_counts[source] += 1
            role_counts[role] += 1
            confidence_by_label[label].append(confidence)
            area_by_label[label].append(area_fraction)
            inventory_rows.append(
                {
                    "frame_index": frame["frame_index"],
                    "frame_key": frame["frame_key"],
                    "label": label,
                    "track_id": track_id,
                    "role": role,
                    "source": source,
                    "confidence": confidence,
                    "area_fraction": area_fraction,
                    "pixel_count": int(binary.sum()),
                    "mask_path": str(mask_path),
                }
            )
        overlap_fraction = float((owner_count > 1).mean())
        maximum_overlap_fraction = max(maximum_overlap_fraction, overlap_fraction)
        if overlap_fraction > 0:
            errors.append(
                {
                    "frame_key": frame["frame_key"],
                    "error": "canonical_masks_overlap",
                    "overlap_fraction": overlap_fraction,
                }
            )
        for label, state in payload.get("label_states", {}).items():
            state_rows.append(
                {
                    "frame_index": frame["frame_index"],
                    "frame_key": frame["frame_key"],
                    "label": label,
                    "state": state.get("state"),
                    "reason": state.get("reason"),
                }
            )

    for label in config["class_prompts"]:
        if label_counts[label] == 0:
            warnings.append({"label": label, "warning": "no_accepted_masks"})
    for track_id, visible_frames in track_frames.items():
        if len(visible_frames) < int(config["minimum_track_visible_frames"]):
            errors.append(
                {
                    "track_id": track_id,
                    "error": "published_track_below_minimum_visible_frames",
                    "visible_frames": len(visible_frames),
                }
            )

    write_csv(paths["report_dir"] / "accepted_mask_inventory.csv", inventory_rows)
    write_csv(paths["report_dir"] / "audited_frame_label_states.csv", state_rows)
    write_csv(paths["report_dir"] / "audit_errors.csv", errors)
    write_csv(paths["report_dir"] / "audit_warnings.csv", warnings)
    cleanup_summary = {
        "enabled": False,
        "removed_file_count": 0,
        "removed_by_area": {},
        "report": None,
    }
    if not errors and bool(config.get("prune_stale_outputs_after_audit", True)):
        cleanup_summary = prune_stale_generated_outputs(
            config,
            paths,
            frames,
            canonical_mask_paths,
            layout_mask_paths,
        )
    label_summary = {
        label: {
            "mask_count": int(label_counts[label]),
            "confidence": summarize_numbers(confidence_by_label[label]),
            "area_fraction": summarize_numbers(area_by_label[label]),
        }
        for label in sorted(label_counts)
    }
    summary = {
        "schema": "tracked_segmentation_audit.v2",
        "pipeline_version": PIPELINE_VERSION,
        "created_at": utc_now(),
        "status": "success" if not errors else "failed",
        "frame_count": len(frames),
        "accepted_mask_count": len(inventory_rows),
        "label_counts": dict(label_counts),
        "source_counts": dict(source_counts),
        "role_counts": dict(role_counts),
        "state_counts": dict(Counter(row["state"] for row in state_rows)),
        "track_visible_frame_counts": {key: len(value) for key, value in track_frames.items()},
        "maximum_overlap_fraction": maximum_overlap_fraction,
        "label_summary": label_summary,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors_report": str(paths["report_dir"] / "audit_errors.csv"),
        "warnings_report": str(paths["report_dir"] / "audit_warnings.csv"),
        "mask_inventory": str(paths["report_dir"] / "accepted_mask_inventory.csv"),
        "stale_output_cleanup": cleanup_summary,
    }
    write_json(paths["report_dir"] / "segmentation_audit_summary.json", summary)
    logger.info(
        "Audit complete: %d accepted masks, %d errors, %d warnings",
        len(inventory_rows),
        len(errors),
        len(warnings),
    )
    if errors:
        print("Audit errors:")
        for error in errors:
            print("  " + json.dumps(error, sort_keys=True))
            logger.error("Audit error: %s", json.dumps(error, sort_keys=True))
        raise RuntimeError(
            f"Segmentation audit found {len(errors)} errors. See "
            f"{paths['report_dir'] / 'audit_errors.csv'}"
        )
    return summary


def run_sam3_pilot(config: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    """Run a gated short-video SAM3 comparison without changing canonical masks."""
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from transformers import Sam3VideoModel, Sam3VideoProcessor

    if not config.get("enable_sam3_pilot", False):
        result = {
            "schema": "sam3_video_pilot.v1",
            "status": "skipped",
            "reason": "enable_sam3_pilot_is_false",
        }
        print(json.dumps(result, indent=2))
        return result
    if not torch.cuda.is_available():
        raise RuntimeError("The SAM3 video pilot requires a GPU runtime.")
    if not paths["tracking_manifest"].exists():
        raise FileNotFoundError("Run the primary SAM2 tracking stage before the SAM3 pilot.")

    _, frames = load_registered_frames(config, paths)
    start = max(0, int(config["sam3_pilot_start_frame"]))
    stop = min(len(frames), start + max(1, int(config["sam3_pilot_frame_count"])))
    pilot_frames = frames[start:stop]
    if not pilot_frames:
        raise RuntimeError(f"SAM3 pilot frame range [{start}, {stop}) is empty.")
    pilot_labels = [
        normalize_label(label)
        for label in config["sam3_pilot_labels"]
        if normalize_label(label) in config["class_prompts"]
    ]
    if not pilot_labels:
        raise RuntimeError("sam3_pilot_labels contains no configured object labels.")
    prompts = [config["class_prompts"][label] for label in pilot_labels]
    prompt_to_label = {
        normalize_label(config["class_prompts"][label]): label for label in pilot_labels
    }
    tracking_manifest = read_json(paths["tracking_manifest"])
    fingerprint = stable_hash(
        {
            "pipeline_version": PIPELINE_VERSION,
            "model_id": config["sam3_model_id"],
            "tracking_fingerprint": tracking_manifest.get("fingerprint"),
            "labels": pilot_labels,
            "frame_keys": [frame["frame_key"] for frame in pilot_frames],
            "score_threshold": config["sam3_pilot_score_threshold"],
        }
    )
    report_dir = paths["report_dir"] / "sam3_pilot"
    output_dir = paths["tracking_dir"] / "sam3_pilot" / fingerprint[:16]
    summary_path = report_dir / "summary.json"
    if config["resume"] and not config["force_stage"] and summary_path.exists():
        previous = read_json(summary_path)
        if previous.get("fingerprint") == fingerprint and previous.get("status") == "success":
            print("Reusing SAM3 pilot:", summary_path)
            return previous
    report_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(report_dir / "sam3_pilot.log", "sam3_pilot")

    capability = torch.cuda.get_device_capability()
    dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
    logger.info(
        "Loading gated SAM3 video model %s with dtype %s", config["sam3_model_id"], dtype
    )
    try:
        model = Sam3VideoModel.from_pretrained(
            config["sam3_model_id"],
            dtype=dtype,
            device_map="auto",
        )
        processor = Sam3VideoProcessor.from_pretrained(config["sam3_model_id"])
    except Exception as error:
        raise RuntimeError(
            "The optional SAM3 pilot could not load facebook/sam3. Accept the model license "
            "on Hugging Face, run huggingface_hub.notebook_login() in Colab, and retry. "
            f"Original error: {error}"
        ) from error

    video_frames = [Image.open(frame["image_path"]).convert("RGB") for frame in pilot_frames]
    inference_session = processor.init_video_session(
        video=video_frames,
        inference_device="cuda",
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=dtype,
    )
    inference_session = processor.add_text_prompt(
        inference_session=inference_session,
        text=prompts,
    )

    rows: list[dict[str, Any]] = []
    frame_instance_counts: Counter[int] = Counter()
    visual_limit = min(12, len(pilot_frames))
    with torch.inference_mode():
        iterator = model.propagate_in_video_iterator(
            inference_session=inference_session,
            max_frame_num_to_track=max(0, len(pilot_frames) - 1),
            show_progress_bar=True,
        )
        for model_outputs in iterator:
            local_index = int(model_outputs.frame_idx)
            if local_index >= len(pilot_frames):
                continue
            frame = pilot_frames[local_index]
            processed = processor.postprocess_outputs(inference_session, model_outputs)
            object_ids = processed.get("object_ids", [])
            scores = processed.get("scores", [])
            boxes = processed.get("boxes", [])
            masks = processed.get("masks", [])
            object_to_prompts: dict[int, list[str]] = defaultdict(list)
            for prompt, prompt_object_ids in processed.get("prompt_to_obj_ids", {}).items():
                for object_id in prompt_object_ids:
                    object_to_prompts[int(object_id)].append(str(prompt))

            primary_payload_path = paths["detection_dir"] / f"{frame['frame_key']}.json"
            primary_payload = read_json(primary_payload_path)
            primary_masks_by_label: dict[str, list[Any]] = defaultdict(list)
            for record in primary_payload.get("detections", []):
                if record.get("segmentation_role") != "tracked_object":
                    continue
                mask = cv2.imread(record["mask_path"], cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    primary_masks_by_label[record["label"]].append(mask > 0)

            frame_visual = cv2.imread(frame["image_path"], cv2.IMREAD_COLOR)
            for output_index, (object_id_value, score_value, box_value, mask_value) in enumerate(
                zip(object_ids, scores, boxes, masks)
            ):
                object_id = int(object_id_value)
                score = float(score_value.detach().float().cpu().item())
                if score < float(config["sam3_pilot_score_threshold"]):
                    continue
                raw_prompts = object_to_prompts.get(object_id, [])
                canonical_candidates = [
                    prompt_to_label.get(normalize_label(prompt)) for prompt in raw_prompts
                ]
                canonical_candidates = [label for label in canonical_candidates if label]
                label = canonical_candidates[0] if canonical_candidates else "unknown"
                mask = mask_value.detach().cpu().numpy().astype(bool)
                if mask.ndim > 2:
                    mask = np.squeeze(mask)
                if not mask.any():
                    continue
                box = [float(value) for value in box_value.detach().float().cpu().tolist()]
                agreement_values = [
                    mask_iou(mask, primary_mask) for primary_mask in primary_masks_by_label.get(label, [])
                ]
                agreement = max(agreement_values) if agreement_values else None
                mask_path = output_dir / (
                    f"{frame['frame_key']}_{label}_{object_id:04d}_{output_index:03d}.png"
                )
                if not cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255):
                    raise RuntimeError(f"Failed to write SAM3 pilot mask: {mask_path}")
                rows.append(
                    {
                        "global_frame_index": frame["frame_index"],
                        "pilot_frame_index": local_index,
                        "frame_key": frame["frame_key"],
                        "object_id": object_id,
                        "label": label,
                        "raw_prompts": ";".join(raw_prompts),
                        "score": score,
                        "area_fraction": float(mask.mean()),
                        "box_xyxy": box,
                        "primary_same_label_max_iou": agreement,
                        "mask_path": str(mask_path),
                    }
                )
                frame_instance_counts[local_index] += 1
                if local_index < visual_limit and frame_visual is not None:
                    color = color_for_track(f"sam3_{label}_{object_id}")
                    colored = np.zeros_like(frame_visual)
                    colored[:, :] = color
                    frame_visual[mask] = (
                        0.45 * colored[mask] + 0.55 * frame_visual[mask]
                    ).astype(np.uint8)
                    x1, y1, x2, y2 = [int(value) for value in box]
                    cv2.rectangle(frame_visual, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame_visual,
                        f"SAM3 {label} {score:.2f}",
                        (x1, max(20, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                    )
            if local_index < visual_limit and frame_visual is not None:
                cv2.imwrite(str(report_dir / f"{frame['frame_key']}_sam3.jpg"), frame_visual)

    write_csv(report_dir / "sam3_mask_comparison.csv", rows)
    agreement_values = [
        float(row["primary_same_label_max_iou"])
        for row in rows
        if row["primary_same_label_max_iou"] is not None
    ]
    summary = {
        "schema": "sam3_video_pilot.v1",
        "pipeline_version": PIPELINE_VERSION,
        "fingerprint": fingerprint,
        "status": "success",
        "created_at": utc_now(),
        "model_id": config["sam3_model_id"],
        "frame_start": start,
        "frame_stop_exclusive": stop,
        "frame_count": len(pilot_frames),
        "labels": pilot_labels,
        "mask_count": len(rows),
        "mask_count_by_label": dict(Counter(row["label"] for row in rows)),
        "same_label_iou_with_primary": summarize_numbers(agreement_values),
        "comparison_report": str(report_dir / "sam3_mask_comparison.csv"),
        "mask_output_dir": str(output_dir),
        "note": "Agreement with the primary pipeline is not ground truth; visually audit both outputs.",
    }
    write_json(summary_path, summary)
    logger.info("SAM3 pilot wrote %d masks across %d frames", len(rows), len(pilot_frames))
    return summary


def run_stage(config: dict[str, Any], stage: str) -> dict[str, Any]:
    paths = project_paths(config)
    ensure_directories(paths)
    if stage == "validate":
        return validate_environment(config, paths)
    if stage == "detect":
        return run_anchor_detection(config, paths)
    if stage == "track":
        return run_sam2_tracking(config, paths)
    if stage == "layout":
        return run_structural_layout(config, paths)
    if stage == "audit":
        return run_audit(config, paths)
    if stage == "sam3_pilot":
        return run_sam3_pilot(config, paths)
    if stage == "all":
        results = {"validate": validate_environment(config, paths)}
        results["detect"] = run_anchor_detection(config, paths)
        results["track"] = run_sam2_tracking(config, paths)
        results["layout"] = run_structural_layout(config, paths)
        results["audit"] = run_audit(config, paths)
        return results
    raise ValueError(f"Unknown stage: {stage}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=["validate", "detect", "track", "layout", "audit", "sam3_pilot", "all"],
        required=True,
    )
    arguments = parser.parse_args(argv)
    config = load_config(arguments.config)
    paths = project_paths(config)
    try:
        result = run_stage(config, arguments.stage)
        print(json.dumps(result, indent=2))
        return 0
    except Exception as error:
        record_fatal(paths, arguments.stage, error)
        print(f"FAILED [{arguments.stage}]: {error}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
