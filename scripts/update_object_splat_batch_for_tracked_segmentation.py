"""Apply the tracked-segmentation v7 contract to the batch notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "object_splat_batch_pipeline.ipynb"


def replace_once(source: str, old: str, new: str, description: str) -> str:
    if new in source:
        return source
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {description} match, found {count}")
    return source.replace(old, new, 1)


notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

config_cell = "".join(notebook["cells"][4]["source"])
config_cell = replace_once(
    config_cell,
    "BATCH_PIPELINE_VERSION = '2026-07-30-object-splat-room-reference-cleanup-v6.1'",
    "BATCH_PIPELINE_VERSION = '2026-08-02-object-splat-tracked-segmentation-v7'",
    "pipeline version",
)
config_cell = replace_once(
    config_cell,
    "MASK_DIRS = [\n    SCAN_DIR / 'semantic_recon' / 'segmentation' / 'masks',\n    SCAN_DIR / 'semantic_recon' / 'layout' / 'masks',\n]\n",
    "MASK_DIRS = [\n    SCAN_DIR / 'semantic_recon' / 'segmentation' / 'masks',\n    SCAN_DIR / 'semantic_recon' / 'layout' / 'masks',\n]\n\n"
    "# The v7 batch trusts only accepted records published by tracked segmentation v2.\n"
    "# Filename fallback is disabled because old/orphaned PNG files can survive a rerun.\n"
    "REQUIRE_TRACKED_SEGMENTATION_V2 = True\n"
    "ALLOW_FILENAME_MASK_FALLBACK = False\n",
    "mask source policy",
)
config_cell = replace_once(
    config_cell,
    "BATCH_ID = 'production_object_splats_v6_room_reference_cleanup'",
    "BATCH_ID = 'production_object_splats_v7_tracked_segmentation'",
    "batch id",
)
config_cell = replace_once(
    config_cell,
    "MASK_EROSION_PIXELS = 1",
    "MASK_EROSION_PIXELS = 0",
    "mask erosion",
)
config_cell = replace_once(
    config_cell,
    "MIN_MASK_AREA_FRACTION = 0.001",
    "MIN_MASK_AREA_FRACTION = 0.0001",
    "minimum training mask area",
)
config_cell = replace_once(
    config_cell,
    "REUSE_RAW_EXPORTS_FROM_BATCH_ID = 'production_object_splats_v3'",
    "REUSE_RAW_EXPORTS_FROM_BATCH_ID = None",
    "raw export reuse",
)
config_cell = replace_once(
    config_cell,
    "RUN_POST_EXPORT_CLEANUP = True\nPREFER_CLEANED_SPLAT_IN_MANIFEST = True",
    "# First evaluate what the improved masks train without a post-export confound.\n"
    "RUN_POST_EXPORT_CLEANUP = False\n"
    "PREFER_CLEANED_SPLAT_IN_MANIFEST = False",
    "cleanup defaults",
)
config_cell = replace_once(
    config_cell,
    "ROOM_REFERENCE_REQUIRED = True",
    "ROOM_REFERENCE_REQUIRED = False",
    "room reference requirement",
)
config_cell = replace_once(
    config_cell,
    "print('Room semantic labels:', ROOM_SEMANTIC_LABELS_PATH)\n",
    "print('Room semantic labels:', ROOM_SEMANTIC_LABELS_PATH)\n"
    "print('Require tracked segmentation v2:', REQUIRE_TRACKED_SEGMENTATION_V2)\n"
    "print('Allow filename mask fallback:', ALLOW_FILENAME_MASK_FALLBACK)\n"
    "print('Post-export cleanup enabled:', RUN_POST_EXPORT_CLEANUP)\n",
    "configuration diagnostics",
)
notebook["cells"][4]["source"] = config_cell.splitlines(keepends=True)


utility_cell = "".join(notebook["cells"][8]["source"])
utility_cell = replace_once(
    utility_cell,
    "def detection_object_id(item):\n"
    "    for key in ['object_id', 'id', 'instance_id', 'track_id']:\n"
    "        if item.get(key) is not None:\n"
    "            return str(item[key])\n"
    "    return None\n\n\n",
    "def detection_object_id(item):\n"
    "    for key in ['object_id', 'id', 'instance_id', 'track_id']:\n"
    "        if item.get(key) is not None:\n"
    "            return str(item[key])\n"
    "    return None\n\n\n"
    "def detection_record_is_authoritative(item, target_label_norm):\n"
    "    # Canonical v2 JSON contains accepted masks only, but these checks make the batch\n"
    "    # fail closed if a hand-edited or older file introduces rejected/unknown records.\n"
    "    if item.get('accepted') is not True or item.get('state') != 'visible':\n"
    "        return False\n"
    "    role = item.get('segmentation_role')\n"
    "    source = item.get('mask_source')\n"
    "    if target_label_norm in {'wall', 'floor', 'ceiling'}:\n"
    "        return role == 'semantic_room_structure' and source == 'segformer_ade_structural'\n"
    "    return role == 'tracked_object' and source == 'grounding_dino_sam2_video'\n\n\n",
    "accepted-record helper",
)
utility_cell = replace_once(
    utility_cell,
    "        default_frame = data.get('frame') or data.get('image') or data.get('image_path') or json_path.stem\n"
    "        for item in iter_detection_items(data):\n",
    "        if REQUIRE_TRACKED_SEGMENTATION_V2 and data.get('schema') != 'tracked_semantic_frame.v2':\n"
    "            fail(\n"
    "                f'Expected tracked_semantic_frame.v2 in {json_path}, found {data.get(\"schema\")!r}. '\n"
    "                'Run the current segmentation.ipynb through its audit stage.'\n"
    "            )\n"
    "        default_frame = data.get('frame') or data.get('image') or data.get('image_path') or json_path.stem\n"
    "        for item in iter_detection_items(data):\n",
    "tracked schema gate",
)
utility_cell = replace_once(
    utility_cell,
    "            if label != target_label_norm:\n"
    "                continue\n"
    "            mask_path = resolve_scan_path(detection_mask_path(item), base_dirs)\n",
    "            if label != target_label_norm:\n"
    "                continue\n"
    "            if not detection_record_is_authoritative(item, target_label_norm):\n"
    "                logger.warning('Ignoring non-authoritative mask record for %s in %s', label, json_path)\n"
    "                continue\n"
    "            mask_path = resolve_scan_path(detection_mask_path(item), base_dirs)\n",
    "authoritative record filter",
)
utility_cell = replace_once(
    utility_cell,
    "                'source': str(json_path),\n"
    "            })\n",
    "                'source': str(json_path),\n"
    "                'segmentation_fingerprint': data.get('segmentation_fingerprint'),\n"
    "                'layout_fingerprint': data.get('layout_fingerprint'),\n"
    "                'segmentation_role': item.get('segmentation_role'),\n"
    "                'mask_source': item.get('mask_source'),\n"
    "            })\n",
    "mask provenance fields",
)
notebook["cells"][8]["source"] = utility_cell.splitlines(keepends=True)


audit_cell = "".join(notebook["cells"][10]["source"])
legacy_fingerprint_block = (
    "detection_json_paths = sorted(DETECTION_DIR.glob('*.json'))\n"
    "if REQUIRE_TRACKED_SEGMENTATION_V2 and not detection_json_paths:\n"
    "    fail(f'No tracked detection JSON files found in {DETECTION_DIR}. Run segmentation.ipynb first.')\n"
    "segmentation_fingerprints = {\n"
    "    load_json(path).get('segmentation_fingerprint') for path in detection_json_paths\n"
    "}\n"
    "segmentation_fingerprints.discard(None)\n"
    "if REQUIRE_TRACKED_SEGMENTATION_V2 and len(segmentation_fingerprints) != 1:\n"
    "    fail(\n"
    "        f'Expected exactly one tracked segmentation fingerprint, found {sorted(segmentation_fingerprints)}. '\n"
    "        'Rerun segmentation.ipynb through the audit stage.'\n"
    "    )\n"
    "SEGMENTATION_FINGERPRINT = next(iter(segmentation_fingerprints), None)\n\n"
)
strict_fingerprint_block = (
    "detection_json_paths = sorted(DETECTION_DIR.glob('*.json'))\n"
    "if REQUIRE_TRACKED_SEGMENTATION_V2 and not detection_json_paths:\n"
    "    fail(f'No tracked detection JSON files found in {DETECTION_DIR}. Run segmentation.ipynb first.')\n"
    "detection_payloads = [(path, load_json(path)) for path in detection_json_paths]\n"
    "invalid_detection_files = [\n"
    "    str(path) for path, data in detection_payloads\n"
    "    if data.get('schema') != 'tracked_semantic_frame.v2'\n"
    "    or not data.get('segmentation_fingerprint')\n"
    "    or not data.get('layout_fingerprint')\n"
    "]\n"
    "if REQUIRE_TRACKED_SEGMENTATION_V2 and invalid_detection_files:\n"
    "    fail(\n"
    "        f'{len(invalid_detection_files)} detection JSON files are stale or incomplete. '\n"
    "        'Run segmentation.ipynb through the audit stage. First invalid file: '\n"
    "        + invalid_detection_files[0]\n"
    "    )\n"
    "segmentation_fingerprints = {\n"
    "    data.get('segmentation_fingerprint') for _, data in detection_payloads\n"
    "}\n"
    "layout_fingerprints = {data.get('layout_fingerprint') for _, data in detection_payloads}\n"
    "if REQUIRE_TRACKED_SEGMENTATION_V2 and (\n"
    "    len(segmentation_fingerprints) != 1 or len(layout_fingerprints) != 1\n"
    "):\n"
    "    fail(\n"
    "        'Expected one tracked segmentation fingerprint and one layout fingerprint; found '\n"
    "        f'{sorted(segmentation_fingerprints)} and {sorted(layout_fingerprints)}. '\n"
    "        'Rerun segmentation.ipynb through the audit stage.'\n"
    "    )\n"
    "SEGMENTATION_FINGERPRINT = next(iter(segmentation_fingerprints), None)\n"
    "LAYOUT_FINGERPRINT = next(iter(layout_fingerprints), None)\n\n"
)
if strict_fingerprint_block not in audit_cell:
    audit_cell = replace_once(
        audit_cell,
        "transforms_data = load_json(TRANSFORMS_PATH)\n",
        strict_fingerprint_block + "transforms_data = load_json(TRANSFORMS_PATH)\n",
        "strict segmentation fingerprint audit",
    )
while legacy_fingerprint_block in audit_cell:
    audit_cell = audit_cell.replace(legacy_fingerprint_block, "", 1)
audit_cell = replace_once(
    audit_cell,
    "    filename_map, filename_rows = collect_target_masks_by_filename(label_norm, frame_keys)\n"
    "    frame_to_masks = merge_mask_maps(detection_map, filename_map)\n",
    "    if ALLOW_FILENAME_MASK_FALLBACK:\n"
    "        filename_map, filename_rows = collect_target_masks_by_filename(label_norm, frame_keys)\n"
    "    else:\n"
    "        filename_map, filename_rows = {}, []\n"
    "    frame_to_masks = merge_mask_maps(detection_map, filename_map)\n",
    "filename fallback gate",
)
audit_cell = replace_once(
    audit_cell,
    "print('Source frames:', len(frames))\n",
    "print('Tracked segmentation fingerprint:', SEGMENTATION_FINGERPRINT)\n"
    "print('Source frames:', len(frames))\n",
    "fingerprint diagnostic",
)
notebook["cells"][10]["source"] = audit_cell.splitlines(keepends=True)


functions_cell = "".join(notebook["cells"][14]["source"])
functions_cell = replace_once(
    functions_cell,
    "def build_masked_dataset_for_label(label_norm, paths, frame_to_masks):\n"
    "    for directory in [paths['root'], paths['ns_data'], paths['mask_dir'], paths['preview_dir'], paths['training_dir'], paths['export_dir'], paths['report_dir']]:\n"
    "        directory.mkdir(parents=True, exist_ok=True)\n"
    "    link_shared_images_into_label_dataset(paths)\n"
    "    dataset_fingerprint = stable_json_hash({\n",
    "def dataset_fingerprint_for_label(label_norm, frame_to_masks):\n"
    "    return stable_json_hash({\n",
    "dataset fingerprint helper start",
)
functions_cell = replace_once(
    functions_cell,
    "        'coordinate_policy': SHARED_COORDINATE_POLICY,\n"
    "    })\n"
    "    if REUSE_COMPLETED_DATASET_STAGE and paths['dataset_stage_manifest'].exists():\n",
    "        'coordinate_policy': SHARED_COORDINATE_POLICY,\n"
    "    })\n\n\n"
    "def build_masked_dataset_for_label(label_norm, paths, frame_to_masks):\n"
    "    for directory in [paths['root'], paths['ns_data'], paths['mask_dir'], paths['preview_dir'], paths['training_dir'], paths['export_dir'], paths['report_dir']]:\n"
    "        directory.mkdir(parents=True, exist_ok=True)\n"
    "    link_shared_images_into_label_dataset(paths)\n"
    "    dataset_fingerprint = dataset_fingerprint_for_label(label_norm, frame_to_masks)\n"
    "    if REUSE_COMPLETED_DATASET_STAGE and paths['dataset_stage_manifest'].exists():\n",
    "dataset fingerprint helper end",
)
functions_cell = replace_once(
    functions_cell,
    "        'source_transforms_hash': SOURCE_TRANSFORMS_HASH,\n"
    "        'label': label_norm,\n",
    "        'source_transforms_hash': SOURCE_TRANSFORMS_HASH,\n"
    "        'segmentation_fingerprint': SEGMENTATION_FINGERPRINT,\n"
    "        'label': label_norm,\n",
    "dataset segmentation fingerprint",
)
functions_cell = replace_once(
    functions_cell,
    "        'source_transforms': str(TRANSFORMS_PATH),\n"
    "        'created_at': datetime.now().isoformat(timespec='seconds'),\n",
    "        'source_transforms': str(TRANSFORMS_PATH),\n"
    "        'segmentation_fingerprint': SEGMENTATION_FINGERPRINT,\n"
    "        'created_at': datetime.now().isoformat(timespec='seconds'),\n",
    "dataset provenance",
)
notebook["cells"][14]["source"] = functions_cell.splitlines(keepends=True)


run_cell = "".join(notebook["cells"][16]["source"])
if "reusable_raw_complete = bool(" not in run_cell:
    run_cell = replace_once(
        run_cell,
        "    reusable_cleanup_complete = bool(\n"
        "        existing_cleanup\n"
        "        and existing_cleanup.get('cleaned_written')\n"
        "        and existing_manifest_splat.exists()\n"
        "        and existing_manifest_splat.stat().st_size > 0\n"
        "    )\n"
        "    if SKIP_EXISTING_SUCCESSFUL_EXPORTS and reusable_cleanup_complete:\n",
        "    reusable_cleanup_complete = bool(\n"
        "        existing_cleanup\n"
        "        and existing_cleanup.get('cleaned_written')\n"
        "        and existing_manifest_splat.exists()\n"
        "        and existing_manifest_splat.stat().st_size > 0\n"
        "    )\n"
        "    reusable_raw_complete = bool(\n"
        "        not RUN_POST_EXPORT_CLEANUP\n"
        "        and existing_splat.exists()\n"
        "        and existing_splat.stat().st_size > 0\n"
        "        and paths['dataset_stage_manifest'].exists()\n"
        "        and (paths['report_dir'] / 'training_stage_manifest.json').exists()\n"
        "    )\n"
        "    reusable_export_complete = reusable_cleanup_complete or reusable_raw_complete\n"
        "    if SKIP_EXISTING_SUCCESSFUL_EXPORTS and reusable_export_complete:\n",
        "raw export resume gate",
    )
run_cell = replace_once(
    run_cell,
    "        and paths['dataset_stage_manifest'].exists()\n"
    "        and (paths['report_dir'] / 'training_stage_manifest.json').exists()\n",
    "        and paths['dataset_stage_manifest'].exists()\n"
    "        and load_json(paths['dataset_stage_manifest']).get('fingerprint')\n"
    "            == dataset_fingerprint_for_label(label_norm, frame_to_masks)\n"
    "        and (paths['report_dir'] / 'training_stage_manifest.json').exists()\n",
    "raw export fingerprint gate",
)
run_cell = replace_once(
    run_cell,
    "            'status': 'existing_cleanup_reused',\n",
    "            'status': 'existing_cleanup_reused' if reusable_cleanup_complete else 'existing_raw_reused',\n",
    "manifest reuse status",
)
run_cell = replace_once(
    run_cell,
    "            'cleaned_splat_path': existing_cleanup.get('cleaned_splat_path'),\n",
    "            'cleaned_splat_path': existing_cleanup.get('cleaned_splat_path') if existing_cleanup else None,\n",
    "optional existing cleanup",
)
run_cell = replace_once(
    run_cell,
    "        status_rows.append({'label': label_norm, 'status': 'existing_cleanup_reused', 'positive_mask_frames': positive_frames, 'splat_path': str(existing_manifest_splat)})\n",
    "        reuse_status = 'existing_cleanup_reused' if reusable_cleanup_complete else 'existing_raw_reused'\n"
    "        status_rows.append({'label': label_norm, 'status': reuse_status, 'positive_mask_frames': positive_frames, 'splat_path': str(existing_manifest_splat)})\n",
    "status-row reuse status",
)
notebook["cells"][16]["source"] = run_cell.splitlines(keepends=True)


intro = "".join(notebook["cells"][0]["source"])
intro = intro.replace(
    "This notebook scales the successful `object_splat_experiment.ipynb` from one target label to multiple semantic/object labels.",
    "This v7 notebook trains fresh object splats from the accepted, tracked masks produced by `segmentation.ipynb`.",
)
intro = intro.replace(
    "Frames where an object is absent get an empty mask.",
    "All registered cameras are retained; frames without an accepted visible mask receive an empty loss mask and preserve shared coordinates.",
)
notebook["cells"][0]["source"] = intro.splitlines(keepends=True)


for cell in notebook["cells"]:
    if cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(f"Updated {NOTEBOOK_PATH}")
