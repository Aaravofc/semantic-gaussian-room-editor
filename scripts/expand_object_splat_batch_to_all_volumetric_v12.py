"""Expand the six-object V11 pilot into an all-instance volumetric V12 batch."""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

from patch_object_splat_runtime_compatibility_v2 import patch_notebook
from patch_object_splat_resume_export_v1 import patch_notebook as patch_resume_export
from patch_object_splat_interruption_recovery_v1 import (
    patch_notebook as patch_interruption_recovery,
)


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "object_splat_batch_pipeline.ipynb"
V11_VERSION = "2026-08-18-object-splat-volumetric-batch-v11"
V12_VERSION = "2026-08-23-object-splat-all-volumetric-instances-v12"


def source_text(cell: dict) -> str:
    return "".join(cell.get("source", []))


def source_lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def cell_containing(cells: list[dict], needle: str, cell_type: str = "code") -> dict:
    matches = [
        cell
        for cell in cells
        if cell.get("cell_type") == cell_type and needle in source_text(cell)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {cell_type} cell containing {needle!r}, found {len(matches)}"
        )
    return matches[0]


def replace_function(source: str, name: str, replacement: str) -> str:
    pattern = re.compile(
        rf"^def {re.escape(name)}\([^\n]*\):\n.*?(?=^def [A-Za-z_]\w*\(|\Z)",
        flags=re.MULTILINE | re.DOTALL,
    )
    matches = list(pattern.finditer(source))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one function {name!r}, found {len(matches)}")
    replacement = textwrap.dedent(replacement).strip() + "\n\n\n"
    return source[: matches[0].start()] + replacement + source[matches[0].end() :]


notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
cells = notebook["cells"]
config_cell = cell_containing(cells, "BATCH_PIPELINE_VERSION =")
config = source_text(config_cell)
match = re.search(r"BATCH_PIPELINE_VERSION\s*=\s*['\"]([^'\"]+)", config)
if match is None:
    raise RuntimeError("Could not read BATCH_PIPELINE_VERSION")
current_version = match.group(1)

if current_version == V12_VERSION:
    repairs = {
        "print('\n' + '=' * 80)": "print('\\n' + '=' * 80)",
        "print('\n' + '-' * 80)": "print('\\n' + '-' * 80)",
        "print('\nVolumetric-object batch complete:": (
            "print('\\nVolumetric-object batch complete:"
        ),
    }
    changed = False
    project_metadata = notebook.setdefault("metadata", {}).setdefault(
        "room_model_project", {}
    )
    if project_metadata.get("object_splat_batch_version") != V12_VERSION:
        project_metadata["object_splat_batch_version"] = V12_VERSION
        changed = True
    for cell in cells:
        source = source_text(cell)
        updated = source
        for old, new in repairs.items():
            updated = updated.replace(old, new)
        if updated != source:
            cell["source"] = source_lines(updated)
            changed = True
    changed = patch_notebook(notebook) or changed
    changed = patch_resume_export(notebook) or changed
    changed = patch_interruption_recovery(notebook) or changed
    if changed:
        NOTEBOOK_PATH.write_text(
            json.dumps(notebook, indent=1) + "\n", encoding="utf-8"
        )
        print(f"Repaired generated V12 source in {NOTEBOOK_PATH}")
    else:
        print(f"{NOTEBOOK_PATH} is already {V12_VERSION}")
    raise SystemExit(0)
if current_version != V11_VERSION:
    raise RuntimeError(
        f"Notebook version {current_version!r} is unsupported; expected {V11_VERSION!r}."
    )

cells[0]["source"] = source_lines(
    """# Batch Object Gaussian Splat Pipeline

This notebook runs the **V12 all-instance bounded-volumetric object batch**. It discovers every
audited persistent instance belonging to the configured compact-object taxonomy and trains the
accepted V9 `ellipsoid_constrained` method for each object in the full-room coordinate system.

Outputs are keyed by persistent instance, so multiple chairs, pillows, bags, or other same-class
objects remain independently selectable. Planar, hanging, deformable, and fixture labels are
reported but excluded because they require structural or class-specific constraints rather than
the compact ellipsoid prior. Existing V9, V10, and V11 Drive outputs are never overwritten.

The dependency preflight pins a tested NumPy ABI before importing NumPy, and geometry sampling is
deterministic. A failed object is recorded without hiding the successful objects, and reruns reuse
completed per-instance dataset, training, and export stages when their fingerprints still match.
"""
)

config = config.replace(V11_VERSION, V12_VERSION, 1)
block_start = config.index("# V11 scales the accepted V9 method")
block_end = config.index("# The three arms share identical cameras")
target_block = """# V12 discovers every audited instance from an explicit compact-volume taxonomy.
VOLUMETRIC_BATCH_MODE = True
AUTO_DISCOVER_VOLUMETRIC_LABELS = True
TARGET_LABELS = []  # Optional class filter; empty means every discovered allowlisted class.
VOLUMETRIC_LABEL_ALLOWLIST = {
    'bed', 'chair', 'table', 'tv', 'wardrobe', 'cabinet', 'drawer', 'shelf',
    'desk', 'sofa', 'lamp', 'pillow', 'bag', 'shoe', 'bottle', 'book', 'plant',
    'miscellaneous',
}
VOLUMETRIC_LABEL_EXCLUSIONS = {
    'wall': 'structural_plane',
    'floor': 'structural_plane',
    'ceiling': 'structural_plane',
    'door': 'architectural_plane',
    'window': 'architectural_plane',
    'curtain': 'hanging_deformable_surface',
    'rug': 'floor_surface',
    'blanket': 'deformable_surface',
    'mirror': 'thin_reflective_surface',
    'picture_frame': 'thin_wall_mounted_surface',
    'fan': 'thin_or_articulated_fixture',
    'light': 'ceiling_or_wall_fixture',
}
STRUCTURAL_OR_THIN_LABELS = set(VOLUMETRIC_LABEL_EXCLUSIONS)
# Optional per-class instance filter. Values may be one instance ID or a list of IDs.
# Empty means every qualifying persistent instance of every selected class.
TARGET_INSTANCE_IDS = {}
INSTANCE_SELECTION_POLICY = 'all_qualifying_persistent_instances'
RESOLVED_INSTANCE_IDS = {}
TARGET_OBJECT_LABELS = {}
TARGET_OBJECTS = []
DISCOVERED_VOLUMETRIC_LABELS = []

"""
config = config[:block_start] + target_block + config[block_end:]
config = config.replace(
    "BATCH_ID = 'volumetric_objects_ellipsoid_batch_v11'",
    "BATCH_ID = 'all_volumetric_instances_ellipsoid_batch_v12'",
    1,
)
config = config.replace(
    "MAX_LABELS_TO_RUN = None  # Example: 2 for a short smoke test.",
    "MAX_OBJECTS_TO_RUN = None  # Example: 2 for a short smoke test.",
    1,
)
config = config.replace(
    "print('Target labels:', TARGET_LABELS)",
    "print('Explicit target-label filter:', TARGET_LABELS or 'all allowlisted labels')",
    1,
)
config_cell["source"] = source_lines(config)

utility_cell = cell_containing(cells, "def load_json")
utility = source_text(utility_cell)
normalizer = """def normalize_label(label):
    return re.sub(r'[^a-z0-9]+', '_', str(label).strip().lower()).strip('_')
"""
helpers = normalizer + """


def object_key_for_instance(label_norm, instance_id):
    label_norm = normalize_label(label_norm)
    instance_norm = normalize_label(instance_id)
    if not instance_norm:
        raise ValueError(f'Empty instance ID for label {label_norm!r}.')
    if instance_norm == label_norm or instance_norm.startswith(label_norm + '_'):
        return instance_norm
    return f'{label_norm}__{instance_norm}'


def target_semantic_label(target_key):
    if target_key not in TARGET_OBJECT_LABELS:
        raise KeyError(f'Unknown target object key: {target_key!r}')
    return TARGET_OBJECT_LABELS[target_key]


def target_instance_id(target_key):
    if target_key not in RESOLVED_INSTANCE_IDS:
        raise KeyError(f'Unknown target object key: {target_key!r}')
    return RESOLVED_INSTANCE_IDS[target_key]


def configured_instance_ids(label_norm):
    configured = TARGET_INSTANCE_IDS.get(label_norm)
    if configured is None:
        return None
    if isinstance(configured, (str, int)):
        return {str(configured)}
    return {str(value) for value in configured}
"""
if utility.count(normalizer) != 1:
    raise RuntimeError("Could not locate normalize_label for V12 helpers")
utility = utility.replace(normalizer, helpers, 1)
utility = replace_function(
    utility,
    "label_paths",
    '''
    def label_paths(target_key):
        target_key = normalize_label(target_key)
        label_norm = target_semantic_label(target_key)
        instance_id = target_instance_id(target_key)
        run_id = f'{target_key}_{BATCH_ID}'
        root = BATCH_ROOT / target_key
        local_root = LOCAL_CACHE_ROOT / BATCH_ID / target_key
        local_ns_data = local_root / 'nerfstudio-data'
        return {
            'label': label_norm,
            'object_key': target_key,
            'instance_id': instance_id,
            'run_id': run_id,
            'root': root,
            'ns_data': local_ns_data,
            'local_root': local_root,
            'image_dir': local_ns_data / 'images',
            'mask_dir': local_ns_data / 'masks',
            'preview_dir': local_root / 'masked_rgb_preview',
            'training_dir': root / 'training',
            'export_dir': root / 'exports',
            'alignment_manifest': root / 'object_splat_alignment_manifest.json',
            'report_dir': REPORT_DIR / target_key,
            'dataset_stage_manifest': root / 'dataset_stage_manifest.json',
        }
    ''',
)
utility = utility.replace(
    "'run_id': f\"{base_paths['label']}_{variant_name}_{BATCH_ID}\"",
    "'run_id': f\"{base_paths['object_key']}_{variant_name}_{BATCH_ID}\"",
    1,
)
utility_cell["source"] = source_lines(utility)

audit_cell = cell_containing(cells, "TRANSFORMS_PATH = SOURCE_NS_DATA_DIR")
audit_cell["source"] = source_lines(
    """TRANSFORMS_PATH = SOURCE_NS_DATA_DIR / 'transforms.json'

if not PROJECT_ROOT.exists():
    fail(f'Missing project root: {PROJECT_ROOT}')
if not SCAN_DIR.exists():
    fail(f'Missing scan dir: {SCAN_DIR}')
if not TRANSFORMS_PATH.exists():
    fail(f'Missing transforms.json: {TRANSFORMS_PATH}. Run reconstruction.ipynb first.')

detection_json_paths = sorted(DETECTION_DIR.glob('*.json'))
if REQUIRE_TRACKED_SEGMENTATION_V2 and not detection_json_paths:
    fail(f'No tracked detection JSON files found in {DETECTION_DIR}. Run segmentation.ipynb first.')
detection_payloads = [(path, load_json(path)) for path in detection_json_paths]
invalid_detection_files = [
    str(path) for path, data in detection_payloads
    if data.get('schema') != 'tracked_semantic_frame.v2'
    or not data.get('segmentation_fingerprint')
    or not data.get('layout_fingerprint')
]
if REQUIRE_TRACKED_SEGMENTATION_V2 and invalid_detection_files:
    fail(
        f'{len(invalid_detection_files)} detection JSON files are stale or incomplete. '
        'Run the current segmentation.ipynb through its audit stage. First invalid file: '
        + invalid_detection_files[0]
    )
segmentation_fingerprints = {
    data.get('segmentation_fingerprint') for _, data in detection_payloads
}
layout_fingerprints = {data.get('layout_fingerprint') for _, data in detection_payloads}
if REQUIRE_TRACKED_SEGMENTATION_V2 and (
    len(segmentation_fingerprints) != 1 or len(layout_fingerprints) != 1
):
    fail(
        'Expected one tracked segmentation fingerprint and one layout fingerprint; found '
        f'{sorted(segmentation_fingerprints)} and {sorted(layout_fingerprints)}. '
        'Rerun segmentation.ipynb through Audit.'
    )
SEGMENTATION_FINGERPRINT = next(iter(segmentation_fingerprints), None)
LAYOUT_FINGERPRINT = next(iter(layout_fingerprints), None)

transforms_data = load_json(TRANSFORMS_PATH)
frames = transforms_data.get('frames', [])
if not frames:
    fail(f'No frames in transforms.json: {TRANSFORMS_PATH}')
frame_keys = [
    frame_key_from_any(frame.get('file_path', f'frame_{idx:06d}'))
    for idx, frame in enumerate(frames)
]
missing_images = []
for frame in frames:
    image_path = frame_image_path(frame)
    if image_path is None or not image_path.exists():
        missing_images.append(str(image_path))
if missing_images:
    safe_write_csv(
        REPORT_DIR / 'missing_source_images.csv',
        [{'image_path': path} for path in missing_images],
    )
    fail(
        f'{len(missing_images)} frame images referenced by transforms.json are missing. '
        'See missing_source_images.csv'
    )

SOURCE_TRANSFORMS_HASH = stable_json_hash(transforms_data)
SHARED_CACHE_MANIFEST = prepare_shared_local_image_cache()

accepted_detection_labels = set()
for _, payload in detection_payloads:
    for item in iter_detection_items(payload):
        label_norm = detection_label(item)
        if label_norm and detection_record_is_authoritative(item, label_norm):
            accepted_detection_labels.add(label_norm)

explicit_targets = {normalize_label(label) for label in TARGET_LABELS}
invalid_explicit_targets = sorted(explicit_targets - VOLUMETRIC_LABEL_ALLOWLIST)
if invalid_explicit_targets:
    fail(
        'Explicit targets are not approved for compact volumetric containment: '
        + ', '.join(invalid_explicit_targets)
    )
selected_labels = sorted(
    accepted_detection_labels & VOLUMETRIC_LABEL_ALLOWLIST
    & (explicit_targets if explicit_targets else VOLUMETRIC_LABEL_ALLOWLIST)
)
DISCOVERED_VOLUMETRIC_LABELS = selected_labels

discovery_rows = []
for label_norm in sorted(accepted_detection_labels | explicit_targets):
    if label_norm in VOLUMETRIC_LABEL_EXCLUSIONS:
        decision = 'excluded_geometry_policy'
        reason = VOLUMETRIC_LABEL_EXCLUSIONS[label_norm]
    elif label_norm not in VOLUMETRIC_LABEL_ALLOWLIST:
        decision = 'excluded_not_allowlisted'
        reason = 'unknown_geometry_policy'
    elif explicit_targets and label_norm not in explicit_targets:
        decision = 'excluded_by_explicit_filter'
        reason = 'not_requested_for_this_run'
    elif label_norm not in accepted_detection_labels:
        decision = 'unavailable'
        reason = 'no_accepted_masks'
    else:
        decision = 'selected_volumetric_class'
        reason = 'compact_bounded_object_policy'
    discovery_rows.append({
        'label': label_norm,
        'accepted_detection_label': label_norm in accepted_detection_labels,
        'decision': decision,
        'reason': reason,
    })
safe_write_csv(REPORT_DIR / 'volumetric_label_discovery.csv', discovery_rows)

object_audit_rows = []
target_mask_maps = {}
target_mask_source_rows = {}
seen_object_keys = set()
for label_norm in selected_labels:
    _, all_detection_rows = collect_target_masks_from_detections(label_norm)
    candidate_rows = summarize_instance_candidates(all_detection_rows)
    configured_ids = configured_instance_ids(label_norm)
    available_ids = {row['instance_id'] for row in candidate_rows}
    if configured_ids is not None:
        missing_configured = sorted(configured_ids - available_ids)
        if missing_configured:
            fail(
                f'Configured instances for {label_norm!r} were not found: '
                + ', '.join(missing_configured)
            )
    for candidate in candidate_rows:
        instance_id = candidate['instance_id']
        selected_by_filter = configured_ids is None or instance_id in configured_ids
        candidate['selected'] = bool(selected_by_filter)
        candidate['selection_policy'] = INSTANCE_SELECTION_POLICY
        target_key = object_key_for_instance(label_norm, instance_id)
        if target_key in seen_object_keys:
            fail(f'Duplicate normalized target object key: {target_key!r}')
        seen_object_keys.add(target_key)

        if selected_by_filter:
            detection_map, detection_rows = collect_target_masks_from_detections(
                label_norm,
                target_object_id=instance_id,
            )
        else:
            detection_map, detection_rows = {}, []
        if ALLOW_FILENAME_MASK_FALLBACK and selected_by_filter:
            filename_map, filename_rows = collect_target_masks_by_filename(
                label_norm, frame_keys
            )
        else:
            filename_map, filename_rows = {}, []
        frame_to_masks = merge_mask_maps(detection_map, filename_map)
        positive_frames = len(frame_to_masks)
        mask_records = sum(len(values) for values in frame_to_masks.values())
        if not selected_by_filter:
            audit_status = 'excluded_by_instance_filter'
        elif positive_frames < MIN_POSITIVE_MASK_FRAMES:
            audit_status = 'too_few_positive_mask_frames'
        else:
            audit_status = 'ready'
        will_run = (
            selected_by_filter
            and (audit_status == 'ready' or not SKIP_LABELS_WITH_TOO_FEW_MASK_FRAMES)
        )

        if selected_by_filter:
            TARGET_OBJECT_LABELS[target_key] = label_norm
            RESOLVED_INSTANCE_IDS[target_key] = instance_id
            target_mask_maps[target_key] = frame_to_masks
            target_mask_source_rows[target_key] = detection_rows + filename_rows
            TARGET_OBJECTS.append({
                'object_key': target_key,
                'label': label_norm,
                'instance_id': instance_id,
                'positive_mask_frames': positive_frames,
                'will_run': bool(will_run),
            })

        object_audit_rows.append({
            'object_key': target_key,
            'label': label_norm,
            'instance_id': instance_id,
            'audit_status': audit_status,
            'positive_mask_frames': positive_frames,
            'mask_records': mask_records,
            'detection_json_records': len(detection_rows),
            'filename_scan_records': len(filename_rows),
            'will_run': bool(will_run),
        })
        safe_write_csv(
            REPORT_DIR / target_key / 'target_mask_sources.csv',
            target_mask_source_rows.get(target_key, []),
        )
    safe_write_csv(REPORT_DIR / label_norm / 'instance_candidates.csv', candidate_rows)

safe_write_csv(REPORT_DIR / 'batch_object_audit.csv', object_audit_rows)
print('Tracked segmentation fingerprint:', SEGMENTATION_FINGERPRINT)
print('Source frames:', len(frames))
print('Missing images:', len(missing_images))
print('Accepted detection labels:', sorted(accepted_detection_labels))
print('Selected volumetric labels:', DISCOVERED_VOLUMETRIC_LABELS)
print('Target object count:', len(TARGET_OBJECTS))
print('Object audit:')
print_rows(
    object_audit_rows,
    columns=[
        'object_key', 'label', 'instance_id', 'audit_status',
        'positive_mask_frames', 'mask_records', 'will_run',
    ],
    limit=200,
)
print('Label discovery CSV:', REPORT_DIR / 'volumetric_label_discovery.csv')
print('Object audit CSV:', REPORT_DIR / 'batch_object_audit.csv')
"""
)

functions_cell = cell_containing(cells, "def dataset_fingerprint_for_label")
functions = source_text(functions_cell)
functions = replace_function(
    functions,
    "dataset_fingerprint_for_label",
    '''
    def dataset_fingerprint_for_label(label_norm, paths, frame_to_masks):
        return stable_json_hash({
            'pipeline_version': BATCH_PIPELINE_VERSION,
            'source_transforms_hash': SOURCE_TRANSFORMS_HASH,
            'segmentation_fingerprint': SEGMENTATION_FINGERPRINT,
            'object_key': paths['object_key'],
            'label': label_norm,
            'instance_id': paths['instance_id'],
            'mask_inventory': file_inventory_hash(
                [path for masks in frame_to_masks.values() for path in masks]
            ),
            'mask_dilation': mask_dilation_for_label(label_norm),
            'mask_erosion': mask_erosion_for_label(label_norm),
            'downscale_factors': DOWNSCALE_FACTORS_TO_WRITE,
            'coordinate_policy': SHARED_COORDINATE_POLICY,
        })
    ''',
)
functions = functions.replace(
    "dataset_fingerprint = dataset_fingerprint_for_label(label_norm, frame_to_masks)",
    "dataset_fingerprint = dataset_fingerprint_for_label(label_norm, paths, frame_to_masks)",
    1,
)
functions = functions.replace(
    "RESOLVED_INSTANCE_IDS.get(label_norm)",
    "paths.get('instance_id')",
)
functions = functions.replace(
    "new_frame['target_label'] = label_norm\n",
    (
        "new_frame['target_label'] = label_norm\n"
        "        new_frame['target_instance_id'] = paths['instance_id']\n"
        "        new_frame['target_object_key'] = paths['object_key']\n"
    ),
    1,
)
functions = functions.replace(
    "'target_label': label_norm,\n        'target_instance_id': paths.get('instance_id'),",
    (
        "'target_object_key': paths['object_key'],\n"
        "        'target_label': label_norm,\n"
        "        'target_instance_id': paths.get('instance_id'),"
    ),
)
functions = functions.replace(
    "'label': label_norm,\n        'instance_id': paths.get('instance_id'),",
    (
        "'object_key': paths['object_key'],\n"
        "        'label': label_norm,\n"
        "        'instance_id': paths.get('instance_id'),"
    ),
)
functions = functions.replace(
    "object_sparse_path = paths['ns_data'] / f'{label_norm}_sparse_pc.ply'",
    "object_sparse_path = paths['ns_data'] / f\"{paths['object_key']}_sparse_pc.ply\"",
    1,
)
functions = functions.replace(
    "f\"ns_train_{BATCH_ID}_{label_norm}_{paths.get('variant_name', method_name)}.log\"",
    "f\"ns_train_{BATCH_ID}_{paths['object_key']}_{paths.get('variant_name', method_name)}.log\"",
    1,
)
functions = functions.replace(
    "Path(f'/content/object-splat-export-{BATCH_ID}-{label_norm}-{variant_name}')",
    "Path(f\"/content/object-splat-export-{BATCH_ID}-{paths['object_key']}-{variant_name}\")",
    1,
)
functions = functions.replace(
    "paths['ns_data'] / f'{label_norm}_visual_support_grid.npz'",
    "paths['ns_data'] / f\"{paths['object_key']}_visual_support_grid.npz\"",
    1,
)
functions = functions.replace(
    "paths['report_dir'] / f'{label_norm}_visual_support_grid.npz'",
    "paths['report_dir'] / f\"{paths['object_key']}_visual_support_grid.npz\"",
    1,
)
functions = replace_function(
    functions,
    "write_batch_manifest",
    '''
    def write_batch_manifest(entries, status_rows):
        viewer_entries = [entry for entry in entries if entry.get('recommended')]
        successful_objects = sorted({entry['object_key'] for entry in entries})
        viewer_objects = sorted({entry['object_key'] for entry in viewer_entries})
        if successful_objects != viewer_objects:
            raise RuntimeError(
                'Every successful object must have exactly one recommended viewer export. '
                f'Successful={successful_objects}; viewer={viewer_objects}'
            )
        for object_key in successful_objects:
            count = sum(
                1 for entry in viewer_entries if entry['object_key'] == object_key
            )
            if count != 1:
                raise RuntimeError(
                    f'Expected one recommended export for {object_key!r}, found {count}.'
                )
        failed = [row for row in status_rows if row.get('status') == 'failed']
        skipped = [row for row in status_rows if row.get('status') == 'skipped']
        manifest = {
            'schema': 'object_splat_manifest.v2',
            'version': BATCH_PIPELINE_VERSION,
            'created_at': datetime.now().isoformat(timespec='seconds'),
            'project_root': str(PROJECT_ROOT),
            'scan_id': SCAN_ID,
            'batch_id': BATCH_ID,
            'source_transforms': str(TRANSFORMS_PATH),
            'coordinate_policy': SHARED_COORDINATE_POLICY,
            'expected_scene_transform_to_full_room_splat': {
                'translation': [0.0, 0.0, 0.0],
                'rotation_quaternion_xyzw': [0.0, 0.0, 0.0, 1.0],
                'scale': 1.0,
            },
            'objects': viewer_entries,
            'status_rows': status_rows,
            'batch_status': (
                'partial_failure' if failed else
                'success_with_skips' if skipped else
                'success'
            ),
            'volumetric_batch': {
                'auto_discovery_enabled': bool(AUTO_DISCOVER_VOLUMETRIC_LABELS),
                'target_labels': list(DISCOVERED_VOLUMETRIC_LABELS),
                'target_object_keys': [item['object_key'] for item in TARGET_OBJECTS],
                'resolved_instance_ids': dict(RESOLVED_INSTANCE_IDS),
                'object_labels': dict(TARGET_OBJECT_LABELS),
                'instance_selection_policy': INSTANCE_SELECTION_POLICY,
                'recommended_variant': PILOT_RECOMMENDED_VARIANT,
                'variants_run': list(PILOT_VARIANTS_TO_RUN),
                'method_family': 'v9_ellipsoid_containment_all_instances',
                'volumetric_label_allowlist': sorted(VOLUMETRIC_LABEL_ALLOWLIST),
                'excluded_label_policies': dict(VOLUMETRIC_LABEL_EXCLUSIONS),
                'post_export_cleanup_enabled': False,
            },
            'ablation_variants': entries,
        }
        write_json(BATCH_MANIFEST_PATH, manifest)
        write_json(LATEST_MANIFEST_PATH, manifest)
        safe_write_csv(REPORT_DIR / 'batch_status.csv', status_rows)
        return manifest
    ''',
)
functions_cell["source"] = source_lines(functions)

run_cell = cell_containing(cells, "if not VOLUMETRIC_BATCH_MODE")
run_cell["source"] = source_lines(
    """if not VOLUMETRIC_BATCH_MODE:
    fail('This notebook version is intentionally configured as a volumetric-object batch.')
if set(VOLUMETRIC_LABEL_ALLOWLIST) & set(VOLUMETRIC_LABEL_EXCLUSIONS):
    fail('Volumetric allowlist and exclusion policy must be disjoint.')
if PILOT_RECOMMENDED_VARIANT not in PILOT_VARIANTS_TO_RUN:
    fail('The recommended variant must be included in PILOT_VARIANTS_TO_RUN.')

audit_by_object = {
    row['object_key']: row for row in object_audit_rows if row.get('object_key')
}
eligible_objects = [
    item for item in TARGET_OBJECTS
    if item.get('will_run') and audit_by_object[item['object_key']].get('will_run')
]
eligible_objects.sort(
    key=lambda item: (
        item['label'],
        -int(item.get('positive_mask_frames', 0)),
        item['object_key'],
    )
)
if MAX_OBJECTS_TO_RUN is not None:
    eligible_objects = eligible_objects[: int(MAX_OBJECTS_TO_RUN)]
if not eligible_objects:
    fail(
        'No persistent volumetric instance has enough positive mask frames. '
        f'Inspect {REPORT_DIR / "batch_object_audit.csv"}.'
    )

estimated_iterations = sum(
    train_iterations_for_label(item['label']) * len(PILOT_VARIANTS_TO_RUN)
    for item in eligible_objects
)
print('Eligible volumetric objects:', len(eligible_objects))
print('Estimated aggregate training iterations:', estimated_iterations)

status_rows = []
variant_entries = []
eligible_keys = {item['object_key'] for item in eligible_objects}
for row in object_audit_rows:
    if row.get('object_key') not in eligible_keys:
        status_rows.append({
            'object_key': row.get('object_key'),
            'label': row['label'],
            'instance_id': row.get('instance_id'),
            'variant': None,
            'method': None,
            'status': 'skipped',
            'stage': 'audit',
            'positive_mask_frames': row.get('positive_mask_frames', 0),
            'message': row.get('audit_status', 'not_selected'),
        })

for target in eligible_objects:
    target_key = target['object_key']
    label_norm = target['label']
    instance_id = target['instance_id']
    frame_to_masks = target_mask_maps.get(target_key, {})
    positive_frames = len(frame_to_masks)
    print('\\n' + '=' * 80)
    print('Volumetric object:', target_key, label_norm, instance_id)
    print('Positive mask frames:', positive_frames)
    base_paths = label_paths(target_key)
    base_paths['report_dir'].mkdir(parents=True, exist_ok=True)

    try:
        dataset_summary = build_masked_dataset_for_label(
            label_norm, base_paths, frame_to_masks
        )
        sparse_written, sparse_count, source_sparse, object_sparse_path = (
            create_sparse_init_for_label(
                label_norm,
                base_paths,
                base_paths['report_dir'] / 'masked_dataset_frame_report.csv',
            )
        )
        if not sparse_written or not object_sparse_path:
            raise RuntimeError(
                'V9 ellipsoid containment requires a non-empty object-specific sparse '
                'initializer. Inspect masks and sparse_point_report.csv before changing '
                'SPARSE_POINT_MIN_MASK_HITS.'
            )
        support_grid_path, support_grid_report = build_visual_support_grid_for_label(
            label_norm,
            base_paths,
            object_sparse_path,
        )
    except Exception as exc:
        logger.exception('Volumetric object setup failed: %s', target_key)
        status_rows.append({
            'object_key': target_key,
            'label': label_norm,
            'instance_id': instance_id,
            'variant': None,
            'method': None,
            'status': 'failed',
            'stage': 'dataset_sparse_or_support',
            'positive_mask_frames': positive_frames,
            'message': str(exc),
        })
        safe_write_csv(REPORT_DIR / 'volumetric_variant_status.csv', status_rows)
        if STOP_ON_LABEL_FAILURE:
            raise
        continue

    for variant_name in PILOT_VARIANTS_TO_RUN:
        print('\\n' + '-' * 80)
        print('Training:', target_key, label_norm, variant_name)
        paths = pilot_variant_paths(
            base_paths,
            variant_name,
            object_sparse_path=object_sparse_path,
            support_grid_path=support_grid_path,
        )
        for directory in [
            paths['root'], paths['training_dir'],
            paths['export_dir'], paths['report_dir'],
        ]:
            directory.mkdir(parents=True, exist_ok=True)
        try:
            train_log = (
                run_training_for_label(label_norm, paths) if RUN_TRAINING else None
            )
            if not RUN_EXPORT:
                raise RuntimeError('RUN_EXPORT must remain enabled for this batch.')
            export_info = export_splat_for_label(label_norm, paths)
            splat_path = Path(export_info['raw_splat_path'])
            entry = {
                'object_id': target_key,
                'object_key': target_key,
                'component_id': f'{target_key}:{variant_name}',
                'label': label_norm,
                'instance_id': instance_id,
                'variant': variant_name,
                'method': paths['method_name'],
                'description': paths['variant_description'],
                'recommended': variant_name == PILOT_RECOMMENDED_VARIANT,
                'run_id': paths['run_id'],
                'splat_path': str(splat_path),
                'raw_splat_path': str(splat_path),
                'splat_path_relative_to_scan': str(splat_path.relative_to(SCAN_DIR)),
                'export_dir': str(paths['export_dir']),
                'training_dir': str(paths['training_dir']),
                'alignment_manifest': str(base_paths['alignment_manifest']),
                'coordinate_policy': SHARED_COORDINATE_POLICY,
                'positive_mask_frame_count': dataset_summary['positive_mask_frame_count'],
                'object_dataset_frame_count': dataset_summary['object_dataset_frame_count'],
                'object_sparse_point_count': int(sparse_count),
                'object_sparse_path': object_sparse_path,
                'visual_support_grid_report': str(
                    base_paths['report_dir'] / 'visual_support_grid_report.json'
                ),
                'post_export_cleanup': False,
                'train_iterations': train_iterations_for_label(label_norm),
                'train_log': train_log,
                'export_log': export_info.get('export_log'),
                'status': 'success',
            }
            variant_entries.append(entry)
            status_rows.append({
                'object_key': target_key,
                'label': label_norm,
                'instance_id': instance_id,
                'variant': variant_name,
                'method': paths['method_name'],
                'status': 'success',
                'stage': 'complete',
                'positive_mask_frames': dataset_summary['positive_mask_frame_count'],
                'object_sparse_point_count': int(sparse_count),
                'splat_path': str(splat_path),
            })
            write_json(paths['report_dir'] / 'variant_summary.json', entry)
            print('SUCCESS:', target_key, label_norm, variant_name, splat_path)
        except Exception as exc:
            logger.exception('Variant failed: %s/%s', target_key, variant_name)
            status_rows.append({
                'object_key': target_key,
                'label': label_norm,
                'instance_id': instance_id,
                'variant': variant_name,
                'method': paths['method_name'],
                'status': 'failed',
                'stage': 'training_or_export',
                'positive_mask_frames': positive_frames,
                'message': str(exc),
            })
            safe_write_csv(REPORT_DIR / 'volumetric_variant_status.csv', status_rows)
            if STOP_ON_LABEL_FAILURE:
                raise

if not variant_entries:
    safe_write_csv(REPORT_DIR / 'volumetric_variant_status.csv', status_rows)
    fail('No volumetric-object variant completed successfully. Inspect the status CSV and logs.')

safe_write_csv(REPORT_DIR / 'volumetric_variant_status.csv', status_rows)
manifest = write_batch_manifest(variant_entries, status_rows)
print('\\nVolumetric-object batch complete:', manifest['batch_status'])
print_rows(
    status_rows,
    columns=[
        'object_key', 'label', 'instance_id', 'variant', 'method',
        'status', 'stage', 'splat_path', 'message',
    ],
    limit=300,
)
print('Viewer-selected variant per successful object:', PILOT_RECOMMENDED_VARIANT)
print('Batch manifest:', BATCH_MANIFEST_PATH)
print('Latest manifest:', LATEST_MANIFEST_PATH)
"""
)

inspection_cell = cell_containing(cells, "manifest = load_json(LATEST_MANIFEST_PATH)")
inspection = source_text(inspection_cell)
inspection = inspection.replace(
    "successful_labels = sorted({entry['label'] for entry in variants})\nfor label in successful_labels:\n    selected = [entry for entry in viewer_objects if entry.get('label') == label]\n    if len(selected) != 1 or selected[0].get('variant') != PILOT_RECOMMENDED_VARIANT:\n        fail(f'Viewer manifest does not select one recommended export for {label!r}.')",
    (
        "successful_objects = sorted({entry['object_key'] for entry in variants})\n"
        "for object_key in successful_objects:\n"
        "    selected = [\n"
        "        entry for entry in viewer_objects\n"
        "        if entry.get('object_key') == object_key\n"
        "    ]\n"
        "    if len(selected) != 1 or selected[0].get('variant') != PILOT_RECOMMENDED_VARIANT:\n"
        "        fail(\n"
        "            f'Viewer manifest does not select one recommended export for {object_key!r}.'\n"
        "        )"
    ),
    1,
)
inspection = inspection.replace(
    "'label': entry['label'],\n        'instance_id': entry['instance_id'],",
    "'object_key': entry['object_key'],\n        'label': entry['label'],\n        'instance_id': entry['instance_id'],",
    1,
)
inspection = inspection.replace(
    "'target_labels': batch.get('target_labels'),\n    'resolved_instance_ids':",
    "'target_labels': batch.get('target_labels'),\n    'target_object_keys': batch.get('target_object_keys'),\n    'resolved_instance_ids':",
    1,
)
inspection = inspection.replace(
    "'label', 'instance_id', 'variant', 'gaussian_count',",
    "'object_key', 'label', 'instance_id', 'variant', 'gaussian_count',",
    1,
)
inspection = inspection.replace(
    "print(entry['label'], entry['instance_id'], entry['splat_path'])",
    "print(entry['object_key'], entry['label'], entry['instance_id'], entry['splat_path'])",
    1,
)
inspection_cell["source"] = source_lines(inspection)

patch_notebook(notebook)
patch_resume_export(notebook)
patch_interruption_recovery(notebook)
notebook.setdefault("metadata", {}).setdefault("room_model_project", {})[
    "object_splat_batch_version"
] = V12_VERSION
NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(f"Expanded {NOTEBOOK_PATH} to {V12_VERSION}")
