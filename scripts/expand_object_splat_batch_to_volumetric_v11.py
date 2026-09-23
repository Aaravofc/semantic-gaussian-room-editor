"""Expand the restored V9 bed pilot into a six-label volumetric-object batch."""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "object_splat_batch_pipeline.ipynb"
V9_VERSION = "2026-08-15-object-splat-ellipsoid-containment-pilot-v9"
V11_VERSION = "2026-08-18-object-splat-volumetric-batch-v11"
NUMPY_ABI_PATCH = "2026-08-21-numpy-abi-preflight-v1"


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
        raise RuntimeError(f"Expected one {cell_type} cell containing {needle!r}, found {len(matches)}")
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
version_match = re.search(r"BATCH_PIPELINE_VERSION\s*=\s*['\"]([^'\"]+)", config)
if version_match is None:
    raise RuntimeError("Could not read BATCH_PIPELINE_VERSION")
current_version = version_match.group(1)

if current_version == V11_VERSION:
    replacements = {
        "print('\n' + '=' * 80)": "print('\\n' + '=' * 80)",
        "print('\n' + '-' * 80)": "print('\\n' + '-' * 80)",
        "print('\nVolumetric-object batch complete:": "print('\\nVolumetric-object batch complete:",
        "print('\nViewer-selected object splats:": "print('\\nViewer-selected object splats:",
    }
    changed = False
    for cell in cells:
        source = source_text(cell)
        updated = source
        for old, new in replacements.items():
            updated = updated.replace(old, new)
        if updated != source:
            cell["source"] = source_lines(updated)
            changed = True

    config_cell = cell_containing(cells, "BATCH_PIPELINE_VERSION =")
    config = source_text(config_cell)
    updated_config = config.replace(
        "import cv2\nimport numpy as np\nfrom tqdm.auto import tqdm\n\n",
        "",
        1,
    )
    if "RUNTIME_COMPATIBILITY_PATCH =" not in updated_config:
        updated_config = updated_config.replace(
            f"BATCH_PIPELINE_VERSION = '{V11_VERSION}'\n",
            (
                f"BATCH_PIPELINE_VERSION = '{V11_VERSION}'\n"
                f"RUNTIME_COMPATIBILITY_PATCH = '{NUMPY_ABI_PATCH}'\n"
            ),
            1,
        )
    if updated_config != config:
        config_cell["source"] = source_lines(updated_config)
        changed = True

    dependency_cell = cell_containing(cells, "NERFSTUDIO_TESTED_VERSION")
    dependency = textwrap.dedent(
        '''
        import importlib.util
        from importlib.metadata import PackageNotFoundError, version as package_version

        NERFSTUDIO_TESTED_VERSION = '1.1.5'
        NUMPY_TESTED_VERSION = '1.26.4'


        def installed_version(package_name):
            try:
                return package_version(package_name)
            except PackageNotFoundError:
                return None


        numpy_version_before = installed_version('numpy')
        numpy_loaded_before_install = 'numpy' in sys.modules
        nerfstudio_version_before = installed_version('nerfstudio')
        needs_install = (
            numpy_version_before != NUMPY_TESTED_VERSION
            or nerfstudio_version_before != NERFSTUDIO_TESTED_VERSION
            or importlib.util.find_spec('plyfile') is None
            or shutil.which('ns-train') is None
        )

        if needs_install:
            print(
                'Installing the tested NumPy/Nerfstudio environment:',
                f'numpy=={NUMPY_TESTED_VERSION},',
                f'nerfstudio=={NERFSTUDIO_TESTED_VERSION}',
            )
            subprocess.run(
                [
                    sys.executable, '-m', 'pip', 'install', '-q',
                    '--upgrade-strategy', 'only-if-needed',
                    f'numpy=={NUMPY_TESTED_VERSION}',
                    'plyfile',
                    f'nerfstudio=={NERFSTUDIO_TESTED_VERSION}',
                ],
                check=True,
            )

        numpy_version_after = installed_version('numpy')
        nerfstudio_version_after = installed_version('nerfstudio')
        if numpy_version_after != NUMPY_TESTED_VERSION:
            raise RuntimeError(
                f'Expected numpy=={NUMPY_TESTED_VERSION}, found {numpy_version_after}. '
                'Use Runtime > Restart session, then rerun from the top.'
            )
        if nerfstudio_version_after != NERFSTUDIO_TESTED_VERSION:
            raise RuntimeError(
                f'Expected nerfstudio=={NERFSTUDIO_TESTED_VERSION}, '
                f'found {nerfstudio_version_after}.'
            )

        if (
            needs_install
            and numpy_loaded_before_install
            and numpy_version_before != numpy_version_after
        ):
            raise RuntimeError(
                'A compatible NumPy build was installed, but the old NumPy binary is already '
                'loaded in this Colab process. Use Runtime > Restart session, '
                'then rerun the notebook from the top. Do not continue this runtime.'
            )

        numpy_probe = subprocess.run(
            [
                sys.executable,
                '-c',
                (
                    'import numpy as np; '
                    'rng = np.random.default_rng(42); '
                    'assert rng.integers(0, 10) >= 0; '
                    'print(np.__version__)'
                ),
            ],
            capture_output=True,
            text=True,
        )
        if numpy_probe.returncode != 0:
            print('NumPy ABI probe stderr:')
            print(numpy_probe.stderr[-4000:])
            subprocess.run(
                [
                    sys.executable, '-m', 'pip', 'install', '-q', '--force-reinstall',
                    '--no-cache-dir', '--no-deps', f'numpy=={NUMPY_TESTED_VERSION}',
                ],
                check=True,
            )
            raise RuntimeError(
                'The NumPy installation was internally inconsistent and has been repaired on '
                'disk. Use Runtime > Restart session, then rerun from the top.'
            )

        import cv2
        import numpy as np
        from tqdm.auto import tqdm
        from plyfile import PlyData, PlyElement

        print('NumPy version:', np.__version__)
        print('NumPy ABI probe:', numpy_probe.stdout.strip())
        print('Nerfstudio version:', nerfstudio_version_after)
        print('ns-train:', shutil.which('ns-train'))
        print('ns-export:', shutil.which('ns-export'))
        if shutil.which('ns-train') is None:
            raise RuntimeError('ns-train is unavailable after install attempt.')
        if shutil.which('ns-export') is None:
            raise RuntimeError('ns-export is unavailable after install attempt.')

        if shutil.which('nvidia-smi') is not None:
            subprocess.run(['nvidia-smi'], check=False)
        else:
            print('WARNING: nvidia-smi was not found. Use Colab Runtime > Change runtime type > GPU before training.')

        try:
            import torch
            print('torch.cuda.is_available():', torch.cuda.is_available())
            if torch.cuda.is_available():
                print('CUDA device:', torch.cuda.get_device_name(0))
        except Exception as exc:
            print('WARNING: Could not import/check torch:', exc)
        '''
    ).strip() + "\n"
    if source_text(dependency_cell) != dependency:
        dependency_cell["source"] = source_lines(dependency)
        changed = True

    functions_cell = cell_containing(cells, "def estimate_sparse_spacing_numpy")
    functions = source_text(functions_cell)
    updated_functions = replace_function(
        functions,
        "estimate_sparse_spacing_numpy",
        '''
        def estimate_sparse_spacing_numpy(sparse_xyz, sample_limit=5000, chunk_size=256):
            if len(sparse_xyz) < 2:
                return 0.0
            sample_count = min(int(sample_limit), len(sparse_xyz))
            if sample_count == len(sparse_xyz):
                sample_index = np.arange(len(sparse_xyz), dtype=np.int64)
            else:
                # Uniform deterministic sampling avoids NumPy's optional random binary while
                # preserving coverage across the input ordering.
                sample_index = np.linspace(
                    0,
                    len(sparse_xyz) - 1,
                    num=sample_count,
                    dtype=np.int64,
                )
                sample_index = np.unique(sample_index)
                sample_count = len(sample_index)
            sample = np.asarray(sparse_xyz[sample_index], dtype=np.float32)
            nearest = np.full(sample_count, np.inf, dtype=np.float32)
            for start_index in range(0, sample_count, int(chunk_size)):
                end_index = min(start_index + int(chunk_size), sample_count)
                delta = sample[start_index:end_index, None, :] - sample[None, :, :]
                distance_sq = np.einsum('ijk,ijk->ij', delta, delta)
                row_index = np.arange(end_index - start_index)
                distance_sq[row_index, np.arange(start_index, end_index)] = np.inf
                nearest[start_index:end_index] = np.sqrt(np.min(distance_sq, axis=1))
            valid = nearest[np.isfinite(nearest) & (nearest > 0)]
            if not valid.size:
                return 0.0
            # A sample has wider spacing than the complete cloud. The square-root correction is
            # appropriate for the mostly surface-like sparse geometry in a room.
            sampling_correction = math.sqrt(sample_count / max(len(sparse_xyz), 1))
            return float(np.median(valid) * sampling_correction)
        ''',
    )
    updated_functions = replace_function(
        updated_functions,
        "robust_surface_plane_keep",
        '''
        def robust_surface_plane_keep(xyz, sigma, sparse_xyz, spacing):
            if sparse_xyz is None or len(sparse_xyz) < 20:
                return np.ones(len(xyz), dtype=bool), None
            threshold = max(float(CLEANUP_PLANE_SPACING_MULTIPLIER) * max(float(spacing or 0.0), 1e-6), 1e-4)
            best_inliers = None
            best_plane = None
            state = 42
            point_count = len(sparse_xyz)
            for _ in range(int(CLEANUP_PLANE_RANSAC_ITERATIONS)):
                # A small deterministic LCG supplies three distinct indices without importing
                # numpy.random. Reproducibility is useful for cleanup diagnostics as well.
                sample_index = []
                attempts = 0
                while len(sample_index) < 3 and attempts < 32:
                    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
                    candidate = int(state % point_count)
                    if candidate not in sample_index:
                        sample_index.append(candidate)
                    attempts += 1
                if len(sample_index) < 3:
                    continue
                sample = sparse_xyz[np.asarray(sample_index, dtype=np.int64)]
                normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
                norm = float(np.linalg.norm(normal))
                if norm < 1e-8:
                    continue
                normal = normal / norm
                offset = -float(np.dot(normal, sample[0]))
                residual = np.abs(sparse_xyz @ normal + offset)
                inliers = residual <= threshold
                if best_inliers is None or int(inliers.sum()) > int(best_inliers.sum()):
                    best_inliers = inliers
                    best_plane = (normal, offset)
            if best_inliers is None or int(best_inliers.sum()) < 10:
                return np.ones(len(xyz), dtype=bool), None
            inlier_xyz = sparse_xyz[best_inliers]
            center = np.mean(inlier_xyz, axis=0)
            _, _, vh = np.linalg.svd(inlier_xyz - center, full_matrices=False)
            normal = vh[-1]
            normal = normal / max(float(np.linalg.norm(normal)), 1e-8)
            offset = -float(np.dot(normal, center))
            distance = np.abs(xyz @ normal + offset)
            allowed = threshold + float(CLEANUP_GAUSSIAN_EXTENT_MULTIPLIER) * sigma
            keep = distance <= allowed
            return keep, {
                'normal': normal.astype(float).tolist(),
                'offset': offset,
                'threshold': threshold,
                'sparse_inliers': int(best_inliers.sum()),
                'sparse_total': int(len(sparse_xyz)),
            }
        ''',
    )
    if updated_functions != functions:
        functions_cell["source"] = source_lines(updated_functions)
        changed = True

    notebook["cells"][0]["source"] = source_lines(
        source_text(notebook["cells"][0]).replace(
            "Existing V9 and V10 Drive outputs are not overwritten\n",
            (
                "The dependency preflight pins a tested NumPy ABI before importing NumPy, and all\n"
                "geometry sampling is deterministic. Existing V9 and V10 Drive outputs are not overwritten\n"
            ),
            1,
        )
    )
    if changed:
        NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
        print(f"Applied V11 runtime compatibility repairs in {NOTEBOOK_PATH}")
    else:
        print(f"{NOTEBOOK_PATH} is already {V11_VERSION} with {NUMPY_ABI_PATCH}")
    raise SystemExit(0)
if current_version != V9_VERSION:
    raise RuntimeError(
        f"Notebook version {current_version!r} is unsupported; restore {V9_VERSION!r} first."
    )

cells[0]["source"] = source_lines(
    """# Batch Object Gaussian Splat Pipeline

This notebook runs the **V11 bounded-volumetric object batch** using the accepted V9 training
logic. It selects the most visible persistent instance for each configured label and trains the
recommended `ellipsoid_constrained` arm in the full-room coordinate system.

The default targets are `bed`, `chair`, `desk`, `table`, `pillow`, and `bag`. They deliberately
exclude wall, floor, ceiling, doors, windows, curtains, and other thin or structural surfaces.
Missing labels are reported and skipped; failures after a label becomes eligible remain explicit.

The V9 method retains masked RGB and alpha supervision, permissive multi-view sparse seeds, the
conservative visual-support grid, and oriented finite 2-sigma ellipsoid containment. It does not
perform post-export Gaussian deletion. Existing V9 and V10 Drive outputs are not overwritten
because this batch uses a new immutable ID.
"""
)

config = config.replace(V9_VERSION, V11_VERSION, 1)
old_target_block_start = config.index("# Phase 1 is deliberately restricted")
old_target_block_end = config.index("# The three arms share identical cameras")
new_target_block = """# V11 scales the accepted V9 method to bounded volumetric objects only.
VOLUMETRIC_BATCH_MODE = True
TARGET_LABELS = ['bed', 'chair', 'desk', 'table', 'pillow', 'bag']
TARGET_INSTANCE_IDS = {}  # Optional overrides, for example {'chair': 'chair_002'}.
INSTANCE_SELECTION_POLICY = 'most_visible_frames'
STRUCTURAL_OR_THIN_LABELS = {'wall', 'floor', 'ceiling', 'door', 'window', 'curtain'}
RESOLVED_INSTANCE_IDS = {}

"""
config = config[:old_target_block_start] + new_target_block + config[old_target_block_end:]
config = config.replace(
    "PILOT_VARIANTS_TO_RUN = ['masked_baseline', 'alpha_constrained', 'ellipsoid_constrained']",
    "PILOT_VARIANTS_TO_RUN = ['ellipsoid_constrained']  # Use all three only for an explicit ablation.",
    1,
)
config = config.replace(
    "BATCH_ID = 'bed_instance_ellipsoid_containment_pilot_v9'",
    "BATCH_ID = 'volumetric_objects_ellipsoid_batch_v11'",
    1,
)
config = config.replace("print('Pilot variants:',", "print('Containment variants:',", 1)
config_cell["source"] = source_lines(config)

utility_cell = cell_containing(cells, "def resolve_pilot_instance")
utility = source_text(utility_cell)
utility = replace_function(
    utility,
    "resolve_pilot_instance",
    '''
    def resolve_target_instance(label_norm, rows):
        candidates = summarize_instance_candidates(rows)
        if not candidates:
            return None, []
        configured = TARGET_INSTANCE_IDS.get(label_norm)
        if configured is not None:
            selected = next(
                (row for row in candidates if row['instance_id'] == str(configured)),
                None,
            )
            if selected is None:
                raise RuntimeError(
                    f'Configured instance {configured!r} for {label_norm!r} was not found. '
                    'Available: ' + ', '.join(row['instance_id'] for row in candidates)
                )
        else:
            selected = sorted(
                candidates,
                key=lambda row: (
                    -row['visible_frames'],
                    -row['mask_records'],
                    row['instance_id'],
                ),
            )[0]
        for row in candidates:
            row['selected'] = row['instance_id'] == selected['instance_id']
            row['selection_policy'] = INSTANCE_SELECTION_POLICY
        return selected['instance_id'], candidates
    ''',
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
        'Run the current segmentation.ipynb through Audit. First invalid file: '
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

label_audit_rows = []
label_mask_maps = {}
label_mask_source_rows = {}
for label in TARGET_LABELS:
    label_norm = normalize_label(label)
    _, all_detection_rows = collect_target_masks_from_detections(label_norm)
    selected_instance, candidate_rows = resolve_target_instance(label_norm, all_detection_rows)
    safe_write_csv(REPORT_DIR / label_norm / 'instance_candidates.csv', candidate_rows)
    if selected_instance:
        RESOLVED_INSTANCE_IDS[label_norm] = selected_instance
        detection_map, detection_rows = collect_target_masks_from_detections(
            label_norm,
            target_object_id=selected_instance,
        )
    else:
        detection_map, detection_rows = {}, []

    if ALLOW_FILENAME_MASK_FALLBACK:
        filename_map, filename_rows = collect_target_masks_by_filename(label_norm, frame_keys)
    else:
        filename_map, filename_rows = {}, []
    frame_to_masks = merge_mask_maps(detection_map, filename_map)
    label_mask_maps[label_norm] = frame_to_masks
    label_mask_source_rows[label_norm] = detection_rows + filename_rows
    positive_frames = len(frame_to_masks)
    mask_records = sum(len(values) for values in frame_to_masks.values())
    if not selected_instance:
        audit_status = 'missing_persistent_instance'
    elif positive_frames < MIN_POSITIVE_MASK_FRAMES:
        audit_status = 'too_few_positive_mask_frames'
    else:
        audit_status = 'ready'
    will_run = audit_status == 'ready' or not SKIP_LABELS_WITH_TOO_FEW_MASK_FRAMES
    label_audit_rows.append({
        'label': label_norm,
        'instance_id': selected_instance,
        'audit_status': audit_status,
        'positive_mask_frames': positive_frames,
        'mask_records': mask_records,
        'detection_json_records': len(detection_rows),
        'filename_scan_records': len(filename_rows),
        'will_run': bool(will_run and selected_instance),
    })
    safe_write_csv(
        REPORT_DIR / label_norm / 'target_mask_sources.csv',
        label_mask_source_rows[label_norm],
    )

safe_write_csv(REPORT_DIR / 'batch_label_audit.csv', label_audit_rows)
print('Tracked segmentation fingerprint:', SEGMENTATION_FINGERPRINT)
print('Source frames:', len(frames))
print('Missing images:', len(missing_images))
print('Resolved target instances:', RESOLVED_INSTANCE_IDS)
print('Label audit:')
print_rows(
    label_audit_rows,
    columns=[
        'label', 'instance_id', 'audit_status',
        'positive_mask_frames', 'mask_records', 'will_run',
    ],
    limit=50,
)
print('Audit CSV:', REPORT_DIR / 'batch_label_audit.csv')
"""
)

functions_cell = cell_containing(cells, "def dataset_fingerprint_for_label")
functions = source_text(functions_cell)
functions = replace_function(
    functions,
    "write_batch_manifest",
    '''
    def write_batch_manifest(entries, status_rows):
        viewer_entries = [entry for entry in entries if entry.get('recommended')]
        successful_labels = sorted({entry['label'] for entry in entries})
        viewer_labels = sorted({entry['label'] for entry in viewer_entries})
        if successful_labels != viewer_labels:
            raise RuntimeError(
                'Every successful label must have exactly one recommended viewer export. '
                f'Successful={successful_labels}; viewer={viewer_labels}'
            )
        for label in successful_labels:
            count = sum(1 for entry in viewer_entries if entry['label'] == label)
            if count != 1:
                raise RuntimeError(
                    f'Expected one recommended export for {label!r}, found {count}.'
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
                'target_labels': [normalize_label(label) for label in TARGET_LABELS],
                'resolved_instance_ids': dict(RESOLVED_INSTANCE_IDS),
                'instance_selection_policy': INSTANCE_SELECTION_POLICY,
                'recommended_variant': PILOT_RECOMMENDED_VARIANT,
                'variants_run': list(PILOT_VARIANTS_TO_RUN),
                'method_family': 'v9_ellipsoid_containment',
                'structural_or_thin_labels_excluded': sorted(STRUCTURAL_OR_THIN_LABELS),
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

run_cell = cell_containing(cells, "Containment pilot instance:")
run_cell["source"] = source_lines(
    """if not VOLUMETRIC_BATCH_MODE:
    fail('This notebook version is intentionally configured as a volumetric-object batch.')
normalized_targets = [normalize_label(label) for label in TARGET_LABELS]
if len(normalized_targets) != len(set(normalized_targets)):
    fail(f'Duplicate target labels are not allowed: {normalized_targets}')
invalid_targets = sorted(set(normalized_targets) & STRUCTURAL_OR_THIN_LABELS)
if invalid_targets:
    fail(
        'V11 uses V9 volumetric containment and cannot train structural/thin labels: '
        + ', '.join(invalid_targets)
    )
if PILOT_RECOMMENDED_VARIANT not in PILOT_VARIANTS_TO_RUN:
    fail('The recommended variant must be included in PILOT_VARIANTS_TO_RUN.')

audit_by_label = {row['label']: row for row in label_audit_rows}
eligible_labels = [
    label for label in normalized_targets
    if audit_by_label[label].get('will_run')
]
if MAX_LABELS_TO_RUN is not None:
    eligible_labels = eligible_labels[: int(MAX_LABELS_TO_RUN)]
if not eligible_labels:
    fail(
        'No target label has a persistent instance with enough positive mask frames. '
        f'Inspect {REPORT_DIR / "batch_label_audit.csv"}.'
    )

status_rows = []
variant_entries = []
for row in label_audit_rows:
    if row['label'] not in eligible_labels:
        status_rows.append({
            'label': row['label'],
            'instance_id': row.get('instance_id'),
            'variant': None,
            'method': None,
            'status': 'skipped',
            'stage': 'audit',
            'positive_mask_frames': row.get('positive_mask_frames', 0),
            'message': row.get('audit_status', 'not_selected'),
        })

for label_norm in eligible_labels:
    instance_id = RESOLVED_INSTANCE_IDS.get(label_norm)
    frame_to_masks = label_mask_maps.get(label_norm, {})
    positive_frames = len(frame_to_masks)
    print('\\n' + '=' * 80)
    print('Volumetric object:', label_norm, instance_id)
    print('Positive mask frames:', positive_frames)
    base_paths = label_paths(label_norm)
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
        logger.exception('Volumetric label setup failed: %s', label_norm)
        status_rows.append({
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
        print('Training:', label_norm, variant_name)
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
                'object_id': instance_id,
                'component_id': f'{instance_id}:{variant_name}',
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
            print('SUCCESS:', label_norm, variant_name, splat_path)
        except Exception as exc:
            logger.exception('Variant failed: %s/%s', label_norm, variant_name)
            status_rows.append({
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
        'label', 'instance_id', 'variant', 'method',
        'status', 'stage', 'splat_path', 'message',
    ],
    limit=100,
)
print('Viewer-selected variant per successful object:', PILOT_RECOMMENDED_VARIANT)
print('Batch manifest:', BATCH_MANIFEST_PATH)
print('Latest manifest:', LATEST_MANIFEST_PATH)
"""
)

inspection_cell = cell_containing(cells, "manifest = load_json(LATEST_MANIFEST_PATH)")
inspection_cell["source"] = source_lines(
    """manifest = load_json(LATEST_MANIFEST_PATH)
if manifest.get('version') != BATCH_PIPELINE_VERSION:
    fail(
        f'Manifest version mismatch: {manifest.get("version")!r} '
        f'!= {BATCH_PIPELINE_VERSION!r}'
    )
batch = manifest.get('volumetric_batch', {})
variants = manifest.get('ablation_variants', [])
viewer_objects = manifest.get('objects', [])
if not variants:
    fail('Manifest contains no successful volumetric-object exports.')
successful_labels = sorted({entry['label'] for entry in variants})
for label in successful_labels:
    selected = [entry for entry in viewer_objects if entry.get('label') == label]
    if len(selected) != 1 or selected[0].get('variant') != PILOT_RECOMMENDED_VARIANT:
        fail(f'Viewer manifest does not select one recommended export for {label!r}.')


def last_containment_log_values(path):
    result = {
        'center_projected_last': None,
        'total_center_projected': None,
        'ellipsoid_shrunk_last': None,
        'total_ellipsoid_shrunk': None,
        'ellipsoid_snapped_last': None,
        'boundary_inside_fraction': None,
    }
    path = Path(path) if path else None
    if path is None or not path.exists():
        return result
    matching = [
        line for line in path.read_text(errors='replace').splitlines()
        if line.startswith('[object-hull] step=')
    ]
    if not matching:
        return result
    values = dict(re.findall(r'([a-z_]+)=([-+0-9.eE]+)', matching[-1]))
    for key in result:
        if key not in values:
            continue
        result[key] = (
            float(values[key])
            if key == 'boundary_inside_fraction'
            else int(float(values[key]))
        )
    return result


inspection_rows = []
for entry in variants:
    splat_path = Path(entry['splat_path'])
    if not splat_path.exists() or splat_path.stat().st_size == 0:
        fail(f'Missing or empty volumetric-object splat: {splat_path}')
    ply = PlyData.read(str(splat_path))
    vertex = ply['vertex'].data
    if len(vertex) <= 0:
        fail(f'Volumetric-object splat has no Gaussians: {splat_path}')
    xyz = vertex_xyz(vertex).astype(np.float64)
    span = np.ptp(xyz, axis=0)
    center = np.mean(xyz, axis=0)
    sigma = gaussian_largest_sigma(vertex).astype(np.float64)
    alpha = gaussian_alpha_from_vertex(vertex)
    alpha = np.asarray(alpha, dtype=np.float64) if alpha is not None else np.asarray([])
    row = {
        'label': entry['label'],
        'instance_id': entry['instance_id'],
        'variant': entry['variant'],
        'method': entry['method'],
        'recommended': entry.get('recommended'),
        'gaussian_count': int(len(vertex)),
        'size_mb': splat_path.stat().st_size / (1024 ** 2),
        'span_x': float(span[0]),
        'span_y': float(span[1]),
        'span_z': float(span[2]),
        'mean_x': float(center[0]),
        'mean_y': float(center[1]),
        'mean_z': float(center[2]),
        'sigma_p50': float(np.percentile(sigma, 50)) if len(sigma) else None,
        'sigma_p90': float(np.percentile(sigma, 90)) if len(sigma) else None,
        'sigma_p99': float(np.percentile(sigma, 99)) if len(sigma) else None,
        'alpha_p50': float(np.percentile(alpha, 50)) if len(alpha) else None,
        'alpha_p90': float(np.percentile(alpha, 90)) if len(alpha) else None,
        'alpha_p99': float(np.percentile(alpha, 99)) if len(alpha) else None,
        'splat_path': str(splat_path),
    }
    row.update(last_containment_log_values(entry.get('train_log')))
    inspection_rows.append(row)

safe_write_csv(REPORT_DIR / 'volumetric_export_inspection.csv', inspection_rows)
write_json(REPORT_DIR / 'volumetric_export_inspection.json', {
    'schema': 'object_splat_volumetric_inspection.v1',
    'pipeline_version': BATCH_PIPELINE_VERSION,
    'batch_id': BATCH_ID,
    'batch_status': manifest.get('batch_status'),
    'target_labels': batch.get('target_labels'),
    'resolved_instance_ids': batch.get('resolved_instance_ids'),
    'post_export_cleanup_enabled': batch.get('post_export_cleanup_enabled'),
    'objects': inspection_rows,
})
print('Batch status:', manifest.get('batch_status'))
print('Resolved instances:', batch.get('resolved_instance_ids'))
print('Post-export cleanup enabled:', batch.get('post_export_cleanup_enabled'))
print_rows(
    inspection_rows,
    columns=[
        'label', 'instance_id', 'variant', 'gaussian_count',
        'span_x', 'span_y', 'span_z', 'total_center_projected',
        'total_ellipsoid_shrunk', 'boundary_inside_fraction',
    ],
    limit=100,
)
print('\\nViewer-selected object splats:')
for entry in viewer_objects:
    print(entry['label'], entry['instance_id'], entry['splat_path'])
print('Inspection CSV:', REPORT_DIR / 'volumetric_export_inspection.csv')
print('Inspection JSON:', REPORT_DIR / 'volumetric_export_inspection.json')
"""
)

metadata = notebook.setdefault("metadata", {}).setdefault("room_model_project", {})
metadata["object_splat_batch_version"] = V11_VERSION
metadata["object_splat_pipeline_version"] = V9_VERSION
metadata["containment_method_family"] = "v9_ellipsoid_containment"

NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(f"Expanded {NOTEBOOK_PATH} to {V11_VERSION}")
