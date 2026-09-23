"""Make the V12 all-object batch recoverable after an interrupted Colab runtime."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "object_splat_batch_pipeline.ipynb"
PATCH_VERSION = "2026-08-23-incremental-manifest-recovery-v1"


def source_text(cell: dict) -> str:
    return "".join(cell.get("source", []))


def source_lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def cell_containing(cells: list[dict], needle: str) -> dict:
    matches = [
        cell
        for cell in cells
        if cell.get("cell_type") == "code" and needle in source_text(cell)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one code cell containing {needle!r}, found {len(matches)}")
    return matches[0]


def patch_notebook(notebook: dict) -> bool:
    cells = notebook["cells"]
    changed = False

    config_cell = cell_containing(cells, "BATCH_PIPELINE_VERSION =")
    config = source_text(config_cell)
    if "INTERRUPTION_RECOVERY_PATCH" not in config:
        anchor = "RESUME_EXPORT_PATCH = '2026-08-23-resumable-local-checkpoint-export-v1'\n"
        replacement = (
            anchor
            + f"INTERRUPTION_RECOVERY_PATCH = '{PATCH_VERSION}'\n"
            + "CLEAN_LOCAL_OBJECT_CACHE_AFTER_SUCCESS = True\n"
        )
        if config.count(anchor) != 1:
            raise RuntimeError("Could not locate the resumed-export patch configuration anchor")
        config = config.replace(anchor, replacement, 1)
        config_cell["source"] = source_lines(config)
        changed = True

    functions_cell = cell_containing(cells, "def write_batch_manifest")
    functions = source_text(functions_cell)

    if "def load_recoverable_variant_entries" not in functions:
        old_signature = "def write_batch_manifest(entries, status_rows):\n"
        new_signature = "def write_batch_manifest(entries, status_rows, batch_complete=True):\n"
        if functions.count(old_signature) != 1:
            raise RuntimeError("Could not locate write_batch_manifest signature")
        functions = functions.replace(old_signature, new_signature, 1)

        old_status = """        'batch_status': (
            'partial_failure' if failed else
            'success_with_skips' if skipped else
            'success'
        ),
"""
        new_status = """        'batch_status': (
            'in_progress_checkpoint' if not batch_complete else
            'partial_failure' if failed else
            'success_with_skips' if skipped else
            'success'
        ),
"""
        if functions.count(old_status) != 1:
            raise RuntimeError("Could not locate batch-status block")
        functions = functions.replace(old_status, new_status, 1)

        old_metadata = """            'post_export_cleanup_enabled': False,
        },
"""
        new_metadata = """            'post_export_cleanup_enabled': False,
            'manifest_checkpointed_incrementally': True,
            'batch_complete': bool(batch_complete),
            'completed_object_count': len(successful_objects),
        },
"""
        if functions.count(old_metadata) != 1:
            raise RuntimeError("Could not locate volumetric manifest metadata block")
        functions = functions.replace(old_metadata, new_metadata, 1)

        helper_anchor = """    safe_write_csv(REPORT_DIR / 'batch_status.csv', status_rows)
    return manifest
"""
        helpers = helper_anchor + """


def load_recoverable_variant_entries(eligible_objects):
    recovered_entries = []
    recovered_status_rows = []
    seen_pairs = set()
    for target in eligible_objects:
        object_key = target['object_key']
        label_norm = target['label']
        instance_id = target['instance_id']
        base_paths = label_paths(object_key)
        for variant_name in PILOT_VARIANTS_TO_RUN:
            summary_path = (
                base_paths['report_dir'] / 'variants' / variant_name /
                'variant_summary.json'
            )
            if not summary_path.exists():
                continue
            try:
                entry = load_json(summary_path)
                expected_run_id = f'{object_key}_{variant_name}_{BATCH_ID}'
                if entry.get('status') != 'success':
                    raise RuntimeError('summary status is not success')
                if entry.get('object_key') != object_key:
                    raise RuntimeError('object key does not match the current target')
                if entry.get('label') != label_norm:
                    raise RuntimeError('semantic label does not match the current target')
                if str(entry.get('instance_id')) != str(instance_id):
                    raise RuntimeError('instance ID does not match the current target')
                if entry.get('variant') != variant_name:
                    raise RuntimeError('variant does not match the expected variant')
                if entry.get('run_id') != expected_run_id:
                    raise RuntimeError('run ID does not match this immutable batch')
                splat_path = Path(entry.get('splat_path', ''))
                try:
                    splat_path.relative_to(BATCH_ROOT)
                except ValueError as exc:
                    raise RuntimeError('splat path is outside the current batch root') from exc
                if not splat_path.is_file() or splat_path.stat().st_size <= 0:
                    raise RuntimeError('persisted splat is missing or empty')
            except Exception as exc:
                print('Ignoring invalid recovery summary:', summary_path, exc)
                continue

            pair = (object_key, variant_name)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            recovered_entries.append(entry)
            recovered_status_rows.append({
                'object_key': object_key,
                'label': label_norm,
                'instance_id': instance_id,
                'variant': variant_name,
                'method': entry.get('method'),
                'status': 'success',
                'stage': 'recovered_complete',
                'positive_mask_frames': entry.get('positive_mask_frame_count'),
                'object_sparse_point_count': entry.get('object_sparse_point_count'),
                'splat_path': str(splat_path),
                'message': f'Recovered from {summary_path}',
            })
    recovered_entries.sort(key=lambda row: (row['object_key'], row['variant']))
    recovered_status_rows.sort(key=lambda row: (row['object_key'], row['variant']))
    return recovered_entries, recovered_status_rows


def checkpoint_batch_progress(entries, status_rows, batch_complete=False):
    safe_write_csv(REPORT_DIR / 'volumetric_variant_status.csv', status_rows)
    if not entries:
        return None
    return write_batch_manifest(
        entries,
        status_rows,
        batch_complete=bool(batch_complete),
    )


def cleanup_local_object_cache(base_paths):
    if not CLEAN_LOCAL_OBJECT_CACHE_AFTER_SUCCESS:
        return
    local_root = Path(base_paths['local_root'])
    if local_root.exists():
        shutil.rmtree(local_root)
        print('Released completed local object cache:', local_root)
"""
        if functions.count(helper_anchor) != 1:
            raise RuntimeError("Could not locate write_batch_manifest return anchor")
        functions = functions.replace(helper_anchor, helpers, 1)
        changed = True

    export_cleanup_anchor = """    drive_files = sorted(str(path) for path in paths['export_dir'].glob('**/*') if path.is_file())
    return {
"""
    export_cleanup = """    drive_files = sorted(str(path) for path in paths['export_dir'].glob('**/*') if path.is_file())
    shutil.rmtree(local_export)
    return {
"""
    if export_cleanup_anchor in functions:
        functions = functions.replace(export_cleanup_anchor, export_cleanup, 1)
        changed = True
    elif "shutil.rmtree(local_export)\n    return {" not in functions:
        raise RuntimeError("Could not locate temporary export cleanup anchor")

    functions_cell["source"] = source_lines(functions)

    run_cell = cell_containing(cells, "if not VOLUMETRIC_BATCH_MODE")
    run = source_text(run_cell)
    if run != RUN_CELL_SOURCE:
        run_cell["source"] = source_lines(RUN_CELL_SOURCE)
        changed = True

    return changed


RUN_CELL_SOURCE = r"""if not VOLUMETRIC_BATCH_MODE:
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

recovered_entries, recovered_status_rows = load_recoverable_variant_entries(
    eligible_objects
)
variant_entries.extend(recovered_entries)
status_rows.extend(recovered_status_rows)
completed_pairs = {
    (entry['object_key'], entry['variant']) for entry in variant_entries
}
if recovered_entries:
    checkpoint_batch_progress(variant_entries, status_rows, batch_complete=False)
print('Recovered completed variants:', len(recovered_entries))

for target in eligible_objects:
    target_key = target['object_key']
    label_norm = target['label']
    instance_id = target['instance_id']
    pending_variants = [
        variant_name for variant_name in PILOT_VARIANTS_TO_RUN
        if (target_key, variant_name) not in completed_pairs
    ]
    base_paths = label_paths(target_key)
    if not pending_variants:
        print('\nReusing completed object exports:', target_key)
        cleanup_local_object_cache(base_paths)
        continue

    frame_to_masks = target_mask_maps.get(target_key, {})
    positive_frames = len(frame_to_masks)
    print('\n' + '=' * 80)
    print('Volumetric object:', target_key, label_norm, instance_id)
    print('Positive mask frames:', positive_frames)
    print('Pending variants:', pending_variants)
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
        checkpoint_batch_progress(variant_entries, status_rows, batch_complete=False)
        if STOP_ON_LABEL_FAILURE:
            raise
        continue

    for variant_name in pending_variants:
        print('\n' + '-' * 80)
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
            write_json(paths['report_dir'] / 'variant_summary.json', entry)
            variant_entries.append(entry)
            completed_pairs.add((target_key, variant_name))
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
            checkpoint_batch_progress(variant_entries, status_rows, batch_complete=False)
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
            checkpoint_batch_progress(variant_entries, status_rows, batch_complete=False)
            if STOP_ON_LABEL_FAILURE:
                raise

    object_complete = all(
        (target_key, variant_name) in completed_pairs
        for variant_name in PILOT_VARIANTS_TO_RUN
    )
    if object_complete:
        cleanup_local_object_cache(base_paths)

if not variant_entries:
    checkpoint_batch_progress(variant_entries, status_rows, batch_complete=False)
    fail('No volumetric-object variant completed successfully. Inspect the status CSV and logs.')

manifest = checkpoint_batch_progress(
    variant_entries,
    status_rows,
    batch_complete=True,
)
print('\nVolumetric-object batch complete:', manifest['batch_status'])
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


def main() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    if patch_notebook(notebook):
        NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
        print(f"Applied {PATCH_VERSION} to {NOTEBOOK_PATH}")
    else:
        print(f"{NOTEBOOK_PATH} already contains {PATCH_VERSION}")


if __name__ == "__main__":
    main()
