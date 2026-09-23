"""Restore the active object-splat batch notebook to the V9 ellipsoid pilot."""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "object_splat_batch_pipeline.ipynb"
V9_VERSION = "2026-08-15-object-splat-ellipsoid-containment-pilot-v9"
V10_VERSION = "2026-08-16-object-splat-depth-visible-containment-pilot-v10"


def source_text(cell: dict) -> str:
    return "".join(cell.get("source", []))


def source_lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def code_cell_containing(cells: list[dict], needle: str) -> dict:
    matches = [
        cell
        for cell in cells
        if cell.get("cell_type") == "code" and needle in source_text(cell)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one code cell containing {needle!r}, found {len(matches)}")
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


def replace_function_range(source: str, first_name: str, next_name: str, replacement: str) -> str:
    pattern = re.compile(
        rf"^def {re.escape(first_name)}\([^\n]*\):\n.*?(?=^def {re.escape(next_name)}\()",
        flags=re.MULTILINE | re.DOTALL,
    )
    matches = list(pattern.finditer(source))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one function range {first_name!r} to {next_name!r}, found {len(matches)}"
        )
    replacement = textwrap.dedent(replacement).strip() + "\n\n\n"
    return source[: matches[0].start()] + replacement + source[matches[0].end() :]


notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
cells = notebook["cells"]
config_cell = code_cell_containing(cells, "BATCH_PIPELINE_VERSION =")
config = source_text(config_cell)
version_match = re.search(r"BATCH_PIPELINE_VERSION\s*=\s*['\"]([^'\"]+)", config)
if version_match is None:
    raise RuntimeError("Could not read BATCH_PIPELINE_VERSION")
current_version = version_match.group(1)

if current_version == V10_VERSION:
    config = config.replace(V10_VERSION, V9_VERSION, 1)
    config = config.replace(
        "BATCH_ID = 'bed_instance_depth_visible_containment_pilot_v10'",
        "BATCH_ID = 'bed_instance_ellipsoid_containment_pilot_v9'",
        1,
    )
    depth_start = config.index("# V10 rejects room points hidden behind the object")
    depth_end = config.index("# Conservative multi-view support-grid construction.")
    config = config[:depth_start] + config[depth_end:]
    config = config.replace("HULL_PADDING_FRACTION = 0.06", "HULL_PADDING_FRACTION = 0.12")
    config = config.replace(
        "HULL_PADDING_SPACING_MULTIPLIER = 2.0",
        "HULL_PADDING_SPACING_MULTIPLIER = 4.0",
    )
    config = config.replace("HULL_MASK_DILATION_PIXELS = 1", "HULL_MASK_DILATION_PIXELS = 3")
    config = config.replace("HULL_MIN_SUPPORT_RATIO = 0.50", "HULL_MIN_SUPPORT_RATIO = 0.35")
    config_cell["source"] = source_lines(config)

    sparse_cell = code_cell_containing(cells, "def resolve_ply_file_path_from_transforms")
    sparse_source = source_text(sparse_cell)
    sparse_source = replace_function_range(
        sparse_source,
        "project_points_with_depth",
        "write_filtered_vertex_ply",
        '''
        def project_points(points_chunk, frame, transforms_obj, image_shape):
            h, w = image_shape[:2]
            fx, fy, cx, cy = frame_intrinsics(frame, transforms_obj, image_shape)
            c2w = np.asarray(frame['transform_matrix'], dtype=np.float64)
            if c2w.shape != (4, 4):
                fail('Frame transform_matrix is not 4x4.')
            w2c = np.linalg.inv(c2w)
            homo = np.ones((len(points_chunk), 4), dtype=np.float64)
            homo[:, :3] = points_chunk.astype(np.float64)
            cam = (w2c @ homo.T).T[:, :3]
            if CAMERA_CONVENTION == 'nerfstudio_opengl':
                depth_z = -cam[:, 2]
                image_x = cam[:, 0]
                image_y = -cam[:, 1]
            elif CAMERA_CONVENTION == 'opencv':
                depth_z = cam[:, 2]
                image_x = cam[:, 0]
                image_y = cam[:, 1]
            else:
                fail(f'Unknown CAMERA_CONVENTION: {CAMERA_CONVENTION}')
            valid = np.isfinite(depth_z) & (depth_z > 0.05) & (depth_z < 50.0)
            u_float = np.full_like(depth_z, np.nan, dtype=np.float64)
            v_float = np.full_like(depth_z, np.nan, dtype=np.float64)
            u_float[valid] = fx * (image_x[valid] / depth_z[valid]) + cx
            v_float[valid] = fy * (image_y[valid] / depth_z[valid]) + cy
            valid &= np.isfinite(u_float) & np.isfinite(v_float)
            valid &= (u_float >= 0) & (u_float < w) & (v_float >= 0) & (v_float < h)
            idx = np.flatnonzero(valid)
            return (
                idx,
                np.floor(u_float[idx]).astype(np.int32),
                np.floor(v_float[idx]).astype(np.int32),
            )
        ''',
    )
    sparse_cell["source"] = source_lines(sparse_source)

    functions_cell = code_cell_containing(cells, "def dataset_fingerprint_for_label")
    functions = source_text(functions_cell)
    visibility_start = functions.index("def depth_frame_key")
    sparse_start = functions.index("def create_sparse_init_for_label")
    functions = functions[:visibility_start] + functions[sparse_start:]

    functions = replace_function(
        functions,
        "create_sparse_init_for_label",
        '''
        def create_sparse_init_for_label(label_norm, paths, mask_stats_path):
            if not CREATE_OBJECT_SPARSE_INIT or source_sparse_path is None or source_sparse_points is None:
                transforms_obj = load_json(paths['ns_data'] / 'transforms.json')
                if REMOVE_FULL_SCENE_PLY_INIT_IF_NO_OBJECT_INIT:
                    transforms_obj.pop('ply_file_path', None)
                    write_json(paths['ns_data'] / 'transforms.json', transforms_obj)
                return False, 0, str(source_sparse_path) if source_sparse_path else None, None

            rows = safe_read_csv(mask_stats_path)
            positive_rows = [
                row for row in rows
                if row.get('status') == 'included_positive_target_mask'
            ]
            frame_lookup = {frame_key_from_any(f.get('file_path')): f for f in frames}
            hit_counts = np.zeros(len(source_sparse_points), dtype=np.uint16)

            for row in tqdm(
                positive_rows,
                total=len(positive_rows),
                desc=f'Filtering sparse init by masks: {label_norm}',
            ):
                frame = frame_lookup.get(row.get('frame_key'))
                mask_path = Path(row.get('output_mask', ''))
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if frame is None or mask is None:
                    continue
                mask = mask > 0
                for start in range(0, len(source_sparse_points), SPARSE_POINT_CHUNK_SIZE):
                    end = min(start + SPARSE_POINT_CHUNK_SIZE, len(source_sparse_points))
                    local, u, v = project_points(
                        source_sparse_points[start:end], frame, transforms_data, mask.shape
                    )
                    if not len(local):
                        continue
                    hits = mask[v, u]
                    if hits.any():
                        hit_counts[start + local[hits]] += 1

            keep = hit_counts >= int(SPARSE_POINT_MIN_MASK_HITS)
            keep_indices = np.flatnonzero(keep)
            if len(keep_indices) > SPARSE_POINT_MAX_OUTPUT_POINTS:
                rank = np.lexsort((keep_indices, -hit_counts[keep_indices]))
                keep_indices = keep_indices[rank[:SPARSE_POINT_MAX_OUTPUT_POINTS]]
                keep = np.zeros(len(keep), dtype=bool)
                keep[keep_indices] = True
            keep_count = int(keep.sum())

            transforms_obj = load_json(paths['ns_data'] / 'transforms.json')
            if keep_count == 0:
                if REMOVE_FULL_SCENE_PLY_INIT_IF_NO_OBJECT_INIT:
                    transforms_obj.pop('ply_file_path', None)
                    write_json(paths['ns_data'] / 'transforms.json', transforms_obj)
                return False, 0, str(source_sparse_path), None

            object_sparse_path = paths['ns_data'] / f'{label_norm}_sparse_pc.ply'
            write_filtered_vertex_ply(source_sparse_path, object_sparse_path, keep)
            transforms_obj['ply_file_path'] = object_sparse_path.name
            write_json(paths['ns_data'] / 'transforms.json', transforms_obj)
            print('Object sparse PLY written:', object_sparse_path)
            print('Object sparse points:', keep_count, '/', len(source_sparse_points))
            return True, keep_count, str(source_sparse_path), str(object_sparse_path)
        ''',
    )

    functions = replace_function(
        functions,
        "build_visual_support_grid_for_label",
        '''
        def build_visual_support_grid_for_label(label_norm, paths, object_sparse_path):
            object_sparse_path = Path(object_sparse_path) if object_sparse_path else None
            if object_sparse_path is None or not object_sparse_path.exists():
                raise RuntimeError('The containment pilot requires a non-empty object sparse initializer.')
            raw_seed = ply_xyz(PlyData.read(str(object_sparse_path))['vertex'].data).astype(np.float64)
            if len(raw_seed) < 3:
                raise RuntimeError(f'Only {len(raw_seed)} object sparse seeds survived.')

            grid_fingerprint = stable_json_hash({
                'pipeline_version': BATCH_PIPELINE_VERSION,
                'dataset_fingerprint': load_json(paths['dataset_stage_manifest'])['fingerprint'],
                'object_sparse_inventory': file_inventory_hash([object_sparse_path]),
                'instance_id': RESOLVED_INSTANCE_IDS.get(label_norm),
                'frame_limit': HULL_POSITIVE_FRAME_LIMIT,
                'voxel_spacing_multiplier': HULL_VOXEL_SPACING_MULTIPLIER,
                'padding_fraction': HULL_PADDING_FRACTION,
                'mask_dilation_pixels': HULL_MASK_DILATION_PIXELS,
                'minimum_supporting_views': HULL_MIN_SUPPORTING_VIEWS,
                'minimum_support_ratio': HULL_MIN_SUPPORT_RATIO,
            })
            local_grid_path = paths['ns_data'] / f'{label_norm}_visual_support_grid.npz'
            persisted_grid_path = paths['report_dir'] / f'{label_norm}_visual_support_grid.npz'
            report_path = paths['report_dir'] / 'visual_support_grid_report.json'
            if report_path.exists() and persisted_grid_path.exists():
                previous = load_json(report_path)
                if previous.get('fingerprint') == grid_fingerprint:
                    shutil.copy2(persisted_grid_path, local_grid_path)
                    print('Reusing visual support grid:', persisted_grid_path)
                    return str(local_grid_path), previous

            center = np.median(raw_seed, axis=0)
            _, _, vh = np.linalg.svd(raw_seed - center, full_matrices=False)
            axes = vh.T
            if np.linalg.det(axes) < 0:
                axes[:, -1] *= -1.0
            seed_local = (raw_seed - center) @ axes
            local_min = seed_local.min(axis=0)
            local_max = seed_local.max(axis=0)
            span = np.maximum(local_max - local_min, 1e-6)
            spacing = estimate_sparse_spacing_numpy(raw_seed)
            if not np.isfinite(spacing) or spacing <= 0:
                raise RuntimeError('Could not estimate sparse seed spacing for support grid.')
            padding = np.maximum(
                span * float(HULL_PADDING_FRACTION),
                spacing * float(HULL_PADDING_SPACING_MULTIPLIER),
            )
            local_min -= padding
            local_max += padding
            expanded_span = np.maximum(local_max - local_min, 1e-6)
            voxel_size = max(
                spacing * float(HULL_VOXEL_SPACING_MULTIPLIER),
                float(expanded_span.max()) / HULL_MAX_AXIS_CELLS,
            )
            shape = np.ceil(expanded_span / voxel_size).astype(int) + 1
            total_voxels = int(np.prod(shape))
            if total_voxels > int(HULL_MAX_VOXELS):
                voxel_size *= (total_voxels / float(HULL_MAX_VOXELS)) ** (1.0 / 3.0)
                shape = np.ceil(expanded_span / voxel_size).astype(int) + 1
                total_voxels = int(np.prod(shape))
            if total_voxels > int(HULL_MAX_VOXELS) * 1.10:
                raise RuntimeError(f'Could not bound support-grid size: {shape.tolist()}')

            origin = center + local_min @ axes.T
            flat_index = np.arange(total_voxels, dtype=np.int64)
            grid_index = np.column_stack(
                np.unravel_index(flat_index, tuple(shape))
            ).astype(np.int32)
            candidate_raw = origin + (grid_index.astype(np.float64) * voxel_size) @ axes.T
            projected_count = np.zeros(total_voxels, dtype=np.uint16)
            hit_count = np.zeros(total_voxels, dtype=np.uint16)

            frame_lookup = {frame_key_from_any(frame.get('file_path')): frame for frame in frames}
            positive_rows = [
                row for row in safe_read_csv(paths['report_dir'] / 'masked_dataset_frame_report.csv')
                if row.get('status') == 'included_positive_target_mask'
            ]
            selected_rows = _sample_rows_evenly(positive_rows, HULL_POSITIVE_FRAME_LIMIT)
            for row in tqdm(selected_rows, desc=f'Visual support grid: {label_norm}'):
                frame = frame_lookup.get(row.get('frame_key'))
                mask_path = Path(row.get('output_mask', ''))
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if frame is None or mask is None:
                    continue
                mask = mask > 0
                if HULL_MASK_DILATION_PIXELS > 0:
                    radius = int(HULL_MASK_DILATION_PIXELS)
                    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
                    mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0
                for start in range(0, total_voxels, int(HULL_PROJECTION_CHUNK_SIZE)):
                    end = min(start + int(HULL_PROJECTION_CHUNK_SIZE), total_voxels)
                    local, u, v = project_points(
                        candidate_raw[start:end], frame, transforms_data, mask.shape
                    )
                    if not len(local):
                        continue
                    absolute = start + local
                    projected_count[absolute] += 1
                    hits = mask[v, u]
                    hit_count[absolute[hits]] += 1

            support_ratio = hit_count.astype(np.float32) / np.maximum(projected_count, 1)
            occupancy = (
                (hit_count >= int(HULL_MIN_SUPPORTING_VIEWS))
                & (support_ratio >= float(HULL_MIN_SUPPORT_RATIO))
            ).reshape(tuple(shape))

            seed_grid = np.rint(((raw_seed - origin) @ axes) / voxel_size).astype(int)
            seed_grid = np.clip(seed_grid, 0, shape - 1)
            protected_seed_cells = 0
            radius = int(HULL_SEED_PROTECTION_RADIUS_VOXELS)
            for coordinate in seed_grid:
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        for dz in range(-radius, radius + 1):
                            cell = coordinate + np.asarray([dx, dy, dz])
                            if np.any(cell < 0) or np.any(cell >= shape):
                                continue
                            cell_tuple = tuple(int(value) for value in cell)
                            if not occupancy[cell_tuple]:
                                protected_seed_cells += 1
                            occupancy[cell_tuple] = True

            occupied_count = int(occupancy.sum())
            if occupied_count < int(HULL_MIN_OCCUPIED_CELLS):
                raise RuntimeError(
                    f'Visual support retained only {occupied_count} cells; '
                    f'{HULL_MIN_OCCUPIED_CELLS} required.'
                )
            nearest = _nearest_occupied_indices_6n(occupancy)
            np.savez_compressed(
                local_grid_path,
                occupancy=occupancy.astype(np.uint8),
                nearest_occupied_index=nearest,
                origin=origin.astype(np.float32),
                axes=axes.astype(np.float32),
                voxel_size=np.asarray([voxel_size], dtype=np.float32),
            )
            shutil.copy2(local_grid_path, persisted_grid_path)
            report = {
                'schema': 'object_visual_support_grid.v1',
                'fingerprint': grid_fingerprint,
                'created_at': datetime.now().isoformat(timespec='seconds'),
                'label': label_norm,
                'instance_id': RESOLVED_INSTANCE_IDS.get(label_norm),
                'object_sparse_path': str(object_sparse_path),
                'object_sparse_point_count': int(len(raw_seed)),
                'selected_positive_frames': int(len(selected_rows)),
                'grid_shape': [int(value) for value in shape],
                'total_cells': total_voxels,
                'occupied_cells': occupied_count,
                'occupied_fraction': occupied_count / float(total_voxels),
                'protected_seed_cells_added': int(protected_seed_cells),
                'raw_sparse_spacing': float(spacing),
                'voxel_size': float(voxel_size),
                'persisted_grid_path': str(persisted_grid_path),
            }
            write_json(report_path, report)
            print('Visual support grid:', report)
            return str(local_grid_path), report
        ''',
    )
    functions_cell["source"] = source_lines(functions)

    run_cell = code_cell_containing(cells, "Containment pilot instance:")
    run_source = source_text(run_cell)
    run_source = run_source.replace(
        "            'sparse_visibility_report': str(base_paths['report_dir'] / 'sparse_visibility_summary.json'),\n",
        "",
        1,
    )
    run_cell["source"] = source_lines(run_source)

    cells[0]["source"] = source_lines(
        """# Batch Object Gaussian Splat Pipeline

This notebook runs a **single-instance bed ellipsoid-containment pilot V9**. It compares three
controlled arms in the same room coordinate system:

1. `masked_baseline`: stock Splatfacto trained on the tracked bed masks.
2. `alpha_constrained`: explicit rendered-opacity supervision and silhouette-gated densification.
3. `ellipsoid_constrained`: the alpha arm plus a conservative multi-view 3D support grid and
   oriented finite-ellipsoid containment.

The V9 arm samples 42 Fibonacci-sphere directions plus six principal axes on two radial shells.
During training it penalizes unsupported finite extent and, after refinement, applies only the
smallest necessary uniform scale correction when the finite ellipsoid crosses the support grid.

Only `ellipsoid_constrained` is placed in the viewer-facing `objects` list. Every raw export remains
available as an ablation. Post-export deletion remains disabled, and earlier Drive outputs are not
overwritten because V9 uses its own immutable batch ID.
"""
    )
elif current_version != V9_VERSION:
    raise RuntimeError(
        f"Notebook version {current_version!r} is unsupported; expected {V9_VERSION!r} or {V10_VERSION!r}."
    )

notebook.setdefault("metadata", {}).setdefault("room_model_project", {})[
    "object_splat_pipeline_version"
] = V9_VERSION
NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(f"Restored {NOTEBOOK_PATH} to {V9_VERSION}")
