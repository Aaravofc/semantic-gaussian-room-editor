"""Upgrade the bed object-splat pilot with depth-aware visibility support."""

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


def replace_once(source: str, old: str, new: str, description: str) -> str:
    if new in source:
        return source
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {description} match, found {count}")
    return source.replace(old, new, 1)


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
config_cell = code_cell_containing(cells, "BATCH_PIPELINE_VERSION =")
config = source_text(config_cell)
version_match = re.search(r"BATCH_PIPELINE_VERSION\s*=\s*['\"]([^'\"]+)", config)
if version_match is None:
    raise RuntimeError("Could not read BATCH_PIPELINE_VERSION")
current_version = version_match.group(1)

if current_version == V9_VERSION:
    config = replace_once(config, V9_VERSION, V10_VERSION, "pipeline version")
    config = replace_once(
        config,
        "BATCH_ID = 'bed_instance_ellipsoid_containment_pilot_v9'",
        "BATCH_ID = 'bed_instance_depth_visible_containment_pilot_v10'",
        "batch ID",
    )
    depth_controls = """
# V10 rejects room points hidden behind the object before they can become protected seeds.
# Rendered room depth is authoritative where valid. A sparse-room z-buffer fills only depth holes.
DEPTH_DIR_CANDIDATES = [
    SCAN_DIR / 'depth_variants' / 'median_depth',
    SCAN_DIR / 'depth_variants' / 'expected_depth',
    SCAN_DIR / 'depths',
]
DEPTH_CONFIDENCE_DIR = SCAN_DIR / 'depth_confidence'
REQUIRE_RENDERED_DEPTH_FOR_PILOT = True
DEPTH_MIN_POSITIVE_FRAME_COVERAGE = 0.75
DEPTH_MIN_M = 0.05
DEPTH_MAX_M = 100.0
DEPTH_CONFIDENCE_THRESHOLD = 0.05
DEPTH_PATCH_RADIUS_PIXELS = 1
DEPTH_SCALE_MIN_OVERLAP_PIXELS = 24
DEPTH_SCALE_MAX_MEDIAN_RELATIVE_ERROR = 0.25
DEPTH_MIN_RELIABLE_POSITIVE_FRAME_FRACTION = 0.60
DEPTH_RELATIVE_TOLERANCE = 0.10
DEPTH_TOLERANCE_SPARSE_SPACING_MULTIPLIER = 2.5
SELF_ZBUFFER_RADIUS_PIXELS = 2
SELF_ZBUFFER_TOLERANCE_SPARSE_SPACING_MULTIPLIER = 2.0

# Sparse seeds require visibility-aware mask agreement and coherent 3D support.
SPARSE_POINT_MIN_VISIBLE_MASK_HITS = 2
SPARSE_POINT_MIN_VISIBLE_MASK_RATIO = 0.50
SPARSE_COMPONENT_VOXEL_SPACING_MULTIPLIER = 3.0
SPARSE_COMPONENT_MIN_POINTS = 24
SPARSE_COMPONENT_MIN_FRACTION = 0.01
SPARSE_COMPONENT_MIN_SUPPORT_FRAMES = 2
SPARSE_COMPONENT_MIN_RELATIVE_SCORE = 0.08
SPARSE_VISIBILITY_MIN_OUTPUT_POINTS = 100
SPARSE_VISIBILITY_PREVIEW_FRAME_LIMIT = 12
SPARSE_VISIBILITY_PREVIEW_MAX_POINTS = 5000

# Depth-carved support keeps near-surface object cells, treats points behind a surface as unknown,
# and rejects cells proven to be free space in front of an observed surface.
HULL_DEPTH_MIN_SURFACE_VIEWS = 2
HULL_DEPTH_MIN_MASK_SUPPORT_RATIO = 0.50
HULL_DEPTH_MAX_FREE_SPACE_RATIO = 0.20
HULL_DEPTH_SEED_MAX_FREE_SPACE_RATIO = 0.35
HULL_COMPONENT_MIN_CELLS = 8
HULL_UNSEEDED_COMPONENT_MIN_CELLS = 24
HULL_UNSEEDED_COMPONENT_MIN_HITS = 4
"""
    config = replace_once(
        config,
        "# Conservative multi-view support-grid construction. Positive visible masks define support;\n",
        depth_controls
        + "\n# Conservative multi-view support-grid construction. Positive visible masks define support;\n",
        "depth visibility controls insertion point",
    )
    config = config.replace("HULL_PADDING_FRACTION = 0.12", "HULL_PADDING_FRACTION = 0.06")
    config = config.replace(
        "HULL_PADDING_SPACING_MULTIPLIER = 4.0",
        "HULL_PADDING_SPACING_MULTIPLIER = 2.0",
    )
    config = config.replace("HULL_MASK_DILATION_PIXELS = 3", "HULL_MASK_DILATION_PIXELS = 1")
    config = config.replace("HULL_MIN_SUPPORT_RATIO = 0.35", "HULL_MIN_SUPPORT_RATIO = 0.50")
    config_cell["source"] = source_lines(config)

    sparse_cell = code_cell_containing(cells, "def resolve_ply_file_path_from_transforms")
    sparse_source = source_text(sparse_cell)
    sparse_source = replace_function(
        sparse_source,
        "project_points",
        '''
        def project_points_with_depth(points_chunk, frame, transforms_obj, image_shape):
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
            valid = np.isfinite(depth_z) & (depth_z > DEPTH_MIN_M) & (depth_z < DEPTH_MAX_M)
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
                depth_z[idx].astype(np.float32),
            )


        def project_points(points_chunk, frame, transforms_obj, image_shape):
            idx, u, v, _ = project_points_with_depth(
                points_chunk, frame, transforms_obj, image_shape
            )
            return idx, u, v
        ''',
    )
    sparse_cell["source"] = source_lines(sparse_source)

    functions_cell = code_cell_containing(cells, "def dataset_fingerprint_for_label")
    functions = source_text(functions_cell)
    visibility_helpers = '''
    def depth_frame_key(value):
        stem = Path(str(value)).stem
        for suffix in ('_median_depth', '_expected_depth', '_rendered_depth', '_depth'):
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
                break
        return frame_key_from_any(stem)


    def build_depth_index():
        index = {}
        source = {}
        for directory in DEPTH_DIR_CANDIDATES:
            if not directory.exists():
                continue
            for path in sorted(directory.iterdir()):
                if path.suffix.lower() not in {'.npy', '.npz', '.png', '.tif', '.tiff', '.exr'}:
                    continue
                key = depth_frame_key(path)
                if key not in index:
                    index[key] = path
                    source[key] = directory.name
        confidence = {}
        if DEPTH_CONFIDENCE_DIR.exists():
            for path in sorted(DEPTH_CONFIDENCE_DIR.iterdir()):
                if path.suffix.lower() in {'.npy', '.npz', '.png', '.tif', '.tiff', '.exr'}:
                    confidence[depth_frame_key(path)] = path
        return index, source, confidence


    DEPTH_INDEX, DEPTH_SOURCE, DEPTH_CONFIDENCE_INDEX = build_depth_index()


    def read_numeric_map(path):
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == '.npy':
            value = np.load(path)
        elif suffix == '.npz':
            payload = np.load(path)
            key = 'depth' if 'depth' in payload else payload.files[0]
            value = payload[key]
        else:
            raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise FileNotFoundError(f'Could not read numeric map: {path}')
            raw_dtype = raw.dtype
            value = raw.astype(np.float32)
            if raw_dtype.kind in {'u', 'i'} and np.nanmax(value) > 255:
                value = value / 1000.0
        value = np.asarray(value, dtype=np.float32)
        if value.ndim == 3:
            value = value[..., 0]
        return value


    def resize_numeric_map(value, image_shape):
        h, w = image_shape[:2]
        if value.shape[:2] == (h, w):
            return value.astype(np.float32, copy=False)
        return cv2.resize(value.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)


    def sample_minimum_patch(image, u, v, radius):
        h, w = image.shape[:2]
        result = np.full(len(u), np.inf, dtype=np.float32)
        for oy in range(-int(radius), int(radius) + 1):
            for ox in range(-int(radius), int(radius) + 1):
                uu = np.clip(u + ox, 0, w - 1)
                vv = np.clip(v + oy, 0, h - 1)
                result = np.minimum(result, image[vv, uu])
        return result


    def sample_best_depth_patch(depth, u, v, z, radius):
        h, w = depth.shape[:2]
        best_ref = np.full(len(u), np.nan, dtype=np.float32)
        best_delta = np.full(len(u), np.inf, dtype=np.float32)
        for oy in range(-int(radius), int(radius) + 1):
            for ox in range(-int(radius), int(radius) + 1):
                uu = np.clip(u + ox, 0, w - 1)
                vv = np.clip(v + oy, 0, h - 1)
                ref = depth[vv, uu]
                valid = np.isfinite(ref) & (ref > DEPTH_MIN_M) & (ref < DEPTH_MAX_M)
                delta = np.abs(z - ref)
                better = valid & (delta < best_delta)
                best_delta[better] = delta[better]
                best_ref[better] = ref[better]
        return best_ref, best_delta


    def source_sparse_zbuffer(frame, image_shape):
        h, w = image_shape[:2]
        zbuffer = np.full((h, w), np.inf, dtype=np.float32)
        projected = 0
        for start in range(0, len(source_sparse_points), SPARSE_POINT_CHUNK_SIZE):
            end = min(start + SPARSE_POINT_CHUNK_SIZE, len(source_sparse_points))
            local, u, v, z = project_points_with_depth(
                source_sparse_points[start:end], frame, transforms_data, image_shape
            )
            if not len(local):
                continue
            projected += len(local)
            np.minimum.at(zbuffer, (v, u), z)
        return zbuffer, projected


    def calibrate_rendered_depth(depth, sparse_zbuffer):
        overlap = (
            np.isfinite(depth)
            & (depth > DEPTH_MIN_M)
            & (depth < DEPTH_MAX_M)
            & np.isfinite(sparse_zbuffer)
        )
        raw_count = int(overlap.sum())
        if raw_count < DEPTH_SCALE_MIN_OVERLAP_PIXELS:
            return depth, {
                'scale': None,
                'overlap_pixels': raw_count,
                'median_relative_error': None,
                'reliable': False,
                'reason': 'insufficient_sparse_depth_overlap',
            }
        ratio = sparse_zbuffer[overlap] / np.maximum(depth[overlap], 1e-8)
        ratio = ratio[np.isfinite(ratio) & (ratio > 1e-4) & (ratio < 1e4)]
        if len(ratio) < DEPTH_SCALE_MIN_OVERLAP_PIXELS:
            return depth, {
                'scale': None,
                'overlap_pixels': int(len(ratio)),
                'median_relative_error': None,
                'reliable': False,
                'reason': 'invalid_depth_scale_ratios',
            }
        log_ratio = np.log(ratio)
        median = float(np.median(log_ratio))
        mad = float(np.median(np.abs(log_ratio - median)))
        if mad > 1e-8:
            keep = np.abs(log_ratio - median) <= 3.5 * 1.4826 * mad
            if int(keep.sum()) >= DEPTH_SCALE_MIN_OVERLAP_PIXELS:
                log_ratio = log_ratio[keep]
        scale = float(np.exp(np.median(log_ratio)))
        aligned = depth * scale
        residual = np.abs(sparse_zbuffer[overlap] - aligned[overlap])
        relative = residual / np.maximum(sparse_zbuffer[overlap], 1e-6)
        median_relative_error = float(np.median(relative))
        reliable = bool(median_relative_error <= DEPTH_SCALE_MAX_MEDIAN_RELATIVE_ERROR)
        return aligned, {
            'scale': scale,
            'overlap_pixels': raw_count,
            'median_relative_error': median_relative_error,
            'reliable': reliable,
            'reason': 'ok' if reliable else 'depth_alignment_residual_too_large',
        }


    def build_visibility_context(frame, image_shape):
        frame_key = frame_key_from_any(frame.get('file_path'))
        zbuffer, projected = source_sparse_zbuffer(frame, image_shape)
        depth_path = DEPTH_INDEX.get(frame_key)
        depth = None
        alignment = {
            'scale': None,
            'overlap_pixels': 0,
            'median_relative_error': None,
            'reliable': False,
            'reason': 'missing_rendered_depth',
        }
        if depth_path is not None:
            depth = resize_numeric_map(read_numeric_map(depth_path), image_shape)
            confidence_path = DEPTH_CONFIDENCE_INDEX.get(frame_key)
            if confidence_path is not None:
                confidence = resize_numeric_map(read_numeric_map(confidence_path), image_shape)
                depth = depth.copy()
                depth[confidence < DEPTH_CONFIDENCE_THRESHOLD] = 0.0
            depth[~np.isfinite(depth)] = 0.0
            depth, alignment = calibrate_rendered_depth(depth, zbuffer)
            if not alignment['reliable']:
                depth = None
        return {
            'frame_key': frame_key,
            'depth': depth,
            'depth_path': str(depth_path) if depth_path is not None else None,
            'depth_source': DEPTH_SOURCE.get(frame_key),
            'zbuffer': zbuffer,
            'sparse_projected': int(projected),
            'alignment': alignment,
        }


    def classify_projected_geometry(u, v, z, context, sparse_spacing):
        rendered_ref = np.full(len(z), np.nan, dtype=np.float32)
        rendered_near = np.zeros(len(z), dtype=bool)
        rendered_free = np.zeros(len(z), dtype=bool)
        rendered_behind = np.zeros(len(z), dtype=bool)
        if context['depth'] is not None:
            rendered_ref, rendered_delta = sample_best_depth_patch(
                context['depth'], u, v, z, DEPTH_PATCH_RADIUS_PIXELS
            )
            rendered_valid = np.isfinite(rendered_ref)
            tolerance = np.maximum(
                DEPTH_RELATIVE_TOLERANCE * rendered_ref,
                DEPTH_TOLERANCE_SPARSE_SPACING_MULTIPLIER * sparse_spacing,
            )
            signed = z - rendered_ref
            rendered_near = rendered_valid & (rendered_delta <= tolerance)
            rendered_free = rendered_valid & (signed < -tolerance)
            rendered_behind = rendered_valid & (signed > tolerance)
        else:
            rendered_valid = np.zeros(len(z), dtype=bool)

        self_ref = sample_minimum_patch(
            context['zbuffer'], u, v, SELF_ZBUFFER_RADIUS_PIXELS
        )
        self_valid = np.isfinite(self_ref)
        self_tolerance = np.maximum(
            DEPTH_RELATIVE_TOLERANCE * self_ref,
            SELF_ZBUFFER_TOLERANCE_SPARSE_SPACING_MULTIPLIER * sparse_spacing,
        )
        self_signed = z - self_ref
        self_near = self_valid & (np.abs(self_signed) <= self_tolerance)
        self_free = self_valid & (self_signed < -self_tolerance)
        self_behind = self_valid & (self_signed > self_tolerance)

        use_rendered = rendered_valid
        near = np.where(use_rendered, rendered_near, self_near)
        free = np.where(use_rendered, rendered_free, self_free)
        behind = np.where(use_rendered, rendered_behind, self_behind)
        geometry_valid = use_rendered | self_valid
        source = np.where(use_rendered, 1, np.where(self_valid, 2, 0)).astype(np.uint8)
        return {
            'near': near.astype(bool),
            'free': free.astype(bool),
            'behind': behind.astype(bool),
            'valid': geometry_valid.astype(bool),
            'source': source,
            'rendered_near': rendered_near,
            'self_near': self_near,
        }


    def sparse_point_component_labels(points, voxel_size):
        if not len(points):
            return np.asarray([], dtype=np.int32), []
        origin = points.min(axis=0)
        cells = np.floor((points - origin) / max(float(voxel_size), 1e-8)).astype(np.int32)
        cell_to_points = {}
        for index, cell in enumerate(map(tuple, cells)):
            cell_to_points.setdefault(cell, []).append(index)
        occupied = set(cell_to_points)
        cell_component = {}
        components = []
        neighbors = [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if (dx, dy, dz) != (0, 0, 0)
        ]
        for start in sorted(occupied):
            if start in cell_component:
                continue
            component_id = len(components)
            queue = deque([start])
            cell_component[start] = component_id
            component_cells = []
            while queue:
                cell = queue.popleft()
                component_cells.append(cell)
                for offset in neighbors:
                    neighbor = tuple(cell[i] + offset[i] for i in range(3))
                    if neighbor in occupied and neighbor not in cell_component:
                        cell_component[neighbor] = component_id
                        queue.append(neighbor)
            components.append(component_cells)
        labels = np.asarray([cell_component[tuple(cell)] for cell in cells], dtype=np.int32)
        return labels, components


    def volume_component_labels(occupancy):
        labels = np.full(occupancy.shape, -1, dtype=np.int32)
        components = []
        neighbors = [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if (dx, dy, dz) != (0, 0, 0)
        ]
        for start_array in np.argwhere(occupancy):
            start = tuple(int(value) for value in start_array)
            if labels[start] >= 0:
                continue
            component_id = len(components)
            queue = deque([start])
            labels[start] = component_id
            cells = []
            while queue:
                cell = queue.popleft()
                cells.append(cell)
                for offset in neighbors:
                    neighbor = tuple(cell[i] + offset[i] for i in range(3))
                    if any(neighbor[i] < 0 or neighbor[i] >= occupancy.shape[i] for i in range(3)):
                        continue
                    if occupancy[neighbor] and labels[neighbor] < 0:
                        labels[neighbor] = component_id
                        queue.append(neighbor)
            components.append(cells)
        return labels, components


    def write_sparse_visibility_preview(path, image, mask, u, v, visible, rng):
        preview = image.copy()
        contour_mask = (mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(preview, contours, -1, (0, 255, 255), 2)
        indices = np.arange(len(u))
        if len(indices) > SPARSE_VISIBILITY_PREVIEW_MAX_POINTS:
            indices = rng.choice(
                indices, size=SPARSE_VISIBILITY_PREVIEW_MAX_POINTS, replace=False
            )
        for local in indices:
            color = (0, 220, 0) if visible[local] else (0, 0, 255)
            cv2.circle(preview, (int(u[local]), int(v[local])), 1, color, -1)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), preview)


    def validate_depth_coverage(positive_rows):
        matched = [
            row for row in positive_rows if row.get('frame_key') in DEPTH_INDEX
        ]
        fraction = len(matched) / max(len(positive_rows), 1)
        report = {
            'positive_mask_frames': int(len(positive_rows)),
            'matched_depth_frames': int(len(matched)),
            'depth_coverage_fraction': float(fraction),
            'minimum_required_fraction': float(DEPTH_MIN_POSITIVE_FRAME_COVERAGE),
            'depth_directories': [str(path) for path in DEPTH_DIR_CANDIDATES],
        }
        if REQUIRE_RENDERED_DEPTH_FOR_PILOT and fraction < DEPTH_MIN_POSITIVE_FRAME_COVERAGE:
            raise RuntimeError(
                'Depth-visible V10 requires rendered room depth for at least '
                f'{DEPTH_MIN_POSITIVE_FRAME_COVERAGE:.0%} of positive mask frames; found '
                f'{len(matched)}/{len(positive_rows)} ({fraction:.1%}). In reconstruction.ipynb, '
                'set RUN_OPTIONAL_DEPTH_RENDERING=True and run sections 24-25 before rerunning V10.'
            )
        return report
    '''
    insertion_anchor = "def create_sparse_init_for_label(label_norm, paths, mask_stats_path):\n"
    if insertion_anchor not in functions:
        raise RuntimeError("Could not find sparse initializer insertion point")
    functions = functions.replace(
        insertion_anchor,
        textwrap.dedent(visibility_helpers).strip() + "\n\n\n" + insertion_anchor,
        1,
    )
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
            positive_rows = [row for row in rows if row.get('status') == 'included_positive_target_mask']
            coverage = validate_depth_coverage(positive_rows)
            frame_lookup = {frame_key_from_any(f.get('file_path')): f for f in frames}
            sparse_spacing = estimate_sparse_spacing_numpy(source_sparse_points)
            if not np.isfinite(sparse_spacing) or sparse_spacing <= 0:
                raise RuntimeError('Could not estimate source sparse-point spacing.')

            point_count = len(source_sparse_points)
            mask_hits = np.zeros(point_count, dtype=np.uint16)
            visible_hits = np.zeros(point_count, dtype=np.uint16)
            rendered_hits = np.zeros(point_count, dtype=np.uint16)
            self_hits = np.zeros(point_count, dtype=np.uint16)
            occluded_hits = np.zeros(point_count, dtype=np.uint16)
            frame_visible_indices = []
            frame_reports = []
            preview_dir = paths['report_dir'] / 'seed_visibility_previews'
            preview_rng = np.random.default_rng(42)

            for frame_number, row in enumerate(
                tqdm(positive_rows, total=len(positive_rows), desc=f'Visible sparse init: {label_norm}')
            ):
                frame_key = row['frame_key']
                frame = frame_lookup.get(frame_key)
                if frame is None:
                    raise RuntimeError(f'Missing transform frame for positive mask: {frame_key}')
                image_path = shared_cached_image_path(frame, 1)
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f'Could not read sparse visibility image: {image_path}')
                mask_path = Path(row['output_mask'])
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise RuntimeError(f'Could not read sparse visibility mask: {mask_path}')
                if mask.shape[:2] != image.shape[:2]:
                    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask = mask > 0
                context = build_visibility_context(frame, image.shape[:2])

                frame_projected = 0
                frame_mask_hits = 0
                frame_visible_hits = 0
                frame_occluded_hits = 0
                preview_u = []
                preview_v = []
                preview_visible = []
                visible_global_parts = []
                for start in range(0, point_count, SPARSE_POINT_CHUNK_SIZE):
                    end = min(start + SPARSE_POINT_CHUNK_SIZE, point_count)
                    local, u, v, z = project_points_with_depth(
                        source_sparse_points[start:end], frame, transforms_data, image.shape[:2]
                    )
                    if not len(local):
                        continue
                    global_index = start + local
                    in_mask = mask[v, u]
                    if not in_mask.any():
                        continue
                    global_index = global_index[in_mask]
                    u = u[in_mask]
                    v = v[in_mask]
                    z = z[in_mask]
                    geometry = classify_projected_geometry(u, v, z, context, sparse_spacing)
                    visible = geometry['near']
                    frame_projected += int(len(local))
                    frame_mask_hits += int(len(global_index))
                    frame_visible_hits += int(visible.sum())
                    frame_occluded_hits += int((geometry['behind'] | geometry['free']).sum())
                    mask_hits[global_index] += 1
                    visible_hits[global_index[visible]] += 1
                    rendered_hits[global_index[visible & (geometry['source'] == 1)]] += 1
                    self_hits[global_index[visible & (geometry['source'] == 2)]] += 1
                    rejected = ~visible & geometry['valid']
                    occluded_hits[global_index[rejected]] += 1
                    if visible.any():
                        visible_global_parts.append(global_index[visible])
                    if frame_number < SPARSE_VISIBILITY_PREVIEW_FRAME_LIMIT:
                        preview_u.append(u)
                        preview_v.append(v)
                        preview_visible.append(visible)

                visible_global = (
                    np.unique(np.concatenate(visible_global_parts))
                    if visible_global_parts else np.asarray([], dtype=np.int64)
                )
                frame_visible_indices.append((frame_key, visible_global))
                alignment = context['alignment']
                frame_reports.append({
                    'frame_key': frame_key,
                    'depth_path': context['depth_path'],
                    'depth_source': context['depth_source'],
                    'depth_scale': alignment.get('scale'),
                    'depth_overlap_pixels': alignment.get('overlap_pixels'),
                    'depth_median_relative_error': alignment.get('median_relative_error'),
                    'depth_reliable': alignment.get('reliable'),
                    'depth_reason': alignment.get('reason'),
                    'source_sparse_projected': context['sparse_projected'],
                    'mask_hit_points': frame_mask_hits,
                    'visible_mask_hit_points': frame_visible_hits,
                    'rejected_geometry_points': frame_occluded_hits,
                })
                if preview_u:
                    write_sparse_visibility_preview(
                        preview_dir / f'{frame_number:03d}_{frame_key}.jpg',
                        image,
                        mask,
                        np.concatenate(preview_u),
                        np.concatenate(preview_v),
                        np.concatenate(preview_visible),
                        preview_rng,
                    )

            reliable_depth_frames = sum(bool(row.get('depth_reliable')) for row in frame_reports)
            reliable_depth_fraction = reliable_depth_frames / max(len(frame_reports), 1)
            if (
                REQUIRE_RENDERED_DEPTH_FOR_PILOT
                and reliable_depth_fraction < DEPTH_MIN_RELIABLE_POSITIVE_FRAME_FRACTION
            ):
                safe_write_csv(
                    paths['report_dir'] / 'sparse_visibility_frame_report.csv', frame_reports
                )
                raise RuntimeError(
                    'Rendered depth calibration succeeded for only '
                    f'{reliable_depth_frames}/{len(frame_reports)} positive frames '
                    f'({reliable_depth_fraction:.1%}); '
                    f'{DEPTH_MIN_RELIABLE_POSITIVE_FRAME_FRACTION:.0%} required. Inspect '
                    'sparse_visibility_frame_report.csv. Do not loosen seed visibility thresholds '
                    'until the depth/camera coordinate mismatch is resolved.'
                )

            visible_ratio = visible_hits.astype(np.float32) / np.maximum(mask_hits, 1)
            mask_only_keep = mask_hits >= int(SPARSE_POINT_MIN_MASK_HITS)
            visible_candidate = (
                (visible_hits >= int(SPARSE_POINT_MIN_VISIBLE_MASK_HITS))
                & (visible_ratio >= float(SPARSE_POINT_MIN_VISIBLE_MASK_RATIO))
            )
            candidate_indices = np.flatnonzero(visible_candidate)
            if len(candidate_indices) < SPARSE_VISIBILITY_MIN_OUTPUT_POINTS:
                raise RuntimeError(
                    f'Visibility filtering retained only {len(candidate_indices)} sparse seeds; '
                    f'{SPARSE_VISIBILITY_MIN_OUTPUT_POINTS} required. Inspect seed_visibility_previews '
                    'and sparse_visibility_frame_report.csv before changing thresholds.'
                )

            component_voxel = sparse_spacing * SPARSE_COMPONENT_VOXEL_SPACING_MULTIPLIER
            local_labels, _ = sparse_point_component_labels(
                source_sparse_points[candidate_indices], component_voxel
            )
            component_ids = sorted(set(int(value) for value in local_labels))
            component_rows = []
            for component_id in component_ids:
                local_member = np.flatnonzero(local_labels == component_id)
                global_member = candidate_indices[local_member]
                member_set = set(int(value) for value in global_member)
                supporting_frames = [
                    frame_key
                    for frame_key, indices in frame_visible_indices
                    if any(int(value) in member_set for value in indices)
                ]
                xyz = source_sparse_points[global_member]
                mean_hits = float(np.mean(visible_hits[global_member]))
                score = float(len(global_member) * max(mean_hits, 1e-6))
                component_rows.append({
                    'component_id': int(component_id),
                    'point_count': int(len(global_member)),
                    'point_fraction': float(len(global_member) / max(len(candidate_indices), 1)),
                    'support_frame_count': int(len(supporting_frames)),
                    'support_frames': ';'.join(supporting_frames),
                    'mean_visible_mask_hits': mean_hits,
                    'score': score,
                    'span_x': float(np.ptp(xyz[:, 0])),
                    'span_y': float(np.ptp(xyz[:, 1])),
                    'span_z': float(np.ptp(xyz[:, 2])),
                    'kept': False,
                })
            primary = max(component_rows, key=lambda row: row['score'])
            for row in component_rows:
                row['kept'] = bool(
                    row['component_id'] == primary['component_id']
                    or (
                        row['point_count'] >= SPARSE_COMPONENT_MIN_POINTS
                        and row['point_fraction'] >= SPARSE_COMPONENT_MIN_FRACTION
                        and row['support_frame_count'] >= SPARSE_COMPONENT_MIN_SUPPORT_FRAMES
                        and row['score'] >= primary['score'] * SPARSE_COMPONENT_MIN_RELATIVE_SCORE
                    )
                )
            kept_components = {row['component_id'] for row in component_rows if row['kept']}
            local_keep = np.asarray(
                [int(value) in kept_components for value in local_labels], dtype=bool
            )
            final_indices = candidate_indices[local_keep]
            if len(final_indices) > SPARSE_POINT_MAX_OUTPUT_POINTS:
                rank = np.lexsort((final_indices, -visible_hits[final_indices]))
                final_indices = final_indices[rank[:SPARSE_POINT_MAX_OUTPUT_POINTS]]
            final_keep = np.zeros(point_count, dtype=bool)
            final_keep[final_indices] = True
            if int(final_keep.sum()) < SPARSE_VISIBILITY_MIN_OUTPUT_POINTS:
                raise RuntimeError('Connected-component filtering removed too many visible sparse seeds.')

            report_dir = paths['report_dir']
            report_dir.mkdir(parents=True, exist_ok=True)
            write_filtered_vertex_ply(
                source_sparse_path, report_dir / f'{label_norm}_sparse_mask_hits_only.ply', mask_only_keep
            )
            write_filtered_vertex_ply(
                source_sparse_path,
                report_dir / f'{label_norm}_sparse_depth_visible_precomponent.ply',
                visible_candidate,
            )
            write_filtered_vertex_ply(
                source_sparse_path, report_dir / f'{label_norm}_sparse_depth_visible_final.ply', final_keep
            )
            safe_write_csv(report_dir / 'sparse_visibility_frame_report.csv', frame_reports)
            safe_write_csv(report_dir / 'sparse_visibility_component_report.csv', component_rows)
            np.savez_compressed(
                report_dir / 'sparse_visibility_evidence.npz',
                mask_hits=mask_hits,
                visible_hits=visible_hits,
                rendered_hits=rendered_hits,
                self_zbuffer_hits=self_hits,
                rejected_geometry_hits=occluded_hits,
                selected=final_keep.astype(np.uint8),
            )
            summary = {
                'schema': 'object_sparse_visibility.v1',
                'pipeline_version': BATCH_PIPELINE_VERSION,
                'label': label_norm,
                'instance_id': RESOLVED_INSTANCE_IDS.get(label_norm),
                'source_sparse_points': int(point_count),
                'mask_only_seed_points': int(mask_only_keep.sum()),
                'depth_visible_precomponent_points': int(visible_candidate.sum()),
                'final_seed_points': int(final_keep.sum()),
                'source_sparse_spacing': float(sparse_spacing),
                'component_voxel_size': float(component_voxel),
                'kept_component_ids': sorted(int(value) for value in kept_components),
                'depth_coverage': coverage,
                'reliable_depth_frames': int(reliable_depth_frames),
                'reliable_depth_fraction': float(reliable_depth_fraction),
                'frame_report': str(report_dir / 'sparse_visibility_frame_report.csv'),
                'component_report': str(report_dir / 'sparse_visibility_component_report.csv'),
                'evidence_npz': str(report_dir / 'sparse_visibility_evidence.npz'),
                'preview_dir': str(preview_dir),
            }
            write_json(report_dir / 'sparse_visibility_summary.json', summary)

            transforms_obj = load_json(paths['ns_data'] / 'transforms.json')
            object_sparse_path = paths['ns_data'] / f'{label_norm}_sparse_pc.ply'
            write_filtered_vertex_ply(source_sparse_path, object_sparse_path, final_keep)
            transforms_obj['ply_file_path'] = object_sparse_path.name
            transforms_obj['object_sparse_visibility_report'] = str(
                report_dir / 'sparse_visibility_summary.json'
            )
            write_json(paths['ns_data'] / 'transforms.json', transforms_obj)
            print('Sparse visibility summary:', summary)
            return True, int(final_keep.sum()), str(source_sparse_path), str(object_sparse_path)
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
            if len(raw_seed) < SPARSE_VISIBILITY_MIN_OUTPUT_POINTS:
                raise RuntimeError(f'Only {len(raw_seed)} visibility-tested object seeds survived.')

            grid_fingerprint = stable_json_hash({
                'pipeline_version': BATCH_PIPELINE_VERSION,
                'dataset_fingerprint': load_json(paths['dataset_stage_manifest'])['fingerprint'],
                'object_sparse_inventory': file_inventory_hash([object_sparse_path]),
                'instance_id': RESOLVED_INSTANCE_IDS.get(label_norm),
                'depth_inventory': file_inventory_hash(list(DEPTH_INDEX.values())),
                'frame_limit': HULL_POSITIVE_FRAME_LIMIT,
                'voxel_spacing_multiplier': HULL_VOXEL_SPACING_MULTIPLIER,
                'padding_fraction': HULL_PADDING_FRACTION,
                'mask_dilation_pixels': HULL_MASK_DILATION_PIXELS,
                'minimum_surface_views': HULL_DEPTH_MIN_SURFACE_VIEWS,
                'minimum_mask_support_ratio': HULL_DEPTH_MIN_MASK_SUPPORT_RATIO,
                'maximum_free_space_ratio': HULL_DEPTH_MAX_FREE_SPACE_RATIO,
            })
            local_grid_path = paths['ns_data'] / f'{label_norm}_visual_support_grid.npz'
            persisted_grid_path = paths['report_dir'] / f'{label_norm}_visual_support_grid.npz'
            report_path = paths['report_dir'] / 'visual_support_grid_report.json'
            if report_path.exists() and persisted_grid_path.exists():
                previous = load_json(report_path)
                if previous.get('fingerprint') == grid_fingerprint:
                    shutil.copy2(persisted_grid_path, local_grid_path)
                    print('Reusing depth-carved support grid:', persisted_grid_path)
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
                raise RuntimeError('Could not estimate visible seed spacing for support grid.')
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
            grid_index = np.column_stack(np.unravel_index(flat_index, tuple(shape))).astype(np.int32)
            candidate_raw = origin + (grid_index.astype(np.float64) * voxel_size) @ axes.T
            geometry_count = np.zeros(total_voxels, dtype=np.uint16)
            surface_count = np.zeros(total_voxels, dtype=np.uint16)
            hit_count = np.zeros(total_voxels, dtype=np.uint16)
            free_count = np.zeros(total_voxels, dtype=np.uint16)
            behind_count = np.zeros(total_voxels, dtype=np.uint16)

            frame_lookup = {frame_key_from_any(frame.get('file_path')): frame for frame in frames}
            positive_rows = [
                row for row in safe_read_csv(paths['report_dir'] / 'masked_dataset_frame_report.csv')
                if row.get('status') == 'included_positive_target_mask'
            ]
            validate_depth_coverage(positive_rows)
            selected_rows = _sample_rows_evenly(positive_rows, HULL_POSITIVE_FRAME_LIMIT)
            frame_report = []
            for row in tqdm(selected_rows, desc=f'Depth-carved support: {label_norm}'):
                frame = frame_lookup.get(row.get('frame_key'))
                mask_path = Path(row.get('output_mask', ''))
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if frame is None or mask is None:
                    raise RuntimeError(f'Unreadable support frame or mask: {row.get("frame_key")}')
                mask = mask > 0
                if HULL_MASK_DILATION_PIXELS > 0:
                    radius = int(HULL_MASK_DILATION_PIXELS)
                    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
                    mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0
                context = build_visibility_context(frame, mask.shape)
                frame_geometry = frame_surface = frame_hits = frame_free = frame_behind = 0
                for start in range(0, total_voxels, int(HULL_PROJECTION_CHUNK_SIZE)):
                    end = min(start + int(HULL_PROJECTION_CHUNK_SIZE), total_voxels)
                    local, u, v, z = project_points_with_depth(
                        candidate_raw[start:end], frame, transforms_data, mask.shape
                    )
                    if not len(local):
                        continue
                    absolute = start + local
                    geometry = classify_projected_geometry(u, v, z, context, spacing)
                    valid = geometry['valid']
                    near = geometry['near']
                    free = geometry['free']
                    behind = geometry['behind']
                    supported = near & mask[v, u]
                    geometry_count[absolute[valid]] += 1
                    surface_count[absolute[near]] += 1
                    hit_count[absolute[supported]] += 1
                    free_count[absolute[free]] += 1
                    behind_count[absolute[behind]] += 1
                    frame_geometry += int(valid.sum())
                    frame_surface += int(near.sum())
                    frame_hits += int(supported.sum())
                    frame_free += int(free.sum())
                    frame_behind += int(behind.sum())
                frame_report.append({
                    'frame_key': row.get('frame_key'),
                    'depth_path': context['depth_path'],
                    'depth_scale': context['alignment'].get('scale'),
                    'depth_reliable': context['alignment'].get('reliable'),
                    'geometry_evaluated_cells': frame_geometry,
                    'near_surface_cells': frame_surface,
                    'mask_supported_surface_cells': frame_hits,
                    'free_space_cells': frame_free,
                    'occluded_behind_surface_cells': frame_behind,
                })

            support_ratio = hit_count.astype(np.float32) / np.maximum(surface_count, 1)
            free_ratio = free_count.astype(np.float32) / np.maximum(geometry_count, 1)
            occupancy = (
                (hit_count >= int(HULL_DEPTH_MIN_SURFACE_VIEWS))
                & (support_ratio >= float(HULL_DEPTH_MIN_MASK_SUPPORT_RATIO))
                & (free_ratio <= float(HULL_DEPTH_MAX_FREE_SPACE_RATIO))
            ).reshape(tuple(shape))

            seed_grid = np.rint(((raw_seed - origin) @ axes) / voxel_size).astype(int)
            seed_grid = np.clip(seed_grid, 0, shape - 1)
            protected_seed_cells = 0
            radius = int(HULL_SEED_PROTECTION_RADIUS_VOXELS)
            flat_free_ratio = free_ratio.reshape(tuple(shape))
            for coordinate in seed_grid:
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        for dz in range(-radius, radius + 1):
                            cell = coordinate + np.asarray([dx, dy, dz])
                            if np.any(cell < 0) or np.any(cell >= shape):
                                continue
                            cell_tuple = tuple(int(value) for value in cell)
                            if flat_free_ratio[cell_tuple] > HULL_DEPTH_SEED_MAX_FREE_SPACE_RATIO:
                                continue
                            if not occupancy[cell_tuple]:
                                protected_seed_cells += 1
                            occupancy[cell_tuple] = True

            labels, components = volume_component_labels(occupancy)
            seed_component_ids = set(
                int(labels[tuple(int(value) for value in cell)])
                for cell in seed_grid
                if labels[tuple(int(value) for value in cell)] >= 0
            )
            component_rows = []
            keep_components = set()
            hit_volume = hit_count.reshape(tuple(shape))
            for component_id, cells_in_component in enumerate(components):
                indices = tuple(np.asarray(cells_in_component, dtype=np.int32).T)
                cell_count = int(len(cells_in_component))
                max_hits = int(hit_volume[indices].max()) if cell_count else 0
                seed_supported = component_id in seed_component_ids
                keep = bool(
                    (seed_supported and cell_count >= HULL_COMPONENT_MIN_CELLS)
                    or (
                        cell_count >= HULL_UNSEEDED_COMPONENT_MIN_CELLS
                        and max_hits >= HULL_UNSEEDED_COMPONENT_MIN_HITS
                    )
                )
                if keep:
                    keep_components.add(component_id)
                component_rows.append({
                    'component_id': component_id,
                    'cell_count': cell_count,
                    'max_mask_support_views': max_hits,
                    'intersects_visible_seed': seed_supported,
                    'kept': keep,
                })
            occupancy &= np.isin(labels, list(keep_components))
            occupied_count = int(occupancy.sum())
            if occupied_count < int(HULL_MIN_OCCUPIED_CELLS):
                raise RuntimeError(
                    f'Depth-carved support retained only {occupied_count} cells; '
                    f'{HULL_MIN_OCCUPIED_CELLS} required. Inspect support reports before tuning.'
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
            safe_write_csv(paths['report_dir'] / 'visual_support_grid_frames.csv', frame_report)
            safe_write_csv(paths['report_dir'] / 'visual_support_grid_components.csv', component_rows)
            report = {
                'schema': 'object_depth_carved_support_grid.v2',
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
                'component_count_before_filter': int(len(components)),
                'component_count_kept': int(len(keep_components)),
                'raw_sparse_spacing': float(spacing),
                'voxel_size': float(voxel_size),
                'frame_report': str(paths['report_dir'] / 'visual_support_grid_frames.csv'),
                'component_report': str(paths['report_dir'] / 'visual_support_grid_components.csv'),
                'persisted_grid_path': str(persisted_grid_path),
            }
            write_json(report_path, report)
            print('Depth-carved visual support grid:', report)
            return str(local_grid_path), report
        ''',
    )
    functions_cell["source"] = source_lines(functions)

    run_cell = code_cell_containing(cells, "Containment pilot instance:")
    run_source = source_text(run_cell)
    run_source = replace_once(
        run_source,
        "            'object_sparse_path': object_sparse_path,\n",
        "            'object_sparse_path': object_sparse_path,\n"
        "            'sparse_visibility_report': str(base_paths['report_dir'] / 'sparse_visibility_summary.json'),\n",
        "sparse visibility manifest field",
    )
    run_cell["source"] = source_lines(run_source)

    cells[0]["source"] = source_lines(
        """# Batch Object Gaussian Splat Pipeline

This notebook runs the **single-instance bed depth-visible containment pilot V10**. It keeps the
same three controlled arms as V9, but fixes the shared upstream support that allowed hidden floor or
wall points to become protected bed seeds.

V10 calibrates full-room rendered depth to the raw camera coordinate scale for every positive mask
frame. A rendered depth sample is authoritative where valid; a room sparse-point z-buffer fills only
depth holes. Sparse points count as object seeds only when they land inside the tracked instance mask
and agree with the first visible surface. Weak disconnected seed components are removed before
training. The ellipsoid arm then uses a depth-carved near-surface support grid that rejects proven
free space, treats geometry behind an observed surface as unknown, and constrains each learned
Gaussian's oriented finite 2-sigma contour.

The notebook writes visibility overlays, frame-level depth calibration, component provenance, three
diagnostic sparse PLY files, and support-grid component reports before training. Post-export deletion
remains disabled. V9 Drive outputs are preserved because V10 has a new immutable batch ID.

Before running V10, `reconstruction.ipynb` must have produced room depth maps in `depths/` or
`depth_variants/`; if coverage is insufficient, the notebook stops with the exact reconstruction
setting to enable. Uploading this notebook alone remains sufficient after those reconstruction outputs
and the current `segmentation.ipynb` audit exist on Drive.
"""
    )
elif current_version != V10_VERSION:
    raise RuntimeError(
        f"Notebook version {current_version!r} is unsupported; expected {V9_VERSION!r} or {V10_VERSION!r}."
    )

# Apply small V10 contract additions to notebooks already carrying the V10 version. Keeping these
# migrations outside the V9 branch makes this updater idempotent across interrupted local edits.
config = source_text(config_cell)
if "DEPTH_MIN_RELIABLE_POSITIVE_FRAME_FRACTION" not in config:
    config = replace_once(
        config,
        "DEPTH_SCALE_MAX_MEDIAN_RELATIVE_ERROR = 0.25\n",
        "DEPTH_SCALE_MAX_MEDIAN_RELATIVE_ERROR = 0.25\n"
        "DEPTH_MIN_RELIABLE_POSITIVE_FRAME_FRACTION = 0.60\n",
        "reliable depth fraction control",
    )
    config_cell["source"] = source_lines(config)

functions_cell = code_cell_containing(cells, "def dataset_fingerprint_for_label")
functions = source_text(functions_cell)
if "reliable_depth_frames = sum(bool(row.get('depth_reliable'))" not in functions:
    reliable_gate = textwrap.indent(textwrap.dedent("""
            reliable_depth_frames = sum(bool(row.get('depth_reliable')) for row in frame_reports)
            reliable_depth_fraction = reliable_depth_frames / max(len(frame_reports), 1)
            if (
                REQUIRE_RENDERED_DEPTH_FOR_PILOT
                and reliable_depth_fraction < DEPTH_MIN_RELIABLE_POSITIVE_FRAME_FRACTION
            ):
                safe_write_csv(
                    paths['report_dir'] / 'sparse_visibility_frame_report.csv', frame_reports
                )
                raise RuntimeError(
                    'Rendered depth calibration succeeded for only '
                    f'{reliable_depth_frames}/{len(frame_reports)} positive frames '
                    f'({reliable_depth_fraction:.1%}); '
                    f'{DEPTH_MIN_RELIABLE_POSITIVE_FRAME_FRACTION:.0%} required. Inspect '
                    'sparse_visibility_frame_report.csv. Do not loosen seed visibility thresholds '
                    'until the depth/camera coordinate mismatch is resolved.'
                )

"""), "    ")
    functions = replace_once(
        functions,
        "    visible_ratio = visible_hits.astype(np.float32) / np.maximum(mask_hits, 1)\n",
        reliable_gate
        + "    visible_ratio = visible_hits.astype(np.float32) / np.maximum(mask_hits, 1)\n",
        "reliable depth runtime gate",
    )
if "'reliable_depth_frames': int(reliable_depth_frames)" not in functions:
    functions = replace_once(
        functions,
        "        'depth_coverage': coverage,\n",
        "        'depth_coverage': coverage,\n"
        "        'reliable_depth_frames': int(reliable_depth_frames),\n"
        "        'reliable_depth_fraction': float(reliable_depth_fraction),\n",
        "reliable depth report fields",
    )

if "def calibrate_depth_candidate" not in functions:
    functions = replace_function(
        functions,
        "calibrate_rendered_depth",
        '''
        def calibrate_depth_candidate(candidate, sparse_zbuffer, mode):
            overlap = (
                np.isfinite(candidate)
                & (candidate > 1e-8)
                & np.isfinite(sparse_zbuffer)
            )
            raw_count = int(overlap.sum())
            if raw_count < DEPTH_SCALE_MIN_OVERLAP_PIXELS:
                return candidate, {
                    'mode': mode,
                    'scale': None,
                    'overlap_pixels': raw_count,
                    'median_relative_error': None,
                    'reliable': False,
                    'reason': 'insufficient_sparse_depth_overlap',
                }
            ratio = sparse_zbuffer[overlap] / np.maximum(candidate[overlap], 1e-8)
            ratio = ratio[np.isfinite(ratio) & (ratio > 1e-4) & (ratio < 1e4)]
            if len(ratio) < DEPTH_SCALE_MIN_OVERLAP_PIXELS:
                return candidate, {
                    'mode': mode,
                    'scale': None,
                    'overlap_pixels': int(len(ratio)),
                    'median_relative_error': None,
                    'reliable': False,
                    'reason': 'invalid_depth_scale_ratios',
                }
            log_ratio = np.log(ratio)
            median = float(np.median(log_ratio))
            mad = float(np.median(np.abs(log_ratio - median)))
            if mad > 1e-8:
                keep = np.abs(log_ratio - median) <= 3.5 * 1.4826 * mad
                if int(keep.sum()) >= DEPTH_SCALE_MIN_OVERLAP_PIXELS:
                    log_ratio = log_ratio[keep]
            scale = float(np.exp(np.median(log_ratio)))
            aligned = candidate * scale
            residual = np.abs(sparse_zbuffer[overlap] - aligned[overlap])
            relative = residual / np.maximum(sparse_zbuffer[overlap], 1e-6)
            median_relative_error = float(np.median(relative))
            reliable = bool(median_relative_error <= DEPTH_SCALE_MAX_MEDIAN_RELATIVE_ERROR)
            return aligned, {
                'mode': mode,
                'scale': scale,
                'overlap_pixels': raw_count,
                'median_relative_error': median_relative_error,
                'reliable': reliable,
                'reason': 'ok' if reliable else 'depth_alignment_residual_too_large',
            }


        def calibrate_rendered_depth(depth, sparse_zbuffer, frame, image_shape):
            h, w = image_shape[:2]
            fx, fy, cx, cy = frame_intrinsics(frame, transforms_data, image_shape)
            pixel_x = (np.arange(w, dtype=np.float32) - float(cx)) / max(float(fx), 1e-6)
            pixel_y = (np.arange(h, dtype=np.float32) - float(cy)) / max(float(fy), 1e-6)
            ray_cosine = 1.0 / np.sqrt(
                1.0 + pixel_y[:, None] ** 2 + pixel_x[None, :] ** 2
            )
            candidates = {
                'native_camera_z': depth,
                'ray_distance_to_camera_z': depth * ray_cosine,
            }
            evaluated = [
                calibrate_depth_candidate(candidate, sparse_zbuffer, mode)
                for mode, candidate in candidates.items()
            ]
            evaluated.sort(
                key=lambda item: (
                    not bool(item[1].get('reliable')),
                    float(item[1].get('median_relative_error'))
                    if item[1].get('median_relative_error') is not None
                    else float('inf'),
                )
            )
            aligned, report = evaluated[0]
            report = dict(report)
            report['candidate_errors'] = {
                item_report['mode']: item_report.get('median_relative_error')
                for _, item_report in evaluated
            }
            return aligned, report
        ''',
    )
    functions = replace_once(
        functions,
        "depth, alignment = calibrate_rendered_depth(depth, zbuffer)\n",
        "depth, alignment = calibrate_rendered_depth(\n"
        "            depth, zbuffer, frame, image_shape\n"
        "        )\n",
        "camera-depth convention calibration call",
    )
if "'depth_mode': alignment.get('mode')" not in functions:
    functions = replace_once(
        functions,
        "            'depth_scale': alignment.get('scale'),\n",
        "            'depth_mode': alignment.get('mode'),\n"
        "            'depth_scale': alignment.get('scale'),\n",
        "sparse frame depth mode report",
    )
functions = functions.replace(
    "        & (candidate > DEPTH_MIN_M)\n"
    "        & (candidate < DEPTH_MAX_M)\n",
    "        & (candidate > 1e-8)\n",
)
functions_cell["source"] = source_lines(functions)

for cell in cells:
    if cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

notebook["metadata"].setdefault("room_model_project", {})[
    "object_splat_batch_version"
] = V10_VERSION
NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(f"Updated {NOTEBOOK_PATH} to {V10_VERSION}")
