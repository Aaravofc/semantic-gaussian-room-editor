"""Upgrade and regenerate the self-contained object-splat containment pilot."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "object_splat_batch_pipeline.ipynb"
PLUGIN_SOURCE_PATH = ROOT / "pipelines" / "object_constrained_splatfacto.py"
V8_VERSION = "2026-08-14-object-splat-containment-pilot-v8"
V9_VERSION = "2026-08-15-object-splat-ellipsoid-containment-pilot-v9"


def source_text(cell: dict) -> str:
    return "".join(cell.get("source", []))


def source_lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def replace_once(source: str, old: str, new: str, description: str) -> str:
    if new in source:
        return source
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {description} match, found {count}")
    return source.replace(old, new, 1)


def code_cell_containing(cells: list[dict], needle: str) -> dict:
    matches = [
        cell
        for cell in cells
        if cell.get("cell_type") == "code" and needle in source_text(cell)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one code cell containing {needle!r}, found {len(matches)}")
    return matches[0]


def tagged_cell(cells: list[dict], tag: str, cell_type: str) -> dict:
    matches = [
        cell
        for cell in cells
        if cell.get("cell_type") == cell_type
        and tag in cell.get("metadata", {}).get("tags", [])
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {cell_type} cell tagged {tag!r}, found {len(matches)}")
    return matches[0]


def plugin_cell_source(plugin_source: str) -> str:
    if "'''" in plugin_source:
        raise RuntimeError("Embedded plugin source unexpectedly contains triple-single quotes")
    return (
        "OBJECT_CONSTRAINED_PLUGIN_SOURCE = r'''"
        + plugin_source
        + "'''\n\n"
        + "PLUGIN_DIR = Path('/content/room-model-object-splat-plugin')\n"
        + "PLUGIN_DIR.mkdir(parents=True, exist_ok=True)\n"
        + "PLUGIN_PATH = PLUGIN_DIR / 'object_constrained_splatfacto.py'\n"
        + "PLUGIN_PATH.write_text(OBJECT_CONSTRAINED_PLUGIN_SOURCE, encoding='utf-8')\n"
        + "PLUGIN_SHA256 = hashlib.sha256(OBJECT_CONSTRAINED_PLUGIN_SOURCE.encode('utf-8')).hexdigest()\n"
        + "CUSTOM_METHOD_DEFINITIONS = (\n"
        + "    'object-alpha-splatfacto=object_constrained_splatfacto:ObjectAlphaSplatfacto,'\n"
        + "    'object-hull-splatfacto=object_constrained_splatfacto:ObjectHullSplatfacto,'\n"
        + "    'object-ellipsoid-splatfacto=object_constrained_splatfacto:ObjectEllipsoidSplatfacto'\n"
        + ")\n\n"
        + "def nerfstudio_subprocess_env():\n"
        + "    environment = os.environ.copy()\n"
        + "    existing_pythonpath = environment.get('PYTHONPATH', '')\n"
        + "    environment['PYTHONPATH'] = str(PLUGIN_DIR) + (os.pathsep + existing_pythonpath if existing_pythonpath else '')\n"
        + "    environment['NERFSTUDIO_METHOD_CONFIGS'] = CUSTOM_METHOD_DEFINITIONS\n"
        + "    environment['MAX_JOBS'] = '1'\n"
        + "    try:\n"
        + "        if torch.cuda.is_available():\n"
        + "            major, minor = torch.cuda.get_device_capability(0)\n"
        + "            environment['TORCH_CUDA_ARCH_LIST'] = f'{major}.{minor}'\n"
        + "    except Exception:\n"
        + "        pass\n"
        + "    return environment\n\n"
        + "for method_name in [\n"
        + "    'object-alpha-splatfacto',\n"
        + "    'object-hull-splatfacto',\n"
        + "    'object-ellipsoid-splatfacto',\n"
        + "]:\n"
        + "    result = subprocess.run(\n"
        + "        ['ns-train', method_name, '--help'],\n"
        + "        env=nerfstudio_subprocess_env(),\n"
        + "        capture_output=True,\n"
        + "        text=True,\n"
        + "    )\n"
        + "    if result.returncode != 0:\n"
        + "        print(result.stdout[-4000:])\n"
        + "        print(result.stderr[-4000:])\n"
        + "        fail(f'Embedded Nerfstudio method registration failed: {method_name}')\n"
        + "    print('Registered method:', method_name)\n"
        + "print('Plugin path:', PLUGIN_PATH)\n"
        + "print('Plugin SHA-256:', PLUGIN_SHA256)\n"
    )


notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
cells = notebook["cells"]
plugin_source = PLUGIN_SOURCE_PATH.read_text(encoding="utf-8")
config_cell = code_cell_containing(cells, "BATCH_PIPELINE_VERSION =")
config = source_text(config_cell)
version_match = re.search(r"BATCH_PIPELINE_VERSION\s*=\s*['\"]([^'\"]+)", config)
if version_match is None:
    raise RuntimeError("Could not read BATCH_PIPELINE_VERSION from the notebook")
current_version = version_match.group(1)

if current_version == V8_VERSION:
    config = replace_once(config, V8_VERSION, V9_VERSION, "pipeline version")
    config = replace_once(
        config,
        "BATCH_ID = 'bed_instance_containment_pilot_v8'",
        "BATCH_ID = 'bed_instance_ellipsoid_containment_pilot_v9'",
        "batch ID",
    )
    config = replace_once(
        config,
        """    'hull_constrained': {
        'method': 'object-hull-splatfacto',
        'description': 'Alpha constraints plus conservative multi-view 3D containment.',
    },""",
        """    'ellipsoid_constrained': {
        'method': 'object-ellipsoid-splatfacto',
        'description': 'Alpha constraints plus oriented 2-sigma ellipsoid containment.',
    },""",
        "ellipsoid variant definition",
    )
    config = replace_once(
        config,
        "PILOT_VARIANTS_TO_RUN = ['masked_baseline', 'alpha_constrained', 'hull_constrained']",
        "PILOT_VARIANTS_TO_RUN = ['masked_baseline', 'alpha_constrained', 'ellipsoid_constrained']",
        "variant run list",
    )
    config = replace_once(
        config,
        "PILOT_RECOMMENDED_VARIANT = 'hull_constrained'",
        "PILOT_RECOMMENDED_VARIANT = 'ellipsoid_constrained'",
        "recommended variant",
    )
    controls_anchor = "PILOT_HULL_PROJECTION_INTERVAL = 100\n"
    ellipsoid_controls = controls_anchor + """

# V9 constrains the learned WXYZ-rotated Gaussian contour, not just its center. The finite
# object boundary is defined at two standard deviations and sampled on two radial shells.
PILOT_ELLIPSOID_SIGMA_MULTIPLIER = 2.0
PILOT_ELLIPSOID_FIBONACCI_DIRECTION_COUNT = 42  # Plus six principal-axis probes.
PILOT_ELLIPSOID_RADIAL_SHELL_COUNT = 2
PILOT_ELLIPSOID_SUPPORT_LOSS_MULT = 0.10
PILOT_ELLIPSOID_BBOX_LOSS_MULT = 0.10
PILOT_ELLIPSOID_MIN_OCCUPANCY = 0.50
PILOT_ELLIPSOID_LOSS_SAMPLE_COUNT = 1024
PILOT_ELLIPSOID_LOSS_EVERY = 2
PILOT_ELLIPSOID_HARD_CHUNK_SIZE = 16384
PILOT_ELLIPSOID_SHRINK_SEARCH_STEPS = 7
PILOT_ELLIPSOID_SNAP_THRESHOLD = 0.10
"""
    config = replace_once(
        config,
        controls_anchor,
        ellipsoid_controls,
        "ellipsoid controls insertion point",
    )
    config_cell["source"] = source_lines(config)

    utility_cell = code_cell_containing(cells, "def pilot_variant_paths")
    utility = source_text(utility_cell)
    utility = replace_once(
        utility,
        "if variant_name == 'hull_constrained':",
        "if variant_name == 'ellipsoid_constrained':",
        "ellipsoid variant condition",
    )
    utility = replace_once(
        utility,
        "The hull-constrained arm requires sparse seed and support-grid paths.",
        "The ellipsoid-constrained arm requires sparse seed and support-grid paths.",
        "ellipsoid requirement message",
    )
    cli_anchor = (
        "            '--pipeline.model.hull-projection-interval', "
        "str(PILOT_HULL_PROJECTION_INTERVAL),\n"
    )
    ellipsoid_cli = cli_anchor + """            '--pipeline.model.ellipsoid-sigma-multiplier', str(PILOT_ELLIPSOID_SIGMA_MULTIPLIER),
            '--pipeline.model.ellipsoid-fibonacci-direction-count', str(PILOT_ELLIPSOID_FIBONACCI_DIRECTION_COUNT),
            '--pipeline.model.ellipsoid-radial-shell-count', str(PILOT_ELLIPSOID_RADIAL_SHELL_COUNT),
            '--pipeline.model.ellipsoid-support-loss-mult', str(PILOT_ELLIPSOID_SUPPORT_LOSS_MULT),
            '--pipeline.model.ellipsoid-bbox-loss-mult', str(PILOT_ELLIPSOID_BBOX_LOSS_MULT),
            '--pipeline.model.ellipsoid-min-occupancy', str(PILOT_ELLIPSOID_MIN_OCCUPANCY),
            '--pipeline.model.ellipsoid-loss-sample-count', str(PILOT_ELLIPSOID_LOSS_SAMPLE_COUNT),
            '--pipeline.model.ellipsoid-loss-every', str(PILOT_ELLIPSOID_LOSS_EVERY),
            '--pipeline.model.ellipsoid-hard-chunk-size', str(PILOT_ELLIPSOID_HARD_CHUNK_SIZE),
            '--pipeline.model.ellipsoid-shrink-search-steps', str(PILOT_ELLIPSOID_SHRINK_SEARCH_STEPS),
            '--pipeline.model.ellipsoid-snap-threshold', str(PILOT_ELLIPSOID_SNAP_THRESHOLD),
"""
    utility = replace_once(
        utility,
        cli_anchor,
        ellipsoid_cli,
        "ellipsoid CLI controls insertion point",
    )
    utility_cell["source"] = source_lines(utility)
elif current_version != V9_VERSION:
    raise RuntimeError(
        f"Notebook version {current_version!r} is unsupported; expected {V8_VERSION!r} or {V9_VERSION!r}."
    )

cells[0]["source"] = source_lines(
    """# Batch Object Gaussian Splat Pipeline

This notebook runs a **single-instance bed ellipsoid-containment pilot V9**. It compares three
controlled training arms with identical registered cameras, RGB frames, masks, and sparse seeds:

1. `masked_baseline`: stock Splatfacto mask-as-ignore behavior.
2. `alpha_constrained`: explicit rendered-opacity supervision and silhouette-gated densification.
3. `ellipsoid_constrained`: the alpha arm plus a conservative multi-view 3D support grid and
   in-training containment of each learned Gaussian's oriented finite 2-sigma volume.

The V9 arm samples 42 Fibonacci-sphere directions plus six principal axes on two radial shells.
It penalizes unsupported probe points differentiably, projects escaped centers, and binary-searches
the smallest necessary uniform scale correction when the finite ellipsoid crosses the support grid.
The mathematical Gaussian still has infinite support; 2-sigma is the explicit finite visual contour.

Only `ellipsoid_constrained` is placed in the viewer-facing `objects` list. Every raw export remains
in `ablation_variants`, and post-export cleanup is disabled. The completed V8 Drive folder is not
overwritten because V9 uses a new immutable batch ID.

The custom Nerfstudio methods are embedded below. Uploading this notebook alone to a fresh Colab GPU
runtime is sufficient after the current `segmentation.ipynb` audit succeeds.
"""
)

plugin_heading = tagged_cell(cells, "object-constrained-plugin-heading", "markdown")
plugin_heading["source"] = source_lines(
    """### 3A. Register Embedded Object-Constrained Splatfacto

This cell writes the maintained V9 plugin to temporary Colab storage and validates the alpha,
center-hull compatibility, and oriented-ellipsoid method registrations. It does not write source
code dependencies to Google Drive.
"""
)
plugin_cell = tagged_cell(cells, "object-constrained-plugin", "code")
plugin_cell["source"] = source_lines(plugin_cell_source(plugin_source))

inspection_cell = code_cell_containing(cells, "manifest = load_json(LATEST_MANIFEST_PATH)")
inspection_cell["source"] = source_lines(
    """manifest = load_json(LATEST_MANIFEST_PATH)
if manifest.get('version') != BATCH_PIPELINE_VERSION:
    fail(
        f'Manifest version mismatch: {manifest.get("version")!r} != {BATCH_PIPELINE_VERSION!r}'
    )
pilot = manifest.get('pilot', {})
variants = manifest.get('ablation_variants', [])
viewer_objects = manifest.get('objects', [])
if len(variants) != len(PILOT_VARIANTS_TO_RUN):
    fail(f'Expected {len(PILOT_VARIANTS_TO_RUN)} successful variants, found {len(variants)}.')
if len(viewer_objects) != 1 or viewer_objects[0].get('variant') != PILOT_RECOMMENDED_VARIANT:
    fail('Viewer manifest does not select exactly the recommended containment variant.')


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
        fail(f'Missing or empty pilot splat: {splat_path}')
    ply = PlyData.read(str(splat_path))
    vertex = ply['vertex'].data
    if len(vertex) <= 0:
        fail(f'Pilot splat has no Gaussians: {splat_path}')
    xyz = vertex_xyz(vertex).astype(np.float64)
    span = np.ptp(xyz, axis=0)
    center = np.mean(xyz, axis=0)
    sigma = gaussian_largest_sigma(vertex).astype(np.float64)
    alpha = gaussian_alpha_from_vertex(vertex)
    alpha = np.asarray(alpha, dtype=np.float64) if alpha is not None else np.asarray([])
    row = {
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

safe_write_csv(REPORT_DIR / 'pilot_export_inspection.csv', inspection_rows)
write_json(REPORT_DIR / 'pilot_export_inspection.json', {
    'schema': 'object_splat_ellipsoid_inspection.v1',
    'pipeline_version': BATCH_PIPELINE_VERSION,
    'batch_id': BATCH_ID,
    'target_instance_id': pilot.get('target_instance_id'),
    'post_export_cleanup_enabled': pilot.get('post_export_cleanup_enabled'),
    'variants': inspection_rows,
})
print('Pilot target:', pilot.get('target_instance_id'))
print('Post-export cleanup enabled:', pilot.get('post_export_cleanup_enabled'))
print_rows(
    inspection_rows,
    columns=[
        'variant', 'gaussian_count', 'span_x', 'span_y', 'span_z',
        'total_center_projected', 'total_ellipsoid_shrunk',
        'ellipsoid_snapped_last', 'boundary_inside_fraction',
    ],
    limit=20,
)
print('\\nOpen these three PLY files separately from identical saved cameras.')
print('The viewer-facing manifest selects only:', viewer_objects[0]['splat_path'])
print('Inspection CSV:', REPORT_DIR / 'pilot_export_inspection.csv')
print('Inspection JSON:', REPORT_DIR / 'pilot_export_inspection.json')
"""
)

for cell in cells:
    if cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

notebook["metadata"].setdefault("room_model_project", {})[
    "object_splat_batch_version"
] = V9_VERSION
NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(f"Updated {NOTEBOOK_PATH} to {V9_VERSION}")
