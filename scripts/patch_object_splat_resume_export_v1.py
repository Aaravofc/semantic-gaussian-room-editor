"""Restore persisted Nerfstudio checkpoints before exporting in a new Colab runtime."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "object_splat_batch_pipeline.ipynb"
PATCH_VERSION = "2026-08-23-resumable-local-checkpoint-export-v1"


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

    config_cell = cell_containing(cells, "RUNTIME_COMPATIBILITY_PATCH =")
    config = source_text(config_cell)
    if "RESUME_EXPORT_PATCH" not in config:
        anchor = "RUNTIME_COMPATIBILITY_PATCH = '2026-08-23-python312-colab-runtime-gate-v2'\n"
        replacement = anchor + f"RESUME_EXPORT_PATCH = '{PATCH_VERSION}'\n"
        if config.count(anchor) != 1:
            raise RuntimeError("Could not locate the runtime patch configuration anchor")
        config_cell["source"] = source_lines(config.replace(anchor, replacement, 1))
        changed = True

    functions_cell = cell_containing(cells, "def run_training_for_label")
    source = source_text(functions_cell)

    if "def prepare_local_training_stage_for_export" not in source:
        anchor = "def run_training_for_label(label_norm, paths):\n"
        helper = """def prepare_local_training_stage_for_export(paths):
    persisted_root = paths['training_dir']
    persisted_configs = sorted(persisted_root.glob('**/config.yml'))
    if not persisted_configs:
        raise RuntimeError(f'No config.yml found under persisted training root {persisted_root}')
    persisted_config = persisted_configs[-1]
    persisted_checkpoints = sorted(persisted_config.parent.glob('**/*.ckpt'))
    if not persisted_checkpoints:
        raise RuntimeError(
            f'No Nerfstudio checkpoint .ckpt files found near persisted config {persisted_config}'
        )

    local_root = paths['local_root'] / 'training'
    relative_run_dir = persisted_config.parent.relative_to(persisted_root)
    local_run_dir = local_root / relative_run_dir
    local_config = local_run_dir / persisted_config.name
    local_checkpoints = sorted(local_run_dir.glob('**/*.ckpt')) if local_run_dir.exists() else []
    if local_config.exists() and local_checkpoints:
        return local_config

    if local_run_dir.exists():
        shutil.rmtree(local_run_dir)
    local_run_dir.parent.mkdir(parents=True, exist_ok=True)
    print('Restoring persisted checkpoint to local Colab cache:', local_run_dir)
    subprocess.run(
        [
            'rsync', '-a', '--info=progress2',
            f'{persisted_config.parent}/', f'{local_run_dir}/',
        ],
        check=True,
    )
    local_checkpoints = sorted(local_run_dir.glob('**/*.ckpt'))
    if not local_config.exists() or not local_checkpoints:
        raise RuntimeError(
            f'Checkpoint restore was incomplete. Expected config/checkpoints under {local_run_dir}'
        )
    return local_config


def run_training_for_label(label_norm, paths):
"""
        if source.count(anchor) != 1:
            raise RuntimeError("Could not locate run_training_for_label insertion point")
        source = source.replace(anchor, helper, 1)
        changed = True

    old_export = """def export_splat_for_label(label_norm, paths):
    config_paths = sorted(paths['training_dir'].glob('**/config.yml'))
    if not config_paths:
        raise RuntimeError(f'No config.yml found under {paths["training_dir"]}')
    config_path = config_paths[-1]
    checkpoint_paths = sorted(config_path.parent.glob('**/*.ckpt'))
    if not checkpoint_paths:
        raise RuntimeError(f'No Nerfstudio checkpoint .ckpt files found near {config_path}')
"""
    new_export = """def export_splat_for_label(label_norm, paths):
    # Nerfstudio serializes the original /content output root into config.yml. A fresh Colab
    # runtime therefore needs the persisted run restored to that same local path before export.
    config_path = prepare_local_training_stage_for_export(paths)
    checkpoint_paths = sorted(config_path.parent.glob('**/*.ckpt'))
    if not checkpoint_paths:
        raise RuntimeError(f'No local Nerfstudio checkpoint .ckpt files found near {config_path}')
"""
    if old_export in source:
        source = source.replace(old_export, new_export, 1)
        changed = True
    elif "config_path = prepare_local_training_stage_for_export(paths)" not in source:
        raise RuntimeError("Could not locate export_splat_for_label configuration block")

    functions_cell["source"] = source_lines(source)
    return changed


def main() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    if patch_notebook(notebook):
        NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
        print(f"Applied {PATCH_VERSION} to {NOTEBOOK_PATH}")
    else:
        print(f"{NOTEBOOK_PATH} already contains {PATCH_VERSION}")


if __name__ == "__main__":
    main()
