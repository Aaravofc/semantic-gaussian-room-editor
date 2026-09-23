"""Add an explicit Python-runtime gate to the V12 object-splat notebook."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "object_splat_batch_pipeline.ipynb"
PATCH_VERSION = "2026-08-23-python312-colab-runtime-gate-v2"


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
    updated_config = re.sub(
        r"RUNTIME_COMPATIBILITY_PATCH\s*=\s*['\"][^'\"]+['\"]",
        f"RUNTIME_COMPATIBILITY_PATCH = '{PATCH_VERSION}'",
        config,
        count=1,
    )
    if updated_config != config:
        config_cell["source"] = source_lines(updated_config)
        changed = True

    dependency_cell = cell_containing(cells, "NERFSTUDIO_TESTED_VERSION")
    dependency = source_text(dependency_cell)

    if "SUPPORTED_PYTHON_VERSIONS" not in dependency:
        anchor = "NUMPY_TESTED_VERSION = '1.26.4'\n"
        runtime_gate = """NUMPY_TESTED_VERSION = '1.26.4'
SUPPORTED_PYTHON_VERSIONS = {(3, 11), (3, 12)}
RECOMMENDED_COLAB_RUNTIME_VERSION = '2026.07'

python_version = sys.version_info[:2]
if python_version not in SUPPORTED_PYTHON_VERSIONS:
    raise RuntimeError(
        f'Python {sys.version.split()[0]} is incompatible with this tested '
        f'Nerfstudio/NumPy stack. In Colab choose Runtime > Change runtime type, '
        f'set Runtime Version to {RECOMMENDED_COLAB_RUNTIME_VERSION} and Hardware '
        'accelerator to T4 GPU, reconnect, then run the notebook from the top. '
        'Do not try to repair the current Python 3.13 session.'
    )
"""
        if dependency.count(anchor) != 1:
            raise RuntimeError("Could not locate the NumPy version anchor")
        dependency = dependency.replace(anchor, runtime_gate, 1)
        changed = True

    old_install = """    subprocess.run(
        [
            sys.executable, '-m', 'pip', 'install', '-q',
            '--upgrade-strategy', 'only-if-needed',
            f'numpy=={NUMPY_TESTED_VERSION}',
            'plyfile',
            f'nerfstudio=={NERFSTUDIO_TESTED_VERSION}',
        ],
        check=True,
    )
"""
    new_install = """    pip_install = subprocess.run(
        [
            sys.executable, '-m', 'pip', 'install', '-q', '--no-input',
            '--upgrade-strategy', 'only-if-needed',
            f'numpy=={NUMPY_TESTED_VERSION}',
            'plyfile',
            f'nerfstudio=={NERFSTUDIO_TESTED_VERSION}',
        ],
        capture_output=True,
        text=True,
    )
    if pip_install.returncode != 0:
        print('pip installation stdout (tail):')
        print(pip_install.stdout[-6000:])
        print('pip installation stderr (tail):')
        print(pip_install.stderr[-6000:])
        raise RuntimeError(
            'Could not install the tested object-splat environment. Confirm that '
            f'Colab Runtime Version is {RECOMMENDED_COLAB_RUNTIME_VERSION}, then '
            'restart the session and rerun from the top.'
        )
"""
    if old_install in dependency:
        dependency = dependency.replace(old_install, new_install, 1)
        changed = True
    elif "pip_install = subprocess.run(" not in dependency:
        raise RuntimeError("Could not locate the dependency-install subprocess block")

    dependency_cell["source"] = source_lines(dependency)
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
