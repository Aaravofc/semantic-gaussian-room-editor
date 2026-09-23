"""Build the checked-in Colab driver for tracked semantic segmentation."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "segmentation.ipynb"
PIPELINE_SOURCE_PATH = ROOT / "pipelines" / "tracked_segmentation.py"


def embedded_pipeline_source() -> str:
    source = PIPELINE_SOURCE_PATH.read_text(encoding="utf-8")
    entrypoint = '\nif __name__ == "__main__":\n'
    if entrypoint not in source:
        raise RuntimeError(f"Could not find the CLI entrypoint in {PIPELINE_SOURCE_PATH}")
    # A notebook has no CLI arguments. Keep every implementation function, including main(),
    # while omitting only the automatic command-line invocation at the end of the module.
    return source.split(entrypoint, 1)[0].rstrip() + "\n"


def markdown(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source.strip().splitlines(keepends=True),
    }


def code(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(keepends=True),
    }


cells = [
    markdown(
        """
# Tracked Semantic Segmentation v2

This notebook replaces independent YOLO-World detections and per-image SAM masks with a
temporally consistent pipeline:

```text
registered room frames
-> Grounding DINO detections on anchor frames
-> persistent same-label instance association
-> bidirectional SAM 2.1 video propagation
-> mask quality and overlap arbitration
-> SegFormer wall/floor/ceiling masks
-> canonical masks + JSON + accepted/rejected audit reports
```

Object labels have one owner: Grounding DINO plus SAM 2.1 video tracking. Structural labels
have one owner: SegFormer. `door` and `curtain` are object labels and are not duplicated by
the structural stage. Important editable objects keep canonical labels, while controlled named
and generic fallback prompts publish uncommon objects under `miscellaneous`.

Run the cells in order. Every stage writes a configuration fingerprint, so unchanged stages
resume safely while stale outputs are rejected automatically.

This notebook is self-contained. You only need to upload this `.ipynb` file to Colab; the
`pipelines` folder is embedded below and is not read from Google Drive.
        """,
        "title",
    ),
    markdown("## 1. Mount Google Drive", "mount-heading"),
    code(
        """
from google.colab import drive

drive.mount('/content/drive')
        """,
        "mount-drive",
    ),
    markdown(
        """
## 2. Embedded Pipeline Implementation

This collapsed cell contains the complete tracked-segmentation implementation. It is embedded
when the notebook is built locally, so Colab does not need `pipelines/tracked_segmentation.py`
or any other project source file. Its only external project dependency is the processed scan
data already stored under `PROJECT_ROOT` in Google Drive.
        """,
        "embedded-heading",
    ),
    code(embedded_pipeline_source(), "embedded-pipeline"),
    markdown(
        """
## 3. Configure the Pipeline

Edit `PROJECT_ROOT`, `SCAN_ID`, or the thresholds here. Keep `MAX_FRAMES = None` for a
production run because the object-splat pipeline depends on the complete registered camera
set. `DETECTION_KEYFRAME_STRIDE = 8` runs Grounding DINO on roughly one eighth of the frames;
SAM2 fills the intervals while preserving instance IDs.

The class dictionary contains one canonical prompt per editable label, while synonyms map back to
that stable taxonomy through `class_prompt_aliases`. Detection is hierarchical: a core pass finds
major room objects, those detections activate relevant bedroom/workspace/living-room detail
profiles, and a second pass runs only the selected detail and fallback prompts. Named uncommon
objects publish as `miscellaneous`; stricter generic fallback detections are retained only where
they do not duplicate a known object.
        """,
        "config-heading",
    ),
    code(
        """
from pathlib import Path
import copy
import json

PROJECT_ROOT = Path('/content/drive/MyDrive/room-model-project')
SCAN_ID = 'scan_001'
MAX_FRAMES = None
DETECTION_KEYFRAME_STRIDE = 8

if not PROJECT_ROOT.exists():
    raise FileNotFoundError(f'Missing Google Drive project root: {PROJECT_ROOT}')

%cd {PROJECT_ROOT}

CONFIG = copy.deepcopy(DEFAULT_CONFIG)
CONFIG.update({
    'pipeline_version': PIPELINE_VERSION,
    'project_root': str(PROJECT_ROOT),
    'scan_id': SCAN_ID,
    'max_frames': MAX_FRAMES,
    'detection_keyframe_stride': DETECTION_KEYFRAME_STRIDE,
    'resume': True,
    'force_stage': False,
    'prune_stale_outputs_after_audit': True,
    'write_all_visual_checks': False,
    'visual_check_limit': 180,
    'visual_check_sampling_mode': 'uniform_full_sequence',
})

# To test fewer object categories, remove entries from both dictionaries below. Keep the keys
# identical. The default production run leaves every configured room-object class enabled.
assert set(CONFIG['class_prompts']) == set(CONFIG['class_detection_thresholds'])
assert set(CONFIG['class_prompt_aliases']).issubset(CONFIG['class_prompts'])
assert 'miscellaneous' in CONFIG['class_prompts']
INITIAL_PROMPT_PLAN = build_scene_prompt_plan(CONFIG, [])

CONFIG_PATH = (
    PROJECT_ROOT / 'scans' / SCAN_ID / 'reports' / 'segmentation_v2' /
    'tracked_segmentation_config.json'
)
write_json(CONFIG_PATH, CONFIG)

print('Pipeline version:', PIPELINE_VERSION)
print('Config:', CONFIG_PATH)
print('Object labels:', sorted(CONFIG['class_prompts']))
print('Detector prompt count:', len(detector_prompt_records(CONFIG)))
print('Core detector labels:', INITIAL_PROMPT_PLAN['core_labels'])
print('Always-active prompt profiles:', CONFIG['always_active_prompt_profiles'])
print('Available conditional profiles:', sorted(CONFIG['class_prompt_profiles']))
print('Miscellaneous named prompts:', CONFIG['class_prompt_aliases']['miscellaneous'])
print('Miscellaneous generic prompts:', CONFIG['miscellaneous_generic_prompts'])
print('Structural labels:', CONFIG['structural_labels'])
print('Max frames:', CONFIG['max_frames'])
        """,
        "configure",
    ),
    markdown(
        """
## 4. Install Grounding DINO, SAM 2.1, and Utilities

Grounding DINO is loaded through Transformers, avoiding the custom CUDA-extension build used
by some Grounded-SAM repositories. SAM2 is installed from Meta's official repository, with
its optional `_C` CUDA post-processing extension deliberately disabled. The pipeline keeps
SAM2's useful stability settings and performs small-hole cleanup portably with OpenCV. The
large SAM2.1 checkpoint is retained under `/content` for the lifetime of this Colab runtime.

This cell does not reinstall NumPy or SciPy, which avoids the binary-compatibility failures
seen in earlier project runs.
        """,
        "install-heading",
    ),
    code(
        """
from pathlib import Path
import importlib
import importlib.util
import os
import shutil
import subprocess
import sys

packages = [
    'transformers>=4.57,<6',
    'accelerate>=1.0',
    'huggingface_hub>=0.28',
    'opencv-python-headless>=4.9',
    'Pillow>=10',
    'tqdm>=4.66',
    'matplotlib>=3.8',
]
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *packages], check=True)

SAM2_REPO = Path(CONFIG['sam2_repo'])
if not SAM2_REPO.exists():
    subprocess.run(
        ['git', 'clone', 'https://github.com/facebookresearch/sam2.git', str(SAM2_REPO)],
        check=True,
    )
else:
    print('Reusing existing SAM2 checkout:', SAM2_REPO)

sam2_install_env = os.environ.copy()
sam2_install_env['SAM2_BUILD_CUDA'] = '0'
subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '-q', '-e', str(SAM2_REPO)],
    check=True,
    env=sam2_install_env,
)

# Colab includes /content on sys.path. A checkout named /content/sam2 is therefore imported as
# a namespace directory before Python reaches Meta's inner sam2 package. Use a non-conflicting
# checkout name, put it first explicitly, and evict any incorrectly cached namespace package.
sam2_repo_text = str(SAM2_REPO.resolve())
sys.path = [value for value in sys.path if value != sam2_repo_text]
sys.path.insert(0, sam2_repo_text)
for module_name in [
    name for name in sys.modules if name == 'sam2' or name.startswith('sam2.')
]:
    del sys.modules[module_name]
importlib.invalidate_caches()

try:
    import sam2
    from sam2.build_sam import build_sam2_video_predictor
except ModuleNotFoundError as error:
    raise RuntimeError(
        f'Official SAM2 was installed but sam2.build_sam is unavailable from {SAM2_REPO}. '
        'Restart the runtime and rerun this cell.'
    ) from error

sam2_origin = Path(sam2.__file__).resolve() if sam2.__file__ else None
if sam2_origin is None or SAM2_REPO.resolve() not in sam2_origin.parents:
    raise RuntimeError(
        f'Wrong sam2 package imported from {sam2_origin}; expected it under {SAM2_REPO}.'
    )

checkpoint = Path(CONFIG['sam2_checkpoint'])
checkpoint.parent.mkdir(parents=True, exist_ok=True)
if not checkpoint.exists():
    checkpoint_url = (
        'https://dl.fbaipublicfiles.com/segment_anything_2/092824/'
        'sam2.1_hiera_large.pt'
    )
    subprocess.run(['wget', '-q', '--show-progress', '-O', str(checkpoint), checkpoint_url], check=True)

required_modules = ['torch', 'transformers', 'cv2', 'PIL']
missing = [name for name in required_modules if importlib.util.find_spec(name) is None]
if missing:
    raise RuntimeError(f'Missing modules after installation: {missing}')

print('SAM2 checkpoint:', checkpoint, checkpoint.stat().st_size, 'bytes')
print('SAM2 package:', sam2_origin)
print('SAM2 CUDA _C extension: deliberately disabled; OpenCV cleanup is enabled.')
print('Dependencies are ready.')
        """,
        "install",
    ),
    markdown(
        """
## 5. Validate Inputs and GPU

This checks `transforms.json`, every registered image path, the configured labels, and the
expected output locations before loading either model.
        """,
        "validate-heading",
    ),
    code(
        """
import torch

def run_pipeline_stage(stage):
    stage_config = load_config(CONFIG_PATH)
    stage_paths = project_paths(stage_config)
    print(f'Running embedded pipeline stage: {stage}')
    try:
        result = run_stage(stage_config, stage)
    except Exception as error:
        record_fatal(stage_paths, stage, error)
        raise
    print(json.dumps(result, indent=2))
    return result

print('torch.cuda.is_available():', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError('Select Runtime > Change runtime type > GPU, then rerun from the install cell.')
print('CUDA device:', torch.cuda.get_device_name(0))

run_pipeline_stage('validate')
        """,
        "validate",
    ),
    markdown(
        """
## 6. Detect Canonical Objects on Anchor Frames

Grounding DINO runs only on anchor frames. Each class has its own confidence threshold and
maximum box size. Same-label non-maximum suppression removes duplicate boxes before tracking.
Accepted and rejected detections are written separately under
`scans/<scan>/reports/segmentation_v2/`.
        """,
        "detect-heading",
    ),
    code(
        """
run_pipeline_stage('detect')
        """,
        "detect",
    ),
    code(
        """
import json

anchor_manifest_path = (
    PROJECT_ROOT / 'scans' / SCAN_ID / 'semantic_recon' / 'tracking' /
    'anchor_detection_manifest.json'
)
anchor_manifest = json.loads(anchor_manifest_path.read_text())
print('Anchor frames:', anchor_manifest['anchor_count'])
print('Accepted detections:', anchor_manifest['accepted_detection_count'])
print('Accepted by label:', anchor_manifest['accepted_by_label'])
print('Rejected by reason:', anchor_manifest['rejected_by_reason'])
        """,
        "inspect-detections",
    ),
    markdown(
        """
## 7. Associate Instances and Track with SAM 2.1

Detections are associated across neighboring anchors using label, box IoU, normalized center
motion, and area consistency. Weak one-frame tracks are rejected. Persistent tracks are then
prompted into SAM2's video predictor and propagated in both directions.

Before publishing, every candidate passes:

- class-specific area limits;
- SAM logit stability;
- connected-component coherence;
- bidirectional agreement;
- temporal visibility and area-jump checks;
- cross-label pixel ownership arbitration;
- a final minimum-visible-frame gate after arbitration, so partial two-frame fragments are
  quarantined instead of entering canonical JSON.

Only accepted masks enter
unknown` states remain available for diagnosis.
        """,
        "track-heading",
    ),
    code(
        """
run_pipeline_stage('track')
        """,
        "track",
    ),
    code(
        """
tracking_manifest_path = (
    PROJECT_ROOT / 'scans' / SCAN_ID / 'semantic_recon' / 'tracking' /
    'tracking_manifest.json'
)
tracking_manifest = json.loads(tracking_manifest_path.read_text())
print('Persistent anchor tracks:', tracking_manifest['track_count'])
print('Published tracks:', tracking_manifest.get('published_track_count'))
print(
    'Suppressed after overlap:',
    tracking_manifest.get('suppressed_after_overlap_tracks', []),
)
print('Accepted tracked masks:', tracking_manifest['accepted_mask_count'])
print('Accepted by label:', tracking_manifest['accepted_by_label'])
print('Rejected by reason:', tracking_manifest['rejected_by_reason'])
print('State counts:', tracking_manifest['state_counts'])
        """,
        "inspect-tracks",
    ),
    markdown(
        """
## 8. Generate Structural Masks

SegFormer is the sole source for `wall`, `floor`, and `ceiling`. It uses per-pixel softmax
confidence rather than assigning confidence `1.0`, resolves morphology-induced overlaps by
class probability, and subtracts only accepted tracked object masks. It does not emit `door`
or `curtain`, preventing duplicate label ownership.
        """,
        "layout-heading",
    ),
    code(
        """
run_pipeline_stage('layout')
        """,
        "layout",
    ),
    markdown(
        """
## 9. Audit the Canonical Outputs

The audit fails if a registered frame JSON is missing, a referenced mask is unreadable, a
track appears twice in one frame, object/structural source ownership is violated, fingerprints
are stale, a published track has fewer than the configured minimum visible frames, or any
published masks overlap. Error records are printed before the exception. Warnings identify
configured classes with no accepted masks; those classes remain `unknown` rather than being
fabricated.

JSON and CSV reports are staged on local Colab storage and committed atomically to Google
Drive. Transient Drive `EIO`, timeout, stale-mount, and disconnected-mount errors retry with
bounded backoff; persistent failures tell you to reconnect Drive and rerun only this stage.

After a zero-error audit, the notebook removes unreferenced files from its managed detection,
mask, layout, tracking-rejection, anchor, and preview directories. It never prunes source
images, reconstruction data, projection outputs, or object-splat experiments.
        """,
        "audit-heading",
    ),
    code(
        """
run_pipeline_stage('audit')
        """,
        "audit",
    ),
    code(
        """
audit_path = (
    PROJECT_ROOT / 'scans' / SCAN_ID / 'reports' / 'segmentation_v2' /
    'segmentation_audit_summary.json'
)
audit = json.loads(audit_path.read_text())
print('Audit status:', audit['status'])
print('Accepted masks:', audit['accepted_mask_count'])
print('Label counts:', audit['label_counts'])
print('State counts:', audit['state_counts'])
print('Maximum mask overlap:', audit['maximum_overlap_fraction'])
print('Warnings:', audit['warning_count'])
print('Stale files removed:', audit['stale_output_cleanup']['removed_file_count'])
print('Cleanup report:', audit['stale_output_cleanup']['report'])
print('Report:', audit_path)
        """,
        "inspect-audit",
    ),
    markdown(
        """
## 10. Inspect Visual Samples

The pipeline writes up to 180 uniformly spaced object and structural overlays across the complete
registered sequence, plus CSV inventories in `reports/segmentation_v2`. Review the room from start
to finish before launching object-splat training. If one class is weak, adjust only that class
threshold and rerun from the detection stage; the changed configuration automatically invalidates
stale tracking outputs.
        """,
        "visual-heading",
    ),
    code(
        """
from PIL import Image
from IPython.display import display

object_visual_dir = (
    PROJECT_ROOT / 'scans' / SCAN_ID / 'semantic_recon' / 'segmentation' / 'visual_checks'
)
layout_visual_dir = (
    PROJECT_ROOT / 'scans' / SCAN_ID / 'semantic_recon' / 'layout' / 'visual_checks'
)

def evenly_spaced_preview(paths, count=12):
    paths = list(paths)
    if len(paths) <= count:
        return paths
    return [paths[round(index * (len(paths) - 1) / (count - 1))] for index in range(count)]

all_object_visuals = sorted(object_visual_dir.glob('*_tracked.jpg'))
all_layout_visuals = sorted(layout_visual_dir.glob('*_structural.jpg'))
object_visuals = evenly_spaced_preview(all_object_visuals)
layout_visuals = evenly_spaced_preview(all_layout_visuals)
print('Object visual checks:', len(all_object_visuals))
print(
    'Object visual inventory:',
    PROJECT_ROOT / 'scans' / SCAN_ID / 'reports' / 'segmentation_v2' /
    'tracked_visual_check_inventory.csv',
)
for path in object_visuals:
    print(path.name)
    display(Image.open(path))
print('Structural visual checks:', len(all_layout_visuals))
print(
    'Structural visual inventory:',
    PROJECT_ROOT / 'scans' / SCAN_ID / 'reports' / 'segmentation_v2' / 'layout' /
    'structural_visual_check_inventory.csv',
)
for path in layout_visuals:
    print(path.name)
    display(Image.open(path))
        """,
        "visuals",
    ),
    markdown(
        """
## 11. Optional SAM3 Video Research Pilot

This does not replace the primary outputs. It runs the gated `facebook/sam3` video model on a
short contiguous frame range, saves its masks separately, and reports same-label IoU against
the Grounding-DINO/SAM2 result. That IoU is agreement, not ground truth; inspect both visual
sets manually.

Before enabling it, accept the SAM3 model license on Hugging Face. The first enabled run asks
for a Hugging Face token. Keep this disabled when using a T4 with insufficient memory or when
you only need the production pipeline.
        """,
        "sam3-heading",
    ),
    code(
        """
RUN_SAM3_PILOT = False

if RUN_SAM3_PILOT:
    from huggingface_hub import notebook_login

    notebook_login()
    CONFIG['enable_sam3_pilot'] = True
    CONFIG['sam3_pilot_labels'] = ['bed', 'desk', 'chair', 'door']
    CONFIG['sam3_pilot_start_frame'] = 0
    CONFIG['sam3_pilot_frame_count'] = 48
    write_json(CONFIG_PATH, CONFIG)
    run_pipeline_stage('sam3_pilot')
else:
    print('SAM3 pilot skipped. Set RUN_SAM3_PILOT = True only after the primary audit succeeds.')
        """,
        "sam3-pilot",
    ),
    markdown(
        """
## 12. Next Step

After the audit succeeds and the visual samples are correct, run the updated
`object_splat_batch_pipeline.ipynb`. Its V11 defaults train the accepted V9 ellipsoid method for
six bounded volumetric labels from current tracked detection JSON records only.
        """,
        "next-step",
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(f"Wrote {OUTPUT} with {len(cells)} cells")
