from __future__ import annotations

import ast
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "object_splat_batch_pipeline.ipynb"
PLUGIN_PATH = ROOT / "pipelines" / "object_constrained_splatfacto.py"
UPDATER_PATH = ROOT / "scripts" / "restore_object_splat_batch_to_v9.py"
V10_UPDATER_PATH = ROOT / "scripts" / "update_object_splat_batch_for_visibility_v10.py"
V11_UPDATER_PATH = ROOT / "scripts" / "expand_object_splat_batch_to_volumetric_v11.py"
V12_UPDATER_PATH = ROOT / "scripts" / "expand_object_splat_batch_to_all_volumetric_v12.py"
RUNTIME_PATCH_PATH = ROOT / "scripts" / "patch_object_splat_runtime_compatibility_v2.py"
RESUME_EXPORT_PATCH_PATH = ROOT / "scripts" / "patch_object_splat_resume_export_v1.py"
INTERRUPTION_RECOVERY_PATCH_PATH = (
    ROOT / "scripts" / "patch_object_splat_interruption_recovery_v1.py"
)
V12_VERSION = "2026-08-23-object-splat-all-volumetric-instances-v12"


def cell_source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def parse_colab_code(source: str, filename: str) -> ast.Module:
    lines = [
        "pass" if line.lstrip().startswith(("%", "!")) else line
        for line in source.splitlines()
    ]
    return ast.parse("\n".join(lines), filename=filename)


def literal_assignment(module: ast.Module, name: str):
    for node in module.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"No literal assignment found for {name}")


class ObjectSplatContainmentNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cls.code_cells = [
            (index, cell_source(cell))
            for index, cell in enumerate(cls.notebook["cells"])
            if cell.get("cell_type") == "code"
        ]

    def source_containing(self, needle: str) -> str:
        matches = [source for _, source in self.code_cells if needle in source]
        self.assertEqual(len(matches), 1, f"Expected one cell containing {needle!r}")
        return matches[0]

    def test_every_code_cell_parses(self) -> None:
        self.assertEqual(
            self.notebook["metadata"]["room_model_project"][
                "object_splat_batch_version"
            ],
            V12_VERSION,
        )
        for index, source in self.code_cells:
            with self.subTest(cell=index):
                parse_colab_code(source, f"object_splat_batch_pipeline.ipynb:cell-{index}")

        ast.parse(PLUGIN_PATH.read_text(encoding="utf-8"), filename=str(PLUGIN_PATH))
        ast.parse(UPDATER_PATH.read_text(encoding="utf-8"), filename=str(UPDATER_PATH))
        ast.parse(V10_UPDATER_PATH.read_text(encoding="utf-8"), filename=str(V10_UPDATER_PATH))
        ast.parse(V11_UPDATER_PATH.read_text(encoding="utf-8"), filename=str(V11_UPDATER_PATH))
        ast.parse(V12_UPDATER_PATH.read_text(encoding="utf-8"), filename=str(V12_UPDATER_PATH))
        ast.parse(RUNTIME_PATCH_PATH.read_text(encoding="utf-8"), filename=str(RUNTIME_PATCH_PATH))
        ast.parse(
            RESUME_EXPORT_PATCH_PATH.read_text(encoding="utf-8"),
            filename=str(RESUME_EXPORT_PATCH_PATH),
        )
        ast.parse(
            INTERRUPTION_RECOVERY_PATCH_PATH.read_text(encoding="utf-8"),
            filename=str(INTERRUPTION_RECOVERY_PATCH_PATH),
        )

    def test_embedded_plugin_matches_maintained_source(self) -> None:
        source = self.source_containing("OBJECT_CONSTRAINED_PLUGIN_SOURCE =")
        module = ast.parse(source)
        embedded = literal_assignment(module, "OBJECT_CONSTRAINED_PLUGIN_SOURCE")
        maintained = PLUGIN_PATH.read_text(encoding="utf-8")
        self.assertEqual(embedded, maintained)
        self.assertEqual(
            hashlib.sha256(embedded.encode("utf-8")).hexdigest(),
            hashlib.sha256(maintained.encode("utf-8")).hexdigest(),
        )

    def test_v12_discovers_all_qualifying_volumetric_instances(self) -> None:
        config_source = self.source_containing("BATCH_PIPELINE_VERSION =")
        config = ast.parse(config_source)
        self.assertEqual(
            literal_assignment(config, "BATCH_PIPELINE_VERSION"),
            "2026-08-23-object-splat-all-volumetric-instances-v12",
        )
        self.assertTrue(literal_assignment(config, "VOLUMETRIC_BATCH_MODE"))
        self.assertTrue(literal_assignment(config, "AUTO_DISCOVER_VOLUMETRIC_LABELS"))
        self.assertEqual(literal_assignment(config, "TARGET_LABELS"), [])
        allowlist = literal_assignment(config, "VOLUMETRIC_LABEL_ALLOWLIST")
        for label in (
            "bed", "chair", "table", "tv", "wardrobe", "cabinet", "drawer",
            "shelf", "desk", "sofa", "lamp", "pillow", "bag", "shoe",
            "bottle", "book", "plant", "miscellaneous",
        ):
            self.assertIn(label, allowlist)
        exclusions = literal_assignment(config, "VOLUMETRIC_LABEL_EXCLUSIONS")
        for label in (
            "wall", "floor", "ceiling", "door", "window", "curtain", "rug",
            "blanket", "mirror", "picture_frame", "fan", "light",
        ):
            self.assertIn(label, exclusions)
        self.assertFalse(allowlist & set(exclusions))
        self.assertEqual(literal_assignment(config, "TARGET_INSTANCE_IDS"), {})
        self.assertEqual(
            literal_assignment(config, "INSTANCE_SELECTION_POLICY"),
            "all_qualifying_persistent_instances",
        )
        self.assertEqual(
            literal_assignment(config, "PILOT_VARIANTS_TO_RUN"),
            ["ellipsoid_constrained"],
        )
        self.assertEqual(
            literal_assignment(config, "PILOT_RECOMMENDED_VARIANT"),
            "ellipsoid_constrained",
        )
        self.assertEqual(literal_assignment(config, "PILOT_ELLIPSOID_SIGMA_MULTIPLIER"), 2.0)
        self.assertEqual(
            literal_assignment(config, "PILOT_ELLIPSOID_FIBONACCI_DIRECTION_COUNT"),
            42,
        )
        self.assertEqual(literal_assignment(config, "PILOT_ELLIPSOID_RADIAL_SHELL_COUNT"), 2)
        self.assertTrue(literal_assignment(config, "KEEP_ALL_REGISTERED_FRAMES_FOR_SHARED_COORDINATES"))
        self.assertFalse(literal_assignment(config, "RUN_POST_EXPORT_CLEANUP"))
        self.assertEqual(literal_assignment(config, "SPARSE_POINT_MIN_MASK_HITS"), 2)
        self.assertEqual(literal_assignment(config, "HULL_PADDING_FRACTION"), 0.12)
        self.assertEqual(
            literal_assignment(config, "HULL_PADDING_SPACING_MULTIPLIER"), 4.0
        )
        self.assertEqual(literal_assignment(config, "HULL_MASK_DILATION_PIXELS"), 3)
        self.assertEqual(literal_assignment(config, "HULL_MIN_SUPPORT_RATIO"), 0.35)
        self.assertNotIn("REQUIRE_RENDERED_DEPTH_FOR_PILOT", config_source)

        audit_source = self.source_containing("Accepted detection labels:")
        self.assertIn("accepted_detection_labels & VOLUMETRIC_LABEL_ALLOWLIST", audit_source)
        self.assertIn("for candidate in candidate_rows", audit_source)
        self.assertIn("target_object_id=instance_id", audit_source)
        self.assertIn("object_key_for_instance(label_norm, instance_id)", audit_source)
        self.assertIn("audit_status = 'too_few_positive_mask_frames'", audit_source)
        self.assertIn("batch_object_audit.csv", audit_source)
        self.assertIn("volumetric_label_discovery.csv", audit_source)

    def test_training_contract_contains_losses_hull_and_no_cleanup(self) -> None:
        functions = self.source_containing("def dataset_fingerprint_for_label")
        for definition in (
            "def run_training_for_label",
            "def export_splat_for_label",
            "def _nearest_occupied_indices_6n",
            "def build_visual_support_grid_for_label",
            "def write_batch_manifest",
        ):
            self.assertEqual(functions.count(definition), 1)

        run_source = self.source_containing("Volumetric object:")
        self.assertLess(
            run_source.index("build_visual_support_grid_for_label"),
            run_source.index("for variant_name in pending_variants"),
        )
        self.assertNotIn("cleanup_exported_splat_for_label(", run_source)
        self.assertIn("'post_export_cleanup': False", run_source)
        self.assertIn("for target in eligible_objects", run_source)
        self.assertIn("'object_key': target_key", run_source)
        self.assertIn("VOLUMETRIC_LABEL_ALLOWLIST", run_source)
        self.assertIn("VOLUMETRIC_LABEL_EXCLUSIONS", run_source)
        self.assertIn("'status': 'skipped'", run_source)
        self.assertIn("'status': 'failed'", run_source)

        manifest_start = functions.index("def write_batch_manifest")
        manifest_source = functions[manifest_start:]
        self.assertIn("entry for entry in entries if entry.get('recommended')", manifest_source)
        self.assertIn("entry['object_key']", manifest_source)
        self.assertIn("'target_object_keys':", manifest_source)
        self.assertIn("'volumetric_batch':", manifest_source)
        self.assertIn("'ablation_variants': entries", manifest_source)
        self.assertIn("'success_with_skips'", manifest_source)

        for v9_contract in (
            "hit_counts >= int(SPARSE_POINT_MIN_MASK_HITS)",
            "object_visual_support_grid.v1",
            "hit_count >= int(HULL_MIN_SUPPORTING_VIEWS)",
            "support_ratio >= float(HULL_MIN_SUPPORT_RATIO)",
            "HULL_SEED_PROTECTION_RADIUS_VOXELS",
        ):
            self.assertIn(v9_contract, functions)
        for v10_contract in (
            "validate_depth_coverage",
            "sparse_visibility_frame_report.csv",
            "object_depth_carved_support_grid.v2",
            "free_space_cells",
        ):
            self.assertNotIn(v10_contract, functions)

        sparse_source = self.source_containing("def resolve_ply_file_path_from_transforms")
        self.assertIn("def project_points", sparse_source)
        self.assertNotIn("def project_points_with_depth", sparse_source)

        inspection_source = self.source_containing("manifest = load_json(LATEST_MANIFEST_PATH)")
        for report_contract in (
            "def last_containment_log_values",
            "volumetric_export_inspection.csv",
            "volumetric_export_inspection.json",
            "total_ellipsoid_shrunk",
            "boundary_inside_fraction",
            "sigma_p99",
        ):
            self.assertIn(report_contract, inspection_source)

    def test_v9_uses_permissive_mask_votes_without_depth_visibility(self) -> None:
        functions = self.source_containing("def dataset_fingerprint_for_label")
        sparse_start = functions.index("def create_sparse_init_for_label")
        sparse_end = functions.index("def run_training_for_label")
        sparse = functions[sparse_start:sparse_end]
        self.assertIn("hit_counts[start + local[hits]] += 1", sparse)
        self.assertIn("keep = hit_counts >= int(SPARSE_POINT_MIN_MASK_HITS)", sparse)
        self.assertNotIn("visible_hits", sparse)
        self.assertNotIn("rendered depth", sparse.lower())

    def test_custom_model_enforces_alpha_densification_and_ellipsoid_extent(self) -> None:
        source = PLUGIN_PATH.read_text(encoding="utf-8")
        module = ast.parse(source)
        class_names = {node.name for node in module.body if isinstance(node, ast.ClassDef)}
        self.assertIn("ObjectConstrainedSplatfactoModelConfig", class_names)
        self.assertIn("ObjectConstrainedSplatfactoModel", class_names)
        for contract in (
            'loss_dict["alpha_positive_loss"]',
            'loss_dict["alpha_outside_loss"]',
            'loss_dict["alpha_dice_loss"]',
            "def _gate_densification_gradients",
            "def _sample_hull_occupancy",
            "def _project_outside_hull",
            "def _ellipsoid_boundary_model_points",
            "def _quat_wxyz_to_rotation_matrix",
            "def _maximum_inside_scale_fraction",
            "def _constrain_ellipsoid_extents",
            'losses["hull_ellipsoid_support_loss"]',
            'metrics["hull_ellipsoid_inside_fraction"]',
            "self.scales.data.clamp_",
        ):
            self.assertIn(contract, source)
        self.assertIn(
            'ObjectAlphaSplatfacto = _build_method_spec("object-alpha-splatfacto"',
            source,
        )
        self.assertIn(
            'ObjectHullSplatfacto = _build_method_spec("object-hull-splatfacto"',
            source,
        )
        self.assertIn(
            'ObjectEllipsoidSplatfacto = _build_method_spec(',
            source,
        )
        self.assertIn('"object-ellipsoid-splatfacto"', source)

        utility_source = self.source_containing("def pilot_variant_paths")
        for cli_option in (
            "--pipeline.model.ellipsoid-sigma-multiplier",
            "--pipeline.model.ellipsoid-fibonacci-direction-count",
            "--pipeline.model.ellipsoid-radial-shell-count",
            "--pipeline.model.ellipsoid-support-loss-mult",
            "--pipeline.model.ellipsoid-shrink-search-steps",
        ):
            self.assertIn(cli_option, utility_source)

    def test_tested_nerfstudio_version_is_recorded(self) -> None:
        dependency_source = self.source_containing("NERFSTUDIO_TESTED_VERSION")
        dependency_module = ast.parse(dependency_source)
        self.assertEqual(literal_assignment(dependency_module, "NERFSTUDIO_TESTED_VERSION"), "1.1.5")
        self.assertEqual(literal_assignment(dependency_module, "NUMPY_TESTED_VERSION"), "1.26.4")
        self.assertEqual(
            literal_assignment(dependency_module, "SUPPORTED_PYTHON_VERSIONS"),
            {(3, 11), (3, 12)},
        )
        self.assertEqual(
            literal_assignment(dependency_module, "RECOMMENDED_COLAB_RUNTIME_VERSION"),
            "2026.07",
        )
        self.assertIn("python_version not in SUPPORTED_PYTHON_VERSIONS", dependency_source)
        self.assertIn("pip_install.returncode != 0", dependency_source)
        self.assertIn("pip installation stderr (tail):", dependency_source)
        self.assertIn("f'nerfstudio=={NERFSTUDIO_TESTED_VERSION}'", dependency_source)
        self.assertIn("f'numpy=={NUMPY_TESTED_VERSION}'", dependency_source)
        self.assertIn("numpy_loaded_before_install = 'numpy' in sys.modules", dependency_source)
        self.assertIn("np.random.default_rng(42)", dependency_source)
        self.assertIn("Use Runtime > Restart session", dependency_source)

        config_source = self.source_containing("BATCH_PIPELINE_VERSION =")
        self.assertNotIn("import numpy as np", config_source)
        self.assertNotIn("import cv2", config_source)
        self.assertLess(
            next(index for index, source in self.code_cells if "NERFSTUDIO_TESTED_VERSION" in source),
            next(index for index, source in self.code_cells if "def load_json" in source),
        )

        functions_source = self.source_containing("def estimate_sparse_spacing_numpy")
        self.assertNotIn("np.random", functions_source)
        self.assertIn("np.linspace(", functions_source)
        self.assertIn("state = (1664525 * state + 1013904223)", functions_source)
        plugin_source = self.source_containing("CUSTOM_METHOD_DEFINITIONS")
        self.assertIn("['ns-train', method_name, '--help']", plugin_source)
        self.assertIn("object-ellipsoid-splatfacto", plugin_source)

    def test_resumed_exports_restore_persisted_checkpoints_locally(self) -> None:
        config_source = self.source_containing("BATCH_PIPELINE_VERSION =")
        self.assertIn(
            "RESUME_EXPORT_PATCH = '2026-08-23-resumable-local-checkpoint-export-v1'",
            config_source,
        )
        functions_source = self.source_containing("def run_training_for_label")
        self.assertIn("def prepare_local_training_stage_for_export(paths):", functions_source)
        self.assertIn("persisted_config.parent.relative_to(persisted_root)", functions_source)
        self.assertIn("Restoring persisted checkpoint to local Colab cache:", functions_source)
        self.assertIn("f'{persisted_config.parent}/', f'{local_run_dir}/'", functions_source)
        self.assertIn(
            "config_path = prepare_local_training_stage_for_export(paths)",
            functions_source,
        )
        export_start = functions_source.index("def export_splat_for_label")
        load_config = functions_source.index("'--load-config', str(config_path)", export_start)
        self.assertLess(export_start, load_config)

    def test_interrupted_batches_recover_and_checkpoint_each_export(self) -> None:
        config_source = self.source_containing("BATCH_PIPELINE_VERSION =")
        self.assertIn(
            "INTERRUPTION_RECOVERY_PATCH = "
            "'2026-08-23-incremental-manifest-recovery-v1'",
            config_source,
        )
        self.assertIn("CLEAN_LOCAL_OBJECT_CACHE_AFTER_SUCCESS = True", config_source)

        functions_source = self.source_containing("def write_batch_manifest")
        for contract in (
            "def write_batch_manifest(entries, status_rows, batch_complete=True):",
            "'in_progress_checkpoint' if not batch_complete else",
            "'manifest_checkpointed_incrementally': True",
            "def load_recoverable_variant_entries(eligible_objects):",
            "entry.get('run_id') != expected_run_id",
            "splat_path.relative_to(BATCH_ROOT)",
            "splat_path.stat().st_size <= 0",
            "def checkpoint_batch_progress(entries, status_rows, batch_complete=False):",
            "def cleanup_local_object_cache(base_paths):",
            "shutil.rmtree(local_root)",
        ):
            self.assertIn(contract, functions_source)

        export_start = functions_source.index("def export_splat_for_label")
        export_end = functions_source.index("def sigmoid", export_start)
        export_source = functions_source[export_start:export_end]
        self.assertIn("shutil.rmtree(local_export)", export_source)

        run_source = self.source_containing("Volumetric object:")
        self.assertIn("load_recoverable_variant_entries(", run_source)
        self.assertIn("completed_pairs = {", run_source)
        self.assertIn("if not pending_variants:", run_source)
        self.assertIn("Reusing completed object exports:", run_source)
        self.assertIn("for variant_name in pending_variants:", run_source)
        summary_write = run_source.index(
            "write_json(paths['report_dir'] / 'variant_summary.json', entry)"
        )
        success_checkpoint = run_source.index(
            "checkpoint_batch_progress(variant_entries, status_rows, batch_complete=False)",
            summary_write,
        )
        self.assertLess(summary_write, success_checkpoint)
        self.assertIn("batch_complete=True", run_source)


if __name__ == "__main__":
    unittest.main()
