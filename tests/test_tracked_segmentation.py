from __future__ import annotations

import copy
import errno
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pipelines.tracked_segmentation import (
    DEFAULT_CONFIG,
    associate_anchor_detections,
    enforce_minimum_published_track_support,
    prune_stale_generated_outputs,
    box_iou,
    build_scene_prompt_plan,
    canonical_detector_label,
    detector_prompt_records,
    deep_merge,
    nms_by_label,
    suppress_generic_miscellaneous_over_known,
    suppress_semantic_conflicts,
    visual_check_frame_indexes,
    write_csv,
    write_json,
)


class TrackedSegmentationTests(unittest.TestCase):
    def test_helper_contracts(self) -> None:
        self.assertAlmostEqual(
            box_iou([0, 0, 10, 10], [5, 5, 15, 15]),
            25 / 175,
        )
        self.assertEqual(
            canonical_detector_label(
                "ceiling fan",
                {"ceiling_fan": "fan"},
                ["fan"],
            ),
            "fan",
        )
        self.assertEqual(
            deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 3}}),
            {"a": {"x": 1, "y": 3}},
        )

    def test_miscellaneous_prompt_taxonomy_and_duplicate_suppression(self) -> None:
        records = detector_prompt_records(copy.deepcopy(DEFAULT_CONFIG))
        prompt_to_record = {record["prompt"]: record for record in records}
        self.assertEqual(prompt_to_record["headboard"]["label"], "miscellaneous")
        self.assertEqual(prompt_to_record["headboard"]["prompt_kind"], "named_fallback")
        self.assertEqual(prompt_to_record["backpack"]["label"], "bag")
        self.assertEqual(prompt_to_record["cushion"]["label"], "pillow")
        self.assertEqual(
            prompt_to_record["small household object"]["prompt_kind"],
            "generic_fallback",
        )
        self.assertIn("window", DEFAULT_CONFIG["class_prompts"])
        self.assertIn("rug", DEFAULT_CONFIG["class_prompts"])

        known_bed = {
            "detection_id": "bed-1",
            "label": "bed",
            "prompt_kind": "canonical",
            "box_xyxy": [0, 0, 100, 100],
        }
        generic_duplicate = {
            "detection_id": "misc-generic",
            "label": "miscellaneous",
            "prompt_kind": "generic_fallback",
            "box_xyxy": [20, 20, 40, 40],
        }
        named_headboard = {
            "detection_id": "misc-headboard",
            "label": "miscellaneous",
            "prompt_kind": "named_fallback",
            "box_xyxy": [0, 0, 100, 30],
        }
        accepted, rejected = suppress_generic_miscellaneous_over_known(
            [known_bed, generic_duplicate, named_headboard],
            iou_threshold=0.20,
            containment_threshold=0.65,
        )
        self.assertEqual(
            {item["detection_id"] for item in accepted},
            {"bed-1", "misc-headboard"},
        )
        self.assertEqual([item["detection_id"] for item in rejected], ["misc-generic"])
        self.assertEqual(
            rejected[0]["rejection_reason"],
            "generic_miscellaneous_duplicates_known_class",
        )

    def test_hierarchical_prompt_plan_activates_scene_details(self) -> None:
        config = copy.deepcopy(DEFAULT_CONFIG)
        empty_plan = build_scene_prompt_plan(config, [])
        self.assertEqual(empty_plan["active_profiles"], ["room_details"])
        self.assertNotIn("pillow", empty_plan["active_labels"])
        self.assertNotIn("headboard", empty_plan["active_named_miscellaneous_prompts"])
        self.assertIn("switchboard", empty_plan["active_named_miscellaneous_prompts"])

        bedroom_plan = build_scene_prompt_plan(
            config,
            [
                {
                    "label": "bed",
                    "frame_key": "frame_000010",
                    "confidence": 0.81,
                }
            ],
        )
        self.assertIn("bedroom", bedroom_plan["active_profiles"])
        self.assertIn("pillow", bedroom_plan["active_labels"])
        self.assertIn("bag", bedroom_plan["active_labels"])
        self.assertIn("headboard", bedroom_plan["active_named_miscellaneous_prompts"])
        active_prompts = {
            record["prompt"] for record in bedroom_plan["active_prompt_records"]
        }
        self.assertIn("bed pillow", active_prompts)
        self.assertIn("headboard", active_prompts)
        self.assertIn("small household object", active_prompts)
        self.assertLess(
            bedroom_plan["active_prompt_count"], bedroom_plan["library_prompt_count"]
        )

    def test_named_miscellaneous_prompt_wins_nms_over_generic_prompt(self) -> None:
        named = {
            "detection_id": "named",
            "label": "miscellaneous",
            "prompt_kind": "named_fallback",
            "confidence": 0.45,
            "box_xyxy": [0, 0, 50, 50],
        }
        generic = {
            "detection_id": "generic",
            "label": "miscellaneous",
            "prompt_kind": "generic_fallback",
            "confidence": 0.92,
            "box_xyxy": [0, 0, 50, 50],
        }
        accepted, rejected = nms_by_label([generic, named], iou_threshold=0.5)
        self.assertEqual([row["detection_id"] for row in accepted], ["named"])
        self.assertEqual([row["detection_id"] for row in rejected], ["generic"])

    def test_curtain_and_blanket_semantic_conflict_gates(self) -> None:
        config = copy.deepcopy(DEFAULT_CONFIG)
        bed = {
            "detection_id": "bed",
            "label": "bed",
            "confidence": 0.82,
            "box_xyxy": [0, 0, 100, 100],
        }
        blanket_duplicate = {
            "detection_id": "blanket-large",
            "label": "blanket",
            "confidence": 0.83,
            "box_xyxy": [5, 5, 95, 95],
        }
        blanket_detail = {
            "detection_id": "blanket-small",
            "label": "blanket",
            "confidence": 0.77,
            "box_xyxy": [10, 10, 45, 35],
        }
        unsupported_curtain = {
            "detection_id": "curtain-panel",
            "label": "curtain",
            "confidence": 0.75,
            "box_xyxy": [120, 0, 180, 180],
        }
        accepted, rejected = suppress_semantic_conflicts(
            [bed, blanket_duplicate, blanket_detail, unsupported_curtain],
            config,
            200,
            200,
        )
        self.assertEqual(
            {row["detection_id"] for row in accepted},
            {"bed", "blanket-small"},
        )
        self.assertEqual(
            {row["rejection_reason"] for row in rejected},
            {"blanket_duplicates_bed_extent", "curtain_without_window_context"},
        )

        window = {
            "detection_id": "window",
            "label": "window",
            "confidence": 0.80,
            "box_xyxy": [118, 0, 182, 180],
        }
        accepted, rejected = suppress_semantic_conflicts(
            [unsupported_curtain, window], config, 200, 200
        )
        self.assertEqual(
            {row["detection_id"] for row in accepted},
            {"curtain-panel", "window"},
        )
        self.assertEqual(rejected, [])

    def test_visual_checks_are_uniform_across_the_complete_sequence(self) -> None:
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["visual_check_limit"] = 180
        frames = [
            {"frame_index": frame_index, "frame_key": f"frame_{frame_index:06d}"}
            for frame_index in range(263)
        ]
        selected = sorted(visual_check_frame_indexes(frames, config))
        self.assertEqual(len(selected), 180)
        self.assertEqual(selected[0], 0)
        self.assertEqual(selected[-1], 262)
        self.assertLessEqual(max(b - a for a, b in zip(selected, selected[1:])), 2)

        config["write_all_visual_checks"] = True
        self.assertEqual(len(visual_check_frame_indexes(frames, config)), 263)

    def test_csv_writer_retries_transient_io_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "reports" / "audit.csv"
            original_copyfile = shutil.copyfile
            attempts = 0

            def flaky_copyfile(source, destination):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise OSError(errno.EIO, "simulated Drive I/O failure")
                return original_copyfile(source, destination)

            with (
                mock.patch(
                    "pipelines.tracked_segmentation.shutil.copyfile",
                    side_effect=flaky_copyfile,
                ),
                mock.patch("pipelines.tracked_segmentation.time.sleep") as sleep,
            ):
                write_csv(output, [{"frame_key": "frame_000001", "error": "example"}])

            self.assertEqual(attempts, 2)
            sleep.assert_called_once_with(1.0)
            self.assertEqual(
                output.read_text(encoding="utf-8").splitlines(),
                ["frame_key,error", "frame_000001,example"],
            )

    def test_anchor_association_rejects_weak_single_frame_fragment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tracking_dir = Path(temporary) / "tracking"
            tracking_dir.mkdir()
            anchor_paths = []
            bed_boxes = {
                0: [10, 10, 50, 40],
                8: [12, 10, 52, 40],
                16: [14, 11, 54, 41],
            }
            for frame_index, bed_box in bed_boxes.items():
                detections = [
                    {
                        "detection_id": f"bed-{frame_index}",
                        "frame_index": frame_index,
                        "frame_key": f"frame_{frame_index:06d}",
                        "label": "bed",
                        "confidence": 0.61,
                        "box_xyxy": bed_box,
                    }
                ]
                if frame_index == 8:
                    detections.append(
                        {
                            "detection_id": "chair-weak",
                            "frame_index": frame_index,
                            "frame_key": f"frame_{frame_index:06d}",
                            "label": "chair",
                            "confidence": 0.40,
                            "box_xyxy": [70, 20, 80, 40],
                        }
                    )
                anchor_path = tracking_dir / f"anchor-{frame_index}.json"
                write_json(
                    anchor_path,
                    {
                        "frame": {"frame_index": frame_index},
                        "image_size": [100, 80],
                        "detections": detections,
                    },
                )
                anchor_paths.append(str(anchor_path))

            write_json(
                tracking_dir / "anchor_detection_manifest.json",
                {"anchor_json_paths": anchor_paths},
            )
            config = copy.deepcopy(DEFAULT_CONFIG)
            tracks, rejected, labels_by_anchor = associate_anchor_detections(
                config,
                {"tracking_dir": tracking_dir},
            )

            self.assertEqual(len(tracks), 1)
            self.assertEqual(tracks[0]["track_id"], "bed_001")
            self.assertEqual(tracks[0]["anchor_count"], 3)
            self.assertEqual(len(rejected), 1)
            self.assertEqual(rejected[0]["label"], "chair")
            self.assertEqual(
                rejected[0]["rejection_reason"],
                "insufficient_anchor_persistence",
            )
            self.assertEqual(labels_by_anchor[8], {"bed", "chair"})

    def test_post_overlap_support_gate_suppresses_two_frame_track(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {
                "mask_dir": root / "masks",
                "rejected_dir": root / "rejected",
                "report_dir": root / "reports",
            }
            for path in paths.values():
                path.mkdir(parents=True, exist_ok=True)

            output_paths = []
            drawer_masks = []
            for frame_index in range(3):
                frame_key = f"frame_{frame_index:06d}"
                records = []
                for track_id, label in [("bed_001", "bed"), ("drawer_004", "drawer")]:
                    if track_id == "drawer_004" and frame_index == 2:
                        continue
                    mask_path = paths["mask_dir"] / f"{frame_key}_{track_id}.png"
                    mask_path.write_bytes(b"synthetic-mask")
                    if track_id == "drawer_004":
                        drawer_masks.append(mask_path)
                    records.append(
                        {
                            "mask_path": str(mask_path),
                            "label": label,
                            "track_id": track_id,
                            "segmentation_role": "tracked_object",
                            "state": "visible",
                            "accepted": True,
                        }
                    )

                json_path = root / "detections" / f"{frame_key}.json"
                rejected_path = paths["rejected_dir"] / f"{frame_key}.json"
                write_json(
                    json_path,
                    {
                        "frame_index": frame_index,
                        "frame_key": frame_key,
                        "detections": records,
                        "label_states": {
                            "bed": {"state": "visible", "reason": "accepted_tracked_mask"},
                            "drawer": {
                                "state": "visible" if frame_index < 2 else "unknown",
                                "reason": "accepted_tracked_mask"
                                if frame_index < 2
                                else "not_visible",
                            },
                        },
                        "rejected_detection_report": str(rejected_path),
                    },
                )
                write_json(rejected_path, {"rejected_candidates": []})
                output_paths.append(json_path)

            config = copy.deepcopy(DEFAULT_CONFIG)
            config["minimum_track_visible_frames"] = 3
            accepted, rejected, _, suppressed = enforce_minimum_published_track_support(
                config, paths, output_paths
            )

            self.assertEqual({row["track_id"] for row in accepted}, {"bed_001"})
            self.assertEqual(len(accepted), 3)
            self.assertEqual(len(rejected), 2)
            self.assertEqual(
                suppressed,
                [
                    {
                        "track_id": "drawer_004",
                        "label": "drawer",
                        "published_visible_frames": 2,
                        "minimum_visible_frames": 3,
                        "removed_mask_count": 2,
                        "reason": "insufficient_visible_track_frames_after_overlap",
                    }
                ],
            )
            self.assertTrue(all(not path.exists() for path in drawer_masks))
            self.assertTrue(
                all(
                    (paths["rejected_dir"] / "under_supported_masks" / path.name).exists()
                    for path in drawer_masks
                )
            )
            first_payload = json.loads(output_paths[0].read_text(encoding="utf-8"))
            self.assertEqual(
                {record["track_id"] for record in first_payload["detections"]},
                {"bed_001"},
            )
            self.assertEqual(first_payload["label_states"]["drawer"]["state"], "unknown")
            first_rejections = json.loads(
                (paths["rejected_dir"] / "frame_000000.json").read_text(encoding="utf-8")
            )["rejected_candidates"]
            self.assertEqual(
                first_rejections[0]["rejection_reasons"],
                "insufficient_visible_track_frames_after_overlap",
            )

    def test_successful_audit_cleanup_prunes_only_unreferenced_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {
                "detection_dir": root / "detections",
                "mask_dir": root / "masks",
                "visual_dir": root / "visual_checks",
                "layout_mask_dir": root / "layout" / "masks",
                "layout_label_dir": root / "layout" / "labels",
                "layout_visual_dir": root / "layout" / "visual_checks",
                "rejected_dir": root / "tracking" / "rejected",
                "tracking_dir": root / "tracking",
                "anchor_dir": root / "tracking" / "anchors",
                "report_dir": root / "reports",
            }
            for directory in paths.values():
                directory.mkdir(parents=True, exist_ok=True)

            frames = [
                {"frame_key": "frame_000000"},
                {"frame_key": "frame_000001"},
            ]
            current_mask = paths["mask_dir"] / "frame_000000_track_0001_bed.png"
            current_layout_mask = paths["mask_dir"] / "frame_000000_wall_layout.png"
            current_layout_copy = paths["layout_mask_dir"] / "frame_000000_wall_layout.png"
            current_files = [
                current_mask,
                current_layout_mask,
                current_layout_copy,
                paths["visual_dir"] / "frame_000000_tracked.jpg",
                paths["layout_visual_dir"] / "frame_000000_structural.jpg",
                paths["anchor_dir"] / "frame_000000.json",
            ]
            current_files.extend(
                paths["detection_dir"] / f"{frame['frame_key']}.json" for frame in frames
            )
            current_files.extend(
                paths["layout_label_dir"] / f"{frame['frame_key']}.json" for frame in frames
            )
            current_files.extend(
                paths["rejected_dir"] / f"{frame['frame_key']}.json" for frame in frames
            )
            for path in current_files:
                path.write_text("current", encoding="utf-8")

            stale_files = [
                paths["detection_dir"] / "frame_999999.json",
                paths["mask_dir"] / "frame_000000_000_bed.png",
                paths["visual_dir"] / "frame_000001_tracked.jpg",
                paths["layout_mask_dir"] / "frame_000001_floor_layout.png",
                paths["layout_label_dir"] / "frame_999999.json",
                paths["layout_visual_dir"] / "frame_000001_structural.jpg",
                paths["rejected_dir"] / "frame_999999.json",
                paths["anchor_dir"] / "frame_999999.json",
            ]
            for path in stale_files:
                path.write_text("stale", encoding="utf-8")

            retained_debug = paths["rejected_dir"] / "under_supported_masks" / "sample.png"
            retained_debug.parent.mkdir(parents=True)
            retained_debug.write_text("debug", encoding="utf-8")
            write_json(
                paths["tracking_dir"] / "anchor_detection_manifest.json",
                {"anchor_json_paths": [str(paths["anchor_dir"] / "frame_000000.json")]},
            )

            config = copy.deepcopy(DEFAULT_CONFIG)
            config["write_all_visual_checks"] = False
            config["visual_check_limit"] = 1
            result = prune_stale_generated_outputs(
                config,
                paths,
                frames,
                [current_mask, current_layout_mask],
                [current_layout_copy],
            )

            self.assertEqual(result["removed_file_count"], len(stale_files))
            self.assertTrue(all(path.exists() for path in current_files))
            self.assertTrue(all(not path.exists() for path in stale_files))
            self.assertTrue(retained_debug.exists())
            self.assertTrue((paths["report_dir"] / "stale_output_cleanup.csv").exists())

    def test_generated_notebook_contracts(self) -> None:
        root = Path(__file__).resolve().parents[1]
        segmentation = json.loads((root / "segmentation.ipynb").read_text(encoding="utf-8"))
        segmentation_source = "\n".join(
            "".join(cell.get("source", [])) for cell in segmentation["cells"]
        )
        embedded_pipeline_cells = [
            "".join(cell.get("source", []))
            for cell in segmentation["cells"]
            if cell.get("cell_type") == "code"
            and "PIPELINE_VERSION =" in "".join(cell.get("source", []))
            and "def run_anchor_detection" in "".join(cell.get("source", []))
        ]
        self.assertEqual(len(embedded_pipeline_cells), 1)
        maintained_pipeline = (
            root / "pipelines" / "tracked_segmentation.py"
        ).read_text(encoding="utf-8")
        maintained_pipeline = maintained_pipeline.split(
            '\nif __name__ == "__main__":\n', 1
        )[0]
        self.assertEqual(
            embedded_pipeline_cells[0].strip(), maintained_pipeline.strip()
        )
        self.assertIn("Grounding DINO", segmentation_source)
        self.assertIn("def run_anchor_detection", segmentation_source)
        self.assertIn("def run_sam2_tracking", segmentation_source)
        self.assertIn("def run_structural_layout", segmentation_source)
        self.assertIn("def repair_tracking_publication_support", segmentation_source)
        self.assertIn("tracks_suppressed_after_overlap.csv", segmentation_source)
        self.assertIn(
            "grounded-sam2-tracked-segmentation-v2.7-room-wide-audit",
            segmentation_source,
        )
        self.assertIn('"miscellaneous": "miscellaneous room object"', segmentation_source)
        self.assertIn('"headboard"', segmentation_source)
        self.assertIn('"switchboard"', segmentation_source)
        self.assertIn("def detector_prompt_records", segmentation_source)
        self.assertIn("def build_scene_prompt_plan", segmentation_source)
        self.assertIn("def suppress_generic_miscellaneous_over_known", segmentation_source)
        self.assertIn("def suppress_semantic_conflicts", segmentation_source)
        self.assertIn("def visual_check_frame_indexes", segmentation_source)
        self.assertIn("tracked_visual_check_inventory.csv", segmentation_source)
        self.assertIn("structural_visual_check_inventory.csv", segmentation_source)
        self.assertIn("'visual_check_limit': 180", segmentation_source)
        self.assertIn("scene_prompt_plan.json", segmentation_source)
        self.assertIn("hierarchical_scene_profiles", segmentation_source)
        self.assertIn("generic_miscellaneous_duplicates_known_class", segmentation_source)
        self.assertIn("Transient storage error while writing", segmentation_source)
        self.assertIn("def prune_stale_generated_outputs", segmentation_source)
        self.assertIn("stale_output_cleanup.csv", segmentation_source)
        self.assertIn("write_json(CONFIG_PATH, CONFIG)", segmentation_source)
        self.assertNotIn("CONFIG_PATH.write_text", segmentation_source)
        self.assertIn("run_pipeline_stage('track')", segmentation_source)
        self.assertIn("run_pipeline_stage('layout')", segmentation_source)
        self.assertIn("run_pipeline_stage('audit')", segmentation_source)
        self.assertIn("/content/facebookresearch_sam2", segmentation_source)
        self.assertIn("from sam2.build_sam import build_sam2_video_predictor", segmentation_source)
        self.assertIn("SAM2_BUILD_CUDA", segmentation_source)
        self.assertIn('\"sam2_use_cuda_postprocessing\": False', segmentation_source)
        self.assertIn('\"maximum_hole_area_pixels\": 64', segmentation_source)
        self.assertIn("apply_postprocessing=use_cuda_postprocessing", segmentation_source)
        self.assertIn("++model.fill_hole_area=0", segmentation_source)
        self.assertNotIn("from pipelines.tracked_segmentation import", segmentation_source)
        self.assertNotIn("SAM2_REPO = Path('/content/sam2')", segmentation_source)
        self.assertNotIn("'-m',\n        'pipelines.tracked_segmentation'", segmentation_source)

        batch = json.loads(
            (root / "object_splat_batch_pipeline.ipynb").read_text(encoding="utf-8")
        )
        batch_source = "\n".join("".join(cell.get("source", [])) for cell in batch["cells"])
        self.assertIn("REQUIRE_TRACKED_SEGMENTATION_V2 = True", batch_source)
        self.assertIn("ALLOW_FILENAME_MASK_FALLBACK = False", batch_source)
        self.assertIn("REUSE_RAW_EXPORTS_FROM_BATCH_ID = None", batch_source)
        self.assertIn("RUN_POST_EXPORT_CLEANUP = False", batch_source)
        self.assertIn("PREFER_CLEANED_SPLAT_IN_MANIFEST = False", batch_source)
        self.assertEqual(batch_source.count("detection_payloads = ["), 1)


if __name__ == "__main__":
    unittest.main()
