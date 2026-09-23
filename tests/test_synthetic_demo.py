from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from scripts.generate_synthetic_demo import GAUSSIAN_STRUCT, OBJECTS, generate


class SyntheticDemoTests(unittest.TestCase):
    def test_demo_is_deterministic_and_self_describing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generate(root, seed=7)

            boxes = json.loads((root / "object_bounding_boxes.json").read_text())
            manifest = json.loads((root / "semantic_scene_manifest.json").read_text())
            self.assertTrue(boxes["synthetic"])
            self.assertTrue(manifest["synthetic"])
            self.assertEqual(len(boxes["boxes"]), len(OBJECTS))
            self.assertEqual(len(manifest["objects"]), len(OBJECTS))

            splat = (root / "splat.ply").read_bytes()
            header, body = splat.split(b"end_header\n", 1)
            expected_count = sum(obj.point_count for obj in OBJECTS)
            self.assertIn(f"element vertex {expected_count}".encode(), header)
            self.assertIn(b"Synthetic portfolio demo", header)
            self.assertEqual(len(body), expected_count * GAUSSIAN_STRUCT.size)
            first_record = struct.unpack("<62f", body[: GAUSSIAN_STRUCT.size])
            self.assertEqual(len(first_record), 62)


if __name__ == "__main__":
    unittest.main()

