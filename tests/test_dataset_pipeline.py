from __future__ import annotations

import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from PIL import Image

from dataset_pipeline import (
    CanonicalBox,
    SampleRecord,
    allocate_splits,
    build_groups,
    load_config,
    parse_annotation_line,
    run_pipeline,
)


ROOT = Path(__file__).resolve().parents[1]


class AnnotationParsingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(ROOT / "config" / "dataset_pipeline.json", False)
        cls.mapping = {5: 0, 6: 1, 8: 2}
        cls.names = {0: "FULL_UID", 1: "MASKED_UID", 2: "QR"}

    def test_bbox_conversion(self) -> None:
        annotation = parse_annotation_line(
            "5 0.5 0.5 0.4 0.2", 1, self.mapping, self.names, self.config
        )
        self.assertEqual(annotation.status, "VALID")
        self.assertEqual(annotation.shape_type, "bbox")
        self.assertEqual(annotation.canonical_class_id, 0)
        self.assertAlmostEqual(annotation.canonical_box.width, 0.4)
        self.assertAlmostEqual(annotation.canonical_box.height, 0.2)

    def test_polygon_conversion(self) -> None:
        annotation = parse_annotation_line(
            "8 0.1 0.2 0.7 0.2 0.8 0.6 0.2 0.7", 1, self.mapping, self.names, self.config
        )
        self.assertEqual(annotation.status, "VALID")
        self.assertEqual(annotation.shape_type, "polygon")
        self.assertEqual(annotation.canonical_class_id, 2)
        self.assertAlmostEqual(annotation.canonical_box.x, 0.45)
        self.assertAlmostEqual(annotation.canonical_box.y, 0.45)
        self.assertAlmostEqual(annotation.canonical_box.width, 0.7)
        self.assertAlmostEqual(annotation.canonical_box.height, 0.5)

    def test_small_boundary_overflow_is_clipped(self) -> None:
        annotation = parse_annotation_line(
            "5 0.05 0.5 0.1001 0.2", 1, self.mapping, self.names, self.config
        )
        self.assertEqual(annotation.status, "VALID")
        self.assertTrue(annotation.clipped)
        self.assertIn("CLIPPED_TO_IMAGE_BOUNDARY", annotation.issue_codes)
        self.assertAlmostEqual(annotation.canonical_box.x - annotation.canonical_box.width / 2, 0.0)

    def test_major_boundary_overflow_is_invalid(self) -> None:
        annotation = parse_annotation_line(
            "5 0.02 0.5 0.2 0.2", 1, self.mapping, self.names, self.config
        )
        self.assertEqual(annotation.status, "INVALID")
        self.assertIn("OUT_OF_BOUNDS_MAJOR", annotation.issue_codes)

    def test_malformed_polygon_is_rejected(self) -> None:
        annotation = parse_annotation_line(
            "8 0.1 0.2 0.3 0.4 0.5", 1, self.mapping, self.names, self.config
        )
        self.assertEqual(annotation.status, "INVALID")
        self.assertIn("UNSAFE_COORDINATE_ARITY", annotation.issue_codes)

    def test_uid_redaction_is_not_masked_uid(self) -> None:
        annotation = parse_annotation_line(
            "9 0.5 0.5 0.2 0.1", 1, self.mapping, self.names, self.config
        )
        self.assertEqual(annotation.status, "VALID")
        self.assertIsNone(annotation.canonical_class_id)


def fake_record(sample_id: str, image_hash: str, lineage: str, phash: str = "0" * 16) -> SampleRecord:
    record = SampleRecord(
        sample_id=sample_id,
        source_partition="train",
        image_path=Path("unused.jpg"),
        label_path=Path("unused.txt"),
        source_path_hmac=sample_id.lower().ljust(64, "0")[:64],
        filename_lineage_hmac=lineage,
        source_lineage_hmac=lineage,
        image_sha256=image_hash,
        label_sha256=(image_hash[::-1]),
        width=640,
        height=640,
        phash=phash,
        original_class_counts=Counter({5: 1, 8: 1}),
        canonical_class_counts=Counter({0: 1, 2: 1}),
    )
    return record


class GroupingAndSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(ROOT / "config" / "dataset_pipeline.json", False)

    def test_exact_duplicate_grouping(self) -> None:
        records = [
            fake_record("IMG_A", "a" * 64, "1" * 64),
            fake_record("IMG_B", "a" * 64, "2" * 64),
            fake_record("IMG_C", "c" * 64, "3" * 64, "f" * 16),
        ]
        groups, reasons, _ = build_groups(records, self.config, {})
        self.assertEqual(sorted(len(members) for members in groups.values()), [1, 2])
        self.assertGreaterEqual(reasons["EXACT_SHA256"], 1)

    def test_lineage_grouping(self) -> None:
        lineage = "9" * 64
        records = [
            fake_record("IMG_A", "a" * 64, lineage),
            fake_record("IMG_B", "b" * 64, lineage),
        ]
        groups, reasons, _ = build_groups(records, self.config, {})
        self.assertEqual(len(groups), 1)
        self.assertGreaterEqual(reasons["SOURCE_LINEAGE"], 1)

    def test_deterministic_group_aware_splits(self) -> None:
        groups = {}
        for index in range(20):
            record = fake_record(f"IMG_{index:02d}", f"{index:064x}", f"{index + 100:064x}")
            group_id = f"GRP_{index:02d}"
            record.group_id = group_id
            groups[group_id] = [record]
        first, first_counts = allocate_splits(groups, self.config["split_ratios"], self.config["seed"])
        second, second_counts = allocate_splits(groups, self.config["split_ratios"], self.config["seed"])
        self.assertEqual(first, second)
        self.assertEqual(first_counts, second_counts)
        self.assertEqual(set(first), set(groups))


class EndToEndDeterminismTests(unittest.TestCase):
    def _create_fixture(self, root: Path) -> None:
        for split in ("train", "valid", "test"):
            (root / split / "images").mkdir(parents=True)
            (root / split / "labels").mkdir(parents=True)
        examples = [
            ("train", "alpha", (255, 0, 0), "5 0.5 0.5 0.4 0.1\n8 0.8 0.7 0.2 0.2\n"),
            ("train", "beta", (0, 255, 0), "6 0.5 0.5 0.4 0.1\n"),
            ("valid", "gamma", (0, 0, 255), "8 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3\n"),
            ("test", "delta", (127, 127, 0), "5 0.5 0.5 0.4 0.1\n"),
        ]
        for split, stem, color, label in examples:
            image = Image.new("RGB", (64, 64), color)
            image.putpixel((0, 0), (0, 0, 0))
            image.save(root / split / "images" / f"{stem}.jpg")
            (root / split / "labels" / f"{stem}.txt").write_text(label, encoding="ascii")

    def test_complete_pipeline_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._create_fixture(root)
            config_path = root / "config.json"
            config = json.loads((ROOT / "config" / "dataset_pipeline.json").read_text(encoding="utf-8"))
            config_path.write_text(json.dumps(config), encoding="utf-8")
            key_path = root / "key.bin"
            key_path.write_bytes(b"k" * 32)

            class Args:
                include_secondary = False
                identity_hmac_map = None

            args = Args()
            args.root = root
            args.config = config_path
            args.hmac_key = key_path
            first = run_pipeline(args)
            first_manifest = (root / "data" / "manifests" / "split_manifest.json").read_bytes()
            first_splits = {
                split: (root / "data" / "splits" / f"{split}.txt").read_bytes()
                for split in ("train", "val", "test")
            }
            second = run_pipeline(args)
            second_manifest = (root / "data" / "manifests" / "split_manifest.json").read_bytes()
            second_splits = {
                split: (root / "data" / "splits" / f"{split}.txt").read_bytes()
                for split in ("train", "val", "test")
            }
            self.assertEqual(first_manifest, second_manifest)
            self.assertEqual(first_splits, second_splits)
            self.assertEqual(first["splits"], second["splits"])
            all_entries = [entry for values in second_splits.values() for entry in values.splitlines()]
            self.assertEqual(len(all_entries), len(set(all_entries)))


if __name__ == "__main__":
    unittest.main()
