"""Canonical, privacy-safe preparation for the local Aadhaar detection dataset.

This tool never OCRs images, decodes QR payloads, or writes source filenames to
reports. Source annotations are preserved in a checksummed JSONL provenance
record. Generated reports contain only aggregate values and pseudonymous IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import shutil
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageStat, UnidentifiedImageError


ORIGINAL_CLASS_NAMES = {
    0: "aadhaar_address",
    1: "aadhaar_dob",
    2: "aadhaar_gender",
    3: "aadhaar_holder_name",
    4: "aadhaar_logo",
    5: "aadhaar_no",
    6: "aadhaar_no_already_masked",
    7: "aadhaar_photo",
    8: "aadhaar_qr",
    9: "aadhar_no_mask",
    10: "emblem",
    11: "gov_logo",
}
PRIMARY_NAMES = {0: "FULL_UID", 1: "MASKED_UID", 2: "QR"}
SPLITS = ("train", "val", "test")
SOURCE_SPLITS = ("train", "valid", "test")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def hmac_hex(key: bytes, domain: str, value: str) -> str:
    message = f"{domain}\0{value}".encode("utf-8", errors="surrogatepass")
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def safe_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite coordinate")
    return result


def format_float(value: float) -> str:
    text = f"{value:.10f}".rstrip("0").rstrip(".")
    return text if text and text != "-0" else "0"


def normalize_lineage(stem: str, aggressive: bool) -> str:
    value = unicodedata.normalize("NFKC", stem).casefold()
    value = re.sub(r"\.rf\.[0-9a-f]{16,}$", "", value)
    value = re.sub(r"^(?:90|180|270)[-_]+", "", value)
    value = re.sub(r"(?:[-_]+copy)+$", "", value)
    value = re.sub(r"[-_]+(?:jpg|jpeg|png|webp)$", "", value)
    if aggressive:
        value = re.sub(r"[-_]20\d{6}[-_]\d{6}[-_]masked$", "", value)
        value = re.sub(r"[-_]masked(?:[-_]output)?(?:[-_]page[-_]\d+)?$", "", value)
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value or "unnamed_source"


@dataclass(frozen=True)
class CanonicalBox:
    x: float
    y: float
    width: float
    height: float

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        small = min(self.width, self.height)
        return math.inf if small <= 0 else max(self.width, self.height) / small

    def yolo(self, class_id: int) -> str:
        return " ".join(
            [str(class_id), format_float(self.x), format_float(self.y), format_float(self.width), format_float(self.height)]
        )


@dataclass
class ParsedAnnotation:
    line_number: int
    original_class_id: int | None
    original_class_name: str | None
    shape_type: str
    original_coordinates: list[float]
    status: str
    issue_codes: list[str] = field(default_factory=list)
    canonical_class_id: int | None = None
    canonical_class_name: str | None = None
    canonical_box: CanonicalBox | None = None
    clipped: bool = False
    suspicious: bool = False

    def provenance_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.canonical_box is not None:
            result["canonical_box"] = asdict(self.canonical_box)
        return result


@dataclass
class SampleRecord:
    sample_id: str
    source_partition: str
    image_path: Path
    label_path: Path
    source_path_hmac: str
    filename_lineage_hmac: str
    source_lineage_hmac: str
    image_sha256: str
    label_sha256: str
    width: int = 0
    height: int = 0
    image_format: str = ""
    image_bytes: int = 0
    phash: str = ""
    blank_image: bool = False
    corrupt_image: bool = False
    annotations: list[ParsedAnnotation] = field(default_factory=list)
    original_class_counts: Counter[int] = field(default_factory=Counter)
    canonical_class_counts: Counter[int] = field(default_factory=Counter)
    invalid_annotation_count: int = 0
    clipped_annotation_count: int = 0
    suspicious_annotation_count: int = 0
    annotation_signature: str = ""
    eligible: bool = True
    group_id: str = ""
    assigned_split: str = ""

    @property
    def aspect_ratio(self) -> float:
        if not self.width or not self.height:
            return 0.0
        return max(self.width, self.height) / min(self.width, self.height)

    @property
    def canonical_object_count(self) -> int:
        return sum(self.canonical_class_counts.values())


class DisjointSet:
    def __init__(self, values: Iterable[str]) -> None:
        materialized = list(values)
        self.parent = {value: value for value in materialized}
        self.rank = {value: 0 for value in materialized}
        self.reasons: defaultdict[str, set[str]] = defaultdict(set)

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: str, right: str, reason: str) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            self.reasons[a].add(reason)
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        self.reasons[a].update(self.reasons.pop(b, set()))
        self.reasons[a].add(reason)
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def dct_matrix(size: int) -> np.ndarray:
    positions = np.arange(size, dtype=np.float64)
    frequencies = positions[:, None]
    matrix = np.cos((math.pi / size) * (positions + 0.5) * frequencies)
    matrix[0, :] *= math.sqrt(1.0 / size)
    matrix[1:, :] *= math.sqrt(2.0 / size)
    return matrix


_DCT32 = dct_matrix(32)


def perceptual_hash(image: Image.Image) -> str:
    hashes: list[int] = []
    grayscale = image.convert("L")
    for angle in (0, 90, 180, 270):
        rotated = grayscale if angle == 0 else grayscale.rotate(angle, expand=True)
        resized = rotated.resize((32, 32), Image.Resampling.LANCZOS)
        array = np.asarray(resized, dtype=np.float64)
        coefficients = _DCT32 @ array @ _DCT32.T
        low = coefficients[:8, :8].flatten()
        median = float(np.median(low[1:]))
        bits = 0
        for coefficient in low:
            bits = (bits << 1) | int(coefficient >= median)
        hashes.append(bits)
    return f"{min(hashes):016x}"


def clip_coordinate(value: float, tolerance: float) -> tuple[float, bool, bool]:
    if 0.0 <= value <= 1.0:
        return value, False, False
    overflow = -value if value < 0 else value - 1.0
    if overflow <= tolerance:
        return min(1.0, max(0.0, value)), True, False
    return value, False, True


def finalize_box(
    left: float,
    top: float,
    right: float,
    bottom: float,
    tolerance: float,
    min_area: float,
    max_area: float,
    max_aspect: float,
) -> tuple[CanonicalBox | None, list[str], bool, bool]:
    issues: list[str] = []
    clipped = False
    major = False
    clipped_values: list[float] = []
    for value in (left, top, right, bottom):
        normalized, was_clipped, was_major = clip_coordinate(value, tolerance)
        clipped_values.append(normalized)
        clipped = clipped or was_clipped
        major = major or was_major
    if major:
        return None, ["OUT_OF_BOUNDS_MAJOR"], False, False
    left, top, right, bottom = clipped_values
    if clipped:
        issues.append("CLIPPED_TO_IMAGE_BOUNDARY")
    if right <= left or bottom <= top:
        return None, issues + ["NON_POSITIVE_BOX"], clipped, False
    box = CanonicalBox(
        x=(left + right) / 2.0,
        y=(top + bottom) / 2.0,
        width=right - left,
        height=bottom - top,
    )
    suspicious = False
    if box.area < min_area:
        issues.append("SUSPICIOUS_TINY_AREA")
        suspicious = True
    if box.area > max_area:
        issues.append("SUSPICIOUS_LARGE_AREA")
        suspicious = True
    if box.aspect_ratio > max_aspect:
        issues.append("SUSPICIOUS_ASPECT_RATIO")
        suspicious = True
    return box, issues, clipped, suspicious


def parse_annotation_line(
    line: str,
    line_number: int,
    class_mapping: Mapping[int, int],
    canonical_names: Mapping[int, str],
    config: Mapping[str, Any],
) -> ParsedAnnotation:
    tokens = line.split()
    if not tokens:
        return ParsedAnnotation(line_number, None, None, "invalid", [], "INVALID", ["EMPTY_LINE"])
    try:
        class_id = int(tokens[0])
    except ValueError:
        return ParsedAnnotation(line_number, None, None, "invalid", [], "INVALID", ["INVALID_CLASS_ID"])
    class_name = ORIGINAL_CLASS_NAMES.get(class_id)
    if class_name is None:
        return ParsedAnnotation(line_number, class_id, None, "invalid", [], "INVALID", ["UNKNOWN_CLASS_ID"])
    try:
        coordinates = [safe_float(value) for value in tokens[1:]]
    except ValueError:
        return ParsedAnnotation(line_number, class_id, class_name, "invalid", [], "INVALID", ["NON_NUMERIC_COORDINATE"])

    shape_type: str
    if len(coordinates) == 4:
        shape_type = "bbox"
        x, y, width, height = coordinates
        if width <= 0 or height <= 0:
            return ParsedAnnotation(
                line_number, class_id, class_name, shape_type, coordinates, "INVALID", ["NON_POSITIVE_SIZE"]
            )
        left, top, right, bottom = x - width / 2.0, y - height / 2.0, x + width / 2.0, y + height / 2.0
    elif len(coordinates) >= 6 and len(coordinates) % 2 == 0:
        shape_type = "polygon"
        xs = coordinates[0::2]
        ys = coordinates[1::2]
        left, top, right, bottom = min(xs), min(ys), max(xs), max(ys)
    else:
        return ParsedAnnotation(
            line_number,
            class_id,
            class_name,
            "invalid",
            coordinates,
            "INVALID",
            ["UNSAFE_COORDINATE_ARITY"],
        )

    box, issues, clipped, suspicious = finalize_box(
        left,
        top,
        right,
        bottom,
        float(config["clip_tolerance"]),
        float(config["suspicious_min_area"]),
        float(config["suspicious_max_area"]),
        float(config["suspicious_max_aspect_ratio"]),
    )
    status = "VALID" if box is not None else "INVALID"
    canonical_id = class_mapping.get(class_id) if box is not None else None
    return ParsedAnnotation(
        line_number=line_number,
        original_class_id=class_id,
        original_class_name=class_name,
        shape_type=shape_type,
        original_coordinates=coordinates,
        status=status,
        issue_codes=issues,
        canonical_class_id=canonical_id,
        canonical_class_name=canonical_names.get(canonical_id) if canonical_id is not None else None,
        canonical_box=box,
        clipped=clipped,
        suspicious=suspicious,
    )


def load_or_create_hmac_key(path: Path) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        key = path.read_bytes()
        if len(key) != 32:
            raise RuntimeError("Local HMAC key has an invalid length")
        return key
    key = secrets.token_bytes(32)
    path.write_bytes(key)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return key


def discover_samples(root: Path) -> list[tuple[str, Path, Path]]:
    discovered: list[tuple[str, Path, Path]] = []
    for partition in SOURCE_SPLITS:
        image_dir = root / partition / "images"
        label_dir = root / partition / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Missing source partition structure for {partition}")
        image_paths = sorted(
            (path for path in image_dir.iterdir() if path.is_file() and path.suffix.casefold() in {".jpg", ".jpeg"}),
            key=lambda path: path.name.casefold(),
        )
        label_by_stem = {path.stem: path for path in label_dir.glob("*.txt")}
        for image_path in image_paths:
            label_path = label_by_stem.get(image_path.stem)
            if label_path is None:
                raise RuntimeError("An image is missing its paired annotation")
            discovered.append((partition, image_path, label_path))
        orphan_count = len(set(label_by_stem) - {path.stem for path in image_paths})
        if orphan_count:
            raise RuntimeError(f"Found {orphan_count} orphan annotations in {partition}")
    return discovered


def build_class_configuration(config: Mapping[str, Any]) -> tuple[dict[int, int], dict[int, str]]:
    mapping = {int(source): int(target) for source, target in config["primary_class_mapping"].items()}
    names = {int(class_id): name for class_id, name in config["canonical_classes"].items()}
    if config.get("include_secondary_classes"):
        mapping.update({int(source): int(target) for source, target in config["secondary_class_mapping"].items()})
        names.update({int(class_id): name for class_id, name in config["secondary_classes"].items()})
    return mapping, names


def inspect_sample(
    root: Path,
    partition: str,
    image_path: Path,
    label_path: Path,
    key: bytes,
    class_mapping: Mapping[int, int],
    canonical_names: Mapping[int, str],
    config: Mapping[str, Any],
) -> SampleRecord:
    image_sha = sha256_file(image_path)
    label_bytes = label_path.read_bytes()
    label_sha = sha256_bytes(label_bytes)
    relative_source = image_path.relative_to(root).as_posix()
    source_path_hmac = hmac_hex(key, "source-path", relative_source)
    filename_lineage = normalize_lineage(image_path.stem, aggressive=False)
    source_lineage = normalize_lineage(image_path.stem, aggressive=True)
    sample_id = "IMG_" + hmac_hex(key, "sample-id", f"{image_sha}|{label_sha}|{source_path_hmac}")[:24].upper()
    record = SampleRecord(
        sample_id=sample_id,
        source_partition=partition,
        image_path=image_path,
        label_path=label_path,
        source_path_hmac=source_path_hmac,
        filename_lineage_hmac=hmac_hex(key, "filename-lineage", filename_lineage),
        source_lineage_hmac=hmac_hex(key, "source-lineage", source_lineage),
        image_sha256=image_sha,
        label_sha256=label_sha,
        image_bytes=image_path.stat().st_size,
    )
    try:
        with Image.open(image_path) as image:
            image.load()
            record.width, record.height = image.size
            record.image_format = image.format or "UNKNOWN"
            record.phash = perceptual_hash(image)
            grayscale = image.convert("L").resize((64, 64), Image.Resampling.BILINEAR)
            record.blank_image = ImageStat.Stat(grayscale).var[0] <= 1.0
    except (UnidentifiedImageError, OSError, ValueError):
        record.corrupt_image = True
        record.eligible = False

    text = label_bytes.decode("utf-8", errors="replace")
    for line_number, line in enumerate(text.splitlines(), 1):
        annotation = parse_annotation_line(line, line_number, class_mapping, canonical_names, config)
        record.annotations.append(annotation)
        if annotation.original_class_id is not None:
            record.original_class_counts[annotation.original_class_id] += 1
        if annotation.status == "INVALID":
            record.invalid_annotation_count += 1
            record.eligible = False
        if annotation.clipped:
            record.clipped_annotation_count += 1
        if annotation.suspicious:
            record.suspicious_annotation_count += 1
        if annotation.canonical_class_id is not None and annotation.canonical_box is not None:
            record.canonical_class_counts[annotation.canonical_class_id] += 1
    if not record.annotations:
        record.invalid_annotation_count += 1
        record.eligible = False

    signature_rows = []
    for annotation in record.annotations:
        if annotation.status != "VALID" or annotation.canonical_box is None:
            continue
        box = annotation.canonical_box
        signature_rows.append(
            (
                annotation.original_class_id,
                annotation.shape_type,
                round(box.x, 6),
                round(box.y, 6),
                round(box.width, 6),
                round(box.height, 6),
            )
        )
    record.annotation_signature = sha256_bytes(canonical_json_bytes(sorted(signature_rows)))
    if record.blank_image:
        record.eligible = False
    return record


def multiset_similarity(left: Counter[int], right: Counter[int]) -> float:
    keys = set(left) | set(right)
    denominator = sum(max(left[key], right[key]) for key in keys)
    if denominator == 0:
        return 1.0
    return sum(min(left[key], right[key]) for key in keys) / denominator


def load_identity_hmac_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Identity HMAC map must be a JSON object keyed by image SHA-256")
    result: dict[str, str] = {}
    for image_hash, identity_hmac in payload.items():
        if not re.fullmatch(r"[0-9a-f]{64}", str(image_hash)) or not re.fullmatch(r"[0-9a-f]{64}", str(identity_hmac)):
            raise ValueError("Identity HMAC map may contain only lowercase SHA-256/HMAC hex values")
        result[str(image_hash)] = str(identity_hmac)
    return result


def build_groups(
    records: Sequence[SampleRecord], config: Mapping[str, Any], identity_map: Mapping[str, str]
) -> tuple[dict[str, list[SampleRecord]], Counter[str], list[tuple[str, str, str]]]:
    eligible = [record for record in records if record.eligible]
    dsu = DisjointSet(record.sample_id for record in eligible)
    edges: list[tuple[str, str, str]] = []

    def union_index(index: Mapping[str, list[SampleRecord]], reason: str) -> None:
        for members in index.values():
            if len(members) < 2:
                continue
            first = members[0]
            for member in members[1:]:
                dsu.union(first.sample_id, member.sample_id, reason)
                edges.append((first.sample_id, member.sample_id, reason))

    exact: defaultdict[str, list[SampleRecord]] = defaultdict(list)
    filename_lineage: defaultdict[str, list[SampleRecord]] = defaultdict(list)
    source_lineage: defaultdict[str, list[SampleRecord]] = defaultdict(list)
    identity: defaultdict[str, list[SampleRecord]] = defaultdict(list)
    for record in eligible:
        exact[record.image_sha256].append(record)
        filename_lineage[record.filename_lineage_hmac].append(record)
        source_lineage[record.source_lineage_hmac].append(record)
        identity_hmac = identity_map.get(record.image_sha256)
        if identity_hmac:
            identity[identity_hmac].append(record)
    union_index(exact, "EXACT_SHA256")
    union_index(filename_lineage, "FILENAME_LINEAGE")
    union_index(source_lineage, "SOURCE_LINEAGE")
    union_index(identity, "AUTHORIZED_IDENTITY_HMAC")

    phash_threshold = int(config["phash_distance_threshold"])
    annotation_threshold = float(config["annotation_similarity_threshold"])
    aspect_tolerance = float(config["aspect_ratio_relative_tolerance"])
    phash_values = [int(record.phash, 16) for record in eligible]
    for index, left in enumerate(eligible):
        left_hash = phash_values[index]
        for right_index in range(index + 1, len(eligible)):
            right = eligible[right_index]
            distance = (left_hash ^ phash_values[right_index]).bit_count()
            if distance > phash_threshold:
                continue
            aspect_denominator = max(left.aspect_ratio, right.aspect_ratio, 1e-9)
            aspect_difference = abs(left.aspect_ratio - right.aspect_ratio) / aspect_denominator
            if aspect_difference > aspect_tolerance:
                continue
            similarity = multiset_similarity(left.original_class_counts, right.original_class_counts)
            if similarity < annotation_threshold:
                continue
            dsu.union(left.sample_id, right.sample_id, "PHASH_ANNOTATION_SIMILARITY")
            edges.append((left.sample_id, right.sample_id, "PHASH_ANNOTATION_SIMILARITY"))

    components: defaultdict[str, list[SampleRecord]] = defaultdict(list)
    for record in eligible:
        components[dsu.find(record.sample_id)].append(record)
    groups: dict[str, list[SampleRecord]] = {}
    reason_counts: Counter[str] = Counter(reason for _, _, reason in edges)
    for root, members in components.items():
        member_ids = sorted(member.sample_id for member in members)
        group_id = "GRP_" + sha256_bytes("\n".join(member_ids).encode("ascii"))[:24].upper()
        groups[group_id] = sorted(members, key=lambda member: member.sample_id)
        for member in members:
            member.group_id = group_id
    return groups, reason_counts, edges


def group_feature_vector(members: Sequence[SampleRecord]) -> dict[str, int]:
    return {
        "images": len(members),
        "full_uid": sum(member.canonical_class_counts[0] for member in members),
        "masked_uid": sum(member.canonical_class_counts[1] for member in members),
        "qr": sum(member.canonical_class_counts[2] for member in members),
        "multi_uid_images": sum(
            member.canonical_class_counts[0] + member.canonical_class_counts[1] > 1 for member in members
        ),
    }


def allocate_splits(
    groups: Mapping[str, Sequence[SampleRecord]], ratios: Mapping[str, float], seed: int
) -> tuple[dict[str, str], dict[str, Counter[str]]]:
    features = {group_id: group_feature_vector(members) for group_id, members in groups.items()}
    total = Counter()
    for vector in features.values():
        total.update(vector)
    targets = {split: {key: total[key] * float(ratios[split]) for key in total} for split in SPLITS}
    current = {split: Counter() for split in SPLITS}

    rarity_denominators = {key: max(total[key], 1) for key in ("masked_uid", "qr", "full_uid", "multi_uid_images")}

    def ordering_key(group_id: str) -> tuple[float, int, str]:
        vector = features[group_id]
        rarity = sum(vector[key] / rarity_denominators[key] for key in rarity_denominators)
        tie = sha256_bytes(f"{seed}:{group_id}".encode("ascii"))
        return (-rarity, -vector["images"], tie)

    assignments: dict[str, str] = {}
    weights = {"images": 5.0, "full_uid": 2.0, "masked_uid": 4.0, "qr": 2.0, "multi_uid_images": 2.0}
    for group_id in sorted(groups, key=ordering_key):
        vector = features[group_id]
        candidates: list[tuple[float, str, str]] = []
        for split in SPLITS:
            cost = 0.0
            for feature, weight in weights.items():
                target = max(targets[split][feature], 1.0)
                before = (current[split][feature] - target) / target
                after = (current[split][feature] + vector[feature] - target) / target
                cost += weight * (after * after - before * before)
            tie = sha256_bytes(f"{seed}:{group_id}:{split}".encode("ascii"))
            candidates.append((cost, tie, split))
        _, _, selected = min(candidates)
        assignments[group_id] = selected
        current[selected].update(vector)
        for member in groups[group_id]:
            member.assigned_split = selected
    return assignments, current


def reset_generated_outputs(data_root: Path) -> None:
    data_root = data_root.resolve()
    for relative in (Path("canonical"), Path("splits"), Path("manifests")):
        target = (data_root / relative).resolve()
        if target.parent != data_root:
            raise RuntimeError("Refusing to clear an output outside the data directory")
        if target.exists():
            shutil.rmtree(target)
    (data_root / "canonical" / "images").mkdir(parents=True)
    (data_root / "canonical" / "labels").mkdir(parents=True)
    (data_root / "splits").mkdir(parents=True)
    (data_root / "manifests").mkdir(parents=True)


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value))


def canonical_annotation_rows(record: SampleRecord) -> Iterator[ParsedAnnotation]:
    for annotation in record.annotations:
        if annotation.status == "VALID" and annotation.canonical_class_id is not None and annotation.canonical_box is not None:
            yield annotation


def materialize_outputs(
    root: Path,
    records: Sequence[SampleRecord],
    groups: Mapping[str, Sequence[SampleRecord]],
    assignments: Mapping[str, str],
    reason_counts: Mapping[str, int],
    edges: Sequence[tuple[str, str, str]],
    config: Mapping[str, Any],
    canonical_names: Mapping[int, str],
    key: bytes,
) -> dict[str, Any]:
    data_root = root / "data"
    reset_generated_outputs(data_root)
    canonical_images = data_root / "canonical" / "images"
    canonical_labels = data_root / "canonical" / "labels"
    manifests = data_root / "manifests"
    splits_dir = data_root / "splits"

    split_lines: dict[str, list[str]] = {split: [] for split in SPLITS}
    image_rows: list[dict[str, Any]] = []
    provenance_path = manifests / "provenance.jsonl"
    with provenance_path.open("wb") as provenance:
        for record in sorted(records, key=lambda item: item.sample_id):
            provenance_record = {
                "sample_id": record.sample_id,
                "source_partition": record.source_partition,
                "source_path_hmac": record.source_path_hmac,
                "filename_lineage_hmac": record.filename_lineage_hmac,
                "source_lineage_hmac": record.source_lineage_hmac,
                "image_sha256": record.image_sha256,
                "label_sha256": record.label_sha256,
                "annotations": [annotation.provenance_dict() for annotation in record.annotations],
            }
            provenance.write(canonical_json_bytes(provenance_record))

            canonical_image_rel: str | None = None
            canonical_label_rel: str | None = None
            if record.eligible:
                image_name = f"{record.sample_id}.jpg"
                label_name = f"{record.sample_id}.txt"
                shutil.copyfile(record.image_path, canonical_images / image_name)
                label_lines = [
                    annotation.canonical_box.yolo(annotation.canonical_class_id)
                    for annotation in canonical_annotation_rows(record)
                    if annotation.canonical_box is not None and annotation.canonical_class_id is not None
                ]
                (canonical_labels / label_name).write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="ascii")
                canonical_image_rel = f"data/canonical/images/{image_name}"
                canonical_label_rel = f"data/canonical/labels/{label_name}"
                split_lines[record.assigned_split].append(canonical_image_rel)

            image_rows.append(
                {
                    "sample_id": record.sample_id,
                    "source_partition": record.source_partition,
                    "source_path_hmac": record.source_path_hmac,
                    "filename_lineage_hmac": record.filename_lineage_hmac,
                    "source_lineage_hmac": record.source_lineage_hmac,
                    "image_sha256": record.image_sha256,
                    "label_sha256": record.label_sha256,
                    "width": record.width,
                    "height": record.height,
                    "image_format": record.image_format,
                    "image_bytes": record.image_bytes,
                    "phash": record.phash,
                    "blank_image": record.blank_image,
                    "corrupt_image": record.corrupt_image,
                    "eligible": record.eligible,
                    "original_object_count": len(record.annotations),
                    "canonical_object_count": record.canonical_object_count,
                    "full_uid_count": record.canonical_class_counts[0],
                    "masked_uid_count": record.canonical_class_counts[1],
                    "qr_count": record.canonical_class_counts[2],
                    "invalid_annotation_count": record.invalid_annotation_count,
                    "clipped_annotation_count": record.clipped_annotation_count,
                    "suspicious_annotation_count": record.suspicious_annotation_count,
                    "annotation_signature": record.annotation_signature,
                    "group_id": record.group_id or None,
                    "assigned_split": record.assigned_split or None,
                    "canonical_image": canonical_image_rel,
                    "canonical_label": canonical_label_rel,
                }
            )

    for split in SPLITS:
        content = "\n".join(sorted(split_lines[split]))
        (splits_dir / f"{split}.txt").write_text(content + ("\n" if content else ""), encoding="ascii")

    group_rows: list[dict[str, Any]] = []
    edge_reasons_by_group: defaultdict[str, set[str]] = defaultdict(set)
    sample_to_group = {member.sample_id: group_id for group_id, members in groups.items() for member in members}
    for left, right, reason in edges:
        if sample_to_group.get(left) == sample_to_group.get(right) and sample_to_group.get(left):
            edge_reasons_by_group[sample_to_group[left]].add(reason)
    for group_id, members in sorted(groups.items()):
        features = group_feature_vector(members)
        group_rows.append(
            {
                "group_id": group_id,
                "assigned_split": assignments[group_id],
                "member_count": len(members),
                "member_ids": [member.sample_id for member in members],
                "evidence_reasons": sorted(edge_reasons_by_group[group_id]),
                "exact_image_hash_count": len({member.image_sha256 for member in members}),
                "filename_lineage_count": len({member.filename_lineage_hmac for member in members}),
                "source_lineage_count": len({member.source_lineage_hmac for member in members}),
                **features,
            }
        )

    metadata = {b"preprocessing_version": str(config["preprocessing_version"]).encode("utf-8")}
    image_table = pa.Table.from_pylist(image_rows).replace_schema_metadata(metadata)
    group_table = pa.Table.from_pylist(group_rows).replace_schema_metadata(metadata)
    pq.write_table(image_table, manifests / "images.parquet", compression="zstd")
    pq.write_table(group_table, manifests / "groups.parquet", compression="zstd")

    eligible = [record for record in records if record.eligible]
    annotation_format_counts = Counter(annotation.shape_type for record in records for annotation in record.annotations)
    canonical_instances = Counter()
    images_with_class = Counter()
    for record in eligible:
        canonical_instances.update(record.canonical_class_counts)
        for class_id, count in record.canonical_class_counts.items():
            if count:
                images_with_class[class_id] += 1
    dimensions = Counter(f"{record.width}x{record.height}" for record in records if not record.corrupt_image)
    original_split_leakage = 0
    for members in groups.values():
        if len({member.source_partition for member in members}) > 1:
            original_split_leakage += 1
    final_edge_leakage = sum(
        1
        for left, right, _ in edges
        if sample_to_group.get(left) and assignments[sample_to_group[left]] != assignments[sample_to_group[right]]
    )
    split_counts = Counter(record.assigned_split for record in eligible)
    split_class_counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        counts = Counter()
        for record in eligible:
            if record.assigned_split == split:
                counts.update(record.canonical_class_counts)
        split_class_counts[split] = {canonical_names[class_id]: counts[class_id] for class_id in sorted(canonical_names)}

    qc_report = {
        "schema_version": "1.0",
        "preprocessing_version": config["preprocessing_version"],
        "generation_timestamp": config["generation_timestamp"],
        "privacy": {
            "contains_source_filenames": False,
            "contains_raw_uid": False,
            "contains_qr_payload": False,
            "pseudonymous_ids_only": True,
        },
        "source": {
            "images": len(records),
            "labels": len(records),
            "partitions": dict(Counter(record.source_partition for record in records)),
        },
        "images": {
            "eligible": len(eligible),
            "excluded": len(records) - len(eligible),
            "blank": sum(record.blank_image for record in records),
            "corrupt": sum(record.corrupt_image for record in records),
            "unique_dimensions": len(dimensions),
            "top_dimensions": dict(dimensions.most_common(20)),
            "min_width": min((record.width for record in records if record.width), default=0),
            "max_width": max((record.width for record in records), default=0),
            "min_height": min((record.height for record in records if record.height), default=0),
            "max_height": max((record.height for record in records), default=0),
        },
        "annotations": {
            "total_source_objects": sum(len(record.annotations) for record in records),
            "format_counts": dict(annotation_format_counts),
            "canonical_instances": {canonical_names[class_id]: canonical_instances[class_id] for class_id in sorted(canonical_names)},
            "images_with_class": {canonical_names[class_id]: images_with_class[class_id] for class_id in sorted(canonical_names)},
            "invalid": sum(record.invalid_annotation_count for record in records),
            "clipped": sum(record.clipped_annotation_count for record in records),
            "suspicious": sum(record.suspicious_annotation_count for record in records),
            "boxes_per_image": dict(
                sorted(Counter(record.canonical_object_count for record in eligible).items(), key=lambda item: item[0])
            ),
            "multiple_uid_occurrence_images": sum(
                record.canonical_class_counts[0] + record.canonical_class_counts[1] > 1 for record in eligible
            ),
            "qr_images": sum(record.canonical_class_counts[2] > 0 for record in eligible),
            "class_imbalance_ratio_max_to_min": round(
                max(canonical_instances[class_id] for class_id in PRIMARY_NAMES)
                / max(1, min(canonical_instances[class_id] for class_id in PRIMARY_NAMES)),
                6,
            ),
        },
        "duplicates_and_leakage": {
            "groups": len(groups),
            "multi_member_groups": sum(len(members) > 1 for members in groups.values()),
            "largest_group": max((len(members) for members in groups.values()), default=0),
            "edge_reason_counts": dict(sorted(reason_counts.items())),
            "original_partition_crossing_groups": original_split_leakage,
            "final_cross_split_edges": final_edge_leakage,
            "final_group_overlap": 0,
        },
        "splits": {
            "seed": config["seed"],
            "target_ratios": config["split_ratios"],
            "image_counts": dict(split_counts),
            "actual_ratios": {split: round(split_counts[split] / max(1, len(eligible)), 6) for split in SPLITS},
            "class_instances": split_class_counts,
        },
    }
    write_json(manifests / "qc_report.json", qc_report)

    split_manifest = {
        "schema_version": "1.0",
        "preprocessing_version": config["preprocessing_version"],
        "generation_timestamp": config["generation_timestamp"],
        "seed": config["seed"],
        "hmac_key_fingerprint": sha256_bytes(key)[:16],
        "configuration_sha256": sha256_bytes(canonical_json_bytes(config)),
        "canonical_classes": {str(key): value for key, value in sorted(canonical_names.items())},
        "source_to_canonical_mapping": config["primary_class_mapping"]
        | (config["secondary_class_mapping"] if config.get("include_secondary_classes") else {}),
        "split_counts": dict(split_counts),
        "groups": {group_id: assignments[group_id] for group_id in sorted(assignments)},
        "leakage_check": {
            "status": "PASS" if final_edge_leakage == 0 else "FAIL",
            "cross_split_edges": final_edge_leakage,
            "group_overlap": 0,
        },
    }
    write_json(manifests / "split_manifest.json", split_manifest)

    checksum_targets = sorted(
        [
            manifests / "images.parquet",
            manifests / "groups.parquet",
            manifests / "split_manifest.json",
            manifests / "qc_report.json",
            provenance_path,
            splits_dir / "train.txt",
            splits_dir / "val.txt",
            splits_dir / "test.txt",
        ],
        key=lambda path: path.as_posix(),
    )
    checksums = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in checksum_targets
    }
    write_json(manifests / "checksums.json", checksums)
    return qc_report


def verify_output_integrity(root: Path, qc_report: Mapping[str, Any]) -> None:
    manifests = root / "data" / "manifests"
    checksums = json.loads((manifests / "checksums.json").read_text(encoding="utf-8"))
    for relative, expected in checksums.items():
        actual = sha256_file(root / relative)
        if not hmac.compare_digest(actual, expected):
            raise RuntimeError("Generated artifact checksum verification failed")
    if qc_report["duplicates_and_leakage"]["final_cross_split_edges"] != 0:
        raise RuntimeError("Cross-split leakage detected after allocation")
    split_sets = []
    for split in SPLITS:
        values = set((root / "data" / "splits" / f"{split}.txt").read_text(encoding="ascii").splitlines())
        split_sets.append(values)
    if any(split_sets[left] & split_sets[right] for left in range(3) for right in range(left + 1, 3)):
        raise RuntimeError("A canonical sample occurs in more than one split")


def privacy_safe_summary(qc_report: Mapping[str, Any]) -> str:
    source = qc_report["source"]
    images = qc_report["images"]
    annotations = qc_report["annotations"]
    leakage = qc_report["duplicates_and_leakage"]
    splits = qc_report["splits"]
    status = "PASS" if leakage["final_cross_split_edges"] == 0 and annotations["invalid"] == 0 else "FAIL"
    lines = [
        f"PIPELINE_STATUS={status}",
        f"SOURCE_IMAGES={source['images']}",
        f"ELIGIBLE_IMAGES={images['eligible']}",
        f"EXCLUDED_IMAGES={images['excluded']}",
        f"CANONICAL_INSTANCES={json.dumps(annotations['canonical_instances'], sort_keys=True)}",
        f"INVALID_ANNOTATIONS={annotations['invalid']}",
        f"CLIPPED_ANNOTATIONS={annotations['clipped']}",
        f"SUSPICIOUS_ANNOTATIONS={annotations['suspicious']}",
        f"MULTIPLE_UID_IMAGES={annotations['multiple_uid_occurrence_images']}",
        f"QR_IMAGES={annotations['qr_images']}",
        f"GROUPS={leakage['groups']}",
        f"ORIGINAL_CROSSING_GROUPS={leakage['original_partition_crossing_groups']}",
        f"FINAL_CROSS_SPLIT_EDGES={leakage['final_cross_split_edges']}",
        f"SPLIT_COUNTS={json.dumps(splits['image_counts'], sort_keys=True)}",
        f"SPLIT_RATIOS={json.dumps(splits['actual_ratios'], sort_keys=True)}",
    ]
    return "\n".join(lines)


def load_config(path: Path, include_secondary: bool) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    config["include_secondary_classes"] = bool(include_secondary or config.get("include_secondary_classes"))
    ratios = config["split_ratios"]
    if set(ratios) != set(SPLITS) or not math.isclose(sum(float(value) for value in ratios.values()), 1.0, abs_tol=1e-9):
        raise ValueError("Split ratios must contain train/val/test and total 1.0")
    return config


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    config = load_config(args.config.resolve(), args.include_secondary)
    key = load_or_create_hmac_key(args.hmac_key.resolve())
    class_mapping, canonical_names = build_class_configuration(config)
    identity_map = load_identity_hmac_map(args.identity_hmac_map.resolve() if args.identity_hmac_map else None)
    samples = discover_samples(root)
    records = [
        inspect_sample(root, partition, image_path, label_path, key, class_mapping, canonical_names, config)
        for partition, image_path, label_path in samples
    ]
    groups, reason_counts, edges = build_groups(records, config, identity_map)
    assignments, _ = allocate_splits(groups, config["split_ratios"], int(config["seed"]))
    qc_report = materialize_outputs(
        root, records, groups, assignments, reason_counts, edges, config, canonical_names, key
    )
    verify_output_integrity(root, qc_report)
    return qc_report


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the canonical local Aadhaar detection dataset")
    parser.add_argument("--root", type=Path, default=Path("."), help="Dataset root containing train/valid/test")
    parser.add_argument(
        "--config", type=Path, default=Path("config/dataset_pipeline.json"), help="Deterministic pipeline configuration"
    )
    parser.add_argument(
        "--hmac-key", type=Path, default=Path(".local_secrets/dataset_hmac.key"), help="Local pseudonymization key"
    )
    parser.add_argument(
        "--identity-hmac-map",
        type=Path,
        default=None,
        help="Optional authorized JSON mapping of image SHA-256 to precomputed identity HMAC",
    )
    parser.add_argument("--include-secondary", action="store_true", help="Include separately mapped secondary classes")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        report = run_pipeline(args)
    except Exception as exc:
        print(f"PIPELINE_STATUS=FAIL\nERROR_CODE={type(exc).__name__}", file=sys.stderr)
        return 2
    print(privacy_safe_summary(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
