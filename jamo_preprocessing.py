#!/usr/bin/env python3
"""Prepare Korean OnomaCap annotations for canonical-Jamo training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


KO_COLUMNS = tuple(f"candidate{i}_ko" for i in range(1, 6))
EN_COLUMNS = tuple(f"candidate{i}_en" for i in range(1, 6))
JAMO_COLUMNS = tuple(f"candidate{i}_jamo" for i in range(1, 6))

DUPLICATE_MARKER = "JUNG BOK DOEN SAEM PEUL"
EXPECTED_LATIN_AUDIO = "21_ (12).mp3"
EXPECTED_OUTPUT_ROWS = 7_961
SPLIT_MANIFEST_VERSION = 1
SPLIT_NAMES = ("train", "val", "test")

CHOSEONG = tuple(chr(codepoint) for codepoint in range(0x1100, 0x1113))
JUNGSEONG = tuple(chr(codepoint) for codepoint in range(0x1161, 0x1176))
JONGSEONG = tuple(chr(codepoint) for codepoint in range(0x11A8, 0x11C3))
JAMO_VOCAB = CHOSEONG + JUNGSEONG + JONGSEONG

CHOSEONG_SET = frozenset(CHOSEONG)
JUNGSEONG_SET = frozenset(JUNGSEONG)
JONGSEONG_SET = frozenset(JONGSEONG)

CHOSEONG_ROMANIZATION = dict(zip(
    CHOSEONG,
    (
        "G", "KK", "N", "D", "TT", "R", "M", "B", "PP", "S",
        "SS", "", "J", "JJ", "CH", "K", "T", "P", "H",
    ),
))
JUNGSEONG_ROMANIZATION = dict(zip(
    JUNGSEONG,
    (
        "A", "AE", "YA", "YAE", "EO", "E", "YEO", "YE", "O", "WA",
        "WAE", "OE", "YO", "U", "WO", "WE", "WI", "YU", "EU", "UI",
        "I",
    ),
))
JONGSEONG_ROMANIZATION = dict(zip(
    JONGSEONG,
    (
        "K", "K", "K", "N", "N", "N", "T", "L", "L", "LM",
        "L", "L", "L", "L", "L", "M", "B", "B", "T", "T",
        "NG", "T", "T", "K", "T", "P", "T",
    ),
))

STANDALONE_REPLACEMENTS = {
    "ㅏ": "아",
    "ㅓ": "어",
    "ㅗ": "오",
    "ㅡ": "으",
    "ㅣ": "이",
    "ㄹ": "르",
    "ㅊ": "츠",
}

PUNCTUATION_RE = re.compile(r"[,.!?;:|*\"]")
LATIN_RE = re.compile(r"[A-Za-z]")
WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class ProcessingReport:
    input_rows: int
    output_rows: int
    duplicate_rows_excluded: tuple[str, ...]
    latin_rows_excluded: tuple[str, ...]
    standalone_replacements: dict[str, int]


def is_precomposed_hangul_syllable(token: str) -> bool:
    return len(token) == 1 and 0xAC00 <= ord(token) <= 0xD7A3


def _replace_controls_with_spaces(text: str) -> str:
    return "".join(
        " " if unicodedata.category(character).startswith("C") else character
        for character in text
    )


def normalize_korean_caption(
    caption: str,
    replacement_counts: Counter[str] | None = None,
) -> str:
    """Clean a spaced Korean caption and repair observed standalone Jamo."""

    text = unicodedata.normalize("NFC", str(caption))
    text = _replace_controls_with_spaces(text)
    text = PUNCTUATION_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()

    normalized_tokens: list[str] = []
    for token in text.split():
        replacement = STANDALONE_REPLACEMENTS.get(token)
        if replacement is not None:
            normalized_tokens.append(replacement)
            if replacement_counts is not None:
                replacement_counts[token] += 1
            continue

        if not is_precomposed_hangul_syllable(token):
            raise ValueError(f"Unsupported Korean annotation token: {token!r}")
        normalized_tokens.append(token)

    if not normalized_tokens:
        raise ValueError("Korean caption is empty after normalization.")
    return " ".join(normalized_tokens)


def hangul_caption_to_jamo(caption: str) -> str:
    """Convert spaced precomposed Hangul to space-delimited canonical Jamo."""

    tokens = caption.split()
    if not tokens or not all(is_precomposed_hangul_syllable(token) for token in tokens):
        raise ValueError(f"Caption must contain spaced Hangul syllables: {caption!r}")

    jamo = unicodedata.normalize("NFD", "".join(tokens))
    unsupported = [character for character in jamo if character not in JAMO_VOCAB]
    if unsupported:
        raise ValueError(f"Unsupported canonical Jamo: {unsupported!r}")
    return " ".join(jamo)


def hangul_caption_to_romanized(caption: str) -> str:
    """Convert spaced Hangul syllables to the OnomaCap Latin notation."""

    tokens = caption.split()
    if not tokens or not all(is_precomposed_hangul_syllable(token) for token in tokens):
        raise ValueError(f"Caption must contain spaced Hangul syllables: {caption!r}")

    romanized_tokens: list[str] = []
    for token in tokens:
        components = unicodedata.normalize("NFD", token)
        onset, vowel = components[:2]
        romanized = (
            CHOSEONG_ROMANIZATION[onset]
            + JUNGSEONG_ROMANIZATION[vowel]
        )
        if len(components) == 3:
            romanized += JONGSEONG_ROMANIZATION[components[2]]
        romanized_tokens.append(romanized)
    return " ".join(romanized_tokens)


def normalize_romanized_caption(caption: str) -> str:
    """Normalize an OnomaCap Latin caption while preserving token boundaries."""

    text = _replace_controls_with_spaces(str(caption)).upper()
    text = PUNCTUATION_RE.sub(" ", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def _dataset_fingerprint(frame) -> str:
    records = sorted(
        (str(audio_file), str(class_name))
        for audio_file, class_name in zip(frame["audio_file"], frame["class"])
    )
    digest = hashlib.sha256()
    for audio_file, class_name in records:
        digest.update(audio_file.encode("utf-8"))
        digest.update(b"\0")
        digest.update(class_name.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_or_create_split_manifest(frame, manifest_path: Path, split_seed: int):
    """Load an immutable Drive split manifest, or create it exactly once."""

    import pandas as pd

    manifest_path = Path(manifest_path)
    metadata_path = manifest_path.with_suffix(manifest_path.suffix + ".meta.json")
    required_columns = {"audio_file", "class"}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing split columns: {sorted(missing_columns)}")
    if frame["audio_file"].isna().any() or frame["class"].isna().any():
        raise ValueError("Split source contains missing audio_file or class values.")
    if frame["audio_file"].astype(str).duplicated().any():
        raise ValueError("Split source contains duplicate audio_file values.")

    source = frame.copy()
    source["audio_file"] = source["audio_file"].astype(str)
    source["class"] = source["class"].astype(str)
    source_fingerprint = _dataset_fingerprint(source)
    manifest_exists = manifest_path.is_file()
    metadata_exists = metadata_path.is_file()

    if manifest_exists != metadata_exists:
        raise RuntimeError(
            "Split manifest is incomplete: both the CSV and metadata JSON "
            "must exist, or neither may exist."
        )

    if not manifest_exists:
        from sklearn.model_selection import train_test_split

        train_frame, rest_frame = train_test_split(
            source,
            test_size=0.2,
            stratify=source["class"],
            random_state=split_seed,
        )
        val_frame, test_frame = train_test_split(
            rest_frame,
            test_size=0.5,
            stratify=rest_frame["class"],
            random_state=split_seed,
        )
        generated_splits = {
            "train": train_frame,
            "val": val_frame,
            "test": test_frame,
        }
        manifest = pd.concat(
            [
                split_frame.assign(split=split_name)[
                    ["audio_file", "class", "split"]
                ]
                for split_name, split_frame in generated_splits.items()
            ],
            ignore_index=True,
        ).sort_values("audio_file", kind="stable")
        split_counts = {
            name: int((manifest["split"] == name).sum())
            for name in SPLIT_NAMES
        }
        metadata = {
            "version": SPLIT_MANIFEST_VERSION,
            "split_seed": int(split_seed),
            "row_count": int(len(source)),
            "dataset_fingerprint": source_fingerprint,
            "split_counts": split_counts,
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_manifest = manifest_path.with_suffix(
            manifest_path.suffix + ".tmp"
        )
        temporary_metadata = metadata_path.with_suffix(
            metadata_path.suffix + ".tmp"
        )
        manifest.to_csv(temporary_manifest, index=False)
        temporary_metadata.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary_manifest, manifest_path)
        os.replace(temporary_metadata, metadata_path)
        print(f"Created split manifest: {manifest_path}")
    else:
        manifest = pd.read_csv(
            manifest_path,
            dtype={"audio_file": str, "class": str, "split": str},
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        print(f"Loaded existing split manifest: {manifest_path}")

    if not {"audio_file", "class", "split"} <= set(manifest.columns):
        raise ValueError("Split manifest is missing required columns.")
    if manifest["audio_file"].isna().any() or manifest["class"].isna().any():
        raise ValueError("Split manifest contains missing values.")
    if manifest["audio_file"].duplicated().any():
        raise ValueError("Split manifest contains duplicate audio_file values.")
    if set(manifest["split"]) != set(SPLIT_NAMES):
        raise ValueError(f"Invalid split names: {sorted(set(manifest['split']))}")
    if int(metadata.get("version", -1)) != SPLIT_MANIFEST_VERSION:
        raise ValueError("Split manifest version does not match.")
    if int(metadata.get("split_seed", -1)) != int(split_seed):
        raise ValueError("Split seed does not match the persisted manifest.")
    if int(metadata.get("row_count", -1)) != len(source):
        raise ValueError("Split manifest row count does not match the dataset.")
    if metadata.get("dataset_fingerprint") != source_fingerprint:
        raise ValueError("Split manifest dataset fingerprint does not match.")
    if _dataset_fingerprint(manifest) != source_fingerprint:
        raise ValueError("Split manifest rows do not match the current dataset.")

    source_classes = dict(zip(source["audio_file"], source["class"]))
    mismatched_classes = [
        audio_file
        for audio_file, class_name in zip(manifest["audio_file"], manifest["class"])
        if source_classes.get(audio_file) != class_name
    ]
    if mismatched_classes:
        raise ValueError(
            f"Split manifest class mismatch: {mismatched_classes[:5]}"
        )

    source_by_name = source.set_index("audio_file", drop=False)
    splits = {
        split_name: source_by_name.loc[
            manifest.loc[manifest["split"] == split_name, "audio_file"]
        ].reset_index(drop=True)
        for split_name in SPLIT_NAMES
    }
    actual_counts = {name: len(split) for name, split in splits.items()}
    expected_counts = {
        name: int(count)
        for name, count in metadata.get("split_counts", {}).items()
    }
    if actual_counts != expected_counts:
        raise ValueError(
            f"Split counts do not match metadata: {actual_counts} != "
            f"{expected_counts}"
        )
    return splits


def jamo_to_hangul_caption(jamo: str) -> str:
    """Parse (choseong jungseong [jongseong]?)* and compose Hangul."""

    syllables: list[str] = []
    current = ""
    state = "choseong"
    compact_jamo = "".join(jamo.split())

    for character in compact_jamo:
        if state == "choseong":
            if character not in CHOSEONG_SET:
                raise ValueError(f"Expected choseong, got {character!r}")
            current = character
            state = "jungseong"
        elif state == "jungseong":
            if character not in JUNGSEONG_SET:
                raise ValueError(f"Expected jungseong, got {character!r}")
            current += character
            state = "optional_jongseong"
        elif character in JONGSEONG_SET:
            syllables.append(unicodedata.normalize("NFC", current + character))
            current = ""
            state = "choseong"
        elif character in CHOSEONG_SET:
            syllables.append(unicodedata.normalize("NFC", current))
            current = character
            state = "jungseong"
        else:
            raise ValueError(
                f"Expected jongseong or next choseong, got {character!r}"
            )

    if state == "optional_jongseong":
        syllables.append(unicodedata.normalize("NFC", current))
    elif state != "choseong":
        raise ValueError("Jamo sequence ends with an incomplete syllable.")

    if not syllables:
        raise ValueError("Jamo sequence is empty.")
    if not all(is_precomposed_hangul_syllable(item) for item in syllables):
        raise ValueError(f"Failed to compose Hangul syllables: {syllables!r}")
    return " ".join(syllables)


def _contains_duplicate_marker(row: dict[str, str]) -> bool:
    return any(DUPLICATE_MARKER in (row.get(column) or "") for column in EN_COLUMNS)


def _contains_latin_korean_annotation(row: dict[str, str]) -> bool:
    return any(LATIN_RE.search(row.get(column) or "") for column in KO_COLUMNS)


def prepare_rows(
    rows: Iterable[dict[str, str]],
) -> tuple[list[dict[str, str]], ProcessingReport]:
    prepared_rows: list[dict[str, str]] = []
    duplicate_rows: list[str] = []
    latin_rows: list[str] = []
    replacement_counts: Counter[str] = Counter()
    input_rows = 0

    for source_row in rows:
        input_rows += 1
        row = dict(source_row)
        audio_file = row["audio_file"]

        if _contains_duplicate_marker(row):
            duplicate_rows.append(audio_file)
            continue
        if _contains_latin_korean_annotation(row):
            latin_rows.append(audio_file)
            continue

        for ko_column, jamo_column in zip(KO_COLUMNS, JAMO_COLUMNS):
            normalized = normalize_korean_caption(
                row[ko_column], replacement_counts=replacement_counts
            )
            jamo = hangul_caption_to_jamo(normalized)
            if jamo_to_hangul_caption(jamo) != normalized:
                raise AssertionError(
                    f"Hangul/Jamo round-trip failed for {audio_file} {ko_column}."
                )
            row[ko_column] = normalized
            row[jamo_column] = jamo

        prepared_rows.append(row)

    report = ProcessingReport(
        input_rows=input_rows,
        output_rows=len(prepared_rows),
        duplicate_rows_excluded=tuple(duplicate_rows),
        latin_rows_excluded=tuple(latin_rows),
        standalone_replacements=dict(sorted(replacement_counts.items())),
    )
    return prepared_rows, report


def process_csv(
    input_path: Path,
    output_path: Path | None = None,
) -> ProcessingReport:
    with input_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {input_path}")
        missing_columns = set(KO_COLUMNS + EN_COLUMNS + ("audio_file",)) - set(
            reader.fieldnames
        )
        if missing_columns:
            raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
        rows, report = prepare_rows(reader)
        output_fieldnames = list(reader.fieldnames) + list(JAMO_COLUMNS)

    if report.latin_rows_excluded != (EXPECTED_LATIN_AUDIO,):
        raise AssertionError(
            "Unexpected Latin annotation rows: "
            f"{report.latin_rows_excluded!r}"
        )
    if report.output_rows != EXPECTED_OUTPUT_ROWS:
        raise AssertionError(
            f"Expected {EXPECTED_OUTPUT_ROWS} output rows, got {report.output_rows}."
        )

    if output_path is not None:
        if output_path.resolve() == input_path.resolve():
            raise ValueError("Refusing to overwrite the source CSV.")
        if output_path.exists():
            raise FileExistsError(f"Output already exists: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=output_fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return report


def parse_args() -> argparse.Namespace:
    default_input = Path(__file__).with_name("sound-to-onomatopoeia_annotation.csv")
    parser = argparse.ArgumentParser(
        description="Normalize OnomaCap Korean captions and derive canonical Jamo."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        type=Path,
        default=default_input,
        help="Source annotation CSV (default: workspace annotation CSV).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional derived CSV path. The source CSV is never overwritten.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = process_csv(args.input_csv, args.output)
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
