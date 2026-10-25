"""Inspect extracted dataset files without modifying them.

Usage:
    python scripts/inspect_dataset.py data/raw
"""
from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import Counter
from pathlib import Path

TEXT_SUFFIXES = {".csv", ".tsv", ".txt", ".json", ".jsonl", ".xlsx", ".parquet"}


def inspect_zip(path: Path) -> None:
    print(f"\n[ZIP] {path} ({path.stat().st_size:,} bytes)")
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            print(f"Members: {len(members)}")
            for member in members[:100]:
                print(f"  {member.filename} | {member.file_size:,} bytes")
            if len(members) > 100:
                print(f"  ... {len(members) - 100} more entries")
            bad = archive.testzip()
            print(f"Integrity: {'OK' if bad is None else 'FAILED at ' + bad}")
    except zipfile.BadZipFile:
        print("ERROR: not a valid ZIP archive")


def inspect_csv(path: Path, delimiter: str = ",") -> None:
    print(f"\n[CSV/TSV] {path} ({path.stat().st_size:,} bytes)")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, delimiter=delimiter)
            if not reader.fieldnames:
                print("No header detected")
                return
            counts = Counter()
            missing = Counter()
            samples = []
            rows = 0
            for row in reader:
                rows += 1
                counts.update({str(k): 1 for k, v in row.items() if v is not None})
                for key, value in row.items():
                    if value is None or not str(value).strip():
                        missing[key] += 1
                if len(samples) < 3:
                    samples.append({k: (str(v)[:180] if v is not None else None) for k, v in row.items()})
            print(f"Rows: {rows}")
            print(f"Columns: {reader.fieldnames}")
            print(f"Blank/missing per column: {dict(missing)}")
            print("First rows (values truncated to 180 chars):")
            print(json.dumps(samples, ensure_ascii=False, indent=2))
    except (UnicodeDecodeError, csv.Error) as exc:
        print(f"Could not parse as UTF-8 delimited text: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Extracted data directory or a dataset archive")
    args = parser.parse_args()
    root = args.path
    if not root.exists():
        raise SystemExit(f"Path does not exist: {root}")
    if root.is_file():
        if root.suffix.lower() == ".zip":
            inspect_zip(root)
        elif root.suffix.lower() in {".csv", ".tsv"}:
            inspect_csv(root, "\t" if root.suffix.lower() == ".tsv" else ",")
        else:
            print(f"File: {root} ({root.stat().st_size:,} bytes)")
        return

    files = sorted(p for p in root.rglob("*") if p.is_file())
    print(f"Root: {root.resolve()}\nFiles found: {len(files)}")
    for path in files:
        suffix = path.suffix.lower()
        if suffix == ".zip":
            inspect_zip(path)
        elif suffix in {".csv", ".tsv"}:
            inspect_csv(path, "\t" if suffix == ".tsv" else ",")
        else:
            print(f"\n[FILE] {path} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
