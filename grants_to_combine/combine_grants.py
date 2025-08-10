#!/usr/bin/env python3
"""
Combine all CSV files in a directory into a single CSV with a timestamped filename.

Default input directory: the directory containing this script
Default output directory: the directory containing this script

Output filename format: all_digitized_grants_YYYYMMDD_HHMMSS.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
from pathlib import Path
from typing import List


def list_csv_files(input_dir: Path) -> List[Path]:
    csv_files = sorted([
        p
        for p in input_dir.glob("*.csv")
        if p.is_file() and not p.name.startswith("all_digitized_grants_")
    ])
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {input_dir}")
    return csv_files


def read_header(csv_path: Path) -> List[str]:
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"CSV file is empty: {csv_path}")
    return header


def combine_csvs(input_dir: Path, output_dir: Path) -> Path:
    csv_files = list_csv_files(input_dir)

    # Prepare output file path
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"all_digitized_grants_{timestamp}.csv"

    # Determine header from the first file and validate others
    canonical_header = read_header(csv_files[0])

    with output_path.open("w", encoding="utf-8", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(canonical_header)

        for csv_path in csv_files:
            header = read_header(csv_path)
            if header != canonical_header:
                raise ValueError(
                    "Header mismatch detected.\n"
                    f"File: {csv_path}\n"
                    f"Found: {header}\n"
                    f"Expected: {canonical_header}"
                )

            with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as in_f:
                reader = csv.reader(in_f)
                # Skip header
                try:
                    next(reader)
                except StopIteration:
                    continue
                for row in reader:
                    writer.writerow(row)

    return output_path


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir
    default_output = script_dir

    parser = argparse.ArgumentParser(description="Combine CSVs in a directory into one file.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input,
        help=f"Directory containing CSVs (default: {default_input})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help=f"Directory to write the combined CSV (default: {default_output})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    output_path = combine_csvs(input_dir, output_dir)
    # Print absolute path for convenience
    print(str(output_path.resolve()))


if __name__ == "__main__":
    main()


