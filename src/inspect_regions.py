#!/usr/bin/env python3
"""Inventory Allen Human Brain Atlas sample regions from SampleAnnot.csv only."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "allen_human_brain_atlas"
INVENTORY_PATH = PROJECT_ROOT / "results" / "brain_region_inventory.csv"
MATCHES_PATH = PROJECT_ROOT / "results" / "disease_region_matches.csv"

DONOR_PREFIX = "normalized_microarray_donor"

SEARCH_TERMS = (
    "substantia nigra",
    "putamen",
    "caudate",
    "globus pallidus",
    "pallidum",
    "striatum",
    "motor cortex",
    "entorhinal",
)


def find_donor_folders(data_dir: Path) -> list[Path]:
    folders = [
        path
        for path in data_dir.iterdir()
        if path.is_dir() and path.name.startswith(DONOR_PREFIX)
    ]
    return sorted(folders, key=lambda path: path.name)


def load_sample_annotations(donor_folders: list[Path]) -> pd.DataFrame:
    frames = []
    for folder in donor_folders:
        annot_path = folder / "SampleAnnot.csv"
        if not annot_path.is_file():
            print(f"Skipping {folder.name}: SampleAnnot.csv not found")
            continue
        frame = pd.read_csv(annot_path)
        frame["donor_id"] = folder.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def matching_terms(row: pd.Series) -> list[str]:
    text = f"{row['structure_acronym']} {row['structure_name']}".lower()
    return [term for term in SEARCH_TERMS if term in text]


def main() -> None:
    donor_folders = find_donor_folders(DATA_DIR)
    print(f"Donor folders found: {len(donor_folders)}")

    samples = load_sample_annotations(donor_folders)
    print(f"Total tissue samples: {len(samples)}")
    print(f"Unique structure IDs: {samples['structure_id'].nunique()}")
    print(f"Unique structure names: {samples['structure_name'].nunique()}")

    inventory = (
        samples.groupby(
            ["structure_id", "structure_acronym", "structure_name"],
            dropna=False,
        )
        .agg(
            total_sample_count=("donor_id", "size"),
            number_of_donors=("donor_id", "nunique"),
        )
        .reset_index()
        .sort_values(
            ["total_sample_count", "structure_name"],
            ascending=[False, True],
        )
    )

    term_lists = inventory.apply(matching_terms, axis=1)
    matches = inventory.loc[term_lists.map(bool)].copy()
    matches["matched_terms"] = term_lists.loc[matches.index].map("; ".join)
    matches = matches.sort_values(
        ["total_sample_count", "structure_name"],
        ascending=[False, True],
    )

    print("\nMatching disease-related structures:")
    if matches.empty:
        print("  none")
    else:
        print(
            matches[
                [
                    "structure_id",
                    "structure_acronym",
                    "structure_name",
                    "total_sample_count",
                    "number_of_donors",
                    "matched_terms",
                ]
            ].to_string(index=False)
        )

    INVENTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(INVENTORY_PATH, index=False)
    matches.to_csv(MATCHES_PATH, index=False)
    print(f"\nWrote {INVENTORY_PATH}")
    print(f"Wrote {MATCHES_PATH}")


if __name__ == "__main__":
    main()
