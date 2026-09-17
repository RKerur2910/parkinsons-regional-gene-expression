#!/usr/bin/env python3
"""Select reliable Allen probes using PACall detection rates."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "allen_human_brain_atlas"
MATCHED_PATH = PROJECT_ROOT / "results" / "parkinsons_genes_matched_to_probes.csv"
BY_DONOR_PATH = PROJECT_ROOT / "results" / "parkinsons_probe_detection_by_donor.csv"
SUMMARY_PATH = PROJECT_ROOT / "results" / "parkinsons_probe_detection_summary.csv"
SELECTED_PATH = PROJECT_ROOT / "results" / "parkinsons_selected_probes.csv"
NO_RELIABLE_PATH = PROJECT_ROOT / "results" / "parkinsons_genes_without_reliable_probe.csv"

DONOR_PREFIX = "normalized_microarray_donor"
DETECTION_THRESHOLD = 0.50


def find_donor_folders(data_dir: Path) -> list[Path]:
    folders = [
        path
        for path in data_dir.iterdir()
        if path.is_dir() and path.name.startswith(DONOR_PREFIX)
    ]
    return sorted(folders, key=lambda path: path.name)


def normalize_probe_id(value: object) -> str:
    text = str(value).strip()
    try:
        return str(int(float(text)))
    except (TypeError, ValueError):
        return text


def process_pacall(path: Path, candidate_ids: set[str]) -> dict[str, tuple[int, int]]:
    found: dict[str, tuple[int, int]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            probe_id = normalize_probe_id(row[0])
            if probe_id not in candidate_ids:
                continue
            calls = row[1:]
            detected = sum(1 for value in calls if value.strip() == "1")
            found[probe_id] = (detected, len(calls))
    return found


def main() -> None:
    matched = pd.read_csv(MATCHED_PATH)
    matched["probe_id_norm"] = matched["probe_id"].map(normalize_probe_id)
    matched["probe_id"] = matched["probe_id_norm"].astype(int)
    candidates = matched.drop_duplicates(subset=["probe_id_norm"])
    candidate_ids = set(candidates["probe_id_norm"])

    print(f"Number of candidate genes: {matched['gene_symbol'].nunique()}")
    print(f"Number of candidate probes: {len(candidate_ids)}")

    donor_folders = find_donor_folders(DATA_DIR)
    print(f"Number of donors processed: {len(donor_folders)}")

    by_donor_rows: list[dict[str, object]] = []
    found_in_every_donor = True
    metadata = matched.loc[
        :, ["gene_symbol", "probe_id", "probe_id_norm", "probe_name"]
    ].drop_duplicates()

    for folder in donor_folders:
        pacall_path = folder / "PACall.csv"
        print(f"Processing {folder.name}...", flush=True)
        found = process_pacall(pacall_path, candidate_ids)
        missing = candidate_ids - set(found)
        if missing:
            found_in_every_donor = False
            print(
                f"  missing {len(missing)} candidate probes in {folder.name}",
                flush=True,
            )
        for _, probe in metadata.iterrows():
            probe_id_norm = probe["probe_id_norm"]
            if probe_id_norm not in found:
                continue
            detected, total = found[probe_id_norm]
            detection_rate = (detected / total) if total else float("nan")
            by_donor_rows.append(
                {
                    "donor_id": folder.name,
                    "gene_symbol": probe["gene_symbol"],
                    "probe_id": probe["probe_id"],
                    "detected_sample_count": detected,
                    "total_sample_count": total,
                    "detection_rate": detection_rate,
                }
            )
        print(f"Finished {folder.name}", flush=True)

    print(
        "Every candidate probe found in every donor: "
        f"{'PASS' if found_in_every_donor else 'FAIL'}"
    )

    by_donor = pd.DataFrame(by_donor_rows)
    by_donor = by_donor.sort_values(
        ["donor_id", "gene_symbol", "probe_id"],
        kind="mergesort",
    )

    probe_totals = (
        by_donor.groupby(["gene_symbol", "probe_id"], as_index=False)
        .agg(
            total_detected_samples=("detected_sample_count", "sum"),
            total_samples=("total_sample_count", "sum"),
            number_of_donors_present=("donor_id", "nunique"),
            donors_with_detection_rate_at_least_0_5=(
                "detection_rate",
                lambda rates: int((rates >= DETECTION_THRESHOLD).sum()),
            ),
        )
    )
    probe_totals["overall_detection_rate"] = (
        probe_totals["total_detected_samples"] / probe_totals["total_samples"]
    )
    probe_totals["reliable_probe"] = (
        probe_totals["overall_detection_rate"] >= DETECTION_THRESHOLD
    )
    summary = probe_totals.merge(
        matched.loc[:, ["gene_symbol", "probe_id", "probe_name"]].drop_duplicates(),
        on=["gene_symbol", "probe_id"],
        how="left",
    )
    summary = summary.loc[
        :,
        [
            "gene_symbol",
            "probe_id",
            "probe_name",
            "total_detected_samples",
            "total_samples",
            "overall_detection_rate",
            "number_of_donors_present",
            "donors_with_detection_rate_at_least_0_5",
            "reliable_probe",
        ],
    ].sort_values(["gene_symbol", "probe_id"], kind="mergesort")

    reliable = summary.loc[summary["reliable_probe"]].copy()
    selected = (
        reliable.sort_values(
            ["overall_detection_rate", "probe_id"],
            ascending=[False, True],
            kind="mergesort",
        )
        .drop_duplicates(subset=["gene_symbol"], keep="first")
        .sort_values("gene_symbol", kind="mergesort")
        .loc[
            :,
            [
                "gene_symbol",
                "probe_id",
                "probe_name",
                "overall_detection_rate",
                "donors_with_detection_rate_at_least_0_5",
            ],
        ]
    )

    all_genes = (
        summary.groupby("gene_symbol", as_index=False)
        .agg(
            number_of_available_probes=("probe_id", "nunique"),
            best_detection_rate=("overall_detection_rate", "max"),
        )
    )
    genes_without = all_genes.loc[
        ~all_genes["gene_symbol"].isin(selected["gene_symbol"])
    ].sort_values("gene_symbol", kind="mergesort")

    n_reliable_probes = int(summary["reliable_probe"].sum())
    n_genes_with = len(selected)
    n_genes_without = len(genes_without)
    print(f"Number of reliable probes: {n_reliable_probes}")
    print(f"Number of genes with at least one reliable probe: {n_genes_with}")
    print(f"Number of genes without a reliable probe: {n_genes_without}")
    print("First 20 selected gene/probe pairs:")
    for row in selected.head(20).itertuples(index=False):
        print(f"  {row.gene_symbol}\t{row.probe_id}")

    BY_DONOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    by_donor.to_csv(BY_DONOR_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    selected.to_csv(SELECTED_PATH, index=False)
    genes_without.to_csv(NO_RELIABLE_PATH, index=False)
    print(f"Wrote {BY_DONOR_PATH}")
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {SELECTED_PATH}")
    print(f"Wrote {NO_RELIABLE_PATH}")


if __name__ == "__main__":
    main()
