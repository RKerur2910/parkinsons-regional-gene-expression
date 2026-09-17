#!/usr/bin/env python3

"""Select one reliable Allen microarray probe for every background gene."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "allen_human_brain_atlas"
RESULTS_DIR = PROJECT_ROOT / "results"

ALL_PROBES_OUTPUT = RESULTS_DIR / "all_probe_detection_summary.csv"
SELECTED_OUTPUT = RESULTS_DIR / "background_selected_probes.csv"
EXCLUDED_OUTPUT = RESULTS_DIR / "background_genes_without_reliable_probe.csv"

DONOR_PREFIX = "normalized_microarray_donor"
DETECTION_THRESHOLD = 0.50
CHUNK_SIZE = 2_000

PLACEHOLDERS = {
    "",
    "NA",
    "N/A",
    "NAN",
    "NONE",
    "NULL",
    "NR",
    "-",
}


def normalize_probe_ids(series: pd.Series) -> pd.Series:
    """Normalize probe IDs without changing their meaning."""

    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def find_donor_folders() -> list[Path]:
    """Find all Allen donor folders."""

    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    folders = [
        path
        for path in DATA_DIR.iterdir()
        if path.is_dir() and path.name.startswith(DONOR_PREFIX)
    ]

    return sorted(folders, key=lambda path: path.name)


def load_probe_metadata(donor_folder: Path) -> pd.DataFrame:
    """Load and clean probe metadata from one donor."""

    probes_path = donor_folder / "Probes.csv"

    if not probes_path.is_file():
        raise FileNotFoundError(f"Missing file: {probes_path}")

    probes = pd.read_csv(probes_path, low_memory=False)

    required_columns = {
        "probe_id",
        "probe_name",
        "gene_id",
        "gene_symbol",
        "gene_name",
        "entrez_id",
        "chromosome",
    }

    missing = required_columns.difference(probes.columns)

    if missing:
        raise ValueError(
            "Probes.csv is missing columns: "
            + ", ".join(sorted(missing))
        )

    probes = probes.copy()
    probes["probe_id_norm"] = normalize_probe_ids(probes["probe_id"])

    if probes["probe_id_norm"].duplicated().any():
        raise ValueError("Duplicate probe IDs found in Probes.csv")

    return probes


def verify_probe_metadata(
    donor_folders: list[Path],
    reference: pd.DataFrame,
) -> None:
    """Verify probe IDs and gene symbols across donors."""

    reference_ids = reference["probe_id_norm"].tolist()

    reference_symbols = (
        reference["gene_symbol"]
        .fillna("")
        .astype(str)
        .str.strip()
        .tolist()
    )

    for folder in donor_folders[1:]:
        current = load_probe_metadata(folder)

        current_ids = current["probe_id_norm"].tolist()

        current_symbols = (
            current["gene_symbol"]
            .fillna("")
            .astype(str)
            .str.strip()
            .tolist()
        )

        if current_ids != reference_ids:
            raise ValueError(
                f"Probe IDs differ in {folder.name}/Probes.csv"
            )

        if current_symbols != reference_symbols:
            raise ValueError(
                f"Gene symbols differ in {folder.name}/Probes.csv"
            )

    print("Probe metadata consistent across donors: PASS")


def process_pacall_file(
    pacall_path: Path,
    expected_probe_ids: np.ndarray,
) -> tuple[np.ndarray, int]:
    """
    Stream one PACall file and return detection counts per probe.

    PACall.csv has no header. Its first column contains probe IDs.
    """

    detected_counts = np.zeros(
        len(expected_probe_ids),
        dtype=np.int64,
    )

    total_sample_count: int | None = None
    row_offset = 0

    for chunk in pd.read_csv(
        pacall_path,
        header=None,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        chunk_rows = len(chunk)
        ending_offset = row_offset + chunk_rows

        if ending_offset > len(expected_probe_ids):
            raise ValueError(
                f"{pacall_path} contains more probe rows than Probes.csv"
            )

        observed_ids = normalize_probe_ids(chunk.iloc[:, 0]).to_numpy()
        expected_ids = expected_probe_ids[row_offset:ending_offset]

        if not np.array_equal(observed_ids, expected_ids):
            mismatch_positions = np.flatnonzero(
                observed_ids != expected_ids
            )

            first_mismatch = int(mismatch_positions[0])

            raise ValueError(
                f"Probe order mismatch in {pacall_path} near row "
                f"{row_offset + first_mismatch + 1}"
            )

        sample_values = chunk.iloc[:, 1:].apply(
            pd.to_numeric,
            errors="coerce",
        )

        if total_sample_count is None:
            total_sample_count = sample_values.shape[1]

        elif sample_values.shape[1] != total_sample_count:
            raise ValueError(
                f"Inconsistent PACall column count in {pacall_path}"
            )

        invalid_values = ~sample_values.isin([0, 1])

        if invalid_values.to_numpy().any():
            raise ValueError(
                f"PACall contains values other than 0 or 1: "
                f"{pacall_path}"
            )

        detected_counts[row_offset:ending_offset] = (
            sample_values.sum(axis=1).to_numpy(dtype=np.int64)
        )

        row_offset = ending_offset

    if row_offset != len(expected_probe_ids):
        raise ValueError(
            f"{pacall_path} contains {row_offset} probe rows; "
            f"expected {len(expected_probe_ids)}"
        )

    if total_sample_count is None:
        raise ValueError(f"No data found in {pacall_path}")

    return detected_counts, total_sample_count


def clean_gene_symbols(probes: pd.DataFrame) -> pd.DataFrame:
    """Clean gene symbols and mark usable records."""

    cleaned = probes.copy()

    cleaned["gene_symbol_clean"] = (
        cleaned["gene_symbol"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    cleaned["valid_gene_symbol"] = ~cleaned[
        "gene_symbol_clean"
    ].isin(PLACEHOLDERS)

    return cleaned


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    donor_folders = find_donor_folders()

    if not donor_folders:
        raise FileNotFoundError("No Allen donor folders were found")

    print(f"Donor folders found: {len(donor_folders)}")

    probe_metadata = load_probe_metadata(donor_folders[0])

    print(f"Total probes: {len(probe_metadata)}")

    verify_probe_metadata(
        donor_folders=donor_folders,
        reference=probe_metadata,
    )

    expected_probe_ids = probe_metadata[
        "probe_id_norm"
    ].to_numpy()

    total_detected = np.zeros(
        len(probe_metadata),
        dtype=np.int64,
    )

    total_samples = 0

    for donor_number, folder in enumerate(donor_folders, start=1):
        pacall_path = folder / "PACall.csv"

        if not pacall_path.is_file():
            raise FileNotFoundError(f"Missing file: {pacall_path}")

        print(
            f"\nProcessing donor {donor_number}/{len(donor_folders)}: "
            f"{folder.name}"
        )

        donor_detected, donor_sample_count = process_pacall_file(
            pacall_path=pacall_path,
            expected_probe_ids=expected_probe_ids,
        )

        total_detected += donor_detected
        total_samples += donor_sample_count

        print(f"  Tissue samples: {donor_sample_count}")
        print(f"Finished {folder.name}")

    summary = clean_gene_symbols(probe_metadata)

    summary["total_detected_samples"] = total_detected
    summary["total_samples"] = total_samples

    summary["overall_detection_rate"] = (
        summary["total_detected_samples"]
        / summary["total_samples"]
    )

    summary["reliable_probe"] = (
        summary["valid_gene_symbol"]
        & (
            summary["overall_detection_rate"]
            >= DETECTION_THRESHOLD
        )
    )

    reliable = summary.loc[summary["reliable_probe"]].copy()

    reliable["probe_id_numeric"] = pd.to_numeric(
        reliable["probe_id_norm"],
        errors="coerce",
    )

    reliable = reliable.sort_values(
        by=[
            "gene_symbol_clean",
            "overall_detection_rate",
            "probe_id_numeric",
            "probe_id_norm",
        ],
        ascending=[True, False, True, True],
        na_position="last",
    )

    selected = (
        reliable
        .drop_duplicates(
            subset=["gene_symbol_clean"],
            keep="first",
        )
        .copy()
    )

    selected["gene_symbol"] = selected["gene_symbol_clean"]
    selected["probe_id"] = selected["probe_id_norm"]

    selected_columns = [
        "gene_symbol",
        "probe_id",
        "probe_name",
        "gene_id",
        "gene_name",
        "entrez_id",
        "chromosome",
        "overall_detection_rate",
        "total_detected_samples",
        "total_samples",
    ]

    selected = (
        selected[selected_columns]
        .sort_values("gene_symbol")
        .reset_index(drop=True)
    )

    valid_gene_probes = summary.loc[
        summary["valid_gene_symbol"]
    ].copy()

    gene_reliability = (
        valid_gene_probes
        .groupby("gene_symbol_clean", as_index=False)
        .agg(
            number_of_available_probes=(
                "probe_id_norm",
                "nunique",
            ),
            number_of_reliable_probes=(
                "reliable_probe",
                "sum",
            ),
            best_detection_rate=(
                "overall_detection_rate",
                "max",
            ),
        )
    )

    excluded = gene_reliability.loc[
        gene_reliability["number_of_reliable_probes"].eq(0)
    ].copy()

    excluded = excluded.rename(
        columns={"gene_symbol_clean": "gene_symbol"}
    )

    excluded = excluded.sort_values(
        "gene_symbol"
    ).reset_index(drop=True)

    all_probe_columns = [
        "probe_id",
        "probe_name",
        "gene_id",
        "gene_symbol",
        "gene_name",
        "entrez_id",
        "chromosome",
        "total_detected_samples",
        "total_samples",
        "overall_detection_rate",
        "valid_gene_symbol",
        "reliable_probe",
    ]

    summary[all_probe_columns].to_csv(
        ALL_PROBES_OUTPUT,
        index=False,
    )

    selected.to_csv(
        SELECTED_OUTPUT,
        index=False,
    )

    excluded.to_csv(
        EXCLUDED_OUTPUT,
        index=False,
    )

    print("\nBackground-probe selection complete")
    print(f"Total tissue samples: {total_samples}")
    print(
        "Probes with valid gene symbols: "
        f"{summary['valid_gene_symbol'].sum()}"
    )
    print(
        f"Reliable probes: {summary['reliable_probe'].sum()}"
    )
    print(
        "Genes with at least one reliable probe: "
        f"{len(selected)}"
    )
    print(
        "Genes without a reliable probe: "
        f"{len(excluded)}"
    )

    print("\nFirst 20 selected background genes:")

    for row in selected.head(20).itertuples(index=False):
        print(
            f"  {row.gene_symbol}\t{row.probe_id}\t"
            f"{row.overall_detection_rate:.3f}"
        )

    print(f"\nWrote {ALL_PROBES_OUTPUT}")
    print(f"Wrote {SELECTED_OUTPUT}")
    print(f"Wrote {EXCLUDED_OUTPUT}")


if __name__ == "__main__":
    main()