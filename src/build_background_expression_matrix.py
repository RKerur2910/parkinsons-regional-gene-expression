#!/usr/bin/env python3

"""Build an all-gene regional expression matrix for permutation testing."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "allen_human_brain_atlas"
RESULTS_DIR = PROJECT_ROOT / "results"

BACKGROUND_PROBES_PATH = RESULTS_DIR / "background_selected_probes.csv"
PARKINSONS_PROBES_PATH = RESULTS_DIR / "parkinsons_selected_probes.csv"

MATRIX_OUTPUT_PATH = (
    RESULTS_DIR / "background_regional_gene_expression_matrix.csv"
)
COVERAGE_OUTPUT_PATH = (
    RESULTS_DIR / "background_expression_coverage.csv"
)

DONOR_PREFIX = "normalized_microarray_donor"

REGION_COLUMNS = [
    "structure_id",
    "structure_acronym",
    "structure_name",
]


def normalize_probe_id(value: object) -> str:
    text = str(value).strip()

    if text.endswith(".0"):
        text = text[:-2]

    return text


def find_donor_folders() -> list[Path]:
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Missing data directory: {DATA_DIR}")

    folders = [
        path
        for path in DATA_DIR.iterdir()
        if path.is_dir() and path.name.startswith(DONOR_PREFIX)
    ]

    return sorted(folders, key=lambda path: path.name)


def load_background_probes() -> pd.DataFrame:
    if not BACKGROUND_PROBES_PATH.is_file():
        raise FileNotFoundError(
            f"Missing background probes: {BACKGROUND_PROBES_PATH}"
        )

    probes = pd.read_csv(BACKGROUND_PROBES_PATH, low_memory=False)

    required = {"gene_symbol", "probe_id"}
    missing = required.difference(probes.columns)

    if missing:
        raise ValueError(
            "Background probe file is missing: "
            + ", ".join(sorted(missing))
        )

    probes = probes.copy()

    probes["gene_symbol"] = (
        probes["gene_symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    probes["probe_id_norm"] = probes["probe_id"].map(
        normalize_probe_id
    )

    if probes["gene_symbol"].duplicated().any():
        raise ValueError(
            "Duplicate gene symbols found in background probe table"
        )

    if probes["probe_id_norm"].duplicated().any():
        raise ValueError(
            "Duplicate probe IDs found in background probe table"
        )

    return probes.sort_values("gene_symbol").reset_index(drop=True)


def load_region_inventory(
    donor_folders: list[Path],
) -> tuple[pd.DataFrame, dict[object, int]]:
    frames = []

    for folder in donor_folders:
        sample_path = folder / "SampleAnnot.csv"
        samples = pd.read_csv(sample_path)

        missing = set(REGION_COLUMNS).difference(samples.columns)

        if missing:
            raise ValueError(
                f"{sample_path} is missing columns: "
                + ", ".join(sorted(missing))
            )

        frames.append(samples[REGION_COLUMNS])

    all_regions = pd.concat(frames, ignore_index=True)

    conflicting_names = (
        all_regions
        .groupby("structure_id", dropna=False)
        .agg(
            acronym_count=("structure_acronym", "nunique"),
            name_count=("structure_name", "nunique"),
        )
    )

    conflicts = conflicting_names.loc[
        (conflicting_names["acronym_count"] > 1)
        | (conflicting_names["name_count"] > 1)
    ]

    if not conflicts.empty:
        raise ValueError(
            "Some structure IDs have conflicting names or acronyms"
        )

    regions = (
        all_regions
        .drop_duplicates(subset=["structure_id"])
        .sort_values("structure_id")
        .reset_index(drop=True)
    )

    region_index = {
        structure_id: index
        for index, structure_id in enumerate(regions["structure_id"])
    }

    return regions, region_index


def verify_parkinsons_probes(
    background_probes: pd.DataFrame,
) -> None:
    """Verify that Parkinson's selected probes are in the background pool."""

    if not PARKINSONS_PROBES_PATH.is_file():
        raise FileNotFoundError(
            f"Missing Parkinson's probes: {PARKINSONS_PROBES_PATH}"
        )

    parkinsons = pd.read_csv(PARKINSONS_PROBES_PATH)

    parkinsons["gene_symbol"] = (
        parkinsons["gene_symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    parkinsons["probe_id_norm"] = parkinsons["probe_id"].map(
        normalize_probe_id
    )

    background_map = dict(
        zip(
            background_probes["gene_symbol"],
            background_probes["probe_id_norm"],
        )
    )

    missing_genes = []
    different_probes = []

    for row in parkinsons.itertuples(index=False):
        gene = row.gene_symbol
        probe = row.probe_id_norm

        if gene not in background_map:
            missing_genes.append(gene)
        elif background_map[gene] != probe:
            different_probes.append(
                (gene, probe, background_map[gene])
            )

    if missing_genes:
        raise ValueError(
            "Parkinson's genes missing from background pool: "
            + ", ".join(sorted(missing_genes))
        )

    if different_probes:
        preview = "; ".join(
            f"{gene}: Parkinson={pd_probe}, background={bg_probe}"
            for gene, pd_probe, bg_probe in different_probes[:20]
        )

        raise ValueError(
            "Parkinson's and background probe selections differ: "
            + preview
        )

    print(
        "Parkinson's selected probes match background probes: PASS"
    )


def process_donor(
    donor_folder: Path,
    probe_to_gene_index: dict[str, int],
    regions: pd.DataFrame,
    region_index: dict[object, int],
    expression_sum: np.ndarray,
    expression_count: np.ndarray,
) -> tuple[int, np.ndarray]:
    donor_id = donor_folder.name
    sample_path = donor_folder / "SampleAnnot.csv"
    expression_path = donor_folder / "MicroarrayExpression.csv"

    samples = pd.read_csv(sample_path)
    sample_count = len(samples)

    local_codes, local_structure_ids = pd.factorize(
        samples["structure_id"],
        sort=True,
    )

    local_structure_ids = list(local_structure_ids)
    local_region_count = len(local_structure_ids)

    samples_per_local_region = np.bincount(
        local_codes,
        minlength=local_region_count,
    )

    global_region_indices = np.asarray(
        [
            region_index[structure_id]
            for structure_id in local_structure_ids
        ],
        dtype=int,
    )

    found_probes: set[str] = set()
    zero_sd_genes = 0

    with expression_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.reader(handle)

        for row_number, row in enumerate(reader, start=1):
            if not row:
                continue

            probe_id = normalize_probe_id(row[0])

            if probe_id not in probe_to_gene_index:
                continue

            if probe_id in found_probes:
                raise ValueError(
                    f"Duplicate selected probe {probe_id} in "
                    f"{expression_path}"
                )

            values = np.fromiter(
                (float(value) for value in row[1:]),
                dtype=np.float64,
                count=sample_count,
            )

            if len(values) != sample_count:
                raise ValueError(
                    f"Probe {probe_id} in {donor_id} has "
                    f"{len(values)} values; expected {sample_count}"
                )

            standard_deviation = values.std(ddof=0)

            if standard_deviation == 0 or not np.isfinite(
                standard_deviation
            ):
                zero_sd_genes += 1
                found_probes.add(probe_id)
                continue

            z_values = (
                values - values.mean()
            ) / standard_deviation

            regional_sums = np.bincount(
                local_codes,
                weights=z_values,
                minlength=local_region_count,
            )

            regional_means = (
                regional_sums / samples_per_local_region
            )

            gene_index = probe_to_gene_index[probe_id]

            expression_sum[
                global_region_indices,
                gene_index,
            ] += regional_means

            expression_count[
                global_region_indices,
                gene_index,
            ] += 1

            found_probes.add(probe_id)

    expected_probes = set(probe_to_gene_index)
    missing_probes = expected_probes.difference(found_probes)

    if missing_probes:
        preview = ", ".join(sorted(missing_probes)[:20])

        raise ValueError(
            f"{donor_id} is missing {len(missing_probes)} "
            f"selected probes. First missing: {preview}"
        )

    donor_region_sample_counts = np.zeros(
        len(regions),
        dtype=np.int64,
    )

    donor_region_sample_counts[global_region_indices] = (
        samples_per_local_region
    )

    print(
        f"  Samples: {sample_count}; "
        f"regions: {local_region_count}; "
        f"probes found: {len(found_probes)}; "
        f"zero-SD genes: {zero_sd_genes}"
    )

    return sample_count, donor_region_sample_counts


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    donor_folders = find_donor_folders()

    if not donor_folders:
        raise FileNotFoundError("No donor folders found")

    background_probes = load_background_probes()
    verify_parkinsons_probes(background_probes)

    regions, region_index = load_region_inventory(
        donor_folders
    )

    gene_symbols = background_probes["gene_symbol"].tolist()

    gene_index_by_probe = {
        probe_id: index
        for index, probe_id in enumerate(
            background_probes["probe_id_norm"]
        )
    }

    number_of_regions = len(regions)
    number_of_genes = len(gene_symbols)

    print(f"Donors: {len(donor_folders)}")
    print(f"Brain structures: {number_of_regions}")
    print(f"Background genes: {number_of_genes}")

    expression_sum = np.zeros(
        (number_of_regions, number_of_genes),
        dtype=np.float64,
    )

    expression_count = np.zeros(
        (number_of_regions, number_of_genes),
        dtype=np.uint8,
    )

    regional_sample_counts = np.zeros(
        number_of_regions,
        dtype=np.int64,
    )

    regional_donor_counts = np.zeros(
        number_of_regions,
        dtype=np.uint8,
    )

    total_samples = 0

    for donor_number, donor_folder in enumerate(
        donor_folders,
        start=1,
    ):
        print(
            f"\nProcessing donor {donor_number}/"
            f"{len(donor_folders)}: {donor_folder.name}"
        )

        sample_count, donor_region_counts = process_donor(
            donor_folder=donor_folder,
            probe_to_gene_index=gene_index_by_probe,
            regions=regions,
            region_index=region_index,
            expression_sum=expression_sum,
            expression_count=expression_count,
        )

        total_samples += sample_count
        regional_sample_counts += donor_region_counts
        regional_donor_counts += (
            donor_region_counts > 0
        ).astype(np.uint8)

        print(f"Finished {donor_folder.name}")

    regional_means = np.full(
        expression_sum.shape,
        np.nan,
        dtype=np.float64,
    )

    valid = expression_count > 0

    regional_means[valid] = (
        expression_sum[valid] / expression_count[valid]
    )

    matrix = pd.DataFrame(
        regional_means,
        columns=gene_symbols,
    )

    matrix = pd.concat(
        [
            regions[REGION_COLUMNS].reset_index(drop=True),
            matrix,
        ],
        axis=1,
    )

    nonmissing_gene_counts = (
        np.isfinite(regional_means)
        .sum(axis=1)
    )

    coverage = regions[REGION_COLUMNS].copy()
    coverage["number_of_donors"] = regional_donor_counts
    coverage["total_sample_count"] = regional_sample_counts
    coverage["number_of_genes"] = nonmissing_gene_counts

    matrix.to_csv(
        MATRIX_OUTPUT_PATH,
        index=False,
    )

    coverage.to_csv(
        COVERAGE_OUTPUT_PATH,
        index=False,
    )

    print("\nBackground expression processing complete")
    print(f"Total samples processed: {total_samples}")
    print(f"Matrix dimensions: {matrix.shape}")
    print(f"Coverage dimensions: {coverage.shape}")

    print("\nStructures by donor coverage:")

    distribution = (
        coverage["number_of_donors"]
        .value_counts()
        .sort_index()
    )

    for donors, structures in distribution.items():
        print(f"  {donors} donor(s): {structures} structures")

    print(
        "\nMissing expression values: "
        f"{np.isnan(regional_means).sum()}"
    )

    print(f"\nWrote {MATRIX_OUTPUT_PATH}")
    print(f"Wrote {COVERAGE_OUTPUT_PATH}")


if __name__ == "__main__":
    main()