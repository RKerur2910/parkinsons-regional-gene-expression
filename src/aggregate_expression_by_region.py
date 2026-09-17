#!/usr/bin/env python3

"""Extract selected Parkinson's probes and aggregate expression by brain region."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "allen_human_brain_atlas"
RESULTS_DIR = PROJECT_ROOT / "results"

SELECTED_PROBES_PATH = RESULTS_DIR / "parkinsons_selected_probes.csv"

DONOR_OUTPUT_PATH = RESULTS_DIR / "donor_region_gene_expression.csv"
REGIONAL_LONG_PATH = RESULTS_DIR / "regional_gene_expression_long.csv"
REGIONAL_MATRIX_PATH = RESULTS_DIR / "regional_gene_expression_matrix.csv"
COVERAGE_PATH = RESULTS_DIR / "regional_expression_coverage.csv"

DONOR_PREFIX = "normalized_microarray_donor"

REGION_COLUMNS = [
    "structure_id",
    "structure_acronym",
    "structure_name",
]


def normalize_probe_id(value: object) -> str:
    """Convert a probe ID into a consistent string."""

    text = str(value).strip()

    if text.endswith(".0"):
        text = text[:-2]

    return text


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


def load_selected_probes() -> pd.DataFrame:
    """Load the selected Parkinson's probes."""

    if not SELECTED_PROBES_PATH.is_file():
        raise FileNotFoundError(
            f"Selected-probe file not found: {SELECTED_PROBES_PATH}"
        )

    probes = pd.read_csv(SELECTED_PROBES_PATH)

    required_columns = {"gene_symbol", "probe_id"}

    missing_columns = required_columns.difference(probes.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Selected-probe file is missing columns: {missing_text}"
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

    probes = probes.loc[
        probes["gene_symbol"].ne("")
        & probes["gene_symbol"].ne("NAN")
        & probes["probe_id_norm"].ne("")
    ].copy()

    if probes["gene_symbol"].duplicated().any():
        duplicated = sorted(
            probes.loc[
                probes["gene_symbol"].duplicated(keep=False),
                "gene_symbol",
            ].unique()
        )

        raise ValueError(
            "More than one selected probe exists for these genes: "
            + ", ".join(duplicated)
        )

    if probes["probe_id_norm"].duplicated().any():
        duplicated = sorted(
            probes.loc[
                probes["probe_id_norm"].duplicated(keep=False),
                "probe_id_norm",
            ].unique()
        )

        raise ValueError(
            "A selected probe is assigned more than once: "
            + ", ".join(duplicated)
        )

    return probes.sort_values("gene_symbol").reset_index(drop=True)


def read_selected_expression_rows(
    expression_path: Path,
    probe_to_gene: dict[str, str],
    expected_sample_count: int,
) -> tuple[dict[str, np.ndarray], set[str]]:
    """
    Stream an expression matrix and retain only selected probes.

    MicroarrayExpression.csv has no header. The first value in each row
    is probe_id, followed by one expression value per tissue sample.
    """

    expression_by_gene: dict[str, np.ndarray] = {}
    found_probe_ids: set[str] = set()

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

            if probe_id not in probe_to_gene:
                continue

            if probe_id in found_probe_ids:
                raise ValueError(
                    f"Duplicate selected probe {probe_id} in "
                    f"{expression_path} at row {row_number}"
                )

            values = np.asarray(row[1:], dtype=float)

            if len(values) != expected_sample_count:
                raise ValueError(
                    f"Probe {probe_id} in {expression_path.name} has "
                    f"{len(values)} expression values, but "
                    f"{expected_sample_count} samples were expected"
                )

            gene_symbol = probe_to_gene[probe_id]
            expression_by_gene[gene_symbol] = values
            found_probe_ids.add(probe_id)

    return expression_by_gene, found_probe_ids


def calculate_z_scores(
    expression: pd.DataFrame,
    donor_id: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Calculate a z-score for every gene within one donor."""

    means = expression.mean(axis=0)
    standard_deviations = expression.std(axis=0, ddof=0)

    zero_standard_deviation_genes = sorted(
        standard_deviations.index[
            standard_deviations.eq(0)
            | standard_deviations.isna()
        ].tolist()
    )

    safe_standard_deviations = standard_deviations.replace(0, np.nan)

    z_expression = (
        expression - means
    ) / safe_standard_deviations

    if zero_standard_deviation_genes:
        print(
            f"  {donor_id}: genes with zero standard deviation: "
            + ", ".join(zero_standard_deviation_genes)
        )
    else:
        print(
            f"  {donor_id}: genes with zero standard deviation: none"
        )

    return z_expression, zero_standard_deviation_genes


def aggregate_one_donor(
    donor_folder: Path,
    selected_probes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Extract and aggregate selected gene expression for one donor."""

    donor_id = donor_folder.name

    sample_path = donor_folder / "SampleAnnot.csv"
    expression_path = donor_folder / "MicroarrayExpression.csv"

    if not sample_path.is_file():
        raise FileNotFoundError(f"Missing file: {sample_path}")

    if not expression_path.is_file():
        raise FileNotFoundError(f"Missing file: {expression_path}")

    samples = pd.read_csv(sample_path)

    missing_region_columns = set(REGION_COLUMNS).difference(
        samples.columns
    )

    if missing_region_columns:
        missing_text = ", ".join(sorted(missing_region_columns))

        raise ValueError(
            f"{sample_path} is missing columns: {missing_text}"
        )

    sample_count = len(samples)

    print(f"  Tissue samples: {sample_count}")

    probe_to_gene = dict(
        zip(
            selected_probes["probe_id_norm"],
            selected_probes["gene_symbol"],
        )
    )

    expression_by_gene, found_probe_ids = (
        read_selected_expression_rows(
            expression_path=expression_path,
            probe_to_gene=probe_to_gene,
            expected_sample_count=sample_count,
        )
    )

    expected_probe_ids = set(probe_to_gene)
    missing_probe_ids = expected_probe_ids.difference(found_probe_ids)

    if missing_probe_ids:
        preview = ", ".join(sorted(missing_probe_ids)[:20])

        raise ValueError(
            f"{donor_id} is missing {len(missing_probe_ids)} "
            f"selected probes. First missing IDs: {preview}"
        )

    print(
        f"  Selected probes found: "
        f"{len(found_probe_ids)}/{len(expected_probe_ids)}"
    )

    gene_order = selected_probes["gene_symbol"].tolist()

    expression = pd.DataFrame(
        {
            gene: expression_by_gene[gene]
            for gene in gene_order
        }
    )

    z_expression, zero_sd_genes = calculate_z_scores(
        expression=expression,
        donor_id=donor_id,
    )

    region_metadata = samples[REGION_COLUMNS].reset_index(drop=True)

    raw_with_regions = pd.concat(
        [region_metadata, expression],
        axis=1,
    )

    z_with_regions = pd.concat(
        [region_metadata, z_expression],
        axis=1,
    )

    raw_means = (
        raw_with_regions
        .groupby(REGION_COLUMNS, dropna=False)[gene_order]
        .mean()
        .reset_index()
    )

    z_means = (
        z_with_regions
        .groupby(REGION_COLUMNS, dropna=False)[gene_order]
        .mean()
        .reset_index()
    )

    region_counts = (
        region_metadata
        .groupby(REGION_COLUMNS, dropna=False)
        .size()
        .rename("sample_count")
        .reset_index()
    )

    raw_long = raw_means.melt(
        id_vars=REGION_COLUMNS,
        value_vars=gene_order,
        var_name="gene_symbol",
        value_name="mean_expression",
    )

    z_long = z_means.melt(
        id_vars=REGION_COLUMNS,
        value_vars=gene_order,
        var_name="gene_symbol",
        value_name="mean_z_expression",
    )

    donor_long = raw_long.merge(
        z_long,
        on=REGION_COLUMNS + ["gene_symbol"],
        how="outer",
        validate="one_to_one",
    )

    donor_long = donor_long.merge(
        region_counts,
        on=REGION_COLUMNS,
        how="left",
        validate="many_to_one",
    )

    donor_long.insert(0, "donor_id", donor_id)

    donor_coverage = region_counts.copy()
    donor_coverage.insert(0, "donor_id", donor_id)

    return donor_long, donor_coverage, zero_sd_genes


def combine_donors(
    donor_results: pd.DataFrame,
) -> pd.DataFrame:
    """Combine donor-level regional results with equal donor weighting."""

    combined = (
        donor_results
        .groupby(
            REGION_COLUMNS + ["gene_symbol"],
            dropna=False,
        )
        .agg(
            mean_expression_across_donors=(
                "mean_expression",
                "mean",
            ),
            mean_z_expression_across_donors=(
                "mean_z_expression",
                "mean",
            ),
            standard_deviation_across_donors=(
                "mean_z_expression",
                "std",
            ),
            number_of_donors=(
                "donor_id",
                "nunique",
            ),
            total_sample_count=(
                "sample_count",
                "sum",
            ),
        )
        .reset_index()
    )

    return combined.sort_values(
        REGION_COLUMNS + ["gene_symbol"]
    ).reset_index(drop=True)


def create_wide_matrix(
    regional_long: pd.DataFrame,
) -> pd.DataFrame:
    """Create a region-by-gene matrix of mean z-expression."""

    matrix = (
        regional_long
        .pivot(
            index=REGION_COLUMNS,
            columns="gene_symbol",
            values="mean_z_expression_across_donors",
        )
        .reset_index()
    )

    matrix.columns.name = None

    gene_columns = sorted(
        column
        for column in matrix.columns
        if column not in REGION_COLUMNS
    )

    return matrix[REGION_COLUMNS + gene_columns]


def create_coverage_table(
    donor_coverage: pd.DataFrame,
    regional_long: pd.DataFrame,
) -> pd.DataFrame:
    """Create one coverage row per brain structure."""

    coverage = (
        donor_coverage
        .groupby(REGION_COLUMNS, dropna=False)
        .agg(
            number_of_donors=("donor_id", "nunique"),
            total_sample_count=("sample_count", "sum"),
        )
        .reset_index()
    )

    gene_counts = (
        regional_long
        .groupby(REGION_COLUMNS, dropna=False)["gene_symbol"]
        .nunique()
        .rename("number_of_genes")
        .reset_index()
    )

    coverage = coverage.merge(
        gene_counts,
        on=REGION_COLUMNS,
        how="left",
        validate="one_to_one",
    )

    coverage["number_of_genes"] = (
        coverage["number_of_genes"]
        .fillna(0)
        .astype(int)
    )

    return coverage.sort_values(
        ["number_of_donors", "total_sample_count", "structure_name"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    selected_probes = load_selected_probes()
    donor_folders = find_donor_folders()

    if not donor_folders:
        raise FileNotFoundError(
            f"No donor folders beginning with "
            f"{DONOR_PREFIX!r} were found in {DATA_DIR}"
        )

    print(f"Number of selected genes: {selected_probes['gene_symbol'].nunique()}")
    print(f"Number of selected probes: {selected_probes['probe_id_norm'].nunique()}")
    print(f"Number of donor folders: {len(donor_folders)}")

    donor_result_frames = []
    donor_coverage_frames = []
    all_zero_sd_genes: dict[str, list[str]] = {}

    for donor_number, donor_folder in enumerate(
        donor_folders,
        start=1,
    ):
        print(
            f"\nProcessing donor {donor_number}/{len(donor_folders)}: "
            f"{donor_folder.name}"
        )

        donor_result, donor_coverage, zero_sd_genes = (
            aggregate_one_donor(
                donor_folder=donor_folder,
                selected_probes=selected_probes,
            )
        )

        donor_result_frames.append(donor_result)
        donor_coverage_frames.append(donor_coverage)

        all_zero_sd_genes[donor_folder.name] = zero_sd_genes

        print(f"Finished {donor_folder.name}")

    donor_results = pd.concat(
        donor_result_frames,
        ignore_index=True,
    )

    donor_coverage = pd.concat(
        donor_coverage_frames,
        ignore_index=True,
    )

    regional_long = combine_donors(donor_results)
    regional_matrix = create_wide_matrix(regional_long)

    coverage = create_coverage_table(
        donor_coverage=donor_coverage,
        regional_long=regional_long,
    )

    donor_results.to_csv(
        DONOR_OUTPUT_PATH,
        index=False,
    )

    regional_long.to_csv(
        REGIONAL_LONG_PATH,
        index=False,
    )

    regional_matrix.to_csv(
        REGIONAL_MATRIX_PATH,
        index=False,
    )

    coverage.to_csv(
        COVERAGE_PATH,
        index=False,
    )

    zero_sd_total = sum(
        len(genes)
        for genes in all_zero_sd_genes.values()
    )

    print("\nProcessing complete")
    print(f"Total zero-standard-deviation cases: {zero_sd_total}")
    print(f"Donor-level output dimensions: {donor_results.shape}")
    print(f"Combined regional output dimensions: {regional_long.shape}")
    print(f"Wide matrix dimensions: {regional_matrix.shape}")
    print(
        f"Number of unique brain structures: "
        f"{coverage['structure_id'].nunique()}"
    )

    print("\nStructures by number of donors:")

    donor_distribution = (
        coverage["number_of_donors"]
        .value_counts()
        .sort_index()
    )

    for number_of_donors, number_of_structures in donor_distribution.items():
        print(
            f"  {number_of_donors} donor(s): "
            f"{number_of_structures} structures"
        )

    print(f"\nWrote {DONOR_OUTPUT_PATH}")
    print(f"Wrote {REGIONAL_LONG_PATH}")
    print(f"Wrote {REGIONAL_MATRIX_PATH}")
    print(f"Wrote {COVERAGE_PATH}")


if __name__ == "__main__":
    main()