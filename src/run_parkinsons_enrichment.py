#!/usr/bin/env python3

"""Test regional Parkinson's gene expression against random gene sets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

MATRIX_PATH = (
    RESULTS_DIR / "background_regional_gene_expression_matrix.csv"
)
COVERAGE_PATH = RESULTS_DIR / "background_expression_coverage.csv"
BACKGROUND_PROBES_PATH = RESULTS_DIR / "background_selected_probes.csv"
PARKINSONS_PROBES_PATH = RESULTS_DIR / "parkinsons_selected_probes.csv"

ENRICHMENT_OUTPUT_PATH = (
    RESULTS_DIR / "parkinsons_regional_enrichment.csv"
)
NULL_OUTPUT_PATH = RESULTS_DIR / "parkinsons_null_scores.npz"
STRATA_OUTPUT_PATH = RESULTS_DIR / "permutation_strata_summary.csv"

REGION_COLUMNS = [
    "structure_id",
    "structure_acronym",
    "structure_name",
]

MINIMUM_DONORS = 4
NUMBER_OF_PERMUTATIONS = 10_000
NUMBER_OF_STRATA = 10
RANDOM_SEED = 42


def clean_gene_symbols(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
    )


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Apply Benjamini-Hochberg FDR correction."""

    _, corrected, _, _ = multipletests(
        p_values,
        alpha=0.05,
        method="fdr_bh",
    )

    return corrected


def main() -> None:
    required_files = [
        MATRIX_PATH,
        COVERAGE_PATH,
        BACKGROUND_PROBES_PATH,
        PARKINSONS_PROBES_PATH,
    ]

    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required file: {path}")

    print("Loading background expression matrix...")
    matrix = pd.read_csv(MATRIX_PATH, low_memory=False)

    coverage = pd.read_csv(COVERAGE_PATH)
    background = pd.read_csv(BACKGROUND_PROBES_PATH)
    parkinsons = pd.read_csv(PARKINSONS_PROBES_PATH)

    for column in REGION_COLUMNS:
        if column not in matrix.columns:
            raise ValueError(f"Expression matrix is missing {column}")

    background["gene_symbol"] = clean_gene_symbols(
        background["gene_symbol"]
    )

    parkinsons["gene_symbol"] = clean_gene_symbols(
        parkinsons["gene_symbol"]
    )

    background["overall_detection_rate"] = pd.to_numeric(
        background["overall_detection_rate"],
        errors="coerce",
    )

    background = background.dropna(
        subset=["gene_symbol", "overall_detection_rate"]
    )

    background = background.drop_duplicates(
        subset=["gene_symbol"],
        keep="first",
    )

    parkinsons_genes = sorted(
        parkinsons["gene_symbol"]
        .dropna()
        .unique()
        .tolist()
    )

    matrix_gene_columns = [
        column
        for column in matrix.columns
        if column not in REGION_COLUMNS
    ]

    matrix_gene_set = set(matrix_gene_columns)
    background_gene_set = set(background["gene_symbol"])

    missing_parkinsons_matrix = sorted(
        set(parkinsons_genes).difference(matrix_gene_set)
    )

    missing_parkinsons_background = sorted(
        set(parkinsons_genes).difference(background_gene_set)
    )

    if missing_parkinsons_matrix:
        raise ValueError(
            "Parkinson's genes missing from expression matrix: "
            + ", ".join(missing_parkinsons_matrix)
        )

    if missing_parkinsons_background:
        raise ValueError(
            "Parkinson's genes missing from background metadata: "
            + ", ".join(missing_parkinsons_background)
        )

    eligible_coverage = coverage.loc[
        coverage["number_of_donors"] >= MINIMUM_DONORS
    ].copy()

    eligible_structure_ids = set(
        eligible_coverage["structure_id"]
    )

    eligible_matrix = matrix.loc[
        matrix["structure_id"].isin(eligible_structure_ids)
    ].copy()

    eligible_matrix = eligible_matrix.merge(
        eligible_coverage[
            REGION_COLUMNS
            + ["number_of_donors", "total_sample_count"]
        ],
        on=REGION_COLUMNS,
        how="inner",
        validate="one_to_one",
    )

    eligible_matrix = eligible_matrix.sort_values(
        "structure_id"
    ).reset_index(drop=True)

    print(f"Total regions: {len(matrix)}")
    print(
        f"Regions represented in at least {MINIMUM_DONORS} donors: "
        f"{len(eligible_matrix)}"
    )
    print(f"Background genes: {len(matrix_gene_columns)}")
    print(f"Parkinson's genes: {len(parkinsons_genes)}")

    expression = eligible_matrix[
        matrix_gene_columns
    ].to_numpy(dtype=np.float32)

    gene_to_column = {
        gene: index
        for index, gene in enumerate(matrix_gene_columns)
    }

    parkinsons_indices = np.asarray(
        [gene_to_column[gene] for gene in parkinsons_genes],
        dtype=int,
    )

    observed_scores = expression[
        :,
        parkinsons_indices,
    ].mean(axis=1)

    background = background.loc[
        background["gene_symbol"].isin(matrix_gene_set)
    ].copy()

    background = background.sort_values(
        "gene_symbol"
    ).reset_index(drop=True)

    ranked_detection = background[
        "overall_detection_rate"
    ].rank(method="first")

    background["detection_stratum"] = pd.qcut(
        ranked_detection,
        q=NUMBER_OF_STRATA,
        labels=False,
    ).astype(int)

    gene_to_stratum = dict(
        zip(
            background["gene_symbol"],
            background["detection_stratum"],
        )
    )

    parkinsons_gene_set = set(parkinsons_genes)

    parkinsons_stratum_counts = (
        pd.Series(
            [gene_to_stratum[gene] for gene in parkinsons_genes]
        )
        .value_counts()
        .sort_index()
    )

    background_pools: dict[int, np.ndarray] = {}
    strata_rows = []

    for stratum in range(NUMBER_OF_STRATA):
        disease_count = int(
            parkinsons_stratum_counts.get(stratum, 0)
        )

        pool_genes = background.loc[
            (background["detection_stratum"] == stratum)
            & ~background["gene_symbol"].isin(parkinsons_gene_set),
            "gene_symbol",
        ].tolist()

        pool_indices = np.asarray(
            [gene_to_column[gene] for gene in pool_genes],
            dtype=int,
        )

        if len(pool_indices) < disease_count:
            raise ValueError(
                f"Stratum {stratum} has {len(pool_indices)} "
                f"background genes but requires {disease_count}"
            )

        background_pools[stratum] = pool_indices

        strata_rows.append(
            {
                "detection_stratum": stratum,
                "parkinsons_gene_count": disease_count,
                "available_background_gene_count": len(pool_indices),
            }
        )

    strata_summary = pd.DataFrame(strata_rows)
    strata_summary.to_csv(STRATA_OUTPUT_PATH, index=False)

    print("\nDetection-matched permutation strata:")
    print(strata_summary.to_string(index=False))

    rng = np.random.default_rng(RANDOM_SEED)

    number_of_regions = len(eligible_matrix)

    null_scores = np.empty(
        (number_of_regions, NUMBER_OF_PERMUTATIONS),
        dtype=np.float32,
    )

    print(
        f"\nRunning {NUMBER_OF_PERMUTATIONS:,} random gene-set "
        "permutations..."
    )

    for permutation in range(NUMBER_OF_PERMUTATIONS):
        selected_indices = []

        for stratum in range(NUMBER_OF_STRATA):
            required_count = int(
                parkinsons_stratum_counts.get(stratum, 0)
            )

            if required_count == 0:
                continue

            selected = rng.choice(
                background_pools[stratum],
                size=required_count,
                replace=False,
            )

            selected_indices.append(selected)

        random_gene_indices = np.concatenate(selected_indices)

        if len(random_gene_indices) != len(parkinsons_genes):
            raise RuntimeError(
                "Random gene-set size does not match Parkinson's set"
            )

        null_scores[:, permutation] = expression[
            :,
            random_gene_indices,
        ].mean(axis=1)

        completed = permutation + 1

        if completed % 1_000 == 0:
            print(
                f"  Completed {completed:,}/"
                f"{NUMBER_OF_PERMUTATIONS:,}"
            )

    null_means = null_scores.mean(axis=1)
    null_standard_deviations = null_scores.std(
        axis=1,
        ddof=1,
    )

    enrichment_z_scores = (
        observed_scores - null_means
    ) / null_standard_deviations

    empirical_p_values = (
        1
        + np.sum(
            null_scores >= observed_scores[:, np.newaxis],
            axis=1,
        )
    ) / (NUMBER_OF_PERMUTATIONS + 1)

    corrected_q_values = benjamini_hochberg(
        empirical_p_values
    )

    results = eligible_matrix[
        REGION_COLUMNS
        + ["number_of_donors", "total_sample_count"]
    ].copy()

    results["parkinsons_expression_score"] = observed_scores
    results["random_mean_score"] = null_means
    results["random_standard_deviation"] = (
        null_standard_deviations
    )
    results["enrichment_z_score"] = enrichment_z_scores
    results["empirical_p_value"] = empirical_p_values
    results["fdr_q_value"] = corrected_q_values
    results["significant_fdr_0_05"] = (
        results["fdr_q_value"] < 0.05
    )

    results["regional_rank"] = (
        results["parkinsons_expression_score"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    results = results.sort_values(
        [
            "parkinsons_expression_score",
            "enrichment_z_score",
        ],
        ascending=[False, False],
    ).reset_index(drop=True)

    results.to_csv(
        ENRICHMENT_OUTPUT_PATH,
        index=False,
    )

    np.savez_compressed(
        NULL_OUTPUT_PATH,
        null_scores=null_scores,
        structure_id=eligible_matrix[
            "structure_id"
        ].to_numpy(),
        structure_acronym=eligible_matrix[
            "structure_acronym"
        ].astype(str).to_numpy(),
        structure_name=eligible_matrix[
            "structure_name"
        ].astype(str).to_numpy(),
        observed_scores=observed_scores,
        random_seed=np.asarray([RANDOM_SEED]),
        number_of_permutations=np.asarray(
            [NUMBER_OF_PERMUTATIONS]
        ),
    )

    significant_count = int(
        results["significant_fdr_0_05"].sum()
    )

    print("\nAnalysis complete")
    print(
        f"Significant regions after FDR correction: "
        f"{significant_count}/{len(results)}"
    )

    print("\nTop 20 regions by Parkinson's expression score:")

    display_columns = [
        "regional_rank",
        "structure_acronym",
        "structure_name",
        "number_of_donors",
        "parkinsons_expression_score",
        "enrichment_z_score",
        "empirical_p_value",
        "fdr_q_value",
        "significant_fdr_0_05",
    ]

    print(
        results[display_columns]
        .head(20)
        .to_string(index=False)
    )

    print(f"\nWrote {ENRICHMENT_OUTPUT_PATH}")
    print(f"Wrote {NULL_OUTPUT_PATH}")
    print(f"Wrote {STRATA_OUTPUT_PATH}")


if __name__ == "__main__":
    main()