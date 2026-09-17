#!/usr/bin/env python3

"""Run coverage and preselected-region sensitivity analyses."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from statsmodels.stats.multitest import multipletests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

ENRICHMENT_PATH = RESULTS_DIR / "parkinsons_regional_enrichment.csv"
TARGET_REGIONS_PATH = RESULTS_DIR / "parkinsons_target_regions.csv"

ALL_SIX_OUTPUT = RESULTS_DIR / "sensitivity_all_six_donors.csv"
TARGET_OUTPUT = RESULTS_DIR / "sensitivity_target_regions.csv"
SUMMARY_OUTPUT = RESULTS_DIR / "sensitivity_analysis_summary.txt"

TARGET_TERMS = (
    "substantia nigra",
    "putamen",
    "caudate",
    "globus pallidus",
)


def apply_fdr(
    dataframe: pd.DataFrame,
    output_column: str,
) -> pd.DataFrame:
    """Apply Benjamini-Hochberg correction to one analysis family."""

    result = dataframe.copy()

    _, q_values, _, _ = multipletests(
        result["empirical_p_value"].to_numpy(),
        alpha=0.05,
        method="fdr_bh",
    )

    result[output_column] = q_values
    result[f"{output_column}_significant_0_05"] = q_values < 0.05

    return result


def format_region_table(dataframe: pd.DataFrame) -> str:
    columns = [
        "structure_acronym",
        "structure_name",
        "number_of_donors",
        "parkinsons_expression_score",
        "enrichment_z_score",
        "empirical_p_value",
    ]

    sensitivity_q_columns = [
        column
        for column in dataframe.columns
        if column.startswith("sensitivity_fdr_q")
    ]

    columns.extend(sensitivity_q_columns)

    return dataframe[columns].to_string(index=False)


def main() -> None:
    if not ENRICHMENT_PATH.is_file():
        raise FileNotFoundError(
            f"Missing enrichment results: {ENRICHMENT_PATH}"
        )

    results = pd.read_csv(ENRICHMENT_PATH)

    numeric_columns = [
        "number_of_donors",
        "parkinsons_expression_score",
        "enrichment_z_score",
        "empirical_p_value",
        "fdr_q_value",
    ]

    for column in numeric_columns:
        results[column] = pd.to_numeric(
            results[column],
            errors="coerce",
        )

    # Sensitivity analysis 1:
    # Include only structures represented in all six donors.
    all_six = results.loc[
        results["number_of_donors"] == 6
    ].copy()

    all_six = apply_fdr(
        all_six,
        "sensitivity_fdr_q_all_six",
    )

    all_six = all_six.sort_values(
        ["enrichment_z_score", "empirical_p_value"],
        ascending=[False, True],
    ).reset_index(drop=True)

    all_six.to_csv(ALL_SIX_OUTPUT, index=False)

    # Sensitivity analysis 2:
    # Correct only across the anatomically preselected Parkinson's regions.
    pattern = "|".join(TARGET_TERMS)

    targets = results.loc[
        results["structure_name"]
        .astype(str)
        .str.contains(
            pattern,
            case=False,
            regex=True,
            na=False,
        )
    ].copy()

    targets = apply_fdr(
        targets,
        "sensitivity_fdr_q_targets",
    )

    targets = targets.sort_values(
        ["enrichment_z_score", "empirical_p_value"],
        ascending=[False, True],
    ).reset_index(drop=True)

    targets.to_csv(TARGET_OUTPUT, index=False)

    all_six_significant = int(
        all_six["sensitivity_fdr_q_all_six_significant_0_05"].sum()
    )

    target_significant = int(
        targets["sensitivity_fdr_q_targets_significant_0_05"].sum()
    )

    summary_lines = [
        "Parkinson's Regional Enrichment Sensitivity Analysis",
        "=" * 55,
        "",
        "Primary analysis",
        f"Regions tested: {len(results)}",
        (
            "FDR-significant regions: "
            f"{int((results['fdr_q_value'] < 0.05).sum())}"
        ),
        "",
        "Sensitivity analysis 1: all six donors",
        f"Regions tested: {len(all_six)}",
        f"FDR-significant regions: {all_six_significant}",
        "",
        "Top 10 all-six-donor regions:",
        format_region_table(all_six.head(10)),
        "",
        "Sensitivity analysis 2: preselected Parkinson's regions",
        f"Regions tested: {len(targets)}",
        f"FDR-significant regions: {target_significant}",
        "",
        format_region_table(targets),
        "",
        "Interpretation note:",
        (
            "The target-region correction is appropriate only as a "
            "predefined hypothesis-family sensitivity analysis. It must "
            "not replace or be presented as the primary whole-brain test."
        ),
    ]

    summary_text = "\n".join(summary_lines)

    SUMMARY_OUTPUT.write_text(
        summary_text,
        encoding="utf-8",
    )

    print(summary_text)

    print(f"\nWrote {ALL_SIX_OUTPUT}")
    print(f"Wrote {TARGET_OUTPUT}")
    print(f"Wrote {SUMMARY_OUTPUT}")


if __name__ == "__main__":
    main()