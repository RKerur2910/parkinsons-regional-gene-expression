#!/usr/bin/env python3

"""Create tables and figures for the Parkinson's enrichment results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

RESULTS_PATH = RESULTS_DIR / "parkinsons_regional_enrichment.csv"
NULL_PATH = RESULTS_DIR / "parkinsons_null_scores.npz"

TARGET_OUTPUT_PATH = RESULTS_DIR / "parkinsons_target_regions.csv"

TOP_ENRICHMENT_FIGURE = FIGURES_DIR / "top_enriched_regions.png"
TARGET_REGIONS_FIGURE = FIGURES_DIR / "parkinsons_target_regions.png"
NULL_FIGURE = FIGURES_DIR / "target_region_null_distributions.png"
SIGNIFICANCE_FIGURE = FIGURES_DIR / "regional_significance_plot.png"

TARGET_TERMS = (
    "substantia nigra",
    "putamen",
    "caudate",
    "globus pallidus",
)

NULL_STRUCTURE_IDS = [
    9074,  # Substantia nigra pars compacta, left
    9075,  # Substantia nigra pars reticulata, left
    4288,  # Putamen, left
    4280,  # Head of caudate nucleus, left
    4295,  # Globus pallidus external segment, left
    4296,  # Globus pallidus internal segment, left
]


def load_results() -> pd.DataFrame:
    if not RESULTS_PATH.is_file():
        raise FileNotFoundError(f"Missing results: {RESULTS_PATH}")

    results = pd.read_csv(RESULTS_PATH)

    numeric_columns = [
        "parkinsons_expression_score",
        "random_mean_score",
        "random_standard_deviation",
        "enrichment_z_score",
        "empirical_p_value",
        "fdr_q_value",
        "number_of_donors",
        "total_sample_count",
    ]

    for column in numeric_columns:
        results[column] = pd.to_numeric(
            results[column],
            errors="coerce",
        )

    results["significant_fdr_0_05"] = (
        results["significant_fdr_0_05"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    return results


def select_target_regions(results: pd.DataFrame) -> pd.DataFrame:
    pattern = "|".join(TARGET_TERMS)

    target_regions = results.loc[
        results["structure_name"]
        .astype(str)
        .str.contains(
            pattern,
            case=False,
            regex=True,
            na=False,
        )
    ].copy()

    target_regions = target_regions.sort_values(
        "enrichment_z_score",
        ascending=False,
    ).reset_index(drop=True)

    return target_regions


def plot_top_enriched_regions(results: pd.DataFrame) -> None:
    top = (
        results
        .sort_values("enrichment_z_score", ascending=False)
        .head(20)
        .sort_values("enrichment_z_score", ascending=True)
    )

    colors = [
        "#c62828" if significant else "#3366a3"
        for significant in top["significant_fdr_0_05"]
    ]

    fig, axis = plt.subplots(figsize=(11, 8))

    axis.barh(
        top["structure_name"],
        top["enrichment_z_score"],
        color=colors,
    )

    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Enrichment z-score")
    axis.set_ylabel("Brain region")
    axis.set_title(
        "Top 20 regions by Parkinson’s gene-set enrichment"
    )

    fig.tight_layout()
    fig.savefig(
        TOP_ENRICHMENT_FIGURE,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_target_regions(targets: pd.DataFrame) -> None:
    ordered = targets.sort_values(
        "enrichment_z_score",
        ascending=True,
    )

    colors = [
        "#c62828" if significant else "#5b8db8"
        for significant in ordered["significant_fdr_0_05"]
    ]

    fig, axis = plt.subplots(figsize=(11, 7))

    axis.barh(
        ordered["structure_name"],
        ordered["enrichment_z_score"],
        color=colors,
    )

    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Enrichment z-score")
    axis.set_ylabel("Parkinson’s-related brain region")
    axis.set_title(
        "Parkinson’s gene-set enrichment in target structures"
    )

    fig.tight_layout()
    fig.savefig(
        TARGET_REGIONS_FIGURE,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_significance(results: pd.DataFrame) -> None:
    plotting = results.copy()

    plotting["negative_log10_p"] = -np.log10(
        plotting["empirical_p_value"].clip(lower=1e-12)
    )

    colors = np.where(
        plotting["significant_fdr_0_05"],
        "#c62828",
        "#607d8b",
    )

    fig, axis = plt.subplots(figsize=(9, 6))

    axis.scatter(
        plotting["enrichment_z_score"],
        plotting["negative_log10_p"],
        c=colors,
        alpha=0.75,
        s=35,
        edgecolors="none",
    )

    axis.axvline(0, color="black", linewidth=0.8)
    axis.axhline(
        -np.log10(0.05),
        color="#c62828",
        linestyle="--",
        linewidth=1,
        label="Uncorrected p = 0.05",
    )

    axis.set_xlabel("Enrichment z-score")
    axis.set_ylabel("−log10 empirical p-value")
    axis.set_title("Regional Parkinson’s enrichment significance")
    axis.legend()

    fig.tight_layout()
    fig.savefig(
        SIGNIFICANCE_FIGURE,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_null_distributions(results: pd.DataFrame) -> None:
    if not NULL_PATH.is_file():
        raise FileNotFoundError(f"Missing null data: {NULL_PATH}")

    null_data = np.load(NULL_PATH)

    null_scores = null_data["null_scores"]
    null_structure_ids = null_data["structure_id"]
    observed_scores = null_data["observed_scores"]

    null_index = {
        int(structure_id): index
        for index, structure_id in enumerate(null_structure_ids)
    }

    available_ids = [
        structure_id
        for structure_id in NULL_STRUCTURE_IDS
        if structure_id in null_index
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(14, 8),
    )

    axes = axes.flatten()

    for axis, structure_id in zip(axes, available_ids):
        index = null_index[structure_id]

        region_row = results.loc[
            results["structure_id"] == structure_id
        ]

        if region_row.empty:
            axis.set_visible(False)
            continue

        region_name = region_row.iloc[0]["structure_name"]
        p_value = region_row.iloc[0]["empirical_p_value"]
        q_value = region_row.iloc[0]["fdr_q_value"]

        axis.hist(
            null_scores[index],
            bins=40,
            color="#9dbbd3",
            edgecolor="white",
        )

        axis.axvline(
            observed_scores[index],
            color="#c62828",
            linewidth=2,
            label="Observed score",
        )

        axis.set_title(region_name, fontsize=10)
        axis.set_xlabel("Random gene-set score")
        axis.set_ylabel("Frequency")
        axis.text(
            0.03,
            0.95,
            f"p = {p_value:.4f}\nq = {q_value:.4f}",
            transform=axis.transAxes,
            verticalalignment="top",
            fontsize=9,
        )
        axis.legend(fontsize=8)

    for axis in axes[len(available_ids):]:
        axis.set_visible(False)

    fig.suptitle(
        "Observed Parkinson’s scores versus matched random gene sets",
        fontsize=14,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(
        NULL_FIGURE,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results = load_results()
    target_regions = select_target_regions(results)

    target_regions.to_csv(
        TARGET_OUTPUT_PATH,
        index=False,
    )

    plot_top_enriched_regions(results)
    plot_target_regions(target_regions)
    plot_significance(results)
    plot_null_distributions(results)

    print(f"Regions analyzed: {len(results)}")
    print(f"FDR-significant regions: {results['significant_fdr_0_05'].sum()}")
    print(f"Parkinson’s target regions found: {len(target_regions)}")

    print("\nParkinson’s-related regions:")
    print(
        target_regions[
            [
                "structure_acronym",
                "structure_name",
                "number_of_donors",
                "parkinsons_expression_score",
                "enrichment_z_score",
                "empirical_p_value",
                "fdr_q_value",
            ]
        ].to_string(index=False)
    )

    print(f"\nWrote {TARGET_OUTPUT_PATH}")
    print(f"Wrote {TOP_ENRICHMENT_FIGURE}")
    print(f"Wrote {TARGET_REGIONS_FIGURE}")
    print(f"Wrote {NULL_FIGURE}")
    print(f"Wrote {SIGNIFICANCE_FIGURE}")


if __name__ == "__main__":
    main()