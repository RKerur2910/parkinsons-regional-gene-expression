#!/usr/bin/env python3

"""Plot Parkinson's regional enrichment results in MNI brain space."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import plotting


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "allen_human_brain_atlas"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

ENRICHMENT_PATH = RESULTS_DIR / "parkinsons_regional_enrichment.csv"
COORDINATE_OUTPUT_PATH = RESULTS_DIR / "parkinsons_region_coordinates.csv"

ENRICHMENT_FIGURE_PATH = (
    FIGURES_DIR / "parkinsons_enrichment_glass_brain.png"
)
EXPRESSION_FIGURE_PATH = (
    FIGURES_DIR / "parkinsons_expression_score_glass_brain.png"
)
TARGET_FIGURE_PATH = (
    FIGURES_DIR / "parkinsons_target_regions_glass_brain.png"
)

DONOR_PREFIX = "normalized_microarray_donor"

ANNOTATION_COLUMNS = [
    "structure_id",
    "structure_acronym",
    "structure_name",
    "mni_x",
    "mni_y",
    "mni_z",
]

REQUIRED_ENRICHMENT_COLUMNS = [
    "structure_id",
    "structure_acronym",
    "structure_name",
    "parkinsons_expression_score",
    "enrichment_z_score",
]

TARGET_TERMS = (
    "substantia nigra",
    "putamen",
    "caudate",
    "globus pallidus",
)


def require_columns(
    frame: pd.DataFrame,
    required: list[str],
    source_name: str,
) -> None:
    """Raise a clear error when required columns are missing."""
    missing = [column for column in required if column not in frame.columns]

    if missing:
        raise ValueError(
            f"{source_name} is missing required columns: "
            + ", ".join(missing)
        )


def find_annotation_files(data_dir: Path) -> list[Path]:
    """Find SampleAnnot.csv inside every donor directory."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    files = [
        folder / "SampleAnnot.csv"
        for folder in data_dir.iterdir()
        if folder.is_dir()
        and folder.name.startswith(DONOR_PREFIX)
        and (folder / "SampleAnnot.csv").is_file()
    ]

    return sorted(files, key=lambda path: path.parent.name)


def load_annotations(annotation_files: list[Path]) -> pd.DataFrame:
    """Load annotation data and attach donor identifiers."""
    frames: list[pd.DataFrame] = []

    for path in annotation_files:
        frame = pd.read_csv(path)
        require_columns(frame, ANNOTATION_COLUMNS, str(path))

        frame = frame.loc[:, ANNOTATION_COLUMNS].copy()
        frame["donor_id"] = path.parent.name
        frames.append(frame)

    if not frames:
        raise RuntimeError("No SampleAnnot.csv files were found.")

    annotations = pd.concat(frames, ignore_index=True)

    numeric_columns = ["structure_id", "mni_x", "mni_y", "mni_z"]

    for column in numeric_columns:
        annotations[column] = pd.to_numeric(
            annotations[column],
            errors="coerce",
        )

    annotations = annotations.dropna(
        subset=["structure_id", "mni_x", "mni_y", "mni_z"]
    ).copy()

    annotations["structure_id"] = annotations["structure_id"].astype("Int64")

    return annotations


def calculate_structure_coordinates(
    annotations: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate representative MNI coordinates while weighting donors equally.

    First, samples are averaged within each donor and structure. Those
    donor-level coordinates are then averaged across donors.
    """
    donor_structure_coordinates = (
        annotations.groupby(
            ["donor_id", "structure_id"],
            as_index=False,
            dropna=False,
        )
        .agg(
            mni_x=("mni_x", "mean"),
            mni_y=("mni_y", "mean"),
            mni_z=("mni_z", "mean"),
        )
    )

    structure_coordinates = (
        donor_structure_coordinates.groupby(
            "structure_id",
            as_index=False,
            dropna=False,
        )
        .agg(
            coordinate_donor_count=("donor_id", "nunique"),
            mni_x=("mni_x", "mean"),
            mni_y=("mni_y", "mean"),
            mni_z=("mni_z", "mean"),
        )
    )

    structure_coordinates["structure_id"] = (
        structure_coordinates["structure_id"].astype("Int64")
    )

    return structure_coordinates


def load_enrichment_results(path: Path) -> pd.DataFrame:
    """Load and validate the regional enrichment table."""
    if not path.is_file():
        raise FileNotFoundError(f"Enrichment file not found: {path}")

    enrichment = pd.read_csv(path)

    require_columns(
        enrichment,
        REQUIRED_ENRICHMENT_COLUMNS,
        str(path),
    )

    numeric_columns = [
        "structure_id",
        "parkinsons_expression_score",
        "enrichment_z_score",
    ]

    optional_numeric_columns = [
        "number_of_donors",
        "empirical_p_value",
        "fdr_q_value",
    ]

    for column in numeric_columns + optional_numeric_columns:
        if column in enrichment.columns:
            enrichment[column] = pd.to_numeric(
                enrichment[column],
                errors="coerce",
            )

    enrichment = enrichment.dropna(
        subset=[
            "structure_id",
            "parkinsons_expression_score",
            "enrichment_z_score",
        ]
    ).copy()

    enrichment["structure_id"] = enrichment["structure_id"].astype("Int64")

    return enrichment


def merge_results_with_coordinates(
    enrichment: pd.DataFrame,
    coordinates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match enrichment results to coordinates using structure ID."""
    merged = enrichment.merge(
        coordinates,
        on="structure_id",
        how="left",
        validate="many_to_one",
    )

    coordinate_columns = ["mni_x", "mni_y", "mni_z"]

    unmatched = merged.loc[
        merged[coordinate_columns].isna().any(axis=1)
    ].copy()

    matched = merged.dropna(subset=coordinate_columns).copy()

    for column in coordinate_columns:
        matched[column] = pd.to_numeric(matched[column], errors="coerce")

    matched = matched.dropna(subset=coordinate_columns).copy()

    return matched, unmatched


def symmetric_limit(values: pd.Series) -> float:
    """Return a safe symmetric color limit."""
    numeric = pd.to_numeric(values, errors="coerce")
    maximum = numeric.abs().max()

    if pd.isna(maximum) or maximum == 0:
        return 1.0

    return float(maximum)


def plot_marker_map(
    frame: pd.DataFrame,
    value_column: str,
    output_path: Path,
    title: str,
    node_size: int,
) -> None:
    """Create and save a Nilearn glass-brain marker map."""
    if frame.empty:
        raise ValueError(f"No regions available for plot: {title}")

    values = pd.to_numeric(frame[value_column], errors="coerce")
    coordinates = frame.loc[:, ["mni_x", "mni_y", "mni_z"]].to_numpy(
        dtype=float
    )

    valid = values.notna().to_numpy() & np.isfinite(coordinates).all(axis=1)

    values = values.loc[valid].to_numpy(dtype=float)
    coordinates = coordinates[valid]

    if len(values) == 0:
        raise ValueError(f"No valid values available for plot: {title}")

    maximum = float(np.max(np.abs(values)))

    if maximum == 0 or not np.isfinite(maximum):
        maximum = 1.0

    plotting.plot_markers(
        node_values=values,
        node_coords=coordinates,
        node_size=node_size,
        node_cmap="coolwarm",
        node_vmin=-maximum,
        node_vmax=maximum,
        alpha=0.8,
        output_file=str(output_path),
        display_mode="lzry",
        title=title,
        annotate=True,
        black_bg=False,
        colorbar=True,
    )


def select_target_regions(frame: pd.DataFrame) -> pd.DataFrame:
    """Select predefined Parkinson's-related anatomical structures."""
    names = frame["structure_name"].fillna("").astype(str).str.lower()

    mask = pd.Series(False, index=frame.index)

    for term in TARGET_TERMS:
        mask |= names.str.contains(term, regex=False)

    return frame.loc[mask].copy()


def ordered_output_columns(frame: pd.DataFrame) -> list[str]:
    """Return requested output columns that exist in the merged table."""
    requested = [
        "structure_id",
        "structure_acronym",
        "structure_name",
        "number_of_donors",
        "coordinate_donor_count",
        "mni_x",
        "mni_y",
        "mni_z",
        "parkinsons_expression_score",
        "enrichment_z_score",
        "empirical_p_value",
        "fdr_q_value",
        "significant_fdr_0_05",
    ]

    return [column for column in requested if column in frame.columns]


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    annotation_files = find_annotation_files(DATA_DIR)
    print(f"Donor annotation files found: {len(annotation_files)}")

    annotations = load_annotations(annotation_files)
    print(
        "Annotated tissue samples with valid MNI coordinates: "
        f"{len(annotations)}"
    )

    coordinates = calculate_structure_coordinates(annotations)
    print(
        "Structures with calculated coordinates: "
        f"{len(coordinates)}"
    )

    enrichment = load_enrichment_results(ENRICHMENT_PATH)
    print(f"Enrichment regions loaded: {len(enrichment)}")

    matched, unmatched = merge_results_with_coordinates(
        enrichment,
        coordinates,
    )

    print(f"Enrichment regions matched to coordinates: {len(matched)}")
    print(f"Unmatched enrichment regions: {len(unmatched)}")

    if not unmatched.empty:
        print("\nUnmatched region names:")

        for name in unmatched["structure_name"].fillna("unknown").astype(str):
            print(f"  {name}")

    output_columns = ordered_output_columns(matched)

    matched.loc[:, output_columns].to_csv(
        COORDINATE_OUTPUT_PATH,
        index=False,
    )

    plot_marker_map(
        frame=matched,
        value_column="enrichment_z_score",
        output_path=ENRICHMENT_FIGURE_PATH,
        title="Parkinson's gene-set enrichment across brain regions",
        node_size=35,
    )

    plot_marker_map(
        frame=matched,
        value_column="parkinsons_expression_score",
        output_path=EXPRESSION_FIGURE_PATH,
        title="Mean expression score of Parkinson's GWAS genes",
        node_size=35,
    )

    target_regions = select_target_regions(matched)
    print(f"Target regions plotted: {len(target_regions)}")

    plot_marker_map(
        frame=target_regions,
        value_column="enrichment_z_score",
        output_path=TARGET_FIGURE_PATH,
        title="Parkinson's target-region enrichment",
        node_size=100,
    )

    print("\nOutputs:")
    print(f"Wrote {COORDINATE_OUTPUT_PATH}")
    print(f"Wrote {ENRICHMENT_FIGURE_PATH}")
    print(f"Wrote {EXPRESSION_FIGURE_PATH}")
    print(f"Wrote {TARGET_FIGURE_PATH}")


if __name__ == "__main__":
    main()