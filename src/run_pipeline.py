#!/usr/bin/env python3

"""Run the complete Parkinson's regional gene-expression pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

PIPELINE_STAGES = [
    (
        "Verify Allen donor data",
        "verify_data.py",
    ),
    (
        "Inventory brain regions",
        "inspect_regions.py",
    ),
    (
        "Inspect Parkinson's GWAS data",
        "inspect_gwas.py",
    ),
    (
        "Extract significant Parkinson's GWAS genes",
        "extract_parkinsons_genes.py",
    ),
    (
        "Match GWAS genes to Allen probes",
        "match_gwas_to_allen_probes.py",
    ),
    (
        "Select reliable Parkinson's probes",
        "select_reliable_probes.py",
    ),
    (
        "Aggregate Parkinson's expression by region",
        "aggregate_expression_by_region.py",
    ),
    (
        "Select reliable background probes",
        "select_background_probes.py",
    ),
    (
        "Build background regional expression matrix",
        "build_background_expression_matrix.py",
    ),
    (
        "Run Parkinson's enrichment analysis",
        "run_parkinsons_enrichment.py",
    ),
    (
        "Create enrichment figures",
        "visualize_parkinsons_results.py",
    ),
    (
        "Run sensitivity analyses",
        "run_sensitivity_analysis.py",
    ),
    (
        "Create anatomical brain maps",
        "plot_brain_maps.py",
    ),
]


def format_duration(seconds: float) -> str:
    """Format elapsed time in a readable form."""
    seconds = int(round(seconds))
    minutes, remaining_seconds = divmod(seconds, 60)
    hours, remaining_minutes = divmod(minutes, 60)

    if hours:
        return (
            f"{hours}h {remaining_minutes}m "
            f"{remaining_seconds}s"
        )

    if minutes:
        return f"{minutes}m {remaining_seconds}s"

    return f"{remaining_seconds}s"


def validate_scripts() -> None:
    """Stop before execution if any pipeline script is missing."""
    missing = [
        str(SRC_DIR / script_name)
        for _, script_name in PIPELINE_STAGES
        if not (SRC_DIR / script_name).is_file()
    ]

    if missing:
        print("Cannot run pipeline because scripts are missing:")
        for path in missing:
            print(f"  - {path}")

        raise SystemExit(1)


def stage_names() -> list[str]:
    """Return script filenames in pipeline order."""
    return [script for _, script in PIPELINE_STAGES]


def select_stages(
    start_from: str | None,
    stop_after: str | None,
) -> list[tuple[str, str]]:
    """Select a continuous range of pipeline stages."""
    scripts = stage_names()
    start_index = 0
    stop_index = len(PIPELINE_STAGES) - 1

    if start_from is not None:
        start_index = scripts.index(start_from)

    if stop_after is not None:
        stop_index = scripts.index(stop_after)

    if start_index > stop_index:
        raise ValueError(
            "--start-from occurs after --stop-after in the pipeline."
        )

    return PIPELINE_STAGES[start_index : stop_index + 1]


def print_pipeline(stages: list[tuple[str, str]]) -> None:
    """Print the stages that will be executed."""
    print("Parkinson's Regional Gene-Expression Pipeline")
    print("=" * 55)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python interpreter: {sys.executable}")
    print(f"Stages to run: {len(stages)}")
    print()

    for number, (description, script_name) in enumerate(stages, start=1):
        print(f"{number:2}. {description}")
        print(f"    src/{script_name}")


def run_stage(
    stage_number: int,
    total_stages: int,
    description: str,
    script_name: str,
) -> float:
    """Run one pipeline stage and stop immediately if it fails."""
    script_path = SRC_DIR / script_name

    print()
    print("=" * 72)
    print(f"STAGE {stage_number}/{total_stages}: {description}")
    print(f"Running: {sys.executable} {script_path}")
    print("=" * 72)
    print(flush=True)

    start_time = time.perf_counter()

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        check=False,
    )

    elapsed = time.perf_counter() - start_time

    if result.returncode != 0:
        print()
        print("=" * 72)
        print(f"PIPELINE FAILED AT STAGE {stage_number}")
        print(f"Description: {description}")
        print(f"Script: src/{script_name}")
        print(f"Exit code: {result.returncode}")
        print(f"Elapsed time: {format_duration(elapsed)}")
        print("=" * 72)

        raise SystemExit(result.returncode)

    print()
    print(
        f"Completed stage {stage_number}/{total_stages} "
        f"in {format_duration(elapsed)}"
    )

    return elapsed


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options."""
    script_choices = stage_names()

    parser = argparse.ArgumentParser(
        description=(
            "Run the complete Parkinson's regional "
            "gene-expression analysis pipeline."
        )
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List pipeline stages without running them.",
    )

    parser.add_argument(
        "--start-from",
        choices=script_choices,
        help=(
            "Start from this script instead of rerunning earlier stages."
        ),
    )

    parser.add_argument(
        "--stop-after",
        choices=script_choices,
        help="Stop after this script finishes successfully.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    validate_scripts()

    try:
        selected_stages = select_stages(
            start_from=args.start_from,
            stop_after=args.stop_after,
        )
    except ValueError as error:
        print(f"Error: {error}")
        raise SystemExit(2) from error

    print_pipeline(selected_stages)

    if args.list:
        return

    pipeline_start = time.perf_counter()
    stage_times: list[tuple[str, float]] = []

    for number, (description, script_name) in enumerate(
        selected_stages,
        start=1,
    ):
        elapsed = run_stage(
            stage_number=number,
            total_stages=len(selected_stages),
            description=description,
            script_name=script_name,
        )

        stage_times.append((description, elapsed))

    total_elapsed = time.perf_counter() - pipeline_start

    print()
    print("=" * 72)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 72)

    for description, elapsed in stage_times:
        print(f"  {description}: {format_duration(elapsed)}")

    print()
    print(f"Total runtime: {format_duration(total_elapsed)}")
    print(f"Results directory: {PROJECT_ROOT / 'results'}")
    print(f"Figures directory: {PROJECT_ROOT / 'figures'}")


if __name__ == "__main__":
    main()