#!/usr/bin/env python3
"""Verify Allen Human Brain Atlas donor folders without loading full matrices."""

from __future__ import annotations

import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "allen_human_brain_atlas"
RESULTS_PATH = PROJECT_ROOT / "results" / "donor_verification.csv"

REQUIRED_FILES = (
    "MicroarrayExpression.csv",
    "PACall.csv",
    "Probes.csv",
    "SampleAnnot.csv",
    "Ontology.csv",
)

DONOR_PREFIX = "normalized_microarray_donor"


def find_donor_folders(data_dir: Path) -> list[Path]:
    folders = [
        path
        for path in data_dir.iterdir()
        if path.is_dir() and path.name.startswith(DONOR_PREFIX)
    ]
    return sorted(folders, key=lambda path: path.name)


def has_readme(folder: Path) -> bool:
    return (folder / "Readme.txt").is_file() or (folder / "README.txt").is_file()


def missing_files(folder: Path) -> list[str]:
    missing = [name for name in REQUIRED_FILES if not (folder / name).is_file()]
    if not has_readme(folder):
        missing.append("Readme.txt/README.txt")
    return missing


def count_data_rows(path: Path) -> int:
    """Count CSV data rows, skipping a header row."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def count_first_row_columns(path: Path) -> int:
    """Read only the first CSV row and return its column count."""
    with path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.reader(handle), [])
    return len(row)


def verify_donor(folder: Path) -> dict[str, object]:
    missing = missing_files(folder)
    n_probes = ""
    n_samples = ""
    n_expr_cols = ""
    n_pacall_cols = ""
    expr_ok = False
    pacall_ok = False
    issues: list[str] = []

    if missing:
        issues.append("missing: " + ";".join(missing))

    probes_path = folder / "Probes.csv"
    samples_path = folder / "SampleAnnot.csv"
    expr_path = folder / "MicroarrayExpression.csv"
    pacall_path = folder / "PACall.csv"

    if probes_path.is_file():
        n_probes = count_data_rows(probes_path)
    if samples_path.is_file():
        n_samples = count_data_rows(samples_path)
    if expr_path.is_file():
        n_expr_cols = count_first_row_columns(expr_path)
    if pacall_path.is_file():
        n_pacall_cols = count_first_row_columns(pacall_path)

    expected_cols = None
    if isinstance(n_samples, int):
        expected_cols = n_samples + 1

    if expected_cols is not None and isinstance(n_expr_cols, int):
        expr_ok = n_expr_cols == expected_cols
        if not expr_ok:
            issues.append(
                f"MicroarrayExpression columns {n_expr_cols} != samples+1 ({expected_cols})"
            )
    elif expr_path.is_file():
        issues.append("could not verify MicroarrayExpression column count")

    if expected_cols is not None and isinstance(n_pacall_cols, int):
        pacall_ok = n_pacall_cols == expected_cols
        if not pacall_ok:
            issues.append(
                f"PACall columns {n_pacall_cols} != samples+1 ({expected_cols})"
            )
    elif pacall_path.is_file():
        issues.append("could not verify PACall column count")

    status = "PASS" if not issues else "FAIL"
    return {
        "donor": folder.name,
        "n_probes": n_probes,
        "n_samples": n_samples,
        "n_expr_cols": n_expr_cols,
        "n_pacall_cols": n_pacall_cols,
        "expected_cols": expected_cols if expected_cols is not None else "",
        "expr_cols_ok": expr_ok,
        "pacall_cols_ok": pacall_ok,
        "missing_files": ";".join(missing),
        "status": status,
        "issues": issues,
    }


def print_donor_report(result: dict[str, object]) -> None:
    print(f"\n{result['donor']}")
    print(f"  missing files: {result['missing_files'] or 'none'}")
    print(f"  Probes.csv data rows: {result['n_probes']}")
    print(f"  SampleAnnot.csv data rows (tissue samples): {result['n_samples']}")
    print(f"  MicroarrayExpression.csv columns (first row): {result['n_expr_cols']}")
    print(f"  PACall.csv columns (first row): {result['n_pacall_cols']}")
    print(f"  expected columns (samples + 1 probe-ID): {result['expected_cols']}")
    print(f"  expression columns match: {result['expr_cols_ok']}")
    print(f"  PACall columns match: {result['pacall_cols_ok']}")
    print(f"  {result['status']}")
    for issue in result["issues"]:
        print(f"    - {issue}")


def write_summary(results: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "donor",
        "n_probes",
        "n_samples",
        "n_expr_cols",
        "n_pacall_cols",
        "expected_cols",
        "expr_cols_ok",
        "pacall_cols_ok",
        "missing_files",
        "status",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({name: result[name] for name in fieldnames})


def main() -> None:
    donor_folders = find_donor_folders(DATA_DIR)
    print(f"Donor folders found: {len(donor_folders)}")
    if not donor_folders:
        print(f"No folders starting with {DONOR_PREFIX!r} under {DATA_DIR}")

    results = [verify_donor(folder) for folder in donor_folders]
    for result in results:
        print_donor_report(result)

    write_summary(results, RESULTS_PATH)
    n_pass = sum(1 for result in results if result["status"] == "PASS")
    n_fail = len(results) - n_pass
    print(f"\nSummary: {n_pass} PASS, {n_fail} FAIL")
    print(f"Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
