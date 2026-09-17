#!/usr/bin/env python3
"""Inspect the GWAS Catalog association TSV without modifying it."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GWAS_DIR = PROJECT_ROOT / "data" / "gwas"

PREVIEW_KEYWORDS = (
    "disease",
    "trait",
    "p-value",
    "pvalue",
    "p_value",
    "snp",
    "variant",
    "mapped_gene",
    "mapped gene",
    "reported gene",
    "reported_gene",
    "accession",
    "pubmed",
)


def find_gwas_tsv(gwas_dir: Path) -> Path:
    files = sorted(gwas_dir.glob("*.tsv"))
    if not files:
        raise FileNotFoundError(f"No TSV file found in {gwas_dir}")
    if len(files) > 1:
        names = ", ".join(path.name for path in files)
        raise RuntimeError(f"Expected one TSV file in {gwas_dir}, found: {names}")
    return files[0]


def normalized_name(column: str) -> str:
    return column.lower().replace("_", " ")


def is_preview_column(column: str) -> bool:
    name = normalized_name(column)
    if "platform" in name:
        return False
    return any(keyword in name for keyword in PREVIEW_KEYWORDS)


def is_gene_column(column: str) -> bool:
    return "gene" in normalized_name(column)


def non_empty_values(series: pd.Series) -> pd.Series:
    values = series.dropna()
    text = values.astype(str).str.strip()
    return values[text.ne("") & text.str.lower().ne("nan")]


def main() -> None:
    tsv_path = find_gwas_tsv(GWAS_DIR)
    print(f"GWAS file: {tsv_path.name}")

    table = pd.read_csv(tsv_path, sep="\t")
    n_rows, n_cols = table.shape
    print(f"Rows: {n_rows}")
    print(f"Columns: {n_cols}")

    print("\nColumns:")
    for index, column in enumerate(table.columns, start=1):
        print(f"  {index}. {column}")

    preview_columns = [column for column in table.columns if is_preview_column(column)]
    print("\nFirst three rows (disease/trait, p-value, variant/SNP, gene, study, PubMed):")
    print(table.loc[:2, preview_columns].to_string(index=False))

    gene_columns = [column for column in table.columns if is_gene_column(column)]
    print("\nMissing values in gene-related columns:")
    for column in gene_columns:
        n_missing = n_rows - len(non_empty_values(table[column]))
        print(f"  {column}: {n_missing}")

    print("\nUnique non-empty values in gene-related columns:")
    for column in gene_columns:
        n_unique = non_empty_values(table[column]).nunique()
        print(f"  {column}: {n_unique}")


if __name__ == "__main__":
    main()
