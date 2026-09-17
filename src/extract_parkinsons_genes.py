#!/usr/bin/env python3
"""Extract genome-wide-significant Parkinson's GWAS gene symbols."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "gwas"
    / "gwas-association-downloaded_2026-09-11-MONDO_0005180.tsv"
)
ASSOCIATION_MAP_PATH = PROJECT_ROOT / "results" / "parkinsons_gwas_association_gene_map.csv"
UNIQUE_GENES_PATH = PROJECT_ROOT / "results" / "parkinsons_gwas_genes.csv"

REQUIRED_COLUMNS = (
    "P-VALUE",
    "MAPPED_GENE",
    "SNPS",
    "STUDY ACCESSION",
    "PUBMEDID",
)

P_VALUE_THRESHOLD = 5e-8
GENE_SPLIT_PATTERN = re.compile(r"[,;]|\s+-\s+")
PLACEHOLDERS = {
    "NR",
    "N/A",
    "NA",
    "NONE",
    "NAN",
    "NULL",
    "INTERGENIC",
    "-",
}


def confirm_required_columns(table: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in table.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def split_mapped_genes(mapped_gene: str) -> list[str]:
    parts = GENE_SPLIT_PATTERN.split(mapped_gene)
    cleaned: list[str] = []
    for part in parts:
        symbol = part.strip().upper()
        if not symbol or symbol in PLACEHOLDERS:
            continue
        cleaned.append(symbol)
    return cleaned


def main() -> None:
    print(f"Input filename: {INPUT_PATH.name}")
    table = pd.read_csv(INPUT_PATH, sep="\t", low_memory=False)
    print(f"Total input associations: {len(table)}")
    confirm_required_columns(table)

    table["P-VALUE"] = pd.to_numeric(table["P-VALUE"], errors="coerce")
    significant = table.loc[table["P-VALUE"] <= P_VALUE_THRESHOLD].copy()
    print(f"Associations passing P-VALUE <= {P_VALUE_THRESHOLD:g}: {len(significant)}")

    mapped = significant["MAPPED_GENE"]
    mapped_text = mapped.astype("string").str.strip()
    has_mapped_gene = mapped.notna() & mapped_text.ne("") & mapped_text.str.upper().ne("NAN")
    with_gene = significant.loc[has_mapped_gene].copy()
    print(f"Significant associations with a non-empty MAPPED_GENE: {len(with_gene)}")

    with_gene["cleaned_gene_symbol"] = with_gene["MAPPED_GENE"].map(split_mapped_genes)
    exploded = with_gene.explode("cleaned_gene_symbol", ignore_index=True)
    exploded = exploded.dropna(subset=["cleaned_gene_symbol"])
    exploded = exploded.loc[exploded["cleaned_gene_symbol"].astype(str).str.strip().ne("")]

    association_map = exploded.loc[
        :,
        [
            "SNPS",
            "P-VALUE",
            "MAPPED_GENE",
            "cleaned_gene_symbol",
            "STUDY ACCESSION",
            "PUBMEDID",
        ],
    ].drop_duplicates()
    association_map = association_map.sort_values(
        ["cleaned_gene_symbol", "P-VALUE"],
        ascending=[True, True],
        kind="mergesort",
    )

    unique_genes = (
        pd.DataFrame({"gene_symbol": association_map["cleaned_gene_symbol"].unique()})
        .sort_values("gene_symbol", kind="mergesort")
        .reset_index(drop=True)
    )
    print(f"Number of unique cleaned genes: {len(unique_genes)}")
    print("First 30 cleaned gene symbols:")
    for symbol in unique_genes["gene_symbol"].head(30):
        print(f"  {symbol}")

    ASSOCIATION_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    association_map.to_csv(ASSOCIATION_MAP_PATH, index=False)
    unique_genes.to_csv(UNIQUE_GENES_PATH, index=False)
    print(f"Wrote {ASSOCIATION_MAP_PATH}")
    print(f"Wrote {UNIQUE_GENES_PATH}")


if __name__ == "__main__":
    main()
