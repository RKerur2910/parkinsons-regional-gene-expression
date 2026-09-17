#!/usr/bin/env python3
"""Match Parkinson's GWAS gene symbols to Allen microarray probes."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "allen_human_brain_atlas"
GWAS_GENES_PATH = PROJECT_ROOT / "results" / "parkinsons_gwas_genes.csv"
MATCHED_PATH = PROJECT_ROOT / "results" / "parkinsons_genes_matched_to_probes.csv"
SUMMARY_PATH = PROJECT_ROOT / "results" / "parkinsons_gene_probe_summary.csv"
UNMATCHED_PATH = PROJECT_ROOT / "results" / "parkinsons_genes_not_in_allen.csv"

DONOR_PREFIX = "normalized_microarray_donor"
PROBE_COLUMNS = (
    "probe_id",
    "probe_name",
    "gene_id",
    "gene_symbol",
    "gene_name",
    "entrez_id",
    "chromosome",
)


def find_donor_folders(data_dir: Path) -> list[Path]:
    folders = [
        path
        for path in data_dir.iterdir()
        if path.is_dir() and path.name.startswith(DONOR_PREFIX)
    ]
    return sorted(folders, key=lambda path: path.name)


def normalize_symbol(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.upper()
    missing = text.isna() | text.eq("") | text.eq("NAN")
    return text.mask(missing)


def probes_are_consistent(probe_tables: list[pd.DataFrame]) -> bool:
    first = probe_tables[0]
    first_ids = first["probe_id"].reset_index(drop=True)
    first_symbols = first["gene_symbol"].reset_index(drop=True)
    for table in probe_tables[1:]:
        if len(table) != len(first):
            return False
        if not table["probe_id"].reset_index(drop=True).equals(first_ids):
            return False
        if not table["gene_symbol"].reset_index(drop=True).equals(first_symbols):
            return False
    return True


def main() -> None:
    donor_folders = find_donor_folders(DATA_DIR)
    probe_paths = [folder / "Probes.csv" for folder in donor_folders]
    probe_paths = [path for path in probe_paths if path.is_file()]
    print(f"Number of donor Probes.csv files found: {len(probe_paths)}")

    probe_tables = [pd.read_csv(path) for path in probe_paths]
    consistent = bool(probe_tables) and probes_are_consistent(probe_tables)
    print(f"All probe files consistent: {'PASS' if consistent else 'FAIL'}")
    if not consistent:
        raise SystemExit("Probe metadata is not consistent across donors.")

    probes = probe_tables[0].copy()
    missing_cols = [column for column in PROBE_COLUMNS if column not in probes.columns]
    if missing_cols:
        raise KeyError(f"Probes.csv missing columns: {missing_cols}")

    gwas_genes = pd.read_csv(GWAS_GENES_PATH)
    gwas_genes["gene_symbol"] = normalize_symbol(gwas_genes["gene_symbol"])
    gwas_genes = gwas_genes.dropna(subset=["gene_symbol"]).drop_duplicates(
        subset=["gene_symbol"]
    )
    print(f"Total Parkinson's GWAS genes: {len(gwas_genes)}")

    probes["gene_symbol_norm"] = normalize_symbol(probes["gene_symbol"])
    allen_genes = probes.dropna(subset=["gene_symbol_norm"])

    matched = gwas_genes.merge(
        allen_genes,
        left_on="gene_symbol",
        right_on="gene_symbol_norm",
        how="inner",
    )
    matched_probes = matched.loc[
        :,
        [
            "gene_symbol_x",
            "probe_id",
            "probe_name",
            "gene_id",
            "gene_name",
            "entrez_id",
            "chromosome",
        ],
    ].rename(columns={"gene_symbol_x": "gene_symbol"})
    matched_probes = matched_probes.drop_duplicates()
    matched_probes = matched_probes.sort_values(
        ["gene_symbol", "probe_id"],
        kind="mergesort",
    )

    matched_symbols = pd.Index(matched_probes["gene_symbol"].unique())
    all_gwas_symbols = pd.Index(gwas_genes["gene_symbol"])
    unmatched_symbols = all_gwas_symbols.difference(matched_symbols).sort_values()
    unmatched = pd.DataFrame({"gene_symbol": unmatched_symbols})

    summary = (
        matched_probes.groupby("gene_symbol", sort=True)
        .agg(
            number_of_probes=("probe_id", "nunique"),
            probe_ids=("probe_id", lambda ids: ";".join(ids.astype(str))),
        )
        .reset_index()
    )

    n_matched_genes = len(summary)
    n_gwas = len(gwas_genes)
    n_unmatched = len(unmatched)
    n_probes = len(matched_probes)
    n_multi = int((summary["number_of_probes"] > 1).sum())
    pct_matched = (100.0 * n_matched_genes / n_gwas) if n_gwas else 0.0

    print(f"Number of genes matched to Allen: {n_matched_genes}")
    print(f"Percentage of GWAS genes matched: {pct_matched:.2f}%")
    print(f"Number of unmatched genes: {n_unmatched}")
    print(f"Total number of matching probes: {n_probes}")
    print(f"Number of matched genes with multiple probes: {n_multi}")

    print("First 20 matched genes:")
    for symbol in summary["gene_symbol"].head(20):
        print(f"  {symbol}")
    print("First 20 unmatched genes:")
    for symbol in unmatched["gene_symbol"].head(20):
        print(f"  {symbol}")

    MATCHED_PATH.parent.mkdir(parents=True, exist_ok=True)
    matched_probes.to_csv(MATCHED_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    unmatched.to_csv(UNMATCHED_PATH, index=False)
    print(f"Wrote {MATCHED_PATH}")
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {UNMATCHED_PATH}")


if __name__ == "__main__":
    main()
