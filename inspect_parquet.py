"""
Inspecte rapidement un fichier Parquet : shape, colonnes, dtypes, apercu.
Usage:
    python inspect_parquet.py data/processed/movies_enriched.parquet
    python inspect_parquet.py data/processed/ratings_clean.parquet --rows 20
"""

import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("path", help="Chemin vers le fichier .parquet")
parser.add_argument("--rows", type=int, default=10, help="Nombre de lignes a afficher")
args = parser.parse_args()

df = pd.read_parquet(args.path)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print(f"Shape: {df.shape}")
print(f"\nColonnes et types:\n{df.dtypes}")
print(f"\nApercu ({args.rows} premieres lignes):")
print(df.head(args.rows))
print(f"\nValeurs manquantes par colonne:\n{df.isna().sum()}")