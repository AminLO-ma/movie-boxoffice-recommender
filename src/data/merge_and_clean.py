"""
Fusionne MovieLens (movies, links, ratings) avec le cache TMDB (data/raw/tmdb_cache/*.json),
nettoie les champs et exporte deux tables Parquet dans data/processed/ :
    - movies_enriched.parquet : un film par ligne, avec metadata TMDB fusionnee
    - ratings_clean.parquet   : notes utilisateur nettoyees, filtrees sur films valides

Usage:
    python -m src.data.merge_and_clean
"""

import json

import pandas as pd

from src.data.config import (
    DATA_PROCESSED,
    LINKS_CSV,
    MOVIES_CSV,
    RATINGS_CSV,
    TMDB_CACHE_DIR,
)


def load_tmdb_cache() -> pd.DataFrame:
    records = []
    for json_file in TMDB_CACHE_DIR.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            records.append(json.load(f))
    df = pd.DataFrame(records)
    print(f"[TMDB] {len(df)} films charges depuis le cache.")
    return df


def load_movielens() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    movies = pd.read_csv(MOVIES_CSV, dtype={"movieId": "int64"})
    movies = movies.rename(columns={"genres": "genres_movielens", "title": "title_movielens"})
    links = pd.read_csv(LINKS_CSV, dtype={"movieId": "int64", "tmdbId": "string"})
    ratings = pd.read_csv(RATINGS_CSV, dtype={"movieId": "int64", "userId": "int64"})
    return movies, links, ratings


def clean_tmdb(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date"].dt.year

    df["genres"] = df["genres"].apply(lambda g: g if isinstance(g, list) else [])
    df["cast"] = df["cast"].apply(lambda c: c if isinstance(c, list) else [])
    df["n_genres"] = df["genres"].apply(len)
    df["n_cast_listed"] = df["cast"].apply(len)

    df["budget"] = pd.to_numeric(df["budget"], errors="coerce")
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")

    # Budget/revenue a 0 = donnee non renseignee sur TMDB, pas un vrai flop -> on met en NaN
    df.loc[df["budget"] == 0, "budget"] = pd.NA
    df.loc[df["revenue"] == 0, "revenue"] = pd.NA

    has_financials = df["budget"].notna() & df["revenue"].notna()
    df["roi"] = pd.NA
    df.loc[has_financials, "roi"] = (
        (df.loc[has_financials, "revenue"] - df.loc[has_financials, "budget"])
        / df.loc[has_financials, "budget"]
    )

    df = df.drop_duplicates(subset="tmdb_id", keep="first")

    return df


def merge_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    tmdb = load_tmdb_cache()
    tmdb = clean_tmdb(tmdb)

    movies, links, ratings = load_movielens()

    links["tmdbId"] = pd.to_numeric(links["tmdbId"], errors="coerce")
    movies_links = movies.merge(links, on="movieId", how="left")

    movies_enriched = movies_links.merge(
        tmdb, left_on="tmdbId", right_on="tmdb_id", how="inner"
    )

    valid_movie_ids = set(movies_enriched["movieId"])
    ratings_clean = ratings[ratings["movieId"].isin(valid_movie_ids)].copy()

    print(f"[MERGE] {len(movies_enriched)} films enrichis, "
          f"{len(ratings_clean)} notes conservees sur {len(ratings)} initiales.")

    return movies_enriched, ratings_clean


def main():
    movies_enriched, ratings_clean = merge_all()

    movies_out = DATA_PROCESSED / "movies_enriched.parquet"
    ratings_out = DATA_PROCESSED / "ratings_clean.parquet"

    # Colonnes liste -> string pour compatibilite Parquet simple
    movies_to_save = movies_enriched.copy()
    movies_to_save["genres_list"] = movies_to_save["genres"].apply(lambda g: "|".join(g))
    movies_to_save["cast_list"] = movies_to_save["cast"].apply(lambda c: "|".join(c))
    movies_to_save = movies_to_save.drop(columns=["genres", "cast"])

    movies_to_save.to_parquet(movies_out, index=False)
    ratings_clean.to_parquet(ratings_out, index=False)

    print(f"[SAVE] {movies_out}")
    print(f"[SAVE] {ratings_out}")


if __name__ == "__main__":
    main()