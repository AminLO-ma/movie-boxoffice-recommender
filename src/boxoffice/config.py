"""Paramètres centralisés : chemins, variante MovieLens, accès TMDB et IMDb."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"

MOVIELENS_VARIANT = "ml-32m"
MOVIELENS_DIR = RAW / "movielens" / MOVIELENS_VARIANT
MIN_RATINGS = 10

TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_LANGUAGE = "en-US"
TMDB_DIR = RAW / "tmdb"
TMDB_CACHE = TMDB_DIR / "movies"
TMDB_EXPLORE_CACHE = TMDB_DIR / "explore"
TMDB_REFERENCE = TMDB_DIR / "reference"
TMDB_APPEND = ("credits", "release_dates", "keywords")
TMDB_APPEND_ALL = (
    "credits", "release_dates", "keywords", "external_ids", "alternative_titles",
    "translations", "images", "videos", "watch/providers", "recommendations",
    "similar", "reviews", "lists", "changes",
)

IMDB_URL = "https://datasets.imdbws.com"
IMDB_RAW = RAW / "imdb"
IMDB_TABLE = INTERIM / "imdb_titles.parquet"

FILM_REFERENCE = PROCESSED / "films_reference.parquet"
FILM_FUNNEL = PROCESSED / "films_funnel.csv"


def tmdb_api_key() -> str:
    key = os.getenv("TMDB_API_KEY", "").strip()
    if not key:
        raise RuntimeError("TMDB_API_KEY absente : copier .env.example en .env et renseigner la clé.")
    return key
