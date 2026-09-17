import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

MOVIELENS_DIR = DATA_RAW / "ml-32m"
LINKS_CSV = MOVIELENS_DIR / "links.csv"
MOVIES_CSV = MOVIELENS_DIR / "movies.csv"
RATINGS_CSV = MOVIELENS_DIR / "ratings.csv"

TMDB_CACHE_DIR = DATA_RAW / "tmdb_cache"
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"

if not TMDB_API_KEY:
    raise EnvironmentError(
        "TMDB_API_KEY manquant. Ajoutez-le dans le fichier .env a la racine du projet."
    )

TMDB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)