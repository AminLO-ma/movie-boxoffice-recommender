"""
Ingestion TMDB : recupere details + casting pour chaque film present dans links.csv.
Cache chaque reponse en JSON pour ne jamais re-appeler un film deja traite.
Usage:
    python -m src.data.fetch_tmdb --limit 500      # test rapide sur 500 films
    python -m src.data.fetch_tmdb                  # run complet
"""

import argparse
import json
import time

import pandas as pd
import requests
from tqdm import tqdm

from src.data.config import LINKS_CSV, TMDB_API_KEY, TMDB_BASE_URL, TMDB_CACHE_DIR

SESSION = requests.Session()
SESSION.headers.update({"accept": "application/json"})

REQUEST_PARAMS = {"api_key": TMDB_API_KEY, "language": "en-US"}


def load_tmdb_ids() -> pd.DataFrame:
    links = pd.read_csv(LINKS_CSV, dtype={"movieId": "int64", "tmdbId": "string"})
    links = links.dropna(subset=["tmdbId"])
    links["tmdbId"] = links["tmdbId"].astype(int)
    return links


def fetch_movie(tmdb_id: int) -> dict | None:
    cache_file = TMDB_CACHE_DIR / f"{tmdb_id}.json"
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    detail_url = f"{TMDB_BASE_URL}/movie/{tmdb_id}"
    credits_url = f"{TMDB_BASE_URL}/movie/{tmdb_id}/credits"

    try:
        detail_resp = SESSION.get(detail_url, params=REQUEST_PARAMS, timeout=10)
        if detail_resp.status_code == 404:
            return None
        detail_resp.raise_for_status()
        detail = detail_resp.json()

        credits_resp = SESSION.get(credits_url, params=REQUEST_PARAMS, timeout=10)
        credits_resp.raise_for_status()
        credits = credits_resp.json()

        record = {
            "tmdb_id": tmdb_id,
            "title": detail.get("title"),
            "budget": detail.get("budget"),
            "revenue": detail.get("revenue"),
            "release_date": detail.get("release_date"),
            "runtime": detail.get("runtime"),
            "original_language": detail.get("original_language"),
            "popularity": detail.get("popularity"),
            "vote_average": detail.get("vote_average"),
            "vote_count": detail.get("vote_count"),
            "genres": [g["name"] for g in detail.get("genres", [])],
            "production_companies": [c["name"] for c in detail.get("production_companies", [])],
            "cast": [c["name"] for c in credits.get("cast", [])[:10]],
            "director": next(
                (c["name"] for c in credits.get("crew", []) if c.get("job") == "Director"),
                None,
            ),
        }

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)

        return record

    except requests.exceptions.RequestException as e:
        print(f"[WARN] tmdb_id={tmdb_id} echec: {e}")
        return None


def main(limit: int | None = None):
    links = load_tmdb_ids()
    if limit:
        links = links.head(limit)

    tmdb_ids = links["tmdbId"].tolist()
    results = []

    for tmdb_id in tqdm(tmdb_ids, desc="Fetching TMDB"):
        record = fetch_movie(tmdb_id)
        if record:
            results.append(record)
        time.sleep(0.02)  # marge sous la limite ~40 req/s de TMDB

    print(f"Termine: {len(results)}/{len(tmdb_ids)} films recuperes avec succes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Nombre de films a tester")
    args = parser.parse_args()
    main(limit=args.limit)