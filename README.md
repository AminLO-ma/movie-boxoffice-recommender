# Recommandation de films & Succès au box-office

## Objectif
Analyse prédictive et système de recommandation pour guider les investissements de production de films.

- **Données** : MovieLens, API TMDB, IMDb
- **ML** : Filtrage collaboratif & modèles de succès (régression/classification)

Tout le groupe travaille sur **la même liste de films** : les films de MovieLens `ml-32m` (notes jusqu'en 10/2023)
notés au moins 10 fois et présents dans TMDB et IMDb.

## 1. Installation

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\Activate.ps1
```

| Dossier | Rôle |
|---|---|
| `src/boxoffice/ingestion/` | collecte : MovieLens, TMDB (cache JSON), IMDb |
| `src/boxoffice/cleaning/` | nettoyage, fusion, référentiel commun |
| `notebooks/` | exploration et restitution |

## 2. Récupérer les données

### Option A — archive du groupe (recommandé)
1. Décompresser `boxoffice_data_AAAA-MM-JJ.zip` dans ce dossier : il crée `data/`.
2. Télécharger MovieLens :
   ```powershell
   python -m boxoffice.ingestion.movielens
   ```

Aucune clé API nécessaire.

### Option B — tout régénérer (~1 h 30, clé TMDB requise)

```powershell
Copy-Item .env.example .env
python -m boxoffice.ingestion.movielens
python -m boxoffice.ingestion.tmdb
python -m boxoffice.ingestion.imdb
python -m boxoffice.cleaning.reference
```

## 3. Données disponibles

| Fichier | Contenu | Clé |
|---|---|---|
| `data/processed/films_reference.parquet` (+ `.csv`) | 1 ligne par film retenu : identifiants, nombre de notes, titres, années, statut TMDB, type IMDb, `in_common_list`, signalements `flag_*` | `movieId` |
| `data/processed/films_funnel.csv` | effectifs à chaque étape de la sélection | — |
| `data/raw/movielens/ml-32m/` | notes, films, tags, liens | `movieId` |
| `data/raw/tmdb/movies/{tmdbId}.json` | réponse TMDB brute : détails, `credits`, `release_dates`, `keywords` | `tmdbId` |
| `data/raw/tmdb/movies/_not_found.jsonl` | films inconnus de TMDB (404) | `tmdb_id` |
| `data/raw/tmdb/reference/*.json` | genres, certifications, pays, langues, métiers | — |
| `data/interim/imdb_titles.parquet` | type de titre, durée, genres, note et nombre de votes IMDb | `tconst` = `imdb_id` |

**Règle commune** : filtrer sur `in_common_list == True`. Pour le modèle de succès, garder en plus `imdb_type == "movie"`.

Les données ne vont jamais sur GitHub (volume, licences IMDb et TMDB) ; la clé API reste dans `.env`.

## 4. Travailler à plusieurs (Git)

- `main` : version stable ; `develop` : intégration. Pas de push direct sur l'une ou l'autre.
- Une branche par fonctionnalité, créée depuis `develop`, supprimée après fusion :

```powershell
git switch develop; git pull
git switch -c feat/recommandation-baseline
git fetch origin; git rebase origin/develop
git push -u origin feat/recommandation-baseline
```

- Pull request vers `develop`, relecture par un autre membre, merge, suppression de la branche.
- Noms de branches : `feat/…`, `fix/…`, `docs/…`, `explo/…`. Commits : `type(module): message`.
- Un notebook par auteur (préfixe par initiales) pour éviter les conflits sur les `.ipynb`.
