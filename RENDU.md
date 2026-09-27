# Recommandation de films & succès au box-office — guide de lecture

Projet Data Visualisation, sujet n°4. Sources : MovieLens `ml-32m`, API TMDB, jeux non commerciaux IMDb, indice des prix FRED.

## Par où commencer

| Ordre | Fichier | Contenu | Audience |
|---|---|---|---|
| 1 | `notebooks/report/01_qualite_donnees.ipynb` | des données brutes au jeu exploitable : sources, notes MovieLens, entonnoir, valeurs manquantes, doublons, aberrations, inflation, biais | technique |
| 2 | `notebooks/report/02_succes_technique.ipynb` | modèle de succès : choix de la cible et du modèle, réglage, diagnostics, seuil de décision, interprétabilité, améliorations | technique |
| 3 | `notebooks/report/03_restitution_metier.ipynb` | recommandations d'investissement, sans vocabulaire technique | direction |
| 3 bis | `streamlit_app.py` | le même propos en interactif : simulateur de projet | direction |

Les notebooks sont livrés **déjà exécutés** : ils se lisent sans rien lancer. Le notebook technique commence par un **résumé exécutif** qui donne les cinq points clés et renvoie aux sections correspondantes.

## Tout relancer

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Les tables nettoyées sont versionnées** (`data/processed`, `data/interim`, 14 Mo) ainsi que le modèle entraîné : les trois notebooks et le tableau de bord s'exécutent directement après l'installation, sans clé API.

```powershell
.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Pour reconstituer les **sources brutes** (2 Go, hors dépôt : volume et licences IMDb et TMDB) :

```powershell
Copy-Item .env.example .env          # puis renseigner une clé TMDB gratuite
python -m boxoffice.ingestion.movielens
python -m boxoffice.ingestion.tmdb       # environ 1 h, relançable sans appel redondant
python -m boxoffice.ingestion.imdb
python -m boxoffice.cleaning.reference
python -m boxoffice.cleaning.movies
```

## Organisation du code

Les notebooks ne contiennent que des appels et des commentaires ; les traitements sont dans `src/boxoffice/`.

| Module | Rôle |
|---|---|
| `ingestion/` | MovieLens, cache TMDB (un JSON par film, relançable, gestion des codes 404 et 429), IMDb, indice des prix |
| `cleaning/reference.py` | référentiel commun : jointure des trois sources et signalements |
| `cleaning/movies.py` | tables propres : films, genres, personnes, certifications, mots-clés |
| `features/success.py` | jeu d'entraînement : cible et 34 variables pré-production |
| `models/entrainement.py` | banc d'essai, réglage, évaluation, seuils, diagnostics, simulation |
| `models/succes.py` | enregistrement du modèle et de sa carte d'identité |
| `audit/`, `viz/` | mesures de qualité et figures (Matplotlib, Seaborn, Plotly) |
| `tests/` | cache TMDB vérifié sans réseau : relance sans appel redondant, 404, 429, clé invalide |

## Chiffres clés

- **31 464 films** présents dans les trois sources, **9 268** exploitables pour le modèle de succès.
- Cible : le film rapporte-t-il **au moins 2,5 fois son budget** en dollars constants 2023 ? 38,6 % y parviennent.
- Modèle retenu : gradient boosting, **AUC 0,749** sur un test temporel (films sortis à partir de 2018, jamais vus à l'entraînement).
- Au seuil retenu, **72 % des films recommandés sont rentables**, contre 35 % de taux de base.
- Variables décisives : appartenance à une saga, budget, notoriété des têtes d'affiche et du réalisateur, pays de production.
