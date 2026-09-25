"""Segmentation des spectateurs MovieLens par profil de genres : profils, choix du k, clusters."""
import argparse
import logging

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.preprocessing import StandardScaler

from boxoffice import config

log = logging.getLogger(__name__)

SEUIL_AIME = 4.0
MIN_FILMS_AIMES = 20
GENRES_EXCLUS = ("TV Movie",)
K_SEGMENTS = 5
KS_TESTES = range(2, 13)
ALEA = 42
ECHANTILLON_SILHOUETTE = 20_000
TIRAGE_STABILITE = 40_000
N_TIRAGES = 5


def _matrice_genres() -> pd.DataFrame:
    genres = pd.read_parquet(config.GENRES_TABLE)
    table = pd.crosstab(genres["movieId"], genres["genre"]).astype("float32")
    table = table.drop(columns=list(GENRES_EXCLUS), errors="ignore")
    table = table[table.sum(axis=1) > 0]
    return table.div(table.sum(axis=1), axis=0)


def _notes_aimees(films_connus: pd.Index) -> pd.DataFrame:
    notes = pd.read_csv(config.MOVIELENS_RATINGS, usecols=["userId", "movieId", "rating"],
                        dtype={"userId": "int32", "movieId": "int32", "rating": "float32"})
    log.info("%s notes brutes", f"{len(notes):,}")
    notes = notes[notes["rating"].ge(SEUIL_AIME) & notes["movieId"].isin(films_connus)]
    taille = notes.groupby("userId").size()
    notes = notes[notes["userId"].isin(taille[taille >= MIN_FILMS_AIMES].index)]
    log.info("%s notes retenues | %s spectateurs", f"{len(notes):,}", f"{notes['userId'].nunique():,}")
    return notes


def build() -> pd.DataFrame:
    genres = _matrice_genres()
    notes = _notes_aimees(genres.index)

    spectateurs, lignes = np.unique(notes["userId"].to_numpy(), return_inverse=True)
    colonnes = genres.index.get_indexer(notes["movieId"].to_numpy())
    vues = sparse.csr_matrix((np.ones(len(notes), dtype="float32"), (lignes, colonnes)),
                             shape=(len(spectateurs), len(genres)))

    profils = pd.DataFrame(vues @ genres.to_numpy(), index=spectateurs, columns=genres.columns)
    profils.index.name = "userId"
    return profils.div(profils.sum(axis=1), axis=0)


def charger_profils(recalculer: bool = False) -> pd.DataFrame:
    """Part de chaque genre dans les films aimés, un spectateur par ligne."""
    if config.USER_PROFILES.exists() and not recalculer:
        return pd.read_parquet(config.USER_PROFILES)
    profils = build()
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    profils.to_parquet(config.USER_PROFILES)
    return profils


def matrice(profils: pd.DataFrame) -> np.ndarray:
    """Profils centrés-réduits par genre : on compare les écarts au goût moyen, pas les parts brutes."""
    return StandardScaler().fit_transform(profils)


def diagnostics_k(profils: pd.DataFrame, ks=KS_TESTES) -> pd.DataFrame:
    """Inertie, silhouette, Calinski-Harabasz et Davies-Bouldin pour chaque k."""
    X = matrice(profils)
    lignes = []
    for k in ks:
        modele = KMeans(n_clusters=k, n_init=10, random_state=ALEA).fit(X)
        tailles = np.bincount(modele.labels_)
        lignes.append({
            "k": k,
            "inertie": modele.inertia_,
            "silhouette": silhouette_score(X, modele.labels_, sample_size=ECHANTILLON_SILHOUETTE,
                                           random_state=ALEA),
            "calinski_harabasz": calinski_harabasz_score(X, modele.labels_),
            "davies_bouldin": davies_bouldin_score(X, modele.labels_),
            "plus_petit_cluster_pct": 100 * tailles.min() / len(X),
        })
        log.info("k=%s traité", k)
    table = pd.DataFrame(lignes).set_index("k")
    table["gain_inertie_pct"] = -100 * table["inertie"].pct_change()
    return table


def stabilite_k(profils: pd.DataFrame, ks=KS_TESTES, n_tirages: int = N_TIRAGES) -> pd.DataFrame:
    """Accord (ARI) entre deux partitions apprises sur des sous-échantillons disjoints tirés au sort."""
    X = matrice(profils)
    alea = np.random.default_rng(ALEA)
    lignes = []
    for k in ks:
        scores = []
        for _ in range(n_tirages):
            a, b = (alea.choice(len(X), TIRAGE_STABILITE, replace=False) for _ in range(2))
            commun = np.intersect1d(a, b)
            modele_a = KMeans(k, n_init=5, random_state=alea.integers(1e6)).fit(X[a])
            modele_b = KMeans(k, n_init=5, random_state=alea.integers(1e6)).fit(X[b])
            scores.append(adjusted_rand_score(modele_a.predict(X[commun]), modele_b.predict(X[commun])))
        lignes.append({"k": k, "ari_moyen": np.mean(scores), "ari_min": np.min(scores),
                       "ari_max": np.max(scores)})
        log.info("k=%s traité", k)
    return pd.DataFrame(lignes).set_index("k")


def segmenter(profils: pd.DataFrame, k: int = K_SEGMENTS) -> pd.DataFrame:
    """Profils augmentés de leur numéro de segment."""
    labels = KMeans(n_clusters=k, n_init=10, random_state=ALEA).fit_predict(matrice(profils))
    return profils.assign(segment=labels)


def charger_segments(recalculer: bool = False, k: int = K_SEGMENTS) -> pd.DataFrame:
    if config.USER_SEGMENTS.exists() and not recalculer:
        return pd.read_parquet(config.USER_SEGMENTS)
    segments = segmenter(charger_profils(), k=k)
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    segments.to_parquet(config.USER_SEGMENTS)
    return segments


def sur_representation(segments: pd.DataFrame) -> pd.DataFrame:
    """Part d'un genre dans le segment rapportée à sa part dans l'ensemble : 1 = goût moyen."""
    genres = segments.columns.drop("segment")
    return segments.groupby("segment")[list(genres)].mean() / segments[genres].mean()


def nommer(ecarts: pd.DataFrame, n_genres: int = 2) -> dict[int, str]:
    """Étiquette chaque segment par les genres qu'il sur-consomme le plus."""
    return {segment: " / ".join(ligne.sort_values(ascending=False).head(n_genres).index)
            for segment, ligne in ecarts.iterrows()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Segmente les spectateurs et justifie le choix du k.")
    parser.add_argument("--recalculer", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--k", type=int, default=K_SEGMENTS)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    profils = charger_profils(recalculer=args.recalculer)
    print(f"{len(profils):,} spectateurs | {profils.shape[1]} genres")

    if args.diagnostics:
        print("\nqualité de la partition")
        print(diagnostics_k(profils).round(3).to_string())
        print("\nstabilité de la partition")
        print(stabilite_k(profils).round(3).to_string())

    segments = charger_segments(recalculer=args.recalculer, k=args.k)
    ecarts = sur_representation(segments)
    noms = nommer(ecarts)
    parts = 100 * segments["segment"].value_counts(normalize=True).sort_index()
    print(f"\n{args.k} segments")
    for numero, nom in noms.items():
        tete = ecarts.loc[numero].sort_values(ascending=False).head(4)
        print(f"  {numero} {nom:<28} {parts[numero]:>5.1f} % | "
              + ", ".join(f"{genre} x{valeur:.2f}" for genre, valeur in tete.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
