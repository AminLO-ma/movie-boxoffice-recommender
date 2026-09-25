"""
Dashboard interactif pour explorer le modèle de recommandation par clustering
MovieLens (voir movielens_clustering_recommender.ipynb et test_model.py).

Installation :
    pip install streamlit plotly requests pandas

Lancement (depuis le même dossier que test_model.py et models/) :
    streamlit run dashboard.py

Affiches des films (optionnel) :
    Pour voir les affiches, il faut :
    1) un fichier links.csv dans le dossier de données (colonnes movieId, imdbId,
       tmdbId) — il est normalement déjà présent dans ml-32m ;
    2) une clé API TMDB gratuite (https://www.themoviedb.org/settings/api),
       à coller dans le champ de la barre latérale ou à mettre dans la
       variable d'environnement TMDB_API_KEY.
    Sans clé, le dashboard fonctionne quand même, simplement sans affiches.
"""

import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from test_model import (
    DATA_DIR,
    MODEL_PATH,
    get_user_cluster,
    load_data,
    load_model,
    recommend,
)

TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w342"
TMDB_API_BASE = "https://api.themoviedb.org/3/movie"

st.set_page_config(page_title="MovieLens — Recommandations & Clusters", layout="wide")


# ----------------------------------------------------------------------------
# Chargement (mis en cache pour ne pas tout recharger à chaque interaction)
# ----------------------------------------------------------------------------

@st.cache_resource
def get_model(path=MODEL_PATH):
    return load_model(path)


@st.cache_data
def get_data(data_dir=DATA_DIR):
    return load_data(data_dir)


@st.cache_data
def get_links(data_dir=DATA_DIR):
    path = os.path.join(data_dir, "links.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, dtype={"movieId": "int32"})


@st.cache_data(show_spinner=False)
def fetch_poster_url(tmdb_id, api_key):
    """Interroge l'API TMDB pour récupérer l'URL de l'affiche d'un film."""
    if not api_key or pd.isna(tmdb_id):
        return None
    try:
        resp = requests.get(
            f"{TMDB_API_BASE}/{int(tmdb_id)}",
            params={"api_key": api_key},
            timeout=5,
        )
        resp.raise_for_status()
        poster_path = resp.json().get("poster_path")
        return f"{TMDB_IMAGE_BASE}{poster_path}" if poster_path else None
    except requests.RequestException:
        return None


def attach_posters(df, links, api_key):
    """Ajoute une colonne poster_url à un DataFrame de films (doit contenir movieId)."""
    if links is None or not api_key:
        return df.assign(poster_url=None)
    merged = df.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")
    merged["poster_url"] = merged["tmdbId"].apply(lambda t: fetch_poster_url(t, api_key))
    return merged


def show_movie_grid(df, columns=5):
    """Affiche une grille d'affiches (ou une pastille par défaut) pour un DataFrame de films."""
    if df.empty:
        st.info("Aucun film à afficher.")
        return
    score_col = "final_score" if "final_score" in df.columns else "weighted_rating"
    rows = [df.iloc[i : i + columns] for i in range(0, len(df), columns)]
    for row in rows:
        cols = st.columns(columns)
        for col, (_, movie) in zip(cols, row.iterrows()):
            with col:
                if movie.get("poster_url"):
                    st.image(movie["poster_url"], use_container_width=True)
                else:
                    st.markdown(
                        "<div style='height:220px;display:flex;align-items:center;"
                        "justify-content:center;background:#eee;border-radius:8px;"
                        "text-align:center;padding:8px;color:#666;'>🎬<br>Pas d'affiche</div>",
                        unsafe_allow_html=True,
                    )
                st.caption(f"**{movie['title']}**")
                st.caption(movie.get("genres", ""))
                if score_col in movie and pd.notna(movie[score_col]):
                    st.caption(f"Score : {movie[score_col]:.2f}")


# ----------------------------------------------------------------------------
# Chargement des données et du modèle
# ----------------------------------------------------------------------------

st.title("🎬 MovieLens — Recommandations & Clusters")

try:
    model = get_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

ratings, movies = get_data()
links = get_links()

genre_profile = model["cluster_genre_profile"]
cluster_stats = model["cluster_movie_stats"]
user_clusters = model["user_clusters"]
cluster_ids = sorted(genre_profile.index.tolist())

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input(
        "Clé API TMDB (optionnelle, pour les affiches)",
        value=os.environ.get("TMDB_API_KEY", "09c183a9e21038d5bc47c4ae460b9559"),
        type="password",
    )
    if links is None:
        st.caption("⚠️ links.csv introuvable dans le dossier de données — pas d'affiches possibles.")

tab_user, tab_cluster, tab_compare = st.tabs(
    ["👤 Recommandations par utilisateur", "📊 Profil d'un cluster", "⚖️ Comparer deux clusters"]
)

# ----------------------------------------------------------------------------
# Onglet 1 : recommandations par utilisateur
# ----------------------------------------------------------------------------

with tab_user:
    all_users = sorted(ratings["userId"].unique().tolist())
    user_id = st.selectbox("Utilisateur", all_users, index=0)
    top_n = st.slider("Nombre de recommandations", 5, 30, 10)

    cluster_id = get_user_cluster(model, user_id)
    if cluster_id is not None:
        cluster_size = (user_clusters["cluster"] == cluster_id).sum()
        st.write(f"Cluster de l'utilisateur **{user_id}** : **{cluster_id}** ({cluster_size:,} utilisateurs)")
    else:
        st.write(f"Utilisateur **{user_id}** inconnu du modèle → repli sur la popularité globale.")

    with st.expander("Historique de l'utilisateur (films les mieux notés)"):
        history = ratings[ratings["userId"] == user_id].merge(movies, on="movieId")
        history = history.sort_values("rating", ascending=False).head(15)
        st.dataframe(history[["title", "genres", "rating"]], use_container_width=True, hide_index=True)

    recs = recommend(model, ratings, movies, user_id, top_n=top_n)
    recs = attach_posters(recs, links, api_key)

    st.subheader("Affiches recommandées")
    show_movie_grid(recs)

    st.subheader("Détail")
    st.dataframe(
        recs.drop(columns=["poster_url", "tmdbId"], errors="ignore"),
        use_container_width=True,
        hide_index=True,
    )

# ----------------------------------------------------------------------------
# Onglet 2 : profil d'un cluster
# ----------------------------------------------------------------------------

with tab_cluster:
    cid = st.selectbox("Cluster", cluster_ids, key="single_cluster")
    size = (user_clusters["cluster"] == cid).sum()
    st.write(f"Taille du cluster : **{size:,}** utilisateurs")

    top_genres = genre_profile.loc[cid].sort_values(ascending=False).head(12)
    fig = px.bar(
        top_genres.iloc[::-1],
        orientation="h",
        labels={"value": "Sur-représentation (lift)", "index": "Genre"},
        title=f"Genres sur-représentés — cluster {cid}",
    )
    fig.add_vline(x=1, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Films les plus distinctifs du cluster")
    top_movies = (
        cluster_stats[cluster_stats["cluster"] == cid]
        .sort_values("distinctive_score", ascending=False)
        .head(15)
    )
    top_movies = attach_posters(top_movies, links, api_key)
    show_movie_grid(top_movies)

# ----------------------------------------------------------------------------
# Onglet 3 : comparer deux clusters
# ----------------------------------------------------------------------------

with tab_compare:
    col_a, col_b = st.columns(2)
    with col_a:
        cid_a = st.selectbox("Cluster A", cluster_ids, index=0, key="cluster_a")
    with col_b:
        default_b_index = 1 if len(cluster_ids) > 1 else 0
        cid_b = st.selectbox("Cluster B", cluster_ids, index=default_b_index, key="cluster_b")

    compare_df = genre_profile.loc[[cid_a, cid_b]].T.reset_index()
    compare_df.columns = ["genre", f"Cluster {cid_a}", f"Cluster {cid_b}"]
    compare_df = compare_df.sort_values(f"Cluster {cid_a}", ascending=False)
    compare_long = compare_df.melt(id_vars="genre", var_name="cluster", value_name="lift")

    fig = px.bar(
        compare_long,
        x="genre",
        y="lift",
        color="cluster",
        barmode="group",
        title=f"Comparaison des genres — cluster {cid_a} vs cluster {cid_b}",
    )
    fig.add_hline(y=1, line_dash="dash", line_color="gray")
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Films distinctifs côte à côte")
    col_a, col_b = st.columns(2)
    for col, cid in [(col_a, cid_a), (col_b, cid_b)]:
        with col:
            st.markdown(f"**Cluster {cid}**")
            top = (
                cluster_stats[cluster_stats["cluster"] == cid]
                .sort_values("distinctive_score", ascending=False)
                .head(8)
            )
            st.dataframe(
                top[["title", "genres", "distinctive_score"]],
                hide_index=True,
                use_container_width=True,
            )
