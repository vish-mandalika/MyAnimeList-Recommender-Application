# Viswas Mandalika

# app.py

import pandas as pd
import numpy as np
import pickle
import json
from flask import Flask, render_template, jsonify, request
import requests as http_requests  # rename to avoid conflict with flask.request
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load precomputed models and data at startup
DATA_DIR = "../data/"

# Anime metadata
anime_df = pd.read_csv(f"{DATA_DIR}anime_clean.csv")
anime_df["title_english"] = anime_df["title_english"].fillna(anime_df["title"])

# Ratings (for user lookups)
ratings_df = pd.read_csv(f"{DATA_DIR}ratings_clean.csv")

# Genre matrix (for new user selections)
genre_matrix = pd.read_pickle(f"{DATA_DIR}genre_matrix_filtered.pkl")
cosine_sim = cosine_similarity(genre_matrix)
cosine_sim_df = pd.DataFrame(
    cosine_sim,
    index=genre_matrix.index,
    columns=genre_matrix.index,
)
print(f"Computed cosine similarity matrix: {cosine_sim_df.shape}")

# Load SVD parameters (no Surprise dependency needed!)
svd_data = np.load(f"{DATA_DIR}svd_params.npz")
with open(f"{DATA_DIR}svd_mappings.json", "r") as f:
    svd_mappings = json.load(f)

svd_pu = svd_data["pu"]
svd_qi = svd_data["qi"]
svd_bu = svd_data["bu"]
svd_bi = svd_data["bi"]
svd_global_mean = svd_data["global_mean"][0]
svd_user_ids = svd_mappings["user_ids"]
svd_item_ids = {int(k): v for k, v in svd_mappings["item_ids"].items()}

# Load covers
with open(f"{DATA_DIR}covers.json", "r") as f:
    covers = json.load(f)
covers = {int(k): v for k, v in covers.items()}

print(f"Loaded {len(anime_df)} anime, {len(ratings_df)} ratings")
print(f"Cosine sim matrix: {cosine_sim_df.shape}")
print(f"SVD: {svd_pu.shape[0]} users, {svd_qi.shape[0]} items, {svd_pu.shape[1]} factors")
print("Models ready!")

def svd_predict(username, anime_id):
    """Predict rating using extracted SVD parameters."""
    uid = svd_user_ids.get(str(username))
    iid = svd_item_ids.get(anime_id)

    if uid is not None and iid is not None:
        pred = svd_global_mean + svd_bu[uid] + svd_bi[iid] + np.dot(svd_pu[uid], svd_qi[iid])
    elif uid is not None:
        pred = svd_global_mean + svd_bu[uid]
    elif iid is not None:
        pred = svd_global_mean + svd_bi[iid]
    else:
        pred = svd_global_mean

    return float(np.clip(pred, 1, 10))

# Helper functions
def get_franchise_key(title):
    words = title.lower().replace(":", "").split()
    return " ".join(words[:2])


# Routes
@app.route("/")
def home():
    return render_template("index.html")

"""
@app.route("/api/popular-anime")
def popular_anime():
    # Sort by number of members (popularity proxy)
    top = anime_df.nlargest(30, "members")[
        ["anime_id", "title", "title_english", "genre", "score"]
    ]
    return jsonify(top.to_dict(orient="records"))
"""

@app.route("/pick-titles")
def pick_titles():
    return render_template("pick_titles.html")

@app.route("/api/recommend", methods=["POST"])
def recommend():
    """Given a list of anime_ids the user liked, return recommendations."""
    data = request.get_json()
    anime_ids = data.get("anime_ids", [])

    if len(anime_ids) == 0:
        return jsonify({"error": "No anime selected"}), 400

    # Get valid IDs that exist in our similarity matrix
    valid_ids = [a for a in anime_ids if a in cosine_sim_df.index]
    if len(valid_ids) == 0:
        return jsonify({"error": "No matching anime found"}), 400

    # Content-based: average similarity to selected anime
    cb_scores = cosine_sim_df[valid_ids].mean(axis=1)

    # Remove anime the user already selected
    cb_scores = cb_scores.drop(valid_ids, errors="ignore")

    # Deduplicate by franchise
    cb_scores = cb_scores.sort_values(ascending=False)
    seen_franchises = set()
    for aid in anime_ids:
        row = anime_df[anime_df["anime_id"] == aid]
        if not row.empty:
            seen_franchises.add(get_franchise_key(row.iloc[0]["title"]))

    results = []
    for aid, score in cb_scores.items():
        if len(results) >= 10:
            break
        row = anime_df[anime_df["anime_id"] == aid]
        if row.empty:
            continue
        info = row.iloc[0]
        franchise = get_franchise_key(info["title"])
        if franchise in seen_franchises:
            continue
        seen_franchises.add(franchise)
        results.append({
            "anime_id": int(aid),
            "title": info["title"],
            "title_english": info["title_english"],
            "genre": info["genre"],
            "score": float(info["score"]),
            "similarity": round(float(score), 3),
            "cover": covers.get(int(aid), None),
        })
    
    # Build explanation context
    selected_anime = anime_df[anime_df["anime_id"].isin(anime_ids)]
    selected_titles = selected_anime["title_english"].fillna(selected_anime["title"]).tolist()

    # Get genre profile of selections
    all_genres = selected_anime["genre"].str.split(r"[|,]").explode().str.strip()
    genre_profile = all_genres.value_counts().head(6)
    top_genres = [{"name": g, "count": int(c)} for g, c in genre_profile.items()]

    return jsonify({
        "recommendations": results,
        "explanation": {
            "method": "content-based",
            "selected_titles": selected_titles,
            "top_genres": top_genres,
            "total_anime_compared": len(cb_scores),
        }
    })

@app.route("/results")
def results():
    return render_template("results.html")

@app.route("/api/genres")
def genres():
    """Return all unique genres with their counts."""
    all_genres = anime_df["genre"].str.split(r"[|,]").explode().str.strip()
    genre_counts = all_genres.value_counts()
    result = [
        {"name": genre, "count": int(count)}
        for genre, count in genre_counts.items()
    ]
    return jsonify(result)


@app.route("/api/recommend-by-genre", methods=["POST"])
def recommend_by_genre():
    """Given a list of genres, return top anime matching those genres."""
    data = request.get_json()
    selected_genres = data.get("genres", [])

    if len(selected_genres) == 0:
        return jsonify({"error": "No genres selected"}), 400

    # Score each anime: how many of the selected genres does it have?
    def genre_match_score(genre_str):
        anime_genres = [g.strip() for g in genre_str.split(",")]
        # Also handle pipe separator
        expanded = []
        for g in anime_genres:
            expanded.extend([x.strip() for x in g.split("|")])
        matches = sum(1 for g in expanded if g in selected_genres)
        return matches / len(selected_genres) if selected_genres else 0

    anime_scored = anime_df.copy()
    anime_scored["match"] = anime_scored["genre"].apply(genre_match_score)

    # Filter to anime that match at least one genre, sort by match then score
    anime_scored = anime_scored[anime_scored["match"] > 0]
    anime_scored = anime_scored.sort_values(
        ["match", "score"], ascending=[False, False]
    )

    # Deduplicate by franchise
    seen_franchises = set()
    results = []
    for _, row in anime_scored.iterrows():
        if len(results) >= 10:
            break
        franchise = get_franchise_key(row["title"])
        if franchise in seen_franchises:
            continue
        seen_franchises.add(franchise)
        results.append({
            "anime_id": int(row["anime_id"]),
            "title": row["title"],
            "title_english": row["title_english"],
            "genre": row["genre"],
            "score": float(row["score"]),
            "match": round(float(row["match"]), 2),
            "cover": covers.get(int(row["anime_id"]), None),
        })

    return jsonify({
        "recommendations": results,
        "explanation": {
            "method": "genre-match",
            "selected_genres": selected_genres,
            "total_anime_scored": len(anime_scored),
        }
    })


@app.route("/pick-genres")
def pick_genres():
    return render_template("pick_genres.html")

@app.route("/results-genre")
def results_genre():
    return render_template("results_genre.html")

@app.route("/mal-user")
def mal_user():
    return render_template("mal_user.html")


@app.route("/api/recommend-by-user", methods=["POST"])
def recommend_by_user():
    """Given a MAL username, return hybrid recommendations."""
    data = request.get_json()
    username = data.get("username", "").strip()

    if not username:
        return jsonify({"error": "No username provided"}), 400

    # Check if user exists in our dataset
    user_ratings = ratings_df[ratings_df["username"] == username]
    if len(user_ratings) == 0:
        return jsonify({
            "error": f"User '{username}' not found in our dataset. "
                     "This demo uses a static snapshot of MAL data, "
                     "so only ~2,300 users are available."
        }), 404

    user_rated = set(user_ratings["anime_id"])
    user_favs = user_ratings[user_ratings["my_score"] >= 8]["anime_id"].values

    # Candidates: anime in similarity matrix that user hasn't rated
    candidates = [a for a in cosine_sim_df.index if a not in user_rated]

    # Content-based scores
    valid_favs = [a for a in user_favs if a in cosine_sim_df.index]
    if valid_favs:
        cb_scores = cosine_sim_df.loc[candidates, valid_favs].mean(axis=1)
    else:
        cb_scores = pd.Series(0, index=candidates)

    # CF scores from SVD
    cf_scores = pd.Series(
        [svd_predict(username, a) for a in candidates],
        index=candidates
    )

    # Rank-based blending (same as hybrid_recommend_v2)
    cf_ranks = cf_scores.rank(pct=True)
    cb_ranks = cb_scores.rank(pct=True)
    hybrid_ranks = 0.7 * cf_ranks + 0.3 * cb_ranks

    sorted_ids = hybrid_ranks.sort_values(ascending=False).index

    # Deduplicate by franchise
    seen_franchises = set()
    results = []
    for aid in sorted_ids:
        if len(results) >= 10:
            break
        row = anime_df[anime_df["anime_id"] == aid]
        if row.empty:
            continue
        info = row.iloc[0]
        franchise = get_franchise_key(info["title"])
        if franchise in seen_franchises:
            continue
        seen_franchises.add(franchise)
        results.append({
            "anime_id": int(aid),
            "title": info["title"],
            "title_english": info["title_english"],
            "genre": info["genre"],
            "score": float(info["score"]),
            "cf_score": round(float(cf_scores[aid]), 2),
            "hybrid_rank": round(float(hybrid_ranks[aid]), 4),
            "cover": covers.get(int(aid), None),
        })

    # Also return some user stats
    user_stats = {
        "username": username,
        "total_rated": len(user_ratings),
        "mean_score": round(float(user_ratings["my_score"].mean()), 2),
        "top_genres": get_user_top_genres(user_ratings),
    }

    return jsonify({
        "recommendations": results,
        "user_stats": user_stats,
        "explanation": {
            "method": "hybrid",
            "n_rated": len(user_ratings),
            "n_favorites": len(valid_favs),
            "n_candidates": len(candidates),
            "cf_weight": 0.7,
            "cb_weight": 0.3,
        }
    })

@app.route("/api/popular-anime")
def popular_anime():
    top = anime_df.nlargest(30, "members")[
        ["anime_id", "title", "title_english", "genre", "score"]
    ]
    records = top.to_dict(orient="records")
    # Add cover URLs
    for r in records:
        r["cover"] = covers.get(r["anime_id"], None)
    return jsonify(records)


def get_user_top_genres(user_ratings):
    """Get a user's most-watched genres."""
    user_anime = user_ratings.merge(
        anime_df[["anime_id", "genre"]], on="anime_id"
    )
    all_genres = user_anime["genre"].str.split(r"[|,]").explode().str.strip()
    top = all_genres.value_counts().head(5)
    return [{"name": g, "count": int(c)} for g, c in top.items()]

@app.route("/api/cover/<int:anime_id>")
def get_cover(anime_id):
    """Return cover URL, fetching from Jikan if not cached."""
    if anime_id not in covers:
        try:
            resp = http_requests.get(
                f"https://api.jikan.moe/v4/anime/{anime_id}",
                timeout=10
            )
            if resp.status_code == 200:
                url = resp.json()["data"]["images"]["jpg"]["large_image_url"]
                covers[anime_id] = url
            else:
                covers[anime_id] = None
        except:
            covers[anime_id] = None

    return jsonify({"url": covers.get(anime_id)})

if __name__ == "__main__":
    app.run(debug=False)

