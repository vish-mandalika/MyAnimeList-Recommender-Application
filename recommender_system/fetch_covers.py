# fetch_covers.py
# Run this once to download cover image URLs from Jikan API
import pandas as pd
import requests
import time
import json

anime_df = pd.read_csv("data/anime_clean.csv")

# We only need covers for popular anime (the ones likely to appear in results)
# Top 500 by members covers most recommendations
top_anime = anime_df.nlargest(500, "members")

covers = {}
total = len(top_anime)

for i, (_, row) in enumerate(top_anime.iterrows()):
    anime_id = int(row["anime_id"])
    print(f"[{i+1}/{total}] Fetching {row['title']}...")

    try:
        resp = requests.get(
            f"https://api.jikan.moe/v4/anime/{anime_id}",
            timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()["data"]
            covers[anime_id] = data["images"]["jpg"]["large_image_url"]
        elif resp.status_code == 429:
            print("  Rate limited — waiting 4 seconds...")
            time.sleep(4)
            continue
    except Exception as e:
        print(f"  Error: {e}")

    # Jikan allows ~3 requests/second
    time.sleep(0.4)

# Save to JSON
with open("data/covers.json", "w") as f:
    json.dump(covers, f)

print(f"\nSaved {len(covers)} cover URLs to data/covers.json")