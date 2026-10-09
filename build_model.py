"""
Movie Recommendation System - Model Artifact Builder
---------------------------------------------------
This script preprocesses the TMDB 5,000 Movies and Credits datasets,
extracts features, applies text stemming, vectorization, and computes
the cosine similarity matrix.

The resulting artifacts are saved as pickle files:
  - movie_dict.pkl / movies.pkl: Processed movie metadata (IDs, titles, tags)
  - similarity.pkl / similarity.pkl.gz: Pairwise cosine similarity matrix
"""

import os
import ast
import gzip
import pickle
import numpy as np
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def parse_names(json_str):
    """Extract list of names from JSON string."""
    try:
        return [item['name'] for item in ast.literal_eval(json_str)]
    except (ValueError, SyntaxError, TypeError):
        return []


def parse_top_cast(json_str, top_n=3):
    """Extract up to top_n cast members from JSON string."""
    try:
        items = ast.literal_eval(json_str)
        return [item['name'] for item in items[:top_n]]
    except (ValueError, SyntaxError, TypeError):
        return []


def parse_director(json_str):
    """Extract the Director name from crew JSON string."""
    try:
        items = ast.literal_eval(json_str)
        for item in items:
            if item.get('job') == 'Director':
                return [item['name']]
        return []
    except (ValueError, SyntaxError, TypeError):
        return []


def collapse_spaces(names_list):
    """Remove spaces from strings to create distinct single-token entities."""
    return [name.replace(" ", "") for name in names_list]


def stem_text(text, stemmer):
    """Apply Porter Stemmer to each whitespace-separated token in text."""
    return " ".join([stemmer.stem(token) for token in text.split()])


def build_recommendation_model(
    movies_path="tmdb_5000_movies.csv",
    credits_path="tmdb_5000_credits.csv",
    output_dir="."
):
    print("=" * 60)
    print("🎬 Building Movie Recommendation System Model")
    print("=" * 60)

    # 1. Load Datasets
    print(f"[1/6] Loading data from '{movies_path}' and '{credits_path}'...")
    movies_df = pd.read_csv(movies_path)
    credits_df = pd.read_csv(credits_path)
    print(f"      - Movies raw shape: {movies_df.shape}")
    print(f"      - Credits raw shape: {credits_df.shape}")

    # 2. Merge Datasets
    print("[2/6] Merging datasets on 'title'...")
    movies = movies_df.merge(credits_df, on="title")
    print(f"      - Merged shape: {movies.shape}")

    # Select relevant columns and drop rows with missing values
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    missing_count = movies.isnull().sum().sum()
    if missing_count > 0:
        print(f"      - Dropping {missing_count} missing records...")
        movies.dropna(inplace=True)
    movies.reset_index(drop=True, inplace=True)
    print(f"      - Cleaned shape: {movies.shape}")

    # 3. Extract & Clean Features
    print("[3/6] Extracting genres, keywords, cast (top 3), and director...")
    movies['genres'] = movies['genres'].apply(parse_names)
    movies['keywords'] = movies['keywords'].apply(parse_names)
    movies['cast'] = movies['cast'].apply(lambda x: parse_top_cast(x, top_n=3))
    movies['crew'] = movies['crew'].apply(parse_director)
    movies['overview'] = movies['overview'].apply(lambda x: x.split() if isinstance(x, str) else [])

    # Remove spaces within entity names to avoid token overlap
    for col in ['genres', 'keywords', 'cast', 'crew']:
        movies[col] = movies[col].apply(collapse_spaces)

    # Combine into unified 'tags'
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

    # Final movie dataframe
    processed_df = movies[['movie_id', 'title', 'tags']].copy()
    processed_df['id'] = processed_df['movie_id']  # compatibility with both 'id' and 'movie_id'
    processed_df['tags'] = processed_df['tags'].apply(lambda x: " ".join(x).lower())

    # 4. Text Stemming & Vectorization
    print("[4/6] Applying Porter Stemming and CountVectorizer...")
    ps = PorterStemmer()
    processed_df['tags'] = processed_df['tags'].apply(lambda text: stem_text(text, ps))

    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(processed_df['tags']).toarray()
    print(f"      - Feature vector matrix shape: {vectors.shape}")

    # 5. Cosine Similarity Matrix
    print("[5/6] Computing Cosine Similarity matrix (float32)...")
    similarity_matrix = cosine_similarity(vectors).astype(np.float32)
    print(f"      - Similarity matrix shape: {similarity_matrix.shape}")

    # 6. Save Artifacts
    print(f"[6/6] Exporting serialized artifacts to '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)

    movie_dict_path = os.path.join(output_dir, "movie_dict.pkl")
    movies_pkl_path = os.path.join(output_dir, "movies.pkl")
    sim_pkl_path = os.path.join(output_dir, "similarity.pkl")
    sim_gz_path = os.path.join(output_dir, "similarity.pkl.gz")

    with open(movie_dict_path, "wb") as f:
        pickle.dump(processed_df.to_dict(), f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"      ✓ Saved: {movie_dict_path} ({os.path.getsize(movie_dict_path) / (1024*1024):.2f} MB)")

    with open(movies_pkl_path, "wb") as f:
        pickle.dump(processed_df, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"      ✓ Saved: {movies_pkl_path} ({os.path.getsize(movies_pkl_path) / (1024*1024):.2f} MB)")

    with gzip.open(sim_gz_path, "wb") as f:
        pickle.dump(similarity_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"      ✓ Saved: {sim_gz_path} ({os.path.getsize(sim_gz_path) / (1024*1024):.2f} MB)")

    with open(sim_pkl_path, "wb") as f:
        pickle.dump(similarity_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"      ✓ Saved: {sim_pkl_path} ({os.path.getsize(sim_pkl_path) / (1024*1024):.2f} MB)")

    print("\n✅ Model build completed successfully!")
    return processed_df, similarity_matrix


if __name__ == "__main__":
    build_recommendation_model()
