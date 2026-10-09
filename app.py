import os
import gzip
import pickle
import requests
import pandas as pd
import streamlit as st

# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="CineMatch | Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Netflix/Cinematic Theme
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Main Header */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #E50914 0%, #FF6B6B 50%, #FFA07A 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        padding-top: 0.5rem;
    }
    .sub-title {
        text-align: center;
        font-size: 1.1rem;
        color: #b3b3b3;
        margin-bottom: 2rem;
    }
    
    /* Movie Card Styling */
    .movie-card {
        background: #1a1f2c;
        border-radius: 12px;
        padding: 10px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        border: 1px solid #2d3748;
    }
    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(229, 9, 20, 0.4);
        border-color: #e50914;
    }
    .movie-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #ffffff;
        margin-top: 8px;
        margin-bottom: 4px;
        text-align: center;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .badge-rating {
        background-color: #e50914;
        color: white;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-year {
        background-color: #2d3748;
        color: #cbd5e0;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 500;
        display: inline-block;
        margin-left: 5px;
    }
    .meta-container {
        text-align: center;
        margin-top: 5px;
        margin-bottom: 8px;
    }
    
    /* Button Customization */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #E50914 0%, #B81D24 100%);
        color: white;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-size: 1.05rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(229, 9, 20, 0.3);
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(90deg, #ff1a25 0%, #E50914 100%);
        transform: scale(1.02);
        box-shadow: 0 6px 18px rgba(229, 9, 20, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# Default TMDB API key fallback
DEFAULT_API_KEY = "8265bd1679663a7ea12ac168da84d2e8"
FALLBACK_POSTER_URL = "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?q=80&w=500&auto=format&fit=crop"


# ==============================================================================
# Helper Functions & Resource Loaders
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_data():
    """Load movies metadata and cosine similarity matrix."""
    movies = None
    similarity = None

    # Load Movies metadata
    if os.path.exists("movies.pkl"):
        with open("movies.pkl", "rb") as f:
            movies = pickle.load(f)
    elif os.path.exists("movie_dict.pkl"):
        with open("movie_dict.pkl", "rb") as f:
            movie_dict = pickle.load(f)
            movies = pd.DataFrame(movie_dict)

    # Load Similarity Matrix
    if os.path.exists("similarity.pkl.gz"):
        with gzip.open("similarity.pkl.gz", "rb") as f:
            similarity = pickle.load(f)
    elif os.path.exists("similarity.pkl"):
        with open("similarity.pkl", "rb") as f:
            similarity = pickle.load(f)

    return movies, similarity


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_movie_details(movie_id, api_key):
    """
    Fetch poster path, rating, release year, and overview from TMDB API.
    Gracefully handles API errors, network timeouts, and missing images.
    """
    default_info = {
        "poster_url": FALLBACK_POSTER_URL,
        "rating": "N/A",
        "year": "N/A",
        "overview": "No synopsis available."
    }

    if not api_key or not movie_id:
        return default_info

    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
    try:
        response = requests.get(url, timeout=4)
        if response.status_code == 200:
            data = response.json()
            poster_path = data.get("poster_path")
            poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else FALLBACK_POSTER_URL
            
            rating = round(data.get("vote_average", 0), 1) if data.get("vote_average") else "N/A"
            release_date = data.get("release_date", "")
            year = release_date.split("-")[0] if release_date else "N/A"
            overview = data.get("overview", "No synopsis available.")

            return {
                "poster_url": poster_url,
                "rating": rating,
                "year": year,
                "overview": overview
            }
    except Exception:
        pass

    return default_info


def get_recommendations(movie_title, movies_df, similarity_matrix, top_n=5):
    """
    Find top_n movies most similar to the selected movie using Cosine Similarity.
    """
    try:
        movie_idx = movies_df[movies_df['title'] == movie_title].index[0]
        distances = similarity_matrix[movie_idx]
        # Sort descending by similarity score, excluding the movie itself (index 0)
        movie_indices = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:top_n + 1]

        recommendations = []
        for idx, score in movie_indices:
            row = movies_df.iloc[idx]
            movie_id = row.get('movie_id', row.get('id'))
            recommendations.append({
                "title": row['title'],
                "movie_id": movie_id,
                "similarity_score": round(float(score) * 100, 1)
            })
        return recommendations
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
        return []


# ==============================================================================
# Application Initialization & Data Verification
# ==============================================================================
movies_df, similarity_matrix = load_data()

# Check if model files need to be generated
if movies_df is None or similarity_matrix is None:
    st.error("⚠️ Model artifacts not found (`movies.pkl`/`similarity.pkl`).")
    st.info("The application requires precomputed artifacts to recommend movies. You can generate them now.")
    if st.button("🔨 Build Model Artifacts Now"):
        with st.spinner("Generating recommendation model and similarity matrix... Please wait."):
            from build_model import build_recommendation_model
            build_recommendation_model()
            st.success("Artifacts created! Reloading app...")
            st.rerun()
    st.stop()


# ==============================================================================
# Sidebar Configuration
# ==============================================================================
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?q=80&w=400&auto=format&fit=crop", use_container_width=True)
    st.title("⚙️ Settings & Info")
    
    # TMDB API Key resolution
    env_api_key = os.getenv("TMDB_API_KEY", "")
    secret_api_key = ""
    try:
        if hasattr(st, "secrets") and "TMDB_API_KEY" in st.secrets:
            secret_api_key = st.secrets["TMDB_API_KEY"]
    except Exception:
        secret_api_key = ""
    default_key = secret_api_key or env_api_key or DEFAULT_API_KEY
    
    api_key = st.text_input(
        "TMDB API Key",
        value=default_key,
        type="password",
        help="Used to fetch official high-definition movie posters and ratings from The Movie Database API."
    )
    if api_key:
        st.success("✓ API Key configured", icon="🔑")
    else:
        st.warning("No API Key. Fallback posters will be shown.", icon="⚠️")

    num_recommendations = st.slider(
        "Number of recommendations",
        min_value=3,
        max_value=10,
        value=5,
        step=1
    )

    st.markdown("---")
    st.subheader("📊 Model Architecture")
    st.markdown("""
    - **Filtering Type:** Content-Based Filtering
    - **Feature Space:** 5,000 top Bag-of-Words features
    - **NLP Pipeline:** Porter Stemming + Stopword Removal
    - **Distance Metric:** Cosine Similarity
    - **Total Movies:** {:,}
    """.format(len(movies_df)))

    st.markdown("---")
    st.markdown("💡 **Quick Pick Suggestions:**")
    col_q1, col_q2 = st.columns(2)
    with col_q1:
        if st.button("Avatar"):
            st.session_state["selected_movie"] = "Avatar"
        if st.button("The Avengers"):
            st.session_state["selected_movie"] = "The Avengers"
    with col_q2:
        if st.button("Interstellar"):
            st.session_state["selected_movie"] = "Interstellar"
        if st.button("The Dark Knight"):
            st.session_state["selected_movie"] = "The Dark Knight"


# ==============================================================================
# Main User Interface
# ==============================================================================
st.markdown("<h1 class='main-title'>🎬 CINEMATCH</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Discover your next favorite movie using Natural Language Processing & Cosine Similarity</p>", unsafe_allow_html=True)

# Movie Selection
movie_list = sorted(movies_df['title'].unique().tolist())
default_selection_index = 0

if "selected_movie" in st.session_state and st.session_state["selected_movie"] in movie_list:
    default_selection_index = movie_list.index(st.session_state["selected_movie"])

selected_movie = st.selectbox(
    "🔍 **Select or search for a movie you love:**",
    movie_list,
    index=default_selection_index,
    help="Type any movie title to filter the dataset"
)

col_center = st.columns([1, 2, 1])[1]
with col_center:
    recommend_button = st.button("🍿 Recommend Similar Movies", use_container_width=True)

# ==============================================================================
# Recommendations Rendering
# ==============================================================================
if recommend_button or ("selected_movie" in st.session_state and st.session_state.get("auto_run")):
    with st.spinner(f"Finding movies similar to '{selected_movie}'..."):
        # Fetch details for the selected reference movie
        selected_row = movies_df[movies_df['title'] == selected_movie].iloc[0]
        selected_id = selected_row.get('movie_id', selected_row.get('id'))
        selected_info = fetch_movie_details(selected_id, api_key)

        # Show selected movie summary
        st.markdown("---")
        st.markdown(f"### 🎯 Because you enjoyed **{selected_movie}**:")
        with st.expander("ℹ️ View Selected Movie Details", expanded=False):
            exp_col1, exp_col2 = st.columns([1, 4])
            with exp_col1:
                st.image(selected_info["poster_url"], width=130)
            with exp_col2:
                st.markdown(f"**Year:** {selected_info['year']} | **Rating:** ⭐ {selected_info['rating']}/10")
                st.markdown(f"**Overview:** {selected_info['overview']}")

        # Fetch recommendations
        recs = get_recommendations(selected_movie, movies_df, similarity_matrix, top_n=num_recommendations)

        if recs:
            st.markdown(f"#### ✨ Top {len(recs)} Recommendations:")
            cols = st.columns(len(recs))

            for idx, item in enumerate(recs):
                details = fetch_movie_details(item['movie_id'], api_key)
                with cols[idx]:
                    st.image(details["poster_url"], use_container_width=True)
                    st.markdown(f"<div class='movie-title' title='{item['title']}'>{item['title']}</div>", unsafe_allow_html=True)
                    st.markdown(
                        f"<div class='meta-container'>"
                        f"<span class='badge-rating'>⭐ {details['rating']}</span>"
                        f"<span class='badge-year'>{details['year']}</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    st.caption(f"Match: **{item['similarity_score']}%**")
                    with st.expander("Synopsis"):
                        st.write(details["overview"])
        else:
            st.warning("No recommendations could be generated for this title.")

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #718096; font-size: 0.9rem; padding: 1rem 0;'>"
    "CineMatch • Powered by Scikit-Learn, Streamlit, and TMDB API"
    "</div>",
    unsafe_allow_html=True
)