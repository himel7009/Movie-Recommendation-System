# 🎬 CineMatch — Content-Based Movie Recommendation System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-3.8%2B-339933.svg)](https://www.nltk.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning web application that recommends movies based on plot synopsis, genres, keywords, lead cast, and director. Built using **Natural Language Processing (NLP)**, **Bag-of-Words Vectorization**, and **Cosine Similarity** on the **TMDB 5,000 Movies Dataset**, with interactive poster and metadata fetching via **The Movie Database (TMDB) API** and a modern **Streamlit** user interface.

---

## 📑 Table of Contents
- [✨ Key Features](#-key-features)
- [🧠 How It Works (Machine Learning Pipeline)](#-how-it-works-machine-learning-pipeline)
  - [1. Data Cleaning & Feature Extraction](#1-data-cleaning--feature-extraction)
  - [2. Entity Normalization](#2-entity-normalization)
  - [3. Natural Language Processing (Porter Stemming)](#3-natural-language-processing-porter-stemming)
  - [4. Vectorization (CountVectorizer)](#4-vectorization-countvectorizer)
  - [5. Cosine Similarity Calculation](#5-cosine-similarity-calculation)
- [📂 Project Structure](#-project-structure)
- [🚀 Quick Start (Local Setup)](#-quick-start-local-setup)
- [🔑 TMDB API Key Configuration](#-tmdb-api-key-configuration)
- [🌐 Deployment Guide](#-deployment-guide)
  - [Streamlit Community Cloud](#option-a-streamlit-community-cloud-recommended)
  - [Docker](#option-b-docker-container)
  - [Render / Heroku](#option-c-render--heroku-via-procfile)
- [📓 Jupyter Notebook Walkthrough](#-jupyter-notebook-walkthrough)
- [🛠️ Troubleshooting & FAQ](#️-troubleshooting--faq)
- [📜 License](#-license)

---

## ✨ Key Features

- **Interactive Search & Typeahead:** Instant dropdown search across 4,806 movies from the TMDB database.
- **Dynamic TMDB API Poster Fetching:** Real-time fetching of official movie posters, release years, vote ratings, and synopses with `@st.cache_data` caching.
- **Resilient Fallback Handling:** Gracefully falls back to placeholder posters if network requests fail or API keys expire, ensuring zero UI crashes.
- **Adjustable Recommendations:** Customize the number of recommended movies (from 3 to 10) dynamically.
- **Cinematic Dark Theme:** Custom CSS styling inspired by Netflix/streaming interfaces.
- **Reproducible Pipeline:** Standalone `build_model.py` script for one-command model retraining without needing Jupyter.
- **Cloud-Ready Serialization:** Compresses the pairwise similarity matrix into `similarity.pkl.gz` (36.5 MB), fitting comfortably within Git limits (<100 MB) while decompressing in under 0.5s.

---

## 🧠 How It Works (Machine Learning Pipeline)

CineMatch uses **Content-Based Filtering**. Rather than relying on collaborative user ratings, it identifies items with similar content attributes.

```
+------------------------------------+       +------------------------------------+
|       tmdb_5000_movies.csv         |       |        tmdb_5000_credits.csv       |
| (id, title, overview, genres, etc) |       | (movie_id, title, cast, crew)      |
+-----------------+------------------+       +-----------------+------------------+
                  |                                            |
                  +---------------------+----------------------+
                                        |
                                        v
                          [ Merge DataFrames on 'title' ]
                                        |
                                        v
                    [ Extract: Genres, Keywords, Cast, Crew ]
                                        |
                                        v
                     [ Entity Normalization (Remove Spaces) ]
                                        |
                                        v
                    [ Concatenate into unified 'tags' text ]
                                        |
                                        v
                       [ NLP: Porter Stemming (NLTK) ]
                                        |
                                        v
                   [ Vectorization: CountVectorizer (5,000-D) ]
                                        |
                                        v
                     [ Compute Cosine Similarity Matrix ]
                                        |
                                        v
                    [ Export: movies.pkl & similarity.pkl.gz ]
                                        |
                                        v
                      [ Streamlit UI + TMDB Poster API ]
```

### 1. Data Cleaning & Feature Extraction
- The movies and credits tables are merged on the `title` column.
- Selected metadata features: `movie_id`, `title`, `overview`, `genres`, `keywords`, `cast`, and `crew`.
- Missing values and duplicates are removed.
- JSON strings are parsed with `ast.literal_eval`:
  - `genres` & `keywords`: Extracted category and plot keyword names.
  - `cast`: Top 3 lead actors (carrying highest star association).
  - `crew`: Extracted the `Director` (`job == 'Director'`).
  - `overview`: Tokenized into words.

### 2. Entity Normalization
If names have spaces, individual tokens can cause false positive similarities.
For example, *Sam Worthington* (actor in *Avatar*) and *Sam Mendes* (director of *Skyfall*) would both match on `'sam'`.
To solve this, spaces within multi-word entities are collapsed:
- `"Sam Worthington"` $\rightarrow$ `"SamWorthington"`
- `"Sam Mendes"` $\rightarrow$ `"SamMendes"`
- `"Science Fiction"` $\rightarrow$ `"ScienceFiction"`

### 3. Natural Language Processing (Porter Stemming)
Words appear in various derived forms (*actions*, *action*, *acting*). Using NLTK's `PorterStemmer`, words are reduced to their root stem:
$$\text{"actions"} \rightarrow \text{"action"}, \quad \text{"loving"} \rightarrow \text{"love"}$$

### 4. Vectorization (CountVectorizer)
All tags are converted into numerical feature vectors using `CountVectorizer`:
- `max_features=5000`: Restricts the feature space to the 5,000 most informative tokens.
- `stop_words='english'`: Strips common non-informative English stopwords (*the*, *is*, *in*, *and*).

### 5. Cosine Similarity Calculation
Rather than Euclidean distance (which is skewed by document length), we calculate the **Cosine Similarity** between every pair of movie vectors:

$$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|} = \frac{\sum_{i=1}^{n} u_i v_i}{\sqrt{\sum_{i=1}^{n} u_i^2} \sqrt{\sum_{i=1}^{n} v_i^2}}$$

- **Score 1.0:** Identical orientation (maximum similarity).
- **Score 0.0:** Orthogonal (no shared terms).

For a query movie, the engine extracts its similarity row, sorts scores descending, and returns the top $N$ closest matches.

---

## 📂 Project Structure

```
Movie_Recomendation_System/
├── .streamlit/
│   └── config.toml             # Custom theme colors and server settings
├── app.py                      # Production Streamlit application
├── build_model.py              # CLI script to preprocess data and generate artifacts
├── jupyter_file_01.ipynb       # Fully annotated step-by-step Jupyter Notebook
├── tmdb_5000_movies.csv        # Raw TMDB movies dataset
├── tmdb_5000_credits.csv       # Raw TMDB credits dataset
├── movies.pkl                  # Processed movie metadata DataFrame
├── movie_dict.pkl              # Processed movie metadata dictionary
├── similarity.pkl.gz           # Compressed Cosine Similarity matrix (36.5 MB)
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container definition for Docker deployment
├── Procfile                    # Web process command for Render / Heroku
├── setup.sh                    # Streamlit setup script for PaaS environments
├── .dockerignore               # Docker build exclusions
├── .gitignore                  # Git tracking exclusions
└── README.md                   # Comprehensive project documentation
```

---

## 🚀 Quick Start (Local Setup)

### Prerequisites
- Python 3.10+ installed
- Git installed

### 1. Clone the Repository
```bash
git clone https://github.com/himel7009/Movie-Recommendation-System.git
cd Movie-Recommendation-System
```

### 2. Create and Activate a Virtual Environment
```bash
# On Linux / macOS:
python3 -m venv venv
source venv/bin/activate

# On Windows:
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Rebuild Model Artifacts
Model pickle files are already included in the repository. If you ever modify the datasets or hyperparameters, regenerate the artifacts by running:
```bash
python build_model.py
```

### 5. Launch the Streamlit App
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 🔑 TMDB API Key Configuration

The app uses **The Movie Database (TMDB) API** to fetch movie posters, release years, and vote ratings. A default fallback key is included, but you can configure your own free key:

1. Create a free account at [themoviedb.org](https://www.themoviedb.org/).
2. Navigate to **Settings > API** and generate an API Key (v3 auth).
3. Supply the key in any of the following three ways:
   - **Method A (UI Sidebar):** Enter it directly into the "TMDB API Key" field in the sidebar.
   - **Method B (Environment Variable):**
     ```bash
     export TMDB_API_KEY="your_api_key_here"
     streamlit run app.py
     ```
   - **Method C (Streamlit Secrets):**
     Create `.streamlit/secrets.toml`:
     ```toml
     TMDB_API_KEY = "your_api_key_here"
     ```

---

## 🌐 Deployment Guide

### Option A: Streamlit Community Cloud (Recommended)
1. Fork or push this repository to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io/).
3. Click **"New app"**, select your repository, branch (`main`), and set **Main file path** to `app.py`.
4. (Optional) In **Advanced settings > Secrets**, add:
   ```toml
   TMDB_API_KEY = "your_tmdb_api_key"
   ```
5. Click **Deploy!**

### Option B: Docker Container
Build and run the app with Docker:
```bash
# Build the Docker image
docker build -t cinematch .

# Run the container
docker run -p 8501:8501 -e TMDB_API_KEY="your_key" cinematch
```
Access the application at `http://localhost:8501`.

### Option C: Render / Heroku (via Procfile)
- **Render:** Create a new **Web Service**, connect your GitHub repository, choose runtime **Python 3**, and set the build command to `pip install -r requirements.txt` and start command to `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`.
- The repository includes `Procfile` and `setup.sh` preconfigured for immediate deployment.

---

## 📓 Jupyter Notebook Walkthrough

The notebook [`jupyter_file_01.ipynb`](jupyter_file_01.ipynb) has been formatted into 13 educational sections:

| Section | Topic | Description |
|---|---|---|
| **1** | Project Overview | Problem statement and content-based vs collaborative filtering |
| **2** | Environment Setup | Library imports (`pandas`, `numpy`, `nltk`, `scikit-learn`) |
| **3** | Data Ingestion | Loading raw movies and credits CSVs and checking shapes |
| **4** | Merging & Feature Selection | Merging on `title` and selecting qualitative metadata |
| **5** | Data Cleaning | Dropping missing values and verifying duplicates |
| **6** | JSON Parsing | Extracting genres, keywords, top 3 cast members, and director |
| **7** | Entity Normalization | Removing whitespace in names to prevent false token overlap |
| **8** | Porter Stemming | Reducing words to root stems with NLTK |
| **9** | Vectorization | Transforming text into a 5,000-D Bag-of-Words space |
| **10** | Cosine Similarity | Calculating pairwise cosine similarity matrix |
| **11** | Recommendation Engine | Sorting and returning top-N recommendations with test cases |
| **12** | Model Serialization | Exporting `movies.pkl`, `movie_dict.pkl`, `similarity.pkl.gz` |
| **13** | Summary & Deployment | Instructions for web app launch |

---

## 🛠️ Troubleshooting & FAQ

#### Q: Why is `similarity.pkl.gz` used instead of `similarity.pkl`?
> The raw float32 similarity matrix is ~88 MB (and float64 is ~176 MB). GitHub restricts files over 100 MB and warns on files over 50 MB. By compressing the matrix with `gzip`, the file size is reduced to **36.5 MB**, which pushes effortlessly to Git and loads in less than 0.5 seconds via `gzip.open()`.

#### Q: What if movie posters fail to load?
> If TMDB is unreachable or you are offline, CineMatch automatically displays an elegant fallback cinema image without raising errors.

#### Q: How can I change the number of recommendations?
> Use the slider in the sidebar of the web app (supports 3 to 10 recommendations), or pass `top_n` to `recommend()` in Python or Jupyter.

---

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).

---

Made with ❤️ for movie lovers and data science enthusiasts.
