# 🎬 Cinematch AI — Advanced Hybrid Movie Recommendation Engine

An end-to-end, production-oriented Movie Recommendation System implementing **Content-Based Filtering (TF-IDF + Metadata Soup)**, **Collaborative Filtering (Latent Factor Matrix Factorization via Truncated SVD)**, and a **Bayesian IMDB Weighted Hybrid Architecture**.

---

## 🌟 Resume Highlights & Key Technical Implementations
- **Hybrid Architecture:** Blends Content Similarity and User-Item Latent vectors using a configurable weighting parameter ($\\alpha$).
- **Cold-Start Problem Resolution:** Gracefully falls back to Bayesian Average weighted ranking (IMDb formula) for unseen users or newly added items.
- **Dimensionality Reduction (SVD):** Implements low-rank Matrix Factorization with mean de-biasing using `scipy.sparse.linalg.svds`.
- **Feature Engineering ("Metadata Soup"):** Extracts weighted tokens across genres, keywords, top cast, and directors to generate rich vector embeddings using TF-IDF.
- **Interactive UI:** Built a full dashboard using **Streamlit** featuring real-time similarity metrics and user profiling.

---

## 🏗️ Project Architecture
```text
movie_recommender_system/
├── data/
│   ├── movies.csv             # Movie metadata (genres, keywords, cast, votes)
│   └── ratings.csv            # User interaction matrix
├── src/
│   ├── __init__.py
│   ├── content_engine.py      # TF-IDF + Cosine Similarity engine
│   ├── collaborative_engine.py# SVD Matrix Factorization engine
│   └── hybrid_engine.py       # Multi-model weighted combiner & cold-start logic
├── app.py                     # Interactive Streamlit Web Application
├── requirements.txt           # Project dependencies
└── README.md                  # Comprehensive documentation
```

---

## 🚀 Quickstart Guide

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/your-username/cinematch-ai.git
cd movie_recommender_system

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Interactive Web App
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📊 Resume Bullet Points (Ready to Copy-Paste)
- *Developed a Hybrid Movie Recommendation System combining Content-Based TF-IDF vectorization and Collaborative Filtering using Truncated SVD matrix factorization.*
- *Engineered a feature-weighted 'metadata soup' incorporating directors, cast, and sub-genres, boosting semantic matching accuracy.*
- *Implemented Bayesian weighted rating algorithms to mitigate the cold-start challenge for new users and sparse rating matrices.*
- *Designed an interactive Streamlit analytics dashboard enabling real-time multi-model comparison and user personalization.*