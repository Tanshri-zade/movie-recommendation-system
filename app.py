import streamlit as st
import pandas as pd
from src.hybrid_engine import HybridRecommender

st.set_page_config(
    page_title="Cinematch AI | Advanced Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1e2130;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        border-left: 5px solid #ff4b4b;
    }
    .stBadge {
        background-color: #2b313e;
        color: #fff;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_engine():
    return HybridRecommender()

engine = load_engine()

st.title("🎬 Cinematch AI — Advanced Recommendation System")
st.caption("⚡ Powered by TF-IDF Content Similarity, Truncated SVD Matrix Factorization & Bayesian Weighted Scoring")

# Sidebar Controls
st.sidebar.header("⚙️ Recommendation Settings")
rec_mode = st.sidebar.radio(
    "Choose Recommendation Engine:",
    ("Hybrid (Content + SVD Latent)", "Pure Content-Based (Metadata Soup)", "Pure Collaborative (User SVD)")
)

top_n = st.sidebar.slider("Number of Recommendations:", min_value=3, max_value=8, value=5)

all_titles = sorted(engine.movies_df['title'].unique())
selected_movie = st.sidebar.selectbox("Select Target Movie:", all_titles, index=all_titles.index("Inception") if "Inception" in all_titles else 0)

available_users = ["None (Cold Start / Guest)"] + [f"User {u}" for u in engine.collab_engine.user_ids]
selected_user_str = st.sidebar.selectbox("Select User Profile for Personalization:", available_users)
user_id = int(selected_user_str.replace("User ", "")) if selected_user_str != "None (Cold Start / Guest)" else None

# Overview of Target Movie
movie_meta = engine.movies_df[engine.movies_df['title'] == selected_movie].iloc[0]
col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
with col1:
    st.subheader(f"🎯 Target Movie: {movie_meta['title']}")
    st.write(f"**Genres:** {movie_meta['genres']} | **Director:** {movie_meta['director']}")
with col2:
    st.metric("IMDb Rating", f"⭐ {movie_meta['vote_average']}")
with col3:
    st.metric("Vote Count", f"{movie_meta['vote_count']:,}")
with col4:
    st.metric("Bayesian Score", f"🏆 {movie_meta['weighted_score']}")

st.divider()

# Output Results
if rec_mode == "Pure Content-Based (Metadata Soup)":
    st.subheader(f"Top {top_n} Recommendations based on Genres, Director & Keywords")
    recs = engine.content_engine.recommend(selected_movie, top_n=top_n)
    
    if not recs.empty:
        for idx, row in recs.iterrows():
            with st.container():
                st.markdown(f"""
                <div class="metric-card">
                    <h4>#{idx+1} {row['title']} <span style="float:right; font-size: 0.9rem; color: #ff7675;">Similarity: {row['score']*100:.1f}%</span></h4>
                    <p><b>Genres:</b> {row['genres']} | <b>Director:</b> {row['director']}</p>
                </div>
                """, unsafe_allow_html=True)

elif rec_mode == "Pure Collaborative (User SVD)":
    st.subheader("Matrix Factorization (Truncated SVD) Predictions")
    if user_id is None:
        st.warning("⚠️ Please select an existing User Profile from the sidebar to view personalized Collaborative predictions.")
    else:
        user_recs = engine.collab_engine.recommend_for_user(user_id, top_n=top_n)
        st.write(f"Showing predicted top recommendations based on **User {user_id}'s** latent ratings:")
        st.dataframe(user_recs, use_container_width=True)

else:  # Hybrid
    st.subheader(f"✨ Hybrid Recommendations for '{selected_movie}' (Personalized for {selected_user_str})")
    recs = engine.recommend(selected_movie, user_id=user_id, top_n=top_n)
    
    cols = st.columns(len(recs))
    for i, (_, row) in enumerate(recs.iterrows()):
        with cols[i]:
            st.markdown(f"### #{i+1}")
            st.markdown(f"**{row['title']}**")
            st.caption(f"🎭 {row['genres']}")
            st.info(f"Hybrid Match: **{row['hybrid_score']*100:.1f}%**")
            st.write(f"Director: *{row['director']}*")
            st.write(f"IMDb Weighted: ⭐ {row['weighted_score']}")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Resume Highlight:** Implements Cold-Start Handling, Truncated SVD, TF-IDF Vectorization, and Bayesian Average ranking.")