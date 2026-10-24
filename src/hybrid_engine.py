import pandas as pd
from src.content_engine import ContentBasedRecommender
from src.collaborative_engine import CollaborativeRecommender

class HybridRecommender:
    """
    Production-ready Hybrid Recommendation Engine:
    Combines Content Similarity, SVD Latent Ratings, and IMDB Weighted Rating (Bayesian Average).
    """
    def __init__(self, movies_path="data/movies.csv", ratings_path="data/ratings.csv"):
        self.movies_df = pd.read_csv(movies_path)
        self.content_engine = ContentBasedRecommender(movies_path)
        self.collab_engine = CollaborativeRecommender(ratings_path, movies_path)
        self._compute_imdb_weighted_ratings()

    def _compute_imdb_weighted_ratings(self):
        # IMDB Formula: WR = (v / (v + m)) * R + (m / (v + m)) * C
        R = self.movies_df['vote_average']
        v = self.movies_df['vote_count']
        m = self.movies_df['vote_count'].quantile(0.60)
        C = self.movies_df['vote_average'].mean()

        self.movies_df['weighted_score'] = (v / (v + m)) * R + (m / (v + m)) * C
        self.movies_df['weighted_score'] = self.movies_df['weighted_score'].round(2)

    def recommend(self, title, user_id=None, alpha=0.6, top_n=5):
        """
        Hybrid Score = alpha * Content_Score + (1 - alpha) * Normalized_SVD_Score
        Falls back to Weighted Average + Content if user has no prior history (Cold-Start).
        """
        content_recs = self.content_engine.recommend(title, top_n=10)
        if content_recs.empty:
            return pd.DataFrame()

        merged = content_recs.merge(self.movies_df[['movie_id', 'weighted_score']], on='movie_id')

        if user_id in self.collab_engine.preds_df.index:
            user_preds = self.collab_engine.preds_df.loc[user_id]
            # Normalize SVD prediction (scale ~ 1-5 to 0-1)
            merged['collab_pred'] = merged['movie_id'].map(user_preds).fillna(3.0)
            merged['norm_collab'] = merged['collab_pred'] / 5.0
            merged['hybrid_score'] = (alpha * merged['score'] + (1 - alpha) * merged['norm_collab']).round(3)
        else:
            # Cold-start fallback: Combine content score with IMDB Bayesian score
            merged['norm_weighted'] = merged['weighted_score'] / 10.0
            merged['hybrid_score'] = (0.7 * merged['score'] + 0.3 * merged['norm_weighted']).round(3)

        return merged.sort_values(by='hybrid_score', ascending=False).head(top_n)