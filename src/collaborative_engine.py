import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds

class CollaborativeRecommender:
    """
    Latent Factor Matrix Factorization Recommender using Truncated SVD (Singular Value Decomposition).
    Learns hidden user preferences and item affinities.
    """
    def __init__(self, ratings_path="data/ratings.csv", movies_path="data/movies.csv", k_factors=4):
        self.ratings_df = pd.read_csv(ratings_path)
        self.movies_df = pd.read_csv(movies_path)
        self.k_factors = k_factors
        self._train_svd()

    def _train_svd(self):
        # Create User-Item Matrix
        self.user_item_matrix = self.ratings_df.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
        self.user_ids = list(self.user_item_matrix.index)
        self.movie_ids = list(self.user_item_matrix.columns)
        
        matrix_values = self.user_item_matrix.values
        # User mean normalization (de-biasing)
        self.user_ratings_mean = np.mean(matrix_values, axis=1)
        matrix_demeaned = matrix_values - self.user_ratings_mean.reshape(-1, 1)

        # Decompose using SVD (U, Sigma, Vt)
        k = min(self.k_factors, min(matrix_demeaned.shape) - 1)
        U, sigma, Vt = svds(matrix_demeaned, k=k)
        sigma_diag = np.diag(sigma)

        # Reconstructed Predicted Ratings
        self.predicted_ratings = np.dot(np.dot(U, sigma_diag), Vt) + self.user_ratings_mean.reshape(-1, 1)
        self.preds_df = pd.DataFrame(self.predicted_ratings, index=self.user_ids, columns=self.movie_ids)

    def recommend_for_user(self, user_id, top_n=5):
        if user_id not in self.preds_df.index:
            return pd.DataFrame()

        user_preds = self.preds_df.loc[user_id]
        rated_movies = self.ratings_df[self.ratings_df['user_id'] == user_id]['movie_id'].values

        # Filter out already rated movies and sort by predicted rating
        unrated_preds = user_preds.drop(index=rated_movies, errors='ignore').sort_values(ascending=False)
        top_movie_ids = unrated_preds.head(top_n).index

        recommendations = self.movies_df[self.movies_df['movie_id'].isin(top_movie_ids)].copy()
        recommendations['predicted_rating'] = recommendations['movie_id'].map(unrated_preds).round(2)
        return recommendations[['movie_id', 'title', 'genres', 'predicted_rating']].sort_values(by='predicted_rating', ascending=False)