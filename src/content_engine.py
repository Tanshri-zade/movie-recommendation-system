import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    """
    Content-Based Recommendation Engine using Feature Soup
    (Genres, Keywords, Cast, and Director) with TF-IDF Vectorization & Cosine Similarity.
    """
    def __init__(self, movies_path="data/movies.csv"):
        self.movies_df = pd.read_csv(movies_path)
        self._prepare_metadata_soup()
        self._compute_similarity()

    def _clean_str(self, text):
        if pd.isna(text):
            return ""
        return str(text).lower().replace(",", " ").replace("-", "")

    def _prepare_metadata_soup(self):
        # Weighted Metadata Soup: Emphasizing genres, director, and keywords
        soup = []
        for _, row in self.movies_df.iterrows():
            genres = self._clean_str(row.get('genres', ''))
            keywords = self._clean_str(row.get('keywords', ''))
            director = self._clean_str(row.get('director', '')).replace(" ", "")  # join name e.g. christophernolan
            cast = " ".join([actor.strip().lower().replace(" ", "") for actor in str(row.get('cast', '')).split(",")[:3]])
            
            # Combine into rich semantic text representation
            combined = f"{genres} {genres} {director} {director} {keywords} {cast}"
            soup.append(combined)

        self.movies_df['soup'] = soup

    def _compute_similarity(self):
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(self.movies_df['soup'])
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

    def recommend(self, title, top_n=5):
        matches = self.movies_df[self.movies_df['title'].str.lower() == title.lower()]
        if matches.empty:
            return pd.DataFrame()

        idx = matches.index[0]
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_indices = [(i, score) for i, score in sim_scores if i != idx][:top_n]

        results = []
        for i, score in top_indices:
            row = self.movies_df.iloc[i]
            results.append({
                "movie_id": row['movie_id'],
                "title": row['title'],
                "genres": row['genres'],
                "director": row['director'],
                "score": round(float(score), 3)
            })

        return pd.DataFrame(results)