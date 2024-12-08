import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommendationEngine:
    def __init__(self):
        """
        Initialize the recommendation engine.
        """
        self.products_df = None
        self.tfidf_matrix = None
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess product descriptions.
        """
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # Remove special characters and digits
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
        return text

    def fit(self, data: pd.DataFrame) -> None:
        """
        Train the recommendation engine on product descriptions.

        Args:
            data: DataFrame containing 'StockCode', 'Description', and 'UnitPrice'.
        """
        try:
            logger.info("Training recommendation engine...")

            # Preprocess descriptions
            data['ProcessedDescription'] = data['Description'].apply(self.preprocess_text)
            self.products_df = data.drop_duplicates(subset=['StockCode']).dropna(subset=['ProcessedDescription'])

            # Compute TF-IDF matrix
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.products_df['ProcessedDescription'])
            logger.info(f"Training completed. Processed {len(self.products_df)} products.")
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

    def get_recommendations(self, product_ids: List[str], n_recommendations: int = 5) -> pd.DataFrame:
        """
        Get product recommendations based on similar descriptions.

        Args:
            product_ids: List of product StockCodes for which to generate recommendations.
            n_recommendations: Number of recommendations to return.
        """
        try:
            if not product_ids:
                return pd.DataFrame()

            # Find indices of input products
            input_indices = []
            for pid in product_ids:
                matches = self.products_df[self.products_df['StockCode'] == pid].index
                if len(matches) > 0:
                    input_indices.append(matches[0])

            if not input_indices:
                return pd.DataFrame()

            # Calculate similarity scores for the input products
            input_vectors = self.tfidf_matrix[input_indices]
            similarity_scores = cosine_similarity(input_vectors, self.tfidf_matrix).mean(axis=0)

            # Add similarity scores to the DataFrame
            recommendations = self.products_df.copy()
            recommendations['similarity_score'] = similarity_scores

            # Exclude input products and sort by similarity
            recommendations = recommendations[~recommendations['StockCode'].isin(product_ids)]
            recommendations = recommendations.sort_values(by='similarity_score', ascending=False).head(
                n_recommendations)

            return recommendations[['StockCode', 'Description', 'UnitPrice', 'similarity_score']]
        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            raise
