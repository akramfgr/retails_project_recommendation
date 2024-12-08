import numpy as np
import pandas as pd
import logging
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RFMMetrics:
    def __init__(self, reference_date=None):
        """
        Initialize the RFM Metrics Calculator
        Args:
            reference_date: The date to calculate recency metrics. Defaults to max date in the dataset.
        """
        self.reference_date = reference_date

    def calculate_rfm(self, df, customer_id=None):
        """
        Calculate RFM (Recency, Frequency, Monetary) metrics for the given dataset.

        Args:
            df (pd.DataFrame): DataFrame containing transaction data.
                               Required columns: 'CustomerID', 'InvoiceDate', 'InvoiceNo', 'TotalPrice'.
            customer_id (int, optional): If provided, calculate RFM metrics for a specific customer.

        Returns:
            pd.DataFrame: DataFrame containing RFM metrics for the specified customer or all customers.
        """
        try:
            # Ensure required columns are present
            required_columns = ['CustomerID', 'InvoiceDate', 'InvoiceNo', 'TotalPrice']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Set the reference date
            if self.reference_date is None:
                self.reference_date = df['InvoiceDate'].max()

            logger.info(f"Using reference date: {self.reference_date}")

            # Filter for the specific customer if customer_id is provided
            if customer_id is not None:
                df = df[df['CustomerID'] == customer_id]
                if df.empty:
                    logger.warning(f"No data found for CustomerID: {customer_id}")
                    return pd.DataFrame()  # Return empty DataFrame if no data for customer

            # Calculate RFM metrics
            rfm_data = df.groupby('CustomerID').agg({
                'InvoiceDate': lambda x: (self.reference_date - x.max()).days,  # Recency
                'InvoiceNo': 'count',  # Frequency
                'TotalPrice': 'sum'  # Monetary
            }).reset_index()

            rfm_data.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

            # Apply log transformation to smooth the metrics
            rfm_data['LogRecency'] = np.log1p(rfm_data['Recency'])
            rfm_data['LogFrequency'] = np.log1p(rfm_data['Frequency'])
            rfm_data['LogMonetary'] = np.log1p(rfm_data['Monetary'])

            logger.info(f"RFM metrics calculated for {len(rfm_data)} customers")

            return rfm_data

        except Exception as e:
            logger.error(f"Error calculating RFM metrics: {str(e)}")
            raise

    def predict_customer_profile(self, rfm_data):
        """Predict customer profiles based on RFM metrics using KMeans clustering."""
        # Use only the numeric RFM columns
        features = rfm_data[['Recency', 'Frequency', 'Monetary']]
        kmeans = KMeans(n_clusters=3, random_state=42)
        rfm_data['Cluster'] = kmeans.fit_predict(features)

        # Map clusters to profiles (e.g., 0: Loyal, 1: At Risk, 2: New)
        cluster_labels = {0: "Loyal", 1: "At Risk", 2: "New"}
        rfm_data['Profile'] = rfm_data['Cluster'].map(cluster_labels)
        return rfm_data