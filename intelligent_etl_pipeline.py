import os
import sqlite3
import logging
import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ==========================================
# 1. LOGGING & CONFIGURATION
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("IntelligentETL")

DB_FILE = "intelligent_warehouse.db"
CSV_SOURCE = "raw_customer_feedback.csv"


# ==========================================
# 2. MOCK DATA GENERATOR (SELF-CONTAINED)
# ==========================================
def generate_mock_raw_data(filename: str) -> None:
    """Generates a raw dataset simulating dirty incoming customer data."""
    logger.info(f"Generating realistic raw data at '{filename}'...")
    
    raw_data = {
        "customer_id": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        "customer_name": ["Alice Smith", "Bob Jones", "Charlie Brown", "Diana Prince", "Evan Wright", 
                         "Fiona Gallagher", "George Clark", "Hannah Abbott", "Ian Malcolm", "Julia Roberts"],
        "monthly_spend": [120.50, np.nan, 340.00, 15.00, 15000.00, 89.90, 210.00, np.nan, 450.00, 110.00],  # Outlier & NaN
        "support_tickets": [1, 5, 0, 12, 1, 8, 2, 0, 15, 3],
        "tenure_months": [24, 6, 36, 2, 48, 1, 18, 12, 3, 30],
        "feedback_text": [
            "Great service, very happy with the quality!",
            "Terrible experience, app keeps crashing and support is slow.",
            "Average service. Works fine most days.",
            "Horrible customer care, I want an immediate refund!",
            "Loved the platform, seamless integration and fast support.",
            "Frustrating interface, broken features everywhere.",
            "It is okay, nothing special but gets the job done.",
            "Superb tool, increased our business productivity by 20%!",
            "Constant downtime and errors, completely unusable.",
            "Good overall value for money."
        ],
        "signup_date": ["2024-01-15", "2024-06-20", "2023-03-10", "2024-11-01", "2022-05-18",
                        "2024-12-01", "2023-09-14", "2024-02-28", "2024-10-10", "2023-11-20"]
    }
    
    df = pd.DataFrame(raw_data)
    df.to_csv(filename, index=False)
    logger.info("Raw CSV data generated successfully.")


# ==========================================
# 3. AI COMPONENT MODULES
# ==========================================
class AIIntelligenceEngine:
    """Encapsulates machine learning models for anomaly detection and text analysis."""
    
    def __init__(self):
        # Mini sentiment classifier training (In-Memory Supervised Model)
        self.vectorizer = TfidfVectorizer()
        self.sentiment_model = MultinomialNB()
        self._train_mini_sentiment_model()
        
    def _train_mini_sentiment_model(self):
        """Train a lightweight Naive Bayes model on initial seed text."""
        training_corpus = [
            "excellent great amazing happy love fast clean solid",
            "wonderful superb fantastic easy awesome efficient",
            "bad terrible horrible crash slow refund poor unusable",
            "frustrating broken error outage downtime hate annoying"
        ]
        labels = ["POSITIVE", "POSITIVE", "NEGATIVE", "NEGATIVE"]
        
        X = self.vectorizer.fit_transform(training_corpus)
        self.sentiment_model.fit(X, labels)

    def predict_sentiment(self, text: str) -> str:
        """Predicts positive/negative sentiment for feedback text."""
        if not text or pd.isna(text):
            return "NEUTRAL"
        X_test = self.vectorizer.transform([text])
        return self.sentiment_model.predict(X_test)[0]

    @staticmethod
    def detect_anomalies(df: pd.DataFrame, feature_col: str) -> pd.Series:
        """Uses Isolation Forest to detect statistical anomalies/outliers."""
        iso = IsolationForest(contamination=0.1, random_state=42)
        
        # Fill temp values for model input
        filled_vals = df[[feature_col]].fillna(df[feature_col].median())
        preds = iso.fit_predict(filled_vals)
        
        # Isolation forest outputs -1 for anomalies, 1 for normal values
        return pd.Series(preds == -1, index=df.index)

    @staticmethod
    def calculate_churn_risk(row: pd.Series) -> float:
        """AI scoring rule: combines sentiment, support tickets, and spend."""
        score = 0.2  # Base risk
        
        if row.get("sentiment") == "NEGATIVE":
            score += 0.4
        if row.get("support_tickets", 0) > 5:
            score += 0.3
        if row.get("tenure_months", 0) < 6:
            score += 0.1
            
        return round(min(score, 1.0), 2)


# ==========================================
# 4. ETL PIPELINE CORE ENGINE
# ==========================================
class IntelligentETLPipeline:
    """Manages Extract, AI-Transform, and Load stages."""
    
    def __init__(self, source_csv: str, db_file: str):
        self.source_csv = source_csv
        self.db_file = db_file
        self.ai = AIIntelligenceEngine()
        self.extracted_df = None
        self.transformed_df = None

    # --- STAGE 1: EXTRACT ---
    def extract(self) -> "IntelligentETLPipeline":
        """Extracts raw data from CSV or fallback source."""
        logger.info("--- [STAGE 1: EXTRACT] Reading source data ---")
        if not os.path.exists(self.source_csv):
            generate_mock_raw_data(self.source_csv)
            
        self.extracted_df = pd.read_csv(self.source_csv)
        logger.info(f"Extracted {len(self.extracted_df)} records from '{self.source_csv}'.")
        return self

    # --- STAGE 2: TRANSFORM (AI-ENRICHED) ---
    def transform(self) -> "IntelligentETLPipeline":
        """Cleans, imputes, enriches, and runs AI inferences on the data."""
        logger.info("--- [STAGE 2: TRANSFORM] Processing & AI Enrichment ---")
        df = self.extracted_df.copy()

        # 1. Smart Missing Value Imputation
        logger.info("Performing smart imputation for missing values...")
        median_spend = df["monthly_spend"].median()
        df["monthly_spend_imputed"] = df["monthly_spend"].isna()
        df["monthly_spend"] = df["monthly_spend"].fillna(median_spend)

        # 2. AI Anomaly Detection (Outlier Detection)
        logger.info("Running Isolation Forest for financial anomaly detection...")
        df["is_spend_anomaly"] = self.ai.detect_anomalies(df, "monthly_spend")

        # 3. AI NLP Sentiment Analysis
        logger.info("Executing NLP Sentiment Analysis on customer feedback...")
        df["sentiment"] = df["feedback_text"].apply(self.ai.predict_sentiment)

        # 4. AI Predictive Churn Risk Scoring
        logger.info("Calculating AI Customer Churn Risk scores...")
        df["churn_risk_score"] = df.apply(self.ai.calculate_churn_risk, axis=1)

        # 5. Pipeline Audit Metadata
        df["processed_at"] = datetime.datetime.now().isoformat()

        self.transformed_df = df
        logger.info("Transformation phase completed successfully.")
        return self

    # --- STAGE 3: LOAD ---
    def load(self) -> "IntelligentETLPipeline":
        """Loads transformed data and aggregated views into SQLite warehouse."""
        logger.info("--- [STAGE 3: LOAD] Writing into SQLite Data Warehouse ---")
        
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()

        # Load raw transformed records
        self.transformed_df.to_sql("dim_customers_enriched", conn, if_exists="replace", index=False)
        
        # Build Analytical Aggregated Table
        high_risk_df = self.transformed_df[self.transformed_df["churn_risk_score"] >= 0.6]
        high_risk_df.to_sql("fact_high_risk_churn", conn, if_exists="replace", index=False)

        # Create Index for optimized database queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_customer_id ON dim_customers_enriched(customer_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_churn_risk ON dim_customers_enriched(churn_risk_score);")
        
        conn.commit()
        conn.close()
        logger.info(f"Loaded records into SQLite DB '{self.db_file}' successfully.")
        return self

    # --- WAREHOUSE QUERY & REPORTING ---
    def generate_report(self):
        """Runs analytics queries directly on SQLite data warehouse."""
        logger.info("--- [POST-ETL ANALYTICS REPORT] ---")
        conn = sqlite3.connect(self.db_file)
        
        print("\n" + "="*60)
        print("          INTELLIGENT ETL PIPELINE REPORT           ")
        print("="*60)
        
        print("\n--- 1. High Churn Risk Customers (AI Enriched) ---")
        query_churn = """
            SELECT customer_id, customer_name, monthly_spend, support_tickets, sentiment, churn_risk_score 
            FROM dim_customers_enriched 
            WHERE churn_risk_score >= 0.5
            ORDER BY churn_risk_score DESC;
        """
        high_risk = pd.read_sql_query(query_churn, conn)
        print(high_risk.to_string(index=False))

        print("\n--- 2. Sentiment Breakdown Summary ---")
        query_sentiment = """
            SELECT sentiment, COUNT(*) as count, ROUND(AVG(monthly_spend), 2) as avg_spend
            FROM dim_customers_enriched
            GROUP BY sentiment;
        """
        sentiment_summary = pd.read_sql_query(query_sentiment, conn)
        print(sentiment_summary.to_string(index=False))

        print("\n--- 3. Anomaly Alerts ---")
        query_anomaly = """
            SELECT customer_id, customer_name, monthly_spend 
            FROM dim_customers_enriched 
            WHERE is_spend_anomaly = 1;
        """
        anomalies = pd.read_sql_query(query_anomaly, conn)
        print(anomalies.to_string(index=False))

        print("="*60 + "\n")
        conn.close()


# ==========================================
# 5. ENTRY POINT EXECUTABLE
# ==========================================
if __name__ == "__main__":
    logger.info("Initializing Intelligent ETL Pipeline Execution...")
    
    # Run the ETL Pipeline
    pipeline = IntelligentETLPipeline(source_csv=CSV_SOURCE, db_file=DB_FILE)
    
    pipeline.extract()\
            .transform()\
            .load()\
            .generate_report()

    logger.info("Intelligent ETL Pipeline executed successfully without errors.")
