# Intelligent-ETL-Pipeline
**AI-powered ETL pipeline** that automates data extraction, cleansing, enrichment, and loading into a SQLite warehouse. Integrates Isolation Forest anomaly detection, NLP sentiment analysis, and churn risk scoring to deliver analytics-ready customer insights with optimized reporting and indexed queries.


# Intelligent AI-Powered ETL Pipeline

A production-style **Intelligent ETL (Extract, Transform, Load) Pipeline** built with **Python**, **Pandas**, **Scikit-learn**, and **SQLite**. This project demonstrates how traditional ETL workflows can be enhanced with Artificial Intelligence by integrating anomaly detection, sentiment analysis, and predictive churn risk scoring into the data transformation process.

The pipeline automatically generates sample customer feedback data (if none exists), cleans and enriches the dataset using machine learning, stores the processed data in a SQLite data warehouse, and generates analytical reports.

## Features

* Automated ETL workflow
* Self-generated mock dataset for quick testing
* Missing value imputation
* AI-powered anomaly detection using Isolation Forest
* Natural Language Processing (NLP) sentiment analysis
* Predictive customer churn risk scoring
* SQLite data warehouse integration
* Optimized database indexing
* Automated analytics reporting
* Modular and object-oriented architecture
* Comprehensive logging throughout execution

## Project Structure

```text
.
├── intelligent_etl_pipeline.py
├── requirements.txt
├── raw_customer_feedback.csv        # Generated automatically
├── intelligent_warehouse.db         # Generated automatically
└── README.md
```

## Technologies Used

* Python 3.x
* Pandas
* NumPy
* Scikit-learn
* SQLite
* Logging

## AI Components

### 1. Sentiment Analysis

A lightweight Naive Bayes classifier is trained in memory using TF-IDF vectorization to classify customer feedback as:

* Positive
* Negative
* Neutral

### 2. Anomaly Detection

The pipeline uses the Isolation Forest algorithm to detect unusual customer spending behavior.

### 3. Predictive Churn Risk

A rule-based AI scoring model estimates customer churn risk using:

* Customer sentiment
* Number of support tickets
* Customer tenure

Each customer receives a churn risk score between **0.0** and **1.0**.

## ETL Workflow

### Extract

* Reads customer data from CSV
* Automatically creates mock data if the source file does not exist

### Transform

* Handles missing values
* Detects spending anomalies
* Performs sentiment analysis
* Calculates churn risk scores
* Adds processing metadata

### Load

* Stores enriched customer records into SQLite
* Creates analytical tables
* Builds database indexes for faster queries

### Reporting

After loading, the pipeline generates:

* High-risk customer report
* Sentiment summary
* Spending anomaly report

## Database Tables

### dim_customers_enriched

Contains the fully transformed and AI-enriched customer dataset.

Additional fields include:

* `monthly_spend_imputed`
* `is_spend_anomaly`
* `sentiment`
* `churn_risk_score`
* `processed_at`

### fact_high_risk_churn

Contains customers whose churn risk score is greater than or equal to **0.6**.

## Installation

Clone the repository:

```bash
git clone https://github.com/Abhinav-cloud482/intelligent-ai-etl-pipeline.git

cd intelligent-ai-etl-pipeline
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Requirements

```text
numpy==2.2.6
pandas==2.3.1
scikit-learn==1.7.1
```

## Running the Project

```bash
python intelligent_etl_pipeline.py
```

## Expected Workflow

When executed, the pipeline performs the following sequence:

1. Generates mock customer data (if required)
2. Extracts records from the CSV source
3. Cleans missing values
4. Detects anomalies using Isolation Forest
5. Performs sentiment classification
6. Calculates churn risk scores
7. Loads enriched data into SQLite
8. Creates optimized database indexes
9. Generates analytical reports

## Example Output

```text
Initializing Intelligent ETL Pipeline Execution...

[STAGE 1: EXTRACT]
Extracted 10 records

[STAGE 2: TRANSFORM]
Missing values imputed
Anomaly detection completed
Sentiment analysis completed
Churn risk scoring completed

[STAGE 3: LOAD]
SQLite warehouse updated successfully

POST-ETL ANALYTICS REPORT

High Churn Risk Customers
Sentiment Breakdown
Anomaly Alerts
```


## Screnshots

### Output :-

<img width="739" height="190" alt="output_1" src="https://github.com/user-attachments/assets/57fb8e75-8d5e-4b7d-8bb7-bc4637e407d1" />


## Machine Learning Models

| Component          | Algorithm                        |
| ------------------ | -------------------------------- |
| Sentiment Analysis | TF-IDF + Multinomial Naive Bayes |
| Anomaly Detection  | Isolation Forest                 |
| Churn Prediction   | Rule-Based AI Scoring            |

## Learning Objectives

This project demonstrates:

* End-to-end ETL pipeline development
* Data cleaning and preprocessing
* AI integration within ETL workflows
* Feature engineering
* Machine learning inference
* SQLite data warehousing
* Analytical reporting
* Python object-oriented design
* Logging and monitoring

## Future Improvements

* REST API integration
* Incremental ETL processing
* Scheduled pipeline execution
* Dashboard visualization with Streamlit
* Real-world sentiment model
* Advanced churn prediction using supervised learning
* Cloud database support
* Docker containerization
* Apache Airflow orchestration
* Unit and integration testing

## License

This project is released under the MIT License.

## Author

Developed as an AI-powered ETL pipeline project demonstrating intelligent data engineering concepts using Python and machine learning.
