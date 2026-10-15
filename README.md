# Customer Segmentation System

An interactive machine learning application that segments customers using Recency, Frequency, and Monetary (RFM) analysis and K-means clustering.

## Project Overview

Businesses need to understand customer purchasing behavior to improve retention, personalize marketing, and allocate resources effectively.

This project transforms historical retail transaction data into actionable customer segments through RFM feature engineering, machine learning, and an interactive Streamlit dashboard.

## Business Objectives

- Identify customers with high purchasing value and loyalty.
- Identify inactive customers who may benefit from re-engagement campaigns.
- Understand customer purchasing frequency and spending behavior.
- Provide practical marketing recommendations for different customer segments.

## Technology Stack

- **Python** — data processing and machine learning
- **Pandas and NumPy** — data manipulation and numerical computing
- **Scikit-learn** — feature scaling and K-means clustering
- **Plotly** — interactive visualizations
- **Streamlit** — interactive web application
- **Joblib** — model artifact persistence

## Methodology

1. Clean and prepare retail transaction data.
2. Calculate RFM features for each customer.
3. Apply logarithmic transformations to reduce skewness.
4. Standardize the features using `StandardScaler`.
5. Evaluate candidate cluster counts using clustering diagnostics.
6. Train a K-means model with four clusters.
7. Profile each cluster and interpret its business characteristics.
8. Build an interactive dashboard for customer predictions and segment analysis.

## Customer Segments

| Cluster | Segment | Customer Count | Revenue Share |
|---|---|---:|---:|
| 0 | High-Value, Loyal Customers | 713 | 64.89% |
| 1 | Lapsed, Low-Value Customers | 1,622 | 6.22% |
| 2 | Recent, Occasional Customers | 837 | 5.25% |
| 3 | Established, Moderate-Value Customers | 1,166 | 23.64% |

The four segments contain 4,338 customers and account for approximately £8.89 million in historical revenue.

## Key Findings

- High-Value, Loyal Customers account for approximately 64.89% of historical revenue.
- Lapsed, Low-Value Customers represent the largest customer segment by count.
- Established, Moderate-Value Customers contribute approximately 23.64% of historical revenue.
- An outlier sensitivity analysis produced an Adjusted Rand Index of 0.9465 between the original and capped-spending cluster assignments, indicating strong agreement between the two segmentations.

## Dashboard Features

- Historical customer and revenue overview.
- Interactive customer-count and revenue-contribution charts.
- Segment filtering and performance statistics.
- Customer segment prediction from RFM inputs.
- Warnings for inputs outside the observed training-data ranges.
- Business recommendations for each predicted segment.
- CSV downloads for segment reports and individual predictions.

## Installation

Clone the repository:

```bash
git clone https://github.com/kayDtechie3/customer-segmentation-system.git
cd customer-segmentation-system
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the application:

```bash
python -m streamlit run app.py
```

The application should be available at `http://localhost:8501`.

## Project Files

- `app.py` — Streamlit dashboard and prediction logic
- `customer_segmentation_model.pkl` — trained model and scaler
- `segment_summary.csv` — historical segment statistics
- `requirements.txt` — required Python packages

## Limitations

- The model assigns customers to learned clusters; it does not predict future revenue or guarantee future customer behavior.
- Segment recommendations are suggestions that should be tested against actual campaign outcomes.
- Predictions outside the observed training-data ranges require additional caution.
- Historical customer segments may change as new transaction data becomes available.

## Future Improvements

- Add automated data ingestion and RFM recalculation.
- Monitor segment stability as new data arrives.
- Evaluate alternative clustering approaches.
- Track the results of targeted marketing campaigns.
- Add automated model retraining when appropriate.

## Author

**Kingsley**

Data Scientist | Python | Machine Learning | Data Visualization
