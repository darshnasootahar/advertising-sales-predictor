# advertising-sales-predictor
Interactive Streamlit dashboard that predicts sales from TV, Radio &amp; Newspaper ad spend using Polynomial Regression (degree 2). Features live sliders, diminishing-returns curves, residual analysis, cross-validation, and feature importance charts.
Live Demo

Run locally — see Getting Started below.


📌 Project Overview
This project builds a machine learning-powered web app that helps marketers and analysts understand how advertising spend across three channels — TV, Radio, and Newspaper — drives product sales.
The model captures non-linear (diminishing) returns on ad spend using Polynomial Regression (Degree 2), which outperforms simple linear regression on this dataset. The entire workflow — from model training to live prediction — runs inside a sleek Streamlit dashboard with interactive controls.

 Features
FeatureDescription🎛️ Live budget slidersAdjust TV, Radio & Newspaper spend in real time📊 Instant sales predictionPredicted sales update dynamically as sliders move📉 Diminishing returns curvesVisual curves showing ROI drop-off per channel🎯 Actual vs Predicted chartScatter plot showing model accuracy on test data🔬 Feature importanceTop 12 polynomial feature coefficients ranked📐 Residual analysisResiduals vs Fitted + distribution histogram (optional)🔁 5-Fold cross-validationCV R² scores per fold with bar chart (optional)⚠️ Budget health badgesWarns when spend exceeds the 75th percentile threshold📋 Raw dataset viewerToggle to inspect all 200 observations with color gradients

🤖 Model Details
Algorithm — Polynomial Regression (Degree 2)
Simple linear regression assumes a straight-line relationship between ad spend and sales. In reality, each extra dollar spent returns less than the previous one — this is diminishing returns. Polynomial Regression captures this curve by adding squared and interaction terms.
Features engineered by PolynomialFeatures (degree=2):
TV, Radio, Newspaper,
TV², Radio², Newspaper²,
TV×Radio, TV×Newspaper, Radio×Newspaper
Training Setup
ParameterValueDatasetAdvertising dataset (200 observations)Train / Test split80% / 20% (random_state=42)Polynomial degree2Cross-validation5-Fold CV on full datasetScoring metricR² (coefficient of determination)
Model Performance
MetricValueR² (test set)~0.98RMSE~0.60K unitsMAE~0.45K unitsCV R² (mean ± std)~0.97 ± 0.01

The model explains ~98% of variance in sales — a strong fit for this dataset.


📁 Project Structure
ad-budget-sales-predictor/
│
├── app.py                  # Main Streamlit application
├── advertising.csv         # Dataset (200 rows, 4 columns)
└── README.md

 Dataset

Source: Classic Advertising dataset (widely used in ML education)
Rows: 200 market observations
Columns: 4

ColumnTypeDescriptionTVFloatTV advertising spend ($K) — range: 0.7 to 296.4RadioFloatRadio advertising spend ($K) — range: 0 to 49.6NewspaperFloatNewspaper advertising spend ($K) — range: 0.3 to 114SalesFloatProduct sales in thousands of units
Key Insight from EDA

TV has the strongest positive correlation with sales
Radio shows a moderate positive effect
Newspaper shows weak correlation — often flagged as low ROI
TV × Radio interaction is the most impactful polynomial feature


 Getting Started
Prerequisites

Python 3.8 or higher
pip

Installation
bash# 1. Clone the repository
git clone https://github.com/your-username/ad-budget-sales-predictor.git
cd ad-budget-sales-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
The app will open automatically in your browser at http://localhost:8501

📦 Requirements
streamlit
pandas
numpy
plotly
scikit-learn
Install all at once:
bashpip install streamlit pandas numpy plotly scikit-learn

 How to Use the App

Open the sidebar — use the three sliders to set your advertising budget for TV, Radio, and Newspaper (in $K)
Read the prediction — the green box shows predicted sales in thousands of units
Check the badges — green badges mean your spend is in the high-ROI zone; yellow badges warn of diminishing returns
Study the charts:

Diminishing returns curves show how each channel's ROI flattens at high spend
Actual vs Predicted confirms model accuracy
Feature importance shows which polynomial terms matter most


Toggle optional views in the sidebar — residual plot, CV details, and raw dataset


 Understanding Diminishing Returns
The dashed orange line on each curve marks the 75th percentile threshold for that channel:
Channel75th Percentile ThresholdMeaningTV~$218KSpend above this shows flattening returnsRadio~$36.5KEfficient up to this pointNewspaper~$45KWeak ROI throughout — use cautiously
Spending beyond these thresholds is not necessarily wrong — but the ROI per dollar drops significantly.

 What I Learned

How to build and deploy an ML model inside a Streamlit web app
Why Polynomial Regression outperforms Linear Regression for non-linear data
How to interpret model metrics: R², RMSE, MAE, and cross-validation scores
How to visualize diminishing returns and feature importances interactively
Streamlit caching with @st.cache_resource for efficient model loading


 Future Improvements

Add budget optimization — auto-suggest the spend mix that maximizes predicted sales
Upload custom CSV to train the model on new data
Compare Linear vs Polynomial regression side-by-side
Add confidence intervals to predictions
Deploy to Streamlit Cloud for public access
