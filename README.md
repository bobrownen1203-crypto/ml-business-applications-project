# Machine Learning Business Applications Project

This project demonstrates practical applications of machine learning in business analytics using real-world datasets (100+ records each).

---

# 📂 Project Structure


---

# ✅ Part 1: House Price Prediction

**Goal:** Predict house sale prices based on square footage and neighborhood.  
**Model:** Linear Regression  
**Dataset Size:** ~1,460 records  

**Dataset Source:**  
Kaggle – Ames Housing Dataset  
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data  

**Key Features Used:**
- GrLivArea (square footage)
- Neighborhood (categorical location variable)

**Outputs:**
- MAE and R² performance metrics
- Predicted price for 2000 sq ft house
- Feature coefficient impact summary

Run:

---

# ✅ Part 2: Customer Churn Prediction

**Goal:** Predict probability that a customer will churn.  
**Model:** Logistic Regression  
**Dataset Size:** 7,000+ customers  

**Dataset Source:**  
Telco Customer Churn (IBM Sample Dataset)  
https://www.kaggle.com/datasets/blastchar/telco-customer-churn  

**Techniques Used:**
- StandardScaler (numerical features)
- OneHotEncoder (categorical features)
- Train/test split with stratification
- Probability threshold classification (0.5)

**Outputs:**
- Accuracy
- ROC AUC
- Classification report
- Churn probability prediction for a new customer
- Feature impact coefficients

Run:

---

# ✅ Part 3: Customer Segmentation

**Goal:** Segment customers using purchasing behavior.  
**Method:** K-Means Clustering with RFM analysis  
**Dataset Size:** 500,000+ transactions  

**Dataset Source:**  
UCI Online Retail Dataset  
https://archive.ics.uci.edu/ml/datasets/Online+Retail  

**Features Created:**
- Recency
- Frequency
- Monetary value

**Outputs:**
- Elbow plot (`elbow_plot.png`)
- Cluster summary statistics
- Marketing strategy suggestions
- CSV file of cluster assignments (`customer_segments.csv`)

Run:

---

# ⭐ Extra Credit: Housing Demand Forecasting

**Goal:** Forecast housing demand for the next 6 months.  
**Model:** Linear Regression (time index feature)  
**Dataset Requirement:** 100+ months of historical housing demand data  

**Example Data Source:**  
Zillow Research Data  
https://www.zillow.com/research/data/  

**Outputs:**
- Demand forecast plot (`demand_forecast.png`)
- 6-month forecast CSV
- Assumptions and improvement discussion

Run:

---

# 📥 Dataset Instructions

Download each dataset and place the file in the root project folder before running scripts.

Required filenames:

- `train.csv`
- `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- `online_retail.csv`
- `housing_demand.csv`

---

# 🛠 Installation

Install required packages:


Or manually:


---

# 📊 Technologies Used

- Python 3
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

---

# 👩‍🎓 Author

Nicole Brownen  
San José State University  
Business MIS Program
