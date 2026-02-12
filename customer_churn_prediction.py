# Customer Churn Prediction using Logistic Regression
# Data Source:
# Kaggle – Telco Customer Churn
# https://www.kaggle.com/datasets/blastchar/telco-customer-churn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Clean data
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

num_features = ["tenure", "MonthlyCharges", "TotalCharges"]
cat_features = ["Contract", "PaymentMethod", "InternetService"]

X = df[num_features + cat_features]
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=2000))
])

model.fit(X_train, y_train)

# Evaluate
proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)

print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print("ROC AUC:", round(roc_auc_score(y_test, proba), 4))
print("\nClassification Report:\n", classification_report(y_test, pred))

# Predict new customer
new_customer = pd.DataFrame([{
    "tenure": 6,
    "MonthlyCharges": 85,
    "TotalCharges": 500,
    "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check",
    "InternetService": "Fiber optic"
}])

new_prob = model.predict_proba(new_customer)[0,1]
print(f"\nNew Customer Churn Probability: {new_prob:.2f}")
