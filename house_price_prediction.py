# House Price Prediction using Linear Regression
# Data Source:
# Kaggle – House Prices: Advanced Regression Techniques
# https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset (must download train.csv first)
df = pd.read_csv("train.csv")

# Select relevant columns
df = df[["SalePrice", "GrLivArea", "Neighborhood"]].dropna()

X = df[["GrLivArea", "Neighborhood"]]
y = df["SalePrice"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("location", OneHotEncoder(handle_unknown="ignore"), ["Neighborhood"])
    ],
    remainder="passthrough"
)

# Pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# Train
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
print("MAE:", round(mean_absolute_error(y_test, predictions), 2))
print("R²:", round(r2_score(y_test, predictions), 4))

# Predict new house
new_house = pd.DataFrame({
    "GrLivArea": [2000],
    "Neighborhood": ["NAmes"]
})
predicted_price = model.predict(new_house)[0]

print(f"\nPredicted price for 2000 sq ft in NAmes: ${predicted_price:,.2f}")

# Coefficients
feature_names = model.named_steps["preprocessor"].get_feature_names_out()
coefficients = model.named_steps["regressor"].coef_

coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
}).sort_values("Coefficient", ascending=False)

print("\nTop Feature Impacts:")
print(coef_df.head(10))
