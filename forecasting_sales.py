# Housing Demand Forecasting Tool (6-Month Forecast)
# Data Source Example:
# Zillow Research Data
# https://www.zillow.com/research/data/
#
# Expected CSV format: columns = ["month", "demand"]
# month should be a monthly date like 2022-01-01, demand is a numeric metric (inventory, new listings, etc.)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load data
df = pd.read_csv("housing_demand.csv")  # columns: month, demand

# Preprocess / cleaning
df["month"] = pd.to_datetime(df["month"], errors="coerce")
df["demand"] = pd.to_numeric(df["demand"], errors="coerce")
df = df.dropna(subset=["month", "demand"]).sort_values("month").reset_index(drop=True)

print("Rows in dataset:", len(df))  # confirms 100+ records (if you have them)

# Create time index feature
df["t"] = np.arange(len(df))

X = df[["t"]]
y = df["demand"]

# Train model
model = LinearRegression()
model.fit(X, y)

# (Optional improvement) quick fit diagnostics on training data
train_pred = model.predict(X)
print("Train MAE:", round(mean_absolute_error(y, train_pred), 2))
print("Train R²:", round(r2_score(y, train_pred), 4))

# Forecast next 6 months
future_t = np.arange(df["t"].max() + 1, df["t"].max() + 7)
future_months = pd.date_range(
    df["month"].max() + pd.offsets.MonthBegin(1),
    periods=6,
    freq="MS"
)

forecast = model.predict(pd.DataFrame({"t": future_t}))

forecast_df = pd.DataFrame({
    "month": future_months,
    "forecast_demand": forecast
})

print("\nForecast (next 6 months):")
print(forecast_df.to_string(index=False))

# Plot + save
plt.figure()
plt.plot(df["month"], y, label="Historical")
plt.plot(forecast_df["month"], forecast_df["forecast_demand"], linestyle="--", label="Forecast")
plt.legend()
plt.title("6-Month Housing Demand Forecast (Linear Regression)")
plt.xlabel("Month")
plt.ylabel("Demand")
plt.grid(True)
plt.tight_layout()
plt.savefig("demand_forecast.png", dpi=200)
plt.show()

# Save forecast results
forecast_df.to_csv("forecast_next_6_months.csv", index=False)
print("\nSaved files: demand_forecast.png, forecast_next_6_months.csv")

# Documentation (required)
print("\nAssumptions:")
print("- Demand is represented by one monthly numeric metric from the CSV (e.g., inventory or new listings).")
print("- Linear trend over time; no seasonality or external drivers included (baseline model).")

print("\nChallenges:")
print("- Housing demand often has strong seasonality and can shift due to interest rates and macroeconomic changes.")
print("- A simple linear model may underfit when patterns are non-linear or include breaks.")

print("\nPotential improvements:")
print("- Add seasonality features (month-of-year), lag features, and rolling averages.")
print("- Include external variables like mortgage rates, unemployment, and inventory supply.")
print("- Try time-series models (ARIMA/Prophet) or non-linear ML models.")
