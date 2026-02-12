# Housing Demand Forecasting Tool
# Data Source Example:
# Zillow Research Data
# https://www.zillow.com/research/data/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("housing_demand.csv")  # columns: month, demand

df["month"] = pd.to_datetime(df["month"])
df = df.sort_values("month")
df["t"] = np.arange(len(df))

X = df[["t"]]
y = df["demand"]

model = LinearRegression()
model.fit(X, y)

# Forecast next 6 months
future_t = np.arange(df["t"].max()+1, df["t"].max()+7)
future_months = pd.date_range(df["month"].max()+pd.offsets.MonthBegin(1), periods=6, freq="MS")

forecast = model.predict(pd.DataFrame({"t": future_t}))

plt.plot(df["month"], y, label="Historical")
plt.plot(future_months, forecast, linestyle="--", label="Forecast")
plt.legend()
plt.title("6-Month Housing Demand Forecast")
plt.show()

print("\nAssumptions:")
print("- Linear trend over time")
print("- No seasonality included")
print("\nImprovements:")
print("- Add seasonal variables")
print("- Include interest rates, inventory levels")
print("- Try ARIMA or Prophet")
