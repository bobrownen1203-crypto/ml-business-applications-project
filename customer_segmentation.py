# Customer Segmentation using K-Means (RFM)
# Data Source:
# UCI Online Retail Dataset
# https://archive.ics.uci.edu/ml/datasets/Online+Retail

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv("online_retail.csv")

df = df.dropna(subset=["CustomerID"])
df = df[df["Quantity"] > 0]
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
    "InvoiceNo": "nunique",
    "TotalPrice": "sum"
}).reset_index()

rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

# Elbow method
inertia = []
for k in range(1, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.plot(range(1,7), inertia)
plt.xlabel("K")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.savefig("elbow_plot.png")
plt.close()

# Final model
kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
rfm["Cluster"] = kmeans.fit_predict(X_scaled)

print("\nCluster Summary:")
print(rfm.groupby("Cluster")[["Recency","Frequency","Monetary"]].mean())

rfm.to_csv("customer_segments.csv", index=False)
