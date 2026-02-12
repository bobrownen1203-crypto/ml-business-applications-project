# Customer Segmentation using K-Means (RFM)
# Data Source:
# UCI Online Retail Dataset
# https://archive.ics.uci.edu/ml/datasets/Online+Retail

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv("online_retail.csv")
print("Total transactions:", len(df))

# Clean data
df = df.dropna(subset=["CustomerID"])
df = df[df["Quantity"] > 0]
df = df[df["UnitPrice"] > 0]

df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Create RFM features
snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
    "InvoiceNo": "nunique",
    "TotalPrice": "sum"
}).reset_index()

rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]
print("Total unique customers:", len(rfm))

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

# Elbow method
inertia = []
for k in range(1, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure()
plt.plot(range(1,7), inertia, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.grid(True)
plt.savefig("elbow_plot.png", dpi=200)
plt.close()

# Final model with K=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
rfm["Cluster"] = kmeans.fit_predict(X_scaled)

# Cluster summary
print("\nCluster Summary:")
cluster_means = rfm.groupby("Cluster")[["Recency","Frequency","Monetary"]].mean()
print(cluster_means)

# Strategy suggestions
print("\nMarketing Strategy Suggestions:")
for cluster in cluster_means.index:
    rec = cluster_means.loc[cluster, "Recency"]
    freq = cluster_means.loc[cluster, "Frequency"]
    mon = cluster_means.loc[cluster, "Monetary"]

    print(f"\nCluster {cluster}:")
    if mon > cluster_means["Monetary"].mean() and freq > cluster_means["Frequency"].mean():
        print("- High value loyal customers → Offer VIP rewards and exclusive promotions.")
    elif rec > cluster_means["Recency"].mean():
        print("- At-risk or inactive customers → Send re-engagement campaigns or discounts.")
    else:
        print("- Mid-tier customers → Encourage cross-selling and loyalty programs.")

# Save output
rfm.to_csv("customer_segments.csv", index=False)
print("\nSaved files: elbow_plot.png, customer_segments.csv")
