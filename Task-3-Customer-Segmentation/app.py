import pandas as pd

# Load the dataset
df = pd.read_csv("data/Mall_Customers.csv")

# Display first 5 rows
print(df.head())

# Display dataset information
print("\nDataset Info:")
print(df.info())

# Display summary statistics
print("\nSummary Statistics:")
print(df.describe())

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("data/Mall_Customers.csv")

# Select features for clustering
X = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]

print("Selected Features:")
print(X.head())

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nScaled Data (First 5 Rows):")
print(X_scaled[:5])

from sklearn.cluster import KMeans

# Create K-Means model
kmeans = KMeans(n_clusters=5, random_state=42)

# Fit the model and predict clusters
df["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)

# Display first 10 customers with their cluster
print("\nK-Means Clusters:")
print(df[["CustomerID", "Age", "Annual Income (k$)",
          "Spending Score (1-100)", "KMeans_Cluster"]].head(10))

from sklearn.cluster import DBSCAN

# Create DBSCAN model
dbscan = DBSCAN(eps=0.8, min_samples=5)

# Fit the model and predict clusters
df["DBSCAN_Cluster"] = dbscan.fit_predict(X_scaled)

print("\nDBSCAN Clusters:")
print(df[["CustomerID",
          "Age",
          "Annual Income (k$)",
          "Spending Score (1-100)",
          "DBSCAN_Cluster"]].head(10))

import plotly.express as px

# Create 3D Scatter Plot
fig = px.scatter_3d(
    df,
    x="Age",
    y="Annual Income (k$)",
    z="Spending Score (1-100)",
    color=df["KMeans_Cluster"].astype(str),
    title="Customer Segmentation using K-Means",
    labels={
        "Age": "Age",
        "Annual Income (k$)": "Annual Income (k$)",
        "Spending Score (1-100)": "Spending Score"
    }
)

fig.show()

# Save clustered dataset
df.to_csv("clustered_customers.csv", index=False)

print("\nClustered dataset saved as 'clustered_customers.csv'")