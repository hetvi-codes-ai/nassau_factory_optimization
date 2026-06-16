import pandas as pd
import os

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# CREATE OUTPUT FOLDER
# =========================
os.makedirs("clustering_outputs", exist_ok=True)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(
    "data/cleaned_data.csv"
)

print("Dataset Loaded!")

# =========================
# CREATE PROFIT MARGIN
# =========================
df["Profit_Margin"] = (
    df["Gross Profit"] / df["Sales"]
) * 100

# =========================
# FEATURES FOR CLUSTERING
# =========================
features = [

    "Distance",

    "Lead_Time",

    "Sales",

    "Profit_Margin"

]

cluster_data = df[features]

# =========================
# SCALE FEATURES
# =========================
scaler = StandardScaler()

scaled_data = scaler.fit_transform(
    cluster_data
)

# =========================
# KMEANS
# =========================
kmeans = KMeans(

    n_clusters=4,

    random_state=42,

    n_init=10
)

df["Cluster"] = kmeans.fit_predict(
    scaled_data
)

# =========================
# SAVE CLUSTERED DATA
# =========================
df.to_csv(
    "clustering_outputs/clustered_data.csv",
    index=False
)

# =========================
# CLUSTER SUMMARY
# =========================
summary = df.groupby("Cluster")[

    ["Distance",
     "Lead_Time",
     "Sales",
     "Profit_Margin"]

].mean()

print("\nCluster Summary")
print(summary)

summary.to_csv(
    "clustering_outputs/cluster_summary.csv"
)

# =========================
# VISUALIZATION
# =========================
plt.figure(figsize=(10,6))

sns.scatterplot(

    x="Distance",

    y="Lead_Time",

    hue="Cluster",

    data=df

)

plt.title(
    "Route Clusters"
)

plt.tight_layout()

plt.savefig(
    "clustering_outputs/route_clusters.png"
)

plt.close()

# =========================
# CLUSTER COUNTS
# =========================
plt.figure(figsize=(8,5))

sns.countplot(
    x="Cluster",
    data=df
)

plt.title(
    "Orders Per Cluster"
)

plt.tight_layout()

plt.savefig(
    "clustering_outputs/cluster_counts.png"
)

plt.close()

print("\nClustering Completed Successfully!")