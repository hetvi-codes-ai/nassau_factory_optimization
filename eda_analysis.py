import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output folder
os.makedirs("eda_outputs", exist_ok=True)

# Load cleaned data
df = pd.read_csv("data/cleaned_data.csv")

print("Dataset Loaded Successfully!")

# -------------------------------
# Lead Time by Region
# -------------------------------

plt.figure(figsize=(10,6))

df.groupby("Region")["Lead_Time"].mean().sort_values().plot(
    kind="bar"
)

plt.title("Average Lead Time by Region")
plt.ylabel("Days")

plt.tight_layout()

plt.savefig(
    "eda_outputs/lead_time_by_region.png"
)

plt.close()

# -------------------------------
# Lead Time by Factory
# -------------------------------

plt.figure(figsize=(10,6))

df.groupby("Factory")["Lead_Time"].mean().sort_values().plot(
    kind="bar"
)

plt.title("Average Lead Time by Factory")
plt.ylabel("Days")

plt.tight_layout()

plt.savefig(
    "eda_outputs/lead_time_by_factory.png"
)

plt.close()

# -------------------------------
# Lead Time by Ship Mode
# -------------------------------

plt.figure(figsize=(10,6))

df.groupby("Ship Mode")["Lead_Time"].mean().sort_values().plot(
    kind="bar"
)

plt.title("Average Lead Time by Ship Mode")
plt.ylabel("Days")

plt.tight_layout()

plt.savefig(
    "eda_outputs/lead_time_by_shipmode.png"
)

plt.close()

# -------------------------------
# Profit by Product
# -------------------------------

plt.figure(figsize=(14,6))

df.groupby("Product Name")["Gross Profit"].sum().sort_values().plot(
    kind="bar"
)

plt.title("Gross Profit by Product")

plt.tight_layout()

plt.savefig(
    "eda_outputs/profit_by_product.png"
)

plt.close()

# -------------------------------
# Sales vs Profit
# -------------------------------

plt.figure(figsize=(8,6))

sns.scatterplot(
    x="Sales",
    y="Gross Profit",
    data=df
)

plt.title("Sales vs Gross Profit")

plt.tight_layout()

plt.savefig(
    "eda_outputs/sales_vs_profit.png"
)

plt.close()

# -------------------------------
# Distance Distribution
# -------------------------------

plt.figure(figsize=(10,6))

sns.histplot(
    df["Distance"],
    bins=30
)

plt.title("Distance Distribution")

plt.tight_layout()

plt.savefig(
    "eda_outputs/distance_distribution.png"
)

plt.close()

# -------------------------------
# Correlation Heatmap
# -------------------------------

numeric_df = df.select_dtypes(
    include=["number"]
)

plt.figure(figsize=(10,8))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    "eda_outputs/correlation_heatmap.png"
)

plt.close()

print("EDA Completed Successfully!")