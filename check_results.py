import pandas as pd

df = pd.read_csv(
    "simulation_outputs/simulation_results.csv"
)

print("Rows:", len(df))

print("\nMaximum Improvement:")
print(df["Lead Time Improvement %"].max())

print("\nMinimum Improvement:")
print(df["Lead Time Improvement %"].min())

print("\nPositive Improvements:")
print(
    len(
        df[
            df["Lead Time Improvement %"] > 0
        ]
    )
)


df = pd.read_csv(
    "data/cleaned_data.csv"
)

print(df["Lead_Time"].describe())
print("\nFirst 10 Rows:")
print(df.head(10))