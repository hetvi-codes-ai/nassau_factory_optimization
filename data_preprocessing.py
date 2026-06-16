import pandas as pd
import numpy as np

from geopy.distance import geodesic

from utils.factory_mapping import factory_map
from utils.factory_coordinates import factory_coordinates
from utils.region_coordinates import region_coordinates

# LOAD DATASET
df = pd.read_csv("data/nassau_data.csv")

# CONVERT DATES
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)

df['Ship Date'] = pd.to_datetime(df['Ship Date'], dayfirst=True)

# CREATE LEAD TIME
df['Lead_Time'] = (
    df['Ship Date'] - df['Order Date']
).dt.days

# REMOVE INVALID LEAD TIMES
df = df[df['Lead_Time'] >= 0]

# MAP PRODUCT TO FACTORY
df['Factory'] = df['Product Name'].map(factory_map)

# PROFIT MARGIN
df['Profit_Margin'] = (
    df['Gross Profit'] / df['Sales']
)

# DISTANCE CALCULATION
def calculate_distance(row):

    factory = row['Factory']

    region = row['Region']

    if (
        factory in factory_coordinates
        and
        region in region_coordinates
    ):

        return geodesic(

            factory_coordinates[factory],

            region_coordinates[region]

        ).km

    return np.nan

# CREATE DISTANCE COLUMN
df['Distance'] = df.apply(
    calculate_distance,
    axis=1
)

# REMOVE NULL VALUES
df.dropna(inplace=True)

# REMOVE OUTLIERS
Q1 = df['Lead_Time'].quantile(0.25)

Q3 = df['Lead_Time'].quantile(0.75)

IQR = Q3 - Q1

df = df[
    ~(
        (df['Lead_Time'] < (Q1 - 1.5 * IQR))
        |
        (df['Lead_Time'] > (Q3 + 1.5 * IQR))
    )
]

# SAVE CLEAN DATA
df.to_csv(
    "data/cleaned_data.csv",
    index=False
)

print("Preprocessing Completed!")
