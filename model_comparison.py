import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =========================
# CREATE MODEL FOLDER
# =========================
os.makedirs("models", exist_ok=True)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(
    "data/cleaned_data.csv"
)

print("Dataset Loaded!")

# =========================
# ENCODING
# =========================
categorical_cols = [

    "Factory",

    "Region",

    "Ship Mode",

    "Product Name"

]

encoders = {}

for col in categorical_cols:

    df[col] = df[col].fillna("Unknown")

    le = LabelEncoder()

    df[col] = le.fit_transform(
        df[col].astype(str)
    )

    encoders[col] = le

# =========================
# FEATURES
# =========================
X = df[
    [
        "Factory",
        "Region",
        "Ship Mode",
        "Product Name",
        "Sales",
        "Units",
        "Distance"
    ]
]

# =========================
# TARGET
# =========================
y = df["Lead_Time"]

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42
)

# =========================
# MODELS
# =========================
models = {

    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    )
}

# =========================
# RESULT STORAGE
# =========================
results = []

best_model = None

best_r2 = -999

best_model_name = ""

# =========================
# TRAIN & EVALUATE
# =========================
for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    results.append({

        "Model": name,

        "MAE": round(mae, 2),

        "RMSE": round(rmse, 2),

        "R2": round(r2, 2)

    })

    if r2 > best_r2:

        best_r2 = r2

        best_model = model

        best_model_name = name

# =========================
# RESULTS TABLE
# =========================
results_df = pd.DataFrame(
    results
)

print("\nMODEL COMPARISON")
print(results_df)

# Save comparison table
results_df.to_csv(
    "models/model_comparison_results.csv",
    index=False
)

# Save best model
joblib.dump(
    best_model,
    "models/best_model.pkl"
)

# Save encoders
joblib.dump(
    encoders,
    "models/label_encoders.pkl"
)

print("\nBest Model:", best_model_name)

print("R2 Score:", round(best_r2,2))

print("\nModel Saved Successfully!")