import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import (
    RandomForestRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# CREATE models FOLDER
os.makedirs("models", exist_ok=True)

# LOAD CLEAN DATA
df = pd.read_csv(
    "data/cleaned_data.csv"
)

# CATEGORICAL COLUMNS
categorical_cols = [

    'Factory',

    'Region',

    'Ship Mode',

    'Product Name'

]

encoders = {}

for col in categorical_cols:

    print(f"Encoding {col}")

    # Fill null values
    df[col] = df[col].fillna("Unknown")

    # Convert to string
    df[col] = df[col].astype(str)

    # Create encoder
    le = LabelEncoder()

    # Encode column
    df[col] = pd.Series(
        le.fit_transform(df[col]),
        index=df.index
    )

    # Save encoder
    encoders[col] = le
# FEATURES
X = df[
    [
        'Factory',
        'Region',
        'Ship Mode',
        'Product Name',
        'Sales',
        'Units',
        'Distance'
    ]
]

# TARGET
y = df['Lead_Time']

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42
)

# MODEL
model = RandomForestRegressor(

    n_estimators=100,

    random_state=42
)

# TRAIN
model.fit(
    X_train,
    y_train
)

# PREDICT
predictions = model.predict(
    X_test
)

# METRICS
mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(
    y_test,
    predictions
)

print("MAE:", round(mae, 2))

print("RMSE:", round(rmse, 2))

print("R2:", round(r2, 2))

# SAVE MODEL
joblib.dump(
    model,
    "models/best_model.pkl"
)

# SAVE ENCODERS
joblib.dump(
    encoders,
    "models/label_encoders.pkl"
)

print("Model Training Completed!")