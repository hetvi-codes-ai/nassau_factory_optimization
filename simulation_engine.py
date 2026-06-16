import pandas as pd
import joblib
import os

# =========================
# CREATE OUTPUT FOLDER
# =========================
os.makedirs(
    "simulation_outputs",
    exist_ok=True
)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(
    "data/cleaned_data.csv"
)

# =========================
# LOAD MODEL
# =========================
model = joblib.load(
    "models/best_model.pkl"
)

encoders = joblib.load(
    "models/label_encoders.pkl"
)

# =========================
# FACTORIES
# =========================
all_factories = [

    "Lot's O' Nuts",

    "Wicked Choccy's",

    "Sugar Shack",

    "Secret Factory",

    "The Other Factory"

]

# =========================
# STORE RESULTS
# =========================
results = []

# =========================
# LOOP PRODUCTS
# =========================
products = df[
    "Product Name"
].unique()

for product in products:

    product_data = df[
        df["Product Name"] == product
    ]

    sample = product_data.iloc[0]

    current_factory = sample[
        "Factory"
    ]

    current_lead_time = sample[
        "Lead_Time"
    ]

    for factory in all_factories:

        if factory == current_factory:
            continue

        try:

            factory_enc = encoders[
                "Factory"
            ].transform(
                [factory]
            )[0]

            region_enc = encoders[
                "Region"
            ].transform(
                [sample["Region"]]
            )[0]

            ship_enc = encoders[
                "Ship Mode"
            ].transform(
                [sample["Ship Mode"]]
            )[0]

            product_enc = encoders[
                "Product Name"
            ].transform(
                [sample["Product Name"]]
            )[0]

            prediction_df = pd.DataFrame({

                "Factory": [factory_enc],

                "Region": [region_enc],

                "Ship Mode": [ship_enc],

                "Product Name": [product_enc],

                "Sales": [sample["Sales"]],

                "Units": [sample["Units"]],

                "Distance": [sample["Distance"]]

            })

            predicted_lead_time = model.predict(
                prediction_df
            )[0]

            # ==================================
            # DEMO IMPROVEMENT CALCULATION
            # ==================================

            improvement = abs(

                (
                    current_lead_time
                    -
                    predicted_lead_time
                )

                /

                current_lead_time

            ) * 100

            profit_impact = round(
                improvement * 0.15,
                2
            )

            confidence_score = round(

                min(
                    70 + improvement,
                    99
                ),

                2

            )

            if improvement >= 20:

                risk = "Low"

            elif improvement >= 10:

                risk = "Medium"

            else:

                risk = "High"

            results.append({

                "Product":
                    product,

                "Current Factory":
                    current_factory,

                "Alternative Factory":
                    factory,

                "Current Lead Time":
                    round(
                        current_lead_time,
                        2
                    ),

                "Predicted Lead Time":
                    round(
                        predicted_lead_time,
                        2
                    ),

                "Lead Time Improvement %":
                    round(
                        improvement,
                        2
                    ),

                "Profit Impact %":
                    profit_impact,

                "Confidence Score":
                    confidence_score,

                "Risk Score":
                    risk

            })

        except Exception as e:

            print(
                f"Error : {e}"
            )

# =========================
# SAVE
# =========================
results_df = pd.DataFrame(
    results
)

results_df.to_csv(

    "simulation_outputs/simulation_results.csv",

    index=False

)

print(
    "\nSimulation Completed Successfully!"
)

print(
    results_df.head()
)