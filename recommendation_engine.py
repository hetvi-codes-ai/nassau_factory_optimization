import pandas as pd
import os

# =========================
# OUTPUT FOLDER
# =========================
os.makedirs(
    "recommendation_outputs",
    exist_ok=True
)

# =========================
# LOAD SIMULATION RESULTS
# =========================
df = pd.read_csv(
    "simulation_outputs/simulation_results.csv"
)

print(
    "Simulation Results Loaded!"
)

# =========================
# RISK SCORE
# =========================
def calculate_risk(profit_impact):

    if profit_impact >= 3:

        return "Low"

    elif profit_impact >= 1:

        return "Medium"

    elif profit_impact == 0:

        return "No Change"

    else:

        return "High"

# =========================
# CONFIDENCE SCORE
# =========================
def calculate_confidence(improvement):

    confidence = (
        70 +
        (max(improvement, 0) * 0.8)
    )

    if confidence > 99:

        confidence = 99

    return round(
        confidence,
        1
    )

# =========================
# APPLY SCORES
# =========================
df["Risk Score"] = df[
    "Profit Impact %"
].apply(
    calculate_risk
)

df["Confidence Score"] = df[
    "Lead Time Improvement %"
].apply(
    calculate_confidence
)

# =========================
# RECOMMENDATION SCORE
# =========================
df["Recommendation Score"] = (

    df["Lead Time Improvement %"] * 0.6

    +

    df["Profit Impact %"] * 0.4

)

# =========================
# BEST FACTORY FOR EACH PRODUCT
# =========================
recommendations = []

products = df[
    "Product"
].unique()

for product in products:

    temp = df[
        df["Product"] == product
    ]

    best = temp.sort_values(
    by="Recommendation Score",
    ascending=False
).iloc[0]
    recommendations.append({

        "Product":
            best["Product"],

        "Current Factory":
            best["Current Factory"],

        "Recommended Factory":
            best["Alternative Factory"],

        "Lead Time Improvement %":
            round(
                best[
                    "Lead Time Improvement %"
                ],
                2
            ),

        "Profit Impact %":
            round(
                best[
                    "Profit Impact %"
                ],
                2
            ),

        "Risk Score":
            best["Risk Score"],

        "Confidence Score":
            best["Confidence Score"]

    })

# =========================
# CREATE FINAL TABLE
# =========================
recommendation_df = pd.DataFrame(
    recommendations
)

if len(recommendation_df) > 0:

    recommendation_df = recommendation_df.sort_values(

        by="Lead Time Improvement %",

        ascending=False

    )

# =========================
# SAVE
# =========================
recommendation_df.to_csv(

    "recommendation_outputs/final_recommendations.csv",

    

)

print(
    "\nFinal Recommendations Generated!"
)

print(
    recommendation_df.head()
)