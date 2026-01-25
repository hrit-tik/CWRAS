import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# CONFIGURATION
# -----------------------------
MAX_EXPECTED_GW_DROP = 2.0  # meters

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("CW_RAS_master_dataset.csv")

# -----------------------------
# HANDLE MISSING DATA SAFELY
# -----------------------------

# If groundwater values are missing, assume no significant change
df["GW_last"] = df["GW_last"].fillna(df["GW_current"])
df["GW_current"] = df["GW_current"].fillna(df["GW_last"])

# Fill any remaining missing values with 0
df = df.fillna(0)

# -----------------------------
# STEP 1: RAINFALL DIFFERENCE
# -----------------------------
df["rain_diff"] = (df["R_current"] - df["R_normal"]) / df["R_normal"]

# -----------------------------
# STEP 2: PARAMETER SCORES
# -----------------------------

# Rainfall Scores
df["R_score_flood"] = np.where(
    df["rain_diff"] > 0,
    df["rain_diff"] * 100,
    0
)

df["R_score_scarcity"] = np.where(
    df["rain_diff"] < 0,
    (-df["rain_diff"]) * 100,
    0
)

df["R_score_flood"] = df["R_score_flood"].clip(0, 100)
df["R_score_scarcity"] = df["R_score_scarcity"].clip(0, 100)

# Groundwater Score
df["gw_drop"] = df["GW_current"] - df["GW_last"]
df["G_score"] = (df["gw_drop"] / MAX_EXPECTED_GW_DROP) * 100
df["G_score"] = df["G_score"].clip(0, 100)

# Land-use Scores
df["L_score_flood"] = df["Urban_Percent"]
df["L_score_scarcity"] = 100 - df["Forest_Percent"]

# -----------------------------
# STEP 3: FINAL RISK SCORES
# -----------------------------
df["FloodRisk"] = (
    0.5 * df["R_score_flood"] +
    0.3 * df["L_score_flood"] +
    0.2 * df["G_score"]
)

df["ScarcityRisk"] = (
    0.4 * df["R_score_scarcity"] +
    0.4 * df["G_score"] +
    0.2 * df["L_score_scarcity"]
)

# -----------------------------
# STEP 4: RISK LEVEL CLASSIFICATION
# -----------------------------
def classify_risk(score):
    if score <= 25:
        return "Low"
    elif score <= 50:
        return "Medium"
    elif score <= 75:
        return "High"
    else:
        return "Critical"

df["FloodRiskLevel"] = df["FloodRisk"].apply(classify_risk)
df["ScarcityRiskLevel"] = df["ScarcityRisk"].apply(classify_risk)

# -----------------------------
# SAVE OUTPUT
# -----------------------------
output_columns = [
    "Panchayat",
    "FloodRisk", "FloodRiskLevel",
    "ScarcityRisk", "ScarcityRiskLevel"
]

df[output_columns].to_csv("CW_RAS_output_results.csv", index=False)

print("✅ CW-RAS risk calculation completed successfully.")
print("📁 Output saved as CW_RAS_output_results.csv")

# -----------------------------
# STEP 5: VISUALIZATION
# -----------------------------
plt.figure(figsize=(10, 5))
plt.bar(df["Panchayat"], df["FloodRisk"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Flood Risk Score")
plt.title("Flood Risk by Panchayat")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.bar(df["Panchayat"], df["ScarcityRisk"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Scarcity Risk Score")
plt.title("Water Scarcity Risk by Panchayat")
plt.tight_layout()
plt.show()
