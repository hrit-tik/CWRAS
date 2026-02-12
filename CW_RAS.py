import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("CW_RAS_master_dataset.csv")

# -----------------------------
# HANDLE MISSING DATA
# -----------------------------
df["GW_last"] = df["GW_last"].fillna(df["GW_current"])
df["GW_current"] = df["GW_current"].fillna(df["GW_last"])
df = df.fillna(0)

# -----------------------------
# STEP 1: RAINFALL ANALYSIS
# -----------------------------
df["rain_diff"] = (df["R_current"] - df["R_normal"]) / df["R_normal"]

# Flood: excess rainfall only
df["R_score_flood"] = np.where(
    df["rain_diff"] > 0,
    df["rain_diff"] * 100,
    0
)

# Scarcity: deficit rainfall only
df["R_score_scarcity"] = np.where(
    df["rain_diff"] < 0,
    (-df["rain_diff"]) * 100,
    0
)

# Prevent extreme dominance
df["R_score_flood"] = df["R_score_flood"].clip(0, 60)
df["R_score_scarcity"] = df["R_score_scarcity"].clip(0, 60)

# -----------------------------
# STEP 2: GROUNDWATER ANALYSIS
# -----------------------------

# Correct drop direction
df["gw_drop"] = df["GW_last"] - df["GW_current"]

# Ignore small seasonal fluctuations (<0.4m)
df["gw_drop"] = df["gw_drop"].apply(lambda x: 0 if x < 0.4 else x)

# Robust scaling using 90th percentile
MAX_EXPECTED_GW_DROP = df["gw_drop"].quantile(0.90)
MAX_EXPECTED_GW_DROP = MAX_EXPECTED_GW_DROP if MAX_EXPECTED_GW_DROP > 0 else 1

df["G_score"] = (df["gw_drop"] / MAX_EXPECTED_GW_DROP) * 100
df["G_score"] = df["G_score"].clip(0, 100)

# -----------------------------
# STEP 3: LAND-USE ANALYSIS
# -----------------------------

# Flood: normalized impervious contribution
df["L_score_flood"] = (df["Urban_Percent"] / 100) * 50

# Scarcity: recharge deficit model
df["L_score_scarcity"] = (
    (df["Urban_Percent"] * 0.6) +
    ((100 - df["Forest_Percent"]) * 0.4)
)

df["L_score_scarcity"] = df["L_score_scarcity"].clip(0, 100)

# -----------------------------
# STEP 4: BASE RISK SCORES
# -----------------------------

df["FloodRisk"] = (
    0.45 * df["R_score_flood"] +
    0.35 * df["L_score_flood"] +
    0.20 * df["G_score"]
)

df["ScarcityRisk"] = (
    0.35 * df["R_score_scarcity"] +
    0.35 * df["G_score"] +
    0.30 * df["L_score_scarcity"]
)

# -----------------------------
# STEP 5: RECHARGE BUFFER LOGIC
# -----------------------------
# Reduce scarcity for rural / forest-dominant recharge zones

def apply_recharge_buffer(row):
    if row["Urban_Percent"] < 30 and row["Forest_Percent"] > 20:
        return row["ScarcityRisk"] * 0.75
    return row["ScarcityRisk"]

df["ScarcityRisk"] = df.apply(apply_recharge_buffer, axis=1)

# Prevent scarcity from exceeding flood unrealistically
df["ScarcityRisk"] = np.minimum(df["ScarcityRisk"], df["FloodRisk"] * 1.1)

# -----------------------------
# STEP 6: CLASSIFICATION
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
# SAVE OUTPUT (UNCHANGED FORMAT)
# -----------------------------

output_columns = [
    "Panchayat",
    "FloodRisk", "FloodRiskLevel",
    "ScarcityRisk", "ScarcityRiskLevel"
]

df[output_columns].to_csv("CW_RAS_output_results.csv", index=False)

print("✅ Calibrated CW-RAS risk calculation completed successfully.")
print("📁 Output saved as CW_RAS_output_results.csv")

# -----------------------------
# VISUALIZATION
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
