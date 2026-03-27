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
# NORMALIZATION FUNCTIONS (0-100)
# -----------------------------

def normalize_rainfall(r_normal, r_current):
    """Rainfall deviation normalized to 0-100, capped at 100."""
    if r_normal == 0:
        return 0
    dev = abs(r_normal - r_current) / r_normal * 100
    return min(dev, 100)


def normalize_groundwater(gw_last, gw_current):
    """Groundwater change normalized to 0-100 using 3m reference max."""
    MAX_GW_CHANGE = 3.0
    if pd.isna(gw_last) or pd.isna(gw_current):
        return 0
    change = abs(gw_last - gw_current)
    score = (change / MAX_GW_CHANGE) * 100
    return min(score, 100)


def normalize_landuse(urban_pct, forest_pct):
    """Land-use score normalized to 0-100. Higher = more runoff-prone."""
    urban_component = (urban_pct / 100) * 50
    forest_deficit = ((100 - forest_pct) / 100) * 50
    return min(urban_component + forest_deficit, 100)


def compute_swf(water_body_pct):
    """Surface Water Factor: moderates scarcity for lake-adjacent areas."""
    if pd.isna(water_body_pct) or water_body_pct <= 0:
        return 1.0
    return max(1.0 - (water_body_pct / 50), 0.1)


# -----------------------------
# STEP 1: COMPUTE NORMALIZED SCORES
# -----------------------------

df["R_score"] = df.apply(
    lambda row: normalize_rainfall(row["R_normal"], row["R_current"]),
    axis=1
)

df["G_score"] = df.apply(
    lambda row: normalize_groundwater(row["GW_last"], row["GW_current"]),
    axis=1
)

df["L_score"] = df.apply(
    lambda row: normalize_landuse(row["Urban_Percent"], row["Forest_Percent"]),
    axis=1
)

# -----------------------------
# STEP 2: WEIGHTED RISK SCORES
# -----------------------------

# ----- FLOOD RISK -----
# Only rising groundwater contributes to flood risk.
# GW (mbgl): Lower value = Higher water table.
# Rise = GW_current < GW_last
df["G_Flood_Score"] = df.apply(
    lambda row: row["G_score"] if (
        pd.notna(row["GW_current"]) and pd.notna(row["GW_last"]) and
        row["GW_current"] < row["GW_last"]
    ) else 0,
    axis=1
)

# Flood Boost: Lake proximity increases flood risk
# Boost = Water_Body_Percent * 1.2
df["FloodBoost"] = df["Water_Body_Percent"].fillna(0) * 1.2

df["FloodRisk_Base"] = (
    0.4 * df["R_score"] +
    0.4 * df["L_score"] +
    0.2 * df["G_Flood_Score"]
)

# Final Flood Score = min(Base + Boost, 100)
df["FloodRisk"] = (df["FloodRisk_Base"] + df["FloodBoost"]).clip(upper=100)


# ----- SCARCITY RISK -----
# Scarcity: 0.4*Rainfall + 0.4*Groundwater + 0.2*Land-use, moderated by SWF
df["SWF"] = df["Water_Body_Percent"].apply(compute_swf) if "Water_Body_Percent" in df.columns else 1.0

df["ScarcityRisk"] = (
    0.4 * df["R_score"] +
    0.4 * df["G_score"] +
    0.2 * df["L_score"]
) * df["SWF"]

# -----------------------------
# STEP 3: CLASSIFICATION
# -----------------------------

def classify_risk(score):
    if score < 30:
        return "Low"
    elif score < 60:
        return "Moderate"
    else:
        return "High"


df["FloodRiskLevel"] = df["FloodRisk"].apply(classify_risk)
df["ScarcityRiskLevel"] = df["ScarcityRisk"].apply(classify_risk)

# -----------------------------
# SAVE OUTPUT
# -----------------------------

output_columns = [
    "Panchayat",
    "R_score", "G_score", "L_score", "SWF",
    "FloodRisk", "FloodRiskLevel",
    "ScarcityRisk", "ScarcityRiskLevel"
]

df[output_columns].to_csv("CW_RAS_output_results.csv", index=False)

print("✅ CW-RAS risk calculation completed successfully.")
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
