from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import requests
import math

app = Flask(__name__)

# Load datasets
risk_data = pd.read_csv("CW_RAS_master_dataset.csv")
location_data = pd.read_csv("panchayat_locations.csv")

panchayat_list = sorted(risk_data["Panchayat"].unique().tolist())


def classify_level(score):
    if score < 30:
        return "Low"
    elif score < 60:
        return "Moderate"
    else:
        return "High"


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
    """Surface Water Factor: moderates scarcity for lake-adjacent areas.
    SWF = max(1.0 - (Water_Body_Percent / 50), 0.1)
    Range: [0.1, 1.0]."""
    if pd.isna(water_body_pct) or water_body_pct <= 0:
        return 1.0
    return max(1.0 - (water_body_pct / 50), 0.1)


# ---------- OSM Geocoding ----------
def get_lat_long(place_name):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": place_name,
        "format": "json",
        "limit": 1
    }
    headers = {
        "User-Agent": "CW-RAS-App"
    }

    response = requests.get(url, params=params, headers=headers)
    data = response.json()

    if len(data) == 0:
        return None, None

    return float(data[0]["lat"]), float(data[0]["lon"])


# ---------- Haversine Distance ----------
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km

    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))

    return R * c


# ---------- Nearest Panchayat ----------
def find_nearest_panchayat(lat, lon):
    min_distance = float("inf")
    nearest_panchayat = None

    for _, row in location_data.iterrows():
        dist = haversine_distance(
            lat, lon,
            row["Latitude"],
            row["Longitude"]
        )

        if dist < min_distance:
            min_distance = dist
            nearest_panchayat = row["Panchayat"]

    return nearest_panchayat


@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "GET":
        return render_template("index.html")

    # ---------- POST LOGIC ----------
    user_place = request.form["panchayat"]
    risk_type = request.form["risk_type"]

    # --- STEP 1: Try exact name match (case-insensitive) ---
    name_lower = user_place.strip().lower()
    exact_match = risk_data[risk_data["Panchayat"].str.lower().str.strip("* ") == name_lower]
    if exact_match.empty:
        # Also try with asterisk-stripped names
        exact_match = risk_data[risk_data["Panchayat"].str.replace("*", "", regex=False).str.lower().str.strip() == name_lower]

    if not exact_match.empty:
        nearest_panchayat = exact_match.iloc[0]["Panchayat"]
    else:
        # --- STEP 2: Fall back to geocoding + Haversine ---
        lat, lon = get_lat_long(user_place)

        if lat is None or lon is None:
            return render_template("index.html", error="Location not found. Please try another name.")

        nearest_panchayat = find_nearest_panchayat(lat, lon)

    row = risk_data[risk_data["Panchayat"] == nearest_panchayat].iloc[0]

    # ----- NORMALIZE COMPONENTS (all 0-100) -----
    rain_score = normalize_rainfall(row["R_normal"], row["R_current"])

    rainfall_normal = round(row["R_normal"], 2)
    rainfall_current = round(row["R_current"], 2)
    rainfall_deviation = round(rain_score, 2)

    # ----- GROUNDWATER -----
    gw_score = normalize_groundwater(row["GW_last"], row["GW_current"])

    gw_last = round(row["GW_last"], 2) if pd.notna(row["GW_last"]) else None
    gw_current = round(row["GW_current"], 2) if pd.notna(row["GW_current"]) else None
    gw_change = round(abs(gw_last - gw_current), 2) if gw_last is not None and gw_current is not None else None

    # ----- LAND USE -----
    urban = row["Urban_Percent"] if pd.notna(row["Urban_Percent"]) else 0
    forest = row["Forest_Percent"] if pd.notna(row["Forest_Percent"]) else 0
    lu_score = normalize_landuse(urban, forest)

    urban_percent = round(urban, 2)
    forest_percent = round(forest, 2)

    if urban_percent >= 50:
        landuse_type = "Urban-dominant"
    elif urban_percent >= 25:
        landuse_type = "Semi-urban"
    else:
        landuse_type = "Rural / Forest-dominant"

    # ----- SURFACE WATER FACTORS -----
    water_body_pct = row["Water_Body_Percent"] if "Water_Body_Percent" in row.index else 0
    swf = compute_swf(water_body_pct)  # For Scarcity
    flood_boost = water_body_pct * 1.2  # For Flood

    # ----- WEIGHTED SCORING -----
    if risk_type == "flood":
        # Flood: Only rising GW contributes (GW_current < GW_last means rise in water level / decrease in depth)
        # Note: Datasets usually use mbgl (meters below ground level).
        # gw_rise_score is calculated based on direction check.
        # However, normalize_groundwater uses abs diff. We need direction.

        is_rising = False
        if pd.notna(gw_last) and pd.notna(gw_current):
            # If current depth < last depth => Water level ROSE
            if gw_current < gw_last:
                is_rising = True

        gw_flood_score = gw_score if is_rising else 0

        rainfall_impact = 0.4 * rain_score
        landuse_impact = 0.4 * lu_score
        groundwater_impact = 0.2 * gw_flood_score
        
        base_score = rainfall_impact + landuse_impact + groundwater_impact
        score = min(base_score + flood_boost, 100)
        risk_label = "Flood Risk"

    else:
        # Scarcity: Absolute fluctuation contributes (instability)
        rainfall_impact = 0.4 * rain_score
        groundwater_impact = 0.4 * gw_score
        landuse_impact = 0.2 * lu_score
        
        raw_score = rainfall_impact + groundwater_impact + landuse_impact
        score = raw_score * swf
        risk_label = "Water Scarcity Risk"

    rainfall = round(rainfall_impact, 2)
    groundwater = round(groundwater_impact, 2)
    landuse = round(landuse_impact, 2)

    level = classify_level(score)

    if rainfall >= groundwater and rainfall >= landuse:
        explanation = "Rainfall variation is the dominant factor influencing the assessed risk in this region."
    elif groundwater >= landuse:
        explanation = "Groundwater level fluctuation significantly contributes to the assessed risk in this region."
    else:
        explanation = "Land-use characteristics such as urbanization influence the assessed risk in this region."

    dashboard_data = {
        "panchayat": user_place,
        "risk_type": risk_label,
        "score": round(score, 2),
        "level": level,
        "rainfall": rainfall,
        "groundwater": groundwater,
        "landuse": landuse,
        "explanation": explanation,

        "rainfall_normal": rainfall_normal,
        "rainfall_current": rainfall_current,
        "rainfall_deviation": rainfall_deviation,

        "gw_last": gw_last,
        "gw_current": gw_current,
        "gw_change": gw_change,

        "urban_percent": urban_percent,
        "forest_percent": forest_percent,
        "landuse_type": landuse_type,

        "water_body_pct": round(water_body_pct, 1) if water_body_pct else 0,
        "swf": round(swf, 2),
        "flood_boost": round(flood_boost, 1)
    }

    return render_template(
        "dashboard.html",
        data=dashboard_data,
        selected_panchayat=user_place,
        selected_risk=risk_type
    )


# ---------- NEW ROUTE (DOCUMENTATION ONLY) ----------
@app.route("/algorithm")
def algorithm():
    return render_template("algorithm.html")


if __name__ == "__main__":
    app.run()
