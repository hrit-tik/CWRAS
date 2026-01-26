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
    if score < 15:
        return "Low"
    elif score < 25:
        return "Moderate"
    else:
        return "High"


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

    lat, lon = get_lat_long(user_place)

    if lat is None or lon is None:
        return render_template("index.html", error="Location not found. Please try another name.")

    nearest_panchayat = find_nearest_panchayat(lat, lon)

    row = risk_data[risk_data["Panchayat"] == nearest_panchayat].iloc[0]

    # ----- RAINFALL DEVIATION -----
    rainfall_dev = abs(row["R_normal"] - row["R_current"]) / row["R_normal"] * 100

    # --- NEW (RAW RAINFALL DATA) ---
    rainfall_normal = round(row["R_normal"], 2)
    rainfall_current = round(row["R_current"], 2)
    rainfall_deviation = round(rainfall_dev, 2)

    # ----- GROUNDWATER DEVIATION -----
    if pd.notna(row["GW_last"]) and pd.notna(row["GW_current"]):
        gw_dev = abs(row["GW_last"] - row["GW_current"]) * 10
    else:
        gw_dev = 0

    # --- NEW (RAW GROUNDWATER DATA) ---
    gw_last = round(row["GW_last"], 2) if pd.notna(row["GW_last"]) else None
    gw_current = round(row["GW_current"], 2) if pd.notna(row["GW_current"]) else None
    gw_change = round(abs(gw_last - gw_current), 2) if gw_last is not None and gw_current is not None else None

    # ----- LAND USE -----
    urban = row["Urban_Percent"] if pd.notna(row["Urban_Percent"]) else 0
    forest = row["Forest_Percent"] if pd.notna(row["Forest_Percent"]) else 0
    landuse_score = urban - (forest * 0.5)

    # --- NEW (RAW LAND-USE DATA + CLASSIFICATION) ---
    urban_percent = round(urban, 2)
    forest_percent = round(forest, 2)

    if urban_percent >= 50:
        landuse_type = "Urban-dominant"
    elif urban_percent >= 25:
        landuse_type = "Semi-urban"
    else:
        landuse_type = "Rural / Forest-dominant"

    if risk_type == "flood":
        score = (0.4 * rainfall_dev) + (0.4 * landuse_score) + (0.2 * gw_dev)
        risk_label = "Flood Risk"

        rainfall = round(0.4 * score, 2)
        groundwater = round(0.2 * score, 2)
        landuse = round(0.4 * score, 2)

    else:
        score = (0.4 * rainfall_dev) + (0.4 * gw_dev) + (0.2 * landuse_score)
        risk_label = "Water Scarcity Risk"

        rainfall = round(0.4 * score, 2)
        groundwater = round(0.4 * score, 2)
        landuse = round(0.2 * score, 2)

    level = classify_level(score)

    # ----- EXPLANATION -----
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

        # --- NEW RAW DATA (ADDITIVE ONLY) ---
        "rainfall_normal": rainfall_normal,
        "rainfall_current": rainfall_current,
        "rainfall_deviation": rainfall_deviation,

        "gw_last": gw_last,
        "gw_current": gw_current,
        "gw_change": gw_change,

        "urban_percent": urban_percent,
        "forest_percent": forest_percent,
        "landuse_type": landuse_type
    }

    return render_template(
        "dashboard.html",
        data=dashboard_data,
        selected_panchayat=user_place,
        selected_risk=risk_type
    )


if __name__ == "__main__":
    app.run()
