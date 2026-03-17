# CW-RAS — Team Code Division (Presentation Reference)

> This file is **NOT** part of the running application.  
> It is a reference document that shows which code belongs to which team member.

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────┐
│                    CW-RAS Application                    │
├──────────────────────────────────────────────────────────┤
│                                                          │
│   ┌──────────────────────────────────────────────────┐   │
│   │  MODULE 1 — app.py  (Hritik Krishna)             │   │
│   │  Flask App Core + Risk Scoring Engine             │   │
│   │  Routes, Weighted Scoring, Result Assembly        │   │
│   └──────┬────────────┬────────────┬─────────────────┘   │
│          │            │            │                      │
│          ▼            ▼            ▼                      │
│   ┌──────────┐ ┌────────────┐ ┌──────────────────────┐   │
│   │ MODULE 2 │ │  MODULE 3  │ │      MODULE 4        │   │
│   │scoring.py│ │geocoding.py│ │data_loader.py        │   │
│   │(Nandu MV)│ │ (Sruthi S) │ │+ templates/          │   │
│   │          │ │            │ │(Ivan John Benny)     │   │
│   │Normalize │ │OSM Geocode │ │CSV Loading           │   │
│   │Classify  │ │Haversine   │ │index.html            │   │
│   │SWF       │ │Nearest     │ │dashboard.html        │   │
│   │          │ │ Panchayat  │ │algorithm.html        │   │
│   └──────────┘ └────────────┘ └──────────────────────┘   │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---
---

# MODULE 1 — Flask App Core + Risk Scoring Engine

**Owner: Hritik Krishna (PRN23CS071)**  
**File: `app.py`**  
**Role:** Main application controller — sets up Flask, defines routes, implements the weighted risk scoring engine, assembles results, and integrates all other modules.

```python
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import requests
import math

# ---------- IMPORTS FROM OTHER MODULES ----------
# from scoring import normalize_rainfall, normalize_groundwater, normalize_landuse, compute_swf, classify_level
# from geocoding import get_lat_long, haversine_distance, find_nearest_panchayat
# from data_loader import load_risk_data, load_location_data, get_panchayat_list

app = Flask(__name__)

# Load datasets
risk_data = pd.read_csv("CW_RAS_master_dataset.csv")
location_data = pd.read_csv("panchayat_locations.csv")

panchayat_list = sorted(risk_data["Panchayat"].unique().tolist())


# ============================================================
# ROUTE: Home Page + Risk Assessment Engine
# ============================================================
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

    # ============================================================
    # RISK SCORING ENGINE — Weighted Calculation
    # ============================================================
    if risk_type == "flood":
        # Flood: Only rising GW contributes
        # (GW_current < GW_last means water level rose / depth decreased)

        is_rising = False
        if pd.notna(gw_last) and pd.notna(gw_current):
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

    # ----- EXPLANATION GENERATION -----
    if rainfall >= groundwater and rainfall >= landuse:
        explanation = "Rainfall variation is the dominant factor influencing the assessed risk in this region."
    elif groundwater >= landuse:
        explanation = "Groundwater level fluctuation significantly contributes to the assessed risk in this region."
    else:
        explanation = "Land-use characteristics such as urbanization influence the assessed risk in this region."

    # ----- RESULT ASSEMBLY -----
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


# ============================================================
# ROUTE: Algorithm Documentation Page
# ============================================================
@app.route("/algorithm")
def algorithm():
    return render_template("algorithm.html")


if __name__ == "__main__":
    app.run()
```

---
---

# MODULE 2 — Risk Assessment Algorithms (Scoring Module)

**Owner: Nandu M V (PRN23CS099)**  
**File: `scoring.py`**  
**Role:** Contains all normalization functions that convert raw environmental data into 0–100 risk scores, the Surface Water Factor (SWF) computation, and the risk level classifier.

```python
"""
scoring.py — CW-RAS Risk Assessment Algorithms
Handles all normalization and classification logic for the SR-SA v2 algorithm.
"""

import pandas as pd


def classify_level(score):
    """Classify a 0-100 risk score into Low / Moderate / High."""
    if score < 30:
        return "Low"
    elif score < 60:
        return "Moderate"
    else:
        return "High"


def normalize_rainfall(r_normal, r_current):
    """
    Rainfall deviation normalized to 0-100, capped at 100.
    
    Formula:
        Deviation = |R_normal - R_current| / R_normal * 100
    
    Higher deviation → higher risk score.
    """
    if r_normal == 0:
        return 0
    dev = abs(r_normal - r_current) / r_normal * 100
    return min(dev, 100)


def normalize_groundwater(gw_last, gw_current):
    """
    Groundwater change normalized to 0-100 using 3m reference max.
    
    Formula:
        Score = |GW_last - GW_current| / 3.0 * 100
    
    Larger fluctuations indicate higher vulnerability.
    """
    MAX_GW_CHANGE = 3.0
    if pd.isna(gw_last) or pd.isna(gw_current):
        return 0
    change = abs(gw_last - gw_current)
    score = (change / MAX_GW_CHANGE) * 100
    return min(score, 100)


def normalize_landuse(urban_pct, forest_pct):
    """
    Land-use score normalized to 0-100. Higher = more runoff-prone.
    
    Formula:
        Urban Component   = (Urban% / 100) * 50
        Forest Deficit    = ((100 - Forest%) / 100) * 50
        Land-use Score    = min(Urban + Forest_Deficit, 100)
    """
    urban_component = (urban_pct / 100) * 50
    forest_deficit = ((100 - forest_pct) / 100) * 50
    return min(urban_component + forest_deficit, 100)


def compute_swf(water_body_pct):
    """
    Surface Water Factor: moderates scarcity for lake-adjacent areas.
    
    Formula:
        SWF = max(1.0 - (Water_Body_Percent / 50), 0.1)
        Range: [0.1, 1.0]
    
    Areas near large water bodies have reduced scarcity risk.
    Example: 35% water body → SWF = 0.30 → scarcity reduced by 70%.
    """
    if pd.isna(water_body_pct) or water_body_pct <= 0:
        return 1.0
    return max(1.0 - (water_body_pct / 50), 0.1)
```

---
---

# MODULE 3 — Geolocation Engine

**Owner: Sruthi S (PRN23CS121)**  
**File: `geocoding.py`**  
**Role:** Handles location lookup via OpenStreetMap Nominatim API, computes distances using the Haversine formula, and finds the nearest panchayat to a user-provided location.

```python
"""
geocoding.py — CW-RAS Geolocation Engine
Handles OSM geocoding, Haversine distance calculation,
and nearest-panchayat matching.
"""

import math
import requests


def get_lat_long(place_name):
    """
    Geocode a place name to latitude/longitude using OpenStreetMap Nominatim.
    
    Args:
        place_name: Name of the location to search for.
    
    Returns:
        (latitude, longitude) tuple, or (None, None) if not found.
    """
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


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on Earth
    using the Haversine formula.
    
    Args:
        lat1, lon1: Latitude and longitude of point 1 (degrees).
        lat2, lon2: Latitude and longitude of point 2 (degrees).
    
    Returns:
        Distance in kilometers.
    
    Formula:
        a = sin²(Δlat/2) + cos(lat1) · cos(lat2) · sin²(Δlon/2)
        c = 2 · arcsin(√a)
        d = R · c,  where R = 6371 km (Earth's radius)
    """
    R = 6371  # Earth radius in km

    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))

    return R * c


def find_nearest_panchayat(lat, lon, location_data):
    """
    Find the nearest panchayat to a given coordinate using brute-force
    nearest-neighbor search with Haversine distance.
    
    Args:
        lat: Latitude of the user's location.
        lon: Longitude of the user's location.
        location_data: DataFrame with columns [Panchayat, Latitude, Longitude].
    
    Returns:
        Name of the nearest panchayat (string).
    """
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
```

---
---

# MODULE 4 — Data Loading + Frontend UI

**Owner: Ivan John Benny (PRN23CS074)**  
**File: `data_loader.py` + `templates/`**  
**Role:** Handles loading and preparing the CSV datasets, and all frontend HTML/CSS templates that form the user interface.

## 4A. Data Loader (`data_loader.py`)

```python
"""
data_loader.py — CW-RAS Data Loading Module
Loads and prepares CSV datasets used by the application.
"""

import pandas as pd


def load_risk_data(filepath="CW_RAS_master_dataset.csv"):
    """
    Load the master risk dataset containing panchayat-level environmental data.
    
    Columns include:
        - Panchayat: Name of the panchayat
        - R_normal: Long-term normal rainfall (mm)
        - R_current: Current observed rainfall (mm)
        - GW_last: Previous groundwater level (meters below ground level)
        - GW_current: Current groundwater level (mbgl)
        - Urban_Percent: Percentage of urban land cover
        - Forest_Percent: Percentage of forest cover
        - Water_Body_Percent: Percentage of water body coverage
    
    Returns:
        pandas DataFrame with 74 panchayat records.
    """
    return pd.read_csv(filepath)


def load_location_data(filepath="panchayat_locations.csv"):
    """
    Load panchayat geographic coordinates for geolocation matching.
    
    Columns:
        - Panchayat: Name
        - Latitude: Decimal degrees
        - Longitude: Decimal degrees
    
    Returns:
        pandas DataFrame with lat/lon for each panchayat.
    """
    return pd.read_csv(filepath)


def get_panchayat_list(risk_data):
    """
    Extract and return a sorted list of unique panchayat names
    from the master dataset.
    
    Args:
        risk_data: DataFrame from load_risk_data().
    
    Returns:
        Sorted list of panchayat name strings.
    """
    return sorted(risk_data["Panchayat"].unique().tolist())
```

---

## 4B. Frontend — Home Page (`templates/index.html`)

The landing page with a glassmorphism dark-theme UI card containing:
- System analysis overview grid
- Input form (panchayat name + risk type selector)
- Group info dropdown + algorithm link button
- Animated CSS background with blob gradients

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CW-RAS · Contextual Water Risk Assessment</title>
    <meta name="description" content="CW-RAS — Contextual Water Risk Assessment System for panchayat-level flood and water-scarcity risk evaluation.">

    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">

    <style>
        /* ===== RESET & BASE ===== */
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'Inter', sans-serif;
            background: #060e18;
            color: #e2edf8;
            min-height: 100vh;
            overflow-x: hidden;
        }

        /* ===== ANIMATED BACKGROUND ===== */
        body::before,
        body::after {
            content: '';
            position: fixed;
            border-radius: 50%;
            filter: blur(120px);
            opacity: .18;
            z-index: 0;
            pointer-events: none;
        }
        body::before {
            width: 600px; height: 600px;
            background: #1e90ff;
            top: -180px; left: -120px;
            animation: blobDrift 18s ease-in-out infinite alternate;
        }
        body::after {
            width: 500px; height: 500px;
            background: #22d3ee;
            bottom: -160px; right: -100px;
            animation: blobDrift 22s ease-in-out infinite alternate-reverse;
        }
        @keyframes blobDrift {
            0%   { transform: translate(0, 0) scale(1); }
            50%  { transform: translate(60px, 40px) scale(1.12); }
            100% { transform: translate(-40px, 20px) scale(.95); }
        }

        /* ===== ANIMATIONS ===== */
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(24px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes shimmer {
            0%   { background-position: -200% center; }
            100% { background-position: 200% center; }
        }

        /* ===== GROUP BUTTON (top-left) ===== */
        .group-wrapper { position: fixed; top: 20px; left: 24px; z-index: 100; }
        .group-btn {
            padding: 9px 18px; border-radius: 22px;
            border: 1px solid rgba(255,255,255,.12); cursor: pointer;
            font-size: 13px; font-weight: 600; font-family: 'Inter', sans-serif;
            color: #fff; background: rgba(255,255,255,.06);
            backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
            box-shadow: 0 4px 20px rgba(0,0,0,.35); transition: all .3s ease;
        }
        .group-btn:hover { background: rgba(255,255,255,.12); box-shadow: 0 6px 28px rgba(0,0,0,.5); }
        .group-box {
            opacity: 0; visibility: hidden; transform: translateY(8px);
            transition: all .35s cubic-bezier(.4,0,.2,1);
            margin-top: 12px; width: 300px; padding: 22px; border-radius: 20px;
            background: rgba(10,24,40,.92); backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(255,255,255,.08);
            box-shadow: 0 20px 50px rgba(0,0,0,.6);
            font-size: 13px; line-height: 1.75;
        }
        .group-box strong { color: #4ade80; font-weight: 600; }
        .group-wrapper:hover .group-box { opacity: 1; visibility: visible; transform: translateY(0); }

        /* ===== ALGO BUTTON (bottom-right) ===== */
        .algo-btn {
            position: fixed; bottom: 28px; right: 28px;
            padding: 11px 20px; font-size: 13px; font-weight: 600;
            font-family: 'Inter', sans-serif; color: #7dd3fc;
            background: rgba(10,24,40,.85); backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border: 1px solid rgba(125,211,252,.2); border-radius: 22px;
            text-decoration: none; box-shadow: 0 8px 30px rgba(0,0,0,.45);
            z-index: 90; transition: all .3s ease;
        }
        .algo-btn:hover {
            background: rgba(15,38,60,.95); border-color: rgba(125,211,252,.45);
            box-shadow: 0 8px 35px rgba(30,144,255,.15); transform: translateY(-2px);
        }

        /* ===== MAIN CARD ===== */
        .card {
            position: relative; z-index: 1; max-width: 720px;
            margin: 80px auto 40px; padding: 48px 44px;
            background: rgba(12,28,46,.55); backdrop-filter: blur(24px);
            -webkit-backdrop-filter: blur(24px);
            border: 1px solid rgba(255,255,255,.07); border-radius: 28px;
            box-shadow: 0 32px 64px rgba(0,0,0,.45), inset 0 1px 0 rgba(255,255,255,.05);
            animation: fadeInUp .7s ease both;
        }

        /* ===== HEADING ===== */
        h1 {
            text-align: center; font-size: 36px; font-weight: 800; letter-spacing: 2px;
            background: linear-gradient(135deg, #60a5fa, #22d3ee, #4ade80);
            background-size: 200% auto;
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text; animation: shimmer 6s linear infinite; margin-bottom: 6px;
        }
        .subtitle {
            text-align: center; color: #8eafc8; font-size: 14px;
            font-weight: 500; margin-bottom: 40px; letter-spacing: .3px;
        }

        /* ===== OVERVIEW BOX ===== */
        .overview {
            border: 1px solid rgba(255,255,255,.06); border-radius: 20px;
            padding: 26px 28px; margin-bottom: 36px;
            background: rgba(255,255,255,.025); animation: fadeInUp .7s ease .15s both;
        }
        .overview h3 { margin: 0 0 18px 0; color: #4ade80; font-size: 15px; font-weight: 600; letter-spacing: .4px; }
        .overview-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 14px 28px; font-size: 14px; color: #b0c9de;
        }
        .overview-grid div { padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,.04); }
        .overview-grid strong { color: #e2edf8; font-weight: 600; }

        /* ===== FORM ===== */
        form { animation: fadeInUp .7s ease .3s both; }
        label {
            display: block; margin-top: 24px; font-size: 13px; font-weight: 600;
            color: #8eafc8; text-transform: uppercase; letter-spacing: .8px;
        }
        input, select {
            width: 100%; padding: 15px 18px; margin-top: 10px; border-radius: 14px;
            border: 1px solid rgba(255,255,255,.08); font-size: 15px;
            font-family: 'Inter', sans-serif; background: rgba(8,20,35,.7);
            color: #e2edf8; outline: none; transition: all .3s ease;
        }
        input:focus, select:focus {
            border-color: rgba(96,165,250,.5);
            box-shadow: 0 0 0 3px rgba(96,165,250,.12), 0 0 20px rgba(96,165,250,.08);
        }
        input::placeholder { color: #5a7a94; }
        select option { background: #0c1c2e; color: #e2edf8; }

        /* ===== SUBMIT BUTTON ===== */
        .submit-btn {
            width: 100%; margin-top: 36px; padding: 16px; border-radius: 16px;
            border: none; font-size: 15px; font-weight: 700;
            font-family: 'Inter', sans-serif; letter-spacing: .5px; cursor: pointer;
            color: #fff; background: linear-gradient(135deg, #2563eb, #0ea5e9, #06b6d4);
            background-size: 200% auto; box-shadow: 0 6px 24px rgba(37,99,235,.3);
            transition: all .35s ease; position: relative; overflow: hidden;
        }
        .submit-btn::after {
            content: ''; position: absolute; inset: 0;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,.15), transparent);
            transform: translateX(-100%); transition: transform .5s ease;
        }
        .submit-btn:hover {
            transform: translateY(-2px); box-shadow: 0 10px 36px rgba(37,99,235,.4);
            background-position: right center;
        }
        .submit-btn:hover::after { transform: translateX(100%); }
        .submit-btn:active { transform: translateY(0) scale(.985); }

        /* ===== ERROR ===== */
        .error {
            margin-top: 22px; text-align: center; color: #f87171;
            font-size: 14px; font-weight: 500; padding: 12px; border-radius: 12px;
            background: rgba(248,113,113,.08); border: 1px solid rgba(248,113,113,.15);
            animation: fadeInUp .4s ease both;
        }

        /* ===== FOOTER ===== */
        .footer { text-align: center; margin-top: 38px; font-size: 12px; color: #5a7a94; letter-spacing: .3px; }

        /* ===== RESPONSIVE ===== */
        @media (max-width: 768px) {
            .card { margin: 60px 16px 32px; padding: 36px 24px; }
            h1 { font-size: 28px; }
            .group-wrapper { top: 14px; left: 16px; }
            .algo-btn { bottom: 18px; right: 18px; font-size: 12px; padding: 9px 16px; }
            .group-box { width: 260px; }
        }
    </style>
</head>

<body>

<div class="group-wrapper">
    <button class="group-btn">Group 5</button>
    <div class="group-box">
        <strong>Group Members</strong><br>
        Nandu M V (PRN23CS099)<br>
        Sruthi S (PRN23CS121)<br>
        Hritik Krishna (PRN23CS071)<br>
        Ivan John Benny (PRN23CS074)
        <br><br>
        <strong>Under the guidance of</strong><br>
        SIJIMOL K<br>
        Assistant Professor<br>
        Department of Computer Science &amp; Engineering<br>
        College of Engineering, Perumon
    </div>
</div>

<a href="/algorithm" class="algo-btn">
    Learn more about our algorithm →
</a>

<div class="card">
    <h1>CW-RAS</h1>
    <div class="subtitle">Contextual Water Risk Assessment System</div>

    <div class="overview">
        <h3>System Analysis Overview</h3>
        <div class="overview-grid">
            <div><strong>Algorithm:</strong> SR-SA v2</div>
            <div><strong>Risk Models:</strong> Flood &amp; Scarcity</div>
            <div><strong>Primary Input:</strong> Panchayat-level Data</div>
            <div><strong>Rainfall:</strong> Deviation Analysis</div>
            <div><strong>Groundwater:</strong> Level Change</div>
            <div><strong>Land-use:</strong> Urban &amp; Forest Cover</div>
        </div>
    </div>

    <form method="POST">
        <label>Location or Panchayat</label>
        <input type="text" name="panchayat" placeholder="Eg: Kottiyam, Chathannoor, Kollam" required>

        <label>Risk Assessment Type</label>
        <select name="risk_type" required>
            <option value="flood">Flood Risk</option>
            <option value="scarcity">Water Scarcity Risk</option>
        </select>

        <button type="submit" class="submit-btn">Run Risk Assessment</button>
    </form>

    {% if error %}
        <div class="error">{{ error }}</div>
    {% endif %}

    <div class="footer">
        Mini Project · CW-RAS
    </div>
</div>

</body>
</html>
```

---

## 4C. Frontend — Results Dashboard (`templates/dashboard.html`)

The results page showing risk scores with animated progress bars, metric cards, and color-coded risk badges.

```html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CW-RAS Dashboard · {{ data.panchayat }}</title>

    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">

    <style>
        /* ===== RESET & BASE ===== */
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'Inter', sans-serif; background: #060e18;
            color: #e2edf8; min-height: 100vh; overflow-x: hidden;
        }

        /* ===== BACKGROUND BLOBS ===== */
        body::before, body::after {
            content: ''; position: fixed; border-radius: 50%;
            filter: blur(130px); opacity: .14; z-index: 0; pointer-events: none;
        }
        body::before {
            width: 650px; height: 650px; background: #3b82f6;
            top: -200px; left: -150px;
            animation: blobDrift 20s ease-in-out infinite alternate;
        }
        body::after {
            width: 500px; height: 500px; background: #06b6d4;
            bottom: -180px; right: -120px;
            animation: blobDrift 24s ease-in-out infinite alternate-reverse;
        }
        @keyframes blobDrift {
            0%   { transform: translate(0, 0) scale(1); }
            50%  { transform: translate(50px, 30px) scale(1.1); }
            100% { transform: translate(-30px, 15px) scale(.94); }
        }

        /* ===== ANIMATIONS ===== */
        @keyframes fadeInUp { from { opacity: 0; transform: translateY(22px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes barFill { from { width: 0; } }
        @keyframes barShine { 0% { left: -40%; } 100% { left: 140%; } }
        @keyframes pulseGlow { 0%, 100% { box-shadow: 0 0 8px currentColor; } 50% { box-shadow: 0 0 20px currentColor; } }

        /* ===== CONTAINER ===== */
        .container { position: relative; z-index: 1; max-width: 1020px; margin: 50px auto; padding: 30px; }
        h1 {
            text-align: center; font-size: 32px; font-weight: 800; letter-spacing: 1.5px;
            background: linear-gradient(135deg, #60a5fa, #22d3ee, #4ade80);
            background-size: 200% auto;
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text; margin-bottom: 6px;
        }
        .subtitle { text-align: center; font-size: 14px; font-weight: 500; color: #8eafc8; margin-bottom: 44px; }

        /* ===== CARDS ===== */
        .card {
            background: rgba(12, 28, 46, .5); backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, .06); border-radius: 22px;
            padding: 28px; margin-top: 26px;
            box-shadow: 0 20px 50px rgba(0, 0, 0, .4);
            animation: fadeInUp .6s ease both;
        }
        .card:nth-child(2) { animation-delay: .08s; }
        .card:nth-child(3) { animation-delay: .16s; }
        .card:nth-child(4) { animation-delay: .24s; }
        .card:nth-child(5) { animation-delay: .32s; }

        /* ===== HEADER CARD ===== */
        .header { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px; }
        .location-title { font-size: 22px; font-weight: 700; letter-spacing: .3px; }
        .risk-type { font-size: 14px; color: #8eafc8; margin-top: 4px; font-weight: 500; }

        /* ===== RISK BADGE ===== */
        .risk-level { font-size: 14px; font-weight: 700; padding: 10px 22px; border-radius: 24px; letter-spacing: .5px; animation: pulseGlow 2.5s ease-in-out infinite; }
        .risk-low { background: rgba(6, 78, 59, .6); color: #4ade80; border: 1px solid rgba(74, 222, 128, .25); }
        .risk-moderate { background: rgba(120, 53, 15, .5); color: #fbbf24; border: 1px solid rgba(251, 191, 36, .25); }
        .risk-high { background: rgba(127, 29, 29, .5); color: #f87171; border: 1px solid rgba(248, 113, 113, .25); }

        /* ===== OVERVIEW GRID ===== */
        .overview { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 20px; }
        .overview-item {
            background: rgba(255, 255, 255, .03); border: 1px solid rgba(255, 255, 255, .05);
            border-radius: 18px; padding: 22px;
            transition: transform .25s ease, box-shadow .25s ease;
        }
        .overview-item:hover { transform: translateY(-3px); box-shadow: 0 12px 30px rgba(0, 0, 0, .3); }
        .overview-item h4 { margin: 0; font-size: 13px; font-weight: 600; color: #8eafc8; text-transform: uppercase; letter-spacing: .6px; margin-bottom: 14px; }
        .overview-item p { font-size: 14px; line-height: 1.8; color: #b0c9de; }
        .overview-item strong { color: #e2edf8; }

        /* ===== METRICS GRID ===== */
        .metrics { display: grid; grid-template-columns: repeat(2, 1fr); gap: 22px; }
        .metric {
            background: rgba(255, 255, 255, .025); border: 1px solid rgba(255, 255, 255, .05);
            border-radius: 18px; padding: 24px;
            transition: transform .25s ease, box-shadow .25s ease;
        }
        .metric:hover { transform: translateY(-3px); box-shadow: 0 12px 30px rgba(0, 0, 0, .3); }
        .metric h3 { margin: 0; font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: .6px; color: #7dd3a7; }
        .metric .value { font-size: 34px; font-weight: 800; margin-top: 10px; letter-spacing: -.5px; }
        .metric .details { margin-top: 10px; font-size: 13px; color: #8eafc8; line-height: 1.6; }

        /* ===== PROGRESS BARS ===== */
        .bar { height: 12px; background: rgba(255, 255, 255, .06); border-radius: 10px; margin-top: 14px; overflow: hidden; }
        .fill { height: 100%; border-radius: 10px; position: relative; animation: barFill 1.2s cubic-bezier(.4, 0, .2, 1) both; overflow: hidden; }
        .fill::after {
            content: ''; position: absolute; top: 0; left: -40%; width: 40%; height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, .3), transparent);
            animation: barShine 2s ease-in-out 1.2s infinite;
        }
        .rain { background: linear-gradient(90deg, #3b82f6, #22d3ee); }
        .ground { background: linear-gradient(90deg, #06b6d4, #34d399); }
        .land { background: linear-gradient(90deg, #4ade80, #86efac); }

        /* ===== SCORE HIGHLIGHT ===== */
        .score-value {
            font-size: 42px; font-weight: 800;
            background: linear-gradient(135deg, #60a5fa, #22d3ee);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text; letter-spacing: -1px;
        }

        /* ===== WEIGHTS ===== */
        .weights { text-align: center; font-size: 13px; color: #6b8fa8; margin-top: 20px; font-weight: 500; }

        /* ===== EXPLANATION ===== */
        .explanation-card h3 { color: #4ade80; font-size: 16px; font-weight: 700; margin-bottom: 14px; }
        .explanation { font-size: 15px; line-height: 1.8; color: #b0c9de; }
        .explanation strong { color: #e2edf8; }

        /* ===== BACK LINK ===== */
        .back { text-align: center; margin-top: 42px; animation: fadeInUp .6s ease .4s both; }
        .back a {
            color: #60a5fa; text-decoration: none; font-weight: 600; font-size: 14px;
            padding: 12px 28px; border-radius: 18px;
            border: 1px solid rgba(96, 165, 250, .2); background: rgba(96, 165, 250, .06);
            transition: all .3s ease; display: inline-block;
        }
        .back a:hover {
            background: rgba(96, 165, 250, .12); border-color: rgba(96, 165, 250, .4);
            transform: translateX(-4px); box-shadow: 0 6px 20px rgba(96, 165, 250, .12);
        }

        /* ===== RESPONSIVE ===== */
        @media (max-width: 768px) {
            .container { margin: 30px auto; padding: 18px; }
            h1 { font-size: 26px; }
            .metrics { grid-template-columns: 1fr; }
            .header { flex-direction: column; align-items: flex-start; }
            .score-value { font-size: 34px; }
        }
    </style>
</head>

<body>

    <div class="container">

        <h1>CW-RAS Dashboard</h1>
        <div class="subtitle">Contextual Water Risk Assessment System</div>

        <!-- HEADER -->
        <div class="card header">
            <div>
                <div class="location-title">📍 {{ data.panchayat }}</div>
                <div class="risk-type">{{ data.risk_type }}</div>
            </div>

            <div class="risk-level
        {% if data.level == 'Low' %}risk-low
        {% elif data.level == 'Moderate' %}risk-moderate
        {% else %}risk-high{% endif %}">
                {{ data.level }} Risk
            </div>
        </div>

        <!-- OVERVIEW WITH RAW DATA -->
        <div class="card overview">
            <div class="overview-item">
                <h4>🌧 Rainfall Characteristics</h4>
                <p>
                    Normal: <strong>{{ data.rainfall_normal }} mm</strong><br>
                    Current: <strong>{{ data.rainfall_current }} mm</strong><br>
                    Deviation: <strong>{{ data.rainfall_deviation }} %</strong>
                </p>
            </div>

            <div class="overview-item">
                <h4>💧 Groundwater Trends</h4>
                <p>
                    Previous: <strong>{{ data.gw_last }} m</strong><br>
                    Current: <strong>{{ data.gw_current }} m</strong><br>
                    Change: <strong>{{ data.gw_change }} m</strong>
                </p>
            </div>

            <div class="overview-item">
                <h4>🌱 Land-use Profile</h4>
                <p>
                    Urban: <strong>{{ data.urban_percent }} %</strong><br>
                    Forest: <strong>{{ data.forest_percent }} %</strong><br>
                    Type: <strong>{{ data.landuse_type }}</strong>
                </p>
            </div>
        </div>

        <!-- METRICS -->
        <div class="card">
            <div class="metrics">
                <div class="metric">
                    <h3>Overall Risk Score</h3>
                    <div class="score-value">{{ data.score }}</div>
                </div>

                <div class="metric">
                    <h3>Rainfall Impact</h3>
                    <div class="value">{{ data.rainfall }}</div>
                    <div class="bar">
                        <div class="fill rain" style="width: {{ data.rainfall * 10 }}%"></div>
                    </div>
                </div>

                <div class="metric">
                    <h3>Groundwater Impact</h3>
                    <div class="value">{{ data.groundwater }}</div>
                    <div class="bar">
                        <div class="fill ground" style="width: {{ data.groundwater * 10 }}%"></div>
                    </div>
                </div>

                <div class="metric">
                    <h3>Land-use Impact</h3>
                    <div class="value">{{ data.landuse }}</div>
                    <div class="bar">
                        <div class="fill land" style="width: {{ data.landuse * 10 }}%"></div>
                    </div>
                </div>
            </div>

            <div class="weights">
                Weight Distribution: Rainfall (40%), Groundwater (40%), Land-use (20%)
            </div>
        </div>

        <!-- EXPLANATION -->
        <div class="card explanation-card">
            <h3>Assessment Explanation</h3>
            <p class="explanation">
                Based on the combined evaluation of rainfall deviation, groundwater fluctuation,
                and land-use characteristics at the panchayat level, the assessed location exhibits
                a <strong>{{ data.level|lower }}</strong> water-related risk profile.
                {{ data.explanation }}
            </p>
        </div>

        <div class="back">
            <a href="/">← Assess another location</a>
        </div>

    </div>
</body>

</html>
```

---

## 4D. Frontend — Algorithm Documentation (`templates/algorithm.html`)

Technical documentation page for the SR-SA v2 algorithm with equation blocks, weight visualization bars, and sample calculation.

> **Note:** This template is 637 lines. Full code is in `templates/algorithm.html`.  
> Key sections it covers:
> - Algorithm Overview
> - Input Parameters (flow grid)
> - Rainfall Deviation Calculation (equation block)
> - Groundwater Fluctuation Assessment (equation block)
> - Land-use Influence (equation block)
> - Risk Score Aggregation — Flood & Scarcity formulas
> - Surface Water Moderation Factor (SWF)
> - Weight Distribution Visualization (colored weight bars)
> - Risk Level Classification thresholds
> - Sample Score Breakdown (interactive sample grid)
> - Key Characteristics of SR-SA v2

---
---

# Quick Reference: Who Explains What

| Member | Module | Key Talking Points |
|--------|--------|--------------------|
| **Hritik Krishna** | `app.py` — Flask Core + Scoring Engine | Flask routes, POST handling, weighted risk scoring (flood vs scarcity), directional GW logic, flood boost, explanation generation, result assembly |
| **Nandu M V** | `scoring.py` — Risk Algorithms | 5 mathematical functions: rainfall normalization, groundwater normalization, land-use scoring, SWF formula, risk classification thresholds |
| **Sruthi S** | `geocoding.py` — Geolocation Engine | OSM Nominatim API integration, Haversine formula derivation, nearest-neighbor panchayat search |
| **Ivan John Benny** | `data_loader.py` + Templates | CSV data loading, DataFrame preparation, HTML/CSS UI design, glassmorphism styling, animated progress bars, responsive layout |
