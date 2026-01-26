from flask import Flask, render_template, request
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load dataset (CSV in same folder as app.py)
risk_data = pd.read_csv("CW_RAS_master_dataset.csv")

panchayat_list = sorted(risk_data["Panchayat"].unique().tolist())


def classify_level(score):
    if score < 15:
        return "Low"
    elif score < 25:
        return "Moderate"
    else:
        return "High"


@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "GET":
        return render_template(
            "index.html",
            panchayats=panchayat_list,
            selected_panchayat=request.args.get("panchayat"),
            selected_risk=request.args.get("risk")
        )

    # ---------- POST LOGIC ----------
    selected_panchayat = request.form["panchayat"]
    risk_type = request.form["risk_type"]

    row = risk_data[risk_data["Panchayat"] == selected_panchayat].iloc[0]

    # ----- RAINFALL DEVIATION -----
    rainfall_dev = abs(row["R_normal"] - row["R_current"]) / row["R_normal"] * 100

    # ----- GROUNDWATER DEVIATION -----
    if pd.notna(row["GW_last"]) and pd.notna(row["GW_current"]):
        gw_dev = abs(row["GW_last"] - row["GW_current"]) * 10
    else:
        gw_dev = 0

    # ----- LAND USE -----
    urban = row["Urban_Percent"] if pd.notna(row["Urban_Percent"]) else 0
    forest = row["Forest_Percent"] if pd.notna(row["Forest_Percent"]) else 0
    landuse_score = urban - (forest * 0.5)

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
        "panchayat": selected_panchayat,
        "risk_type": risk_label,
        "score": round(score, 2),
        "level": level,
        "rainfall": rainfall,
        "groundwater": groundwater,
        "landuse": landuse,
        "explanation": explanation
    }

    return render_template(
        "dashboard.html",
        data=dashboard_data,
        selected_panchayat=selected_panchayat,
        selected_risk=risk_type
    )


if __name__ == "__main__":
    app.run()
