import pandas as pd
import requests
import time

# Load master dataset
risk_data = pd.read_csv("CW_RAS_master_dataset.csv")

panchayats = risk_data["Panchayat"].unique().tolist()

results = []

for panchayat in panchayats:
    print(f"Fetching location for: {panchayat}")

    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": f"{panchayat}, Kerala, India",
        "format": "json",
        "limit": 1
    }
    headers = {
        "User-Agent": "CW-RAS-Location-Generator"
    }

    response = requests.get(url, params=params, headers=headers)

    if response.status_code == 200:
        data = response.json()
        if len(data) > 0:
            lat = float(data[0]["lat"])
            lon = float(data[0]["lon"])
        else:
            lat, lon = None, None
    else:
        lat, lon = None, None

    results.append({
        "Panchayat": panchayat,
        "Latitude": lat,
        "Longitude": lon
    })

    time.sleep(1)  # IMPORTANT: respect OSM usage policy

# Save to CSV
df = pd.DataFrame(results)
df.to_csv("panchayat_locations.csv", index=False)

print("panchayat_locations.csv generated successfully")
