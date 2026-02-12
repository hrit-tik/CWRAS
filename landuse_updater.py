import pandas as pd
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point

# -----------------------------
# SETTINGS
# -----------------------------
BUFFER_RADIUS = 3000  # meters
OUTPUT_FILE = "CW_RAS_master_dataset_updated.csv"

print("Loading datasets...")

locations = pd.read_csv("panchayat_locations.csv")
master = pd.read_csv("CW_RAS_master_dataset.csv")

locations = locations.dropna(subset=["Latitude", "Longitude"])

tags = {
    "landuse": True,
    "natural": True
}

print("Processing panchayats...")

for index, row in locations.iterrows():

    name = row["Panchayat"]
    lat = row["Latitude"]
    lon = row["Longitude"]

    print(f"Processing {name}...")

    try:
        # Fetch features within 3km
        gdf = ox.features_from_point((lat, lon), tags=tags, dist=BUFFER_RADIUS)

        if gdf.empty:
            urban_percent = 0
            forest_percent = 0
        else:
            # Keep only polygons
            gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]

            if gdf.empty:
                urban_percent = 0
                forest_percent = 0
            else:
                # Project to meters
                gdf = gdf.to_crs(epsg=3857)

                # Create buffer
                point = gpd.GeoDataFrame(
                    geometry=[Point(lon, lat)],
                    crs="EPSG:4326"
                ).to_crs(epsg=3857)

                buffer = point.buffer(BUFFER_RADIUS).iloc[0]
                buffer_area = buffer.area

                # Clip geometries manually (faster than overlay)
                gdf["clipped"] = gdf.geometry.intersection(buffer)
                gdf["area"] = gdf["clipped"].area

                # -------------------------
                # URBAN DETECTION
                # -------------------------
                urban_types = [
                    "residential",
                    "commercial",
                    "industrial",
                    "retail",
                    "construction"
                ]

                urban = gdf[gdf["landuse"].isin(urban_types)]
                urban_area = urban["area"].sum()

                # -------------------------
                # FOREST DETECTION
                # -------------------------
                forest = gdf[
                    (gdf["landuse"] == "forest") |
                    (gdf["natural"] == "wood") |
                    (gdf["natural"] == "scrub")
                ]

                forest_area = forest["area"].sum()

                urban_percent = (urban_area / buffer_area) * 100
                forest_percent = (forest_area / buffer_area) * 100

        master.loc[master["Panchayat"] == name, "Urban_Percent"] = round(urban_percent, 2)
        master.loc[master["Panchayat"] == name, "Forest_Percent"] = round(forest_percent, 2)

    except Exception as e:
        print(f"Error processing {name}: {e}")
        master.loc[master["Panchayat"] == name, "Urban_Percent"] = 0
        master.loc[master["Panchayat"] == name, "Forest_Percent"] = 0

print("Saving updated dataset...")

master.to_csv(OUTPUT_FILE, index=False)

print("✅ Land-use update completed successfully.")
