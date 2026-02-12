import pandas as pd
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point

# Load data
locations = pd.read_csv("panchayat_locations.csv")
master = pd.read_csv("CW_RAS_master_dataset.csv")

# Settings
BUFFER_RADIUS = 3000  # meters (3 km)

def calculate_landuse_percentages(lat, lon):
    try:
        # Create point and buffer
        point = Point(lon, lat)
        gdf_point = gpd.GeoDataFrame(geometry=[point], crs="EPSG:4326")
        gdf_buffer = gdf_point.to_crs(epsg=3857).buffer(BUFFER_RADIUS).to_crs(epsg=4326)

        # Fetch landuse data
        tags = {
            "landuse": True,
            "natural": True
        }

        landuse = ox.features_from_polygon(gdf_buffer.iloc[0], tags)

        if landuse.empty:
            return 0, 0

        landuse = landuse.to_crs(epsg=3857)
        total_area = landuse.geometry.area.sum()

        if total_area == 0:
            return 0, 0

        # Urban calculation
        urban_types = ["residential", "commercial", "industrial", "retail"]
        urban = landuse[
            (landuse.get("landuse").isin(urban_types))
        ]
        urban_area = urban.geometry.area.sum()

        # Forest calculation
        forest = landuse[
            (landuse.get("landuse") == "forest") |
            (landuse.get("natural") == "wood")
        ]
        forest_area = forest.geometry.area.sum()

        urban_percent = (urban_area / total_area) * 100
        forest_percent = (forest_area / total_area) * 100

        return round(urban_percent, 2), round(forest_percent, 2)

    except Exception as e:
        print("Error:", e)
        return 0, 0


# Update master dataset
for index, row in locations.iterrows():
    name = row["Panchayat"]
    lat = row["Latitude"]
    lon = row["Longitude"]

    print(f"Processing {name}...")

    urban_p, forest_p = calculate_landuse_percentages(lat, lon)

    master.loc[master["Panchayat"] == name, "Urban_Percent"] = urban_p
    master.loc[master["Panchayat"] == name, "Forest_Percent"] = forest_p


# Save updated dataset
master.to_csv("CW_RAS_master_dataset_updated.csv", index=False)

print("✅ Land-use updated successfully.")
