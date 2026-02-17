import ee
import pandas as pd

# Initialize Earth Engine
ee.Initialize(project="cwras-landuse")

# Load local files
locations = pd.read_csv("panchayat_locations.csv")
master = pd.read_csv("CW_RAS_master_dataset.csv")

# Load ESA WorldCover 2020 (10m resolution)
landcover = ee.Image("ESA/WorldCover/v100/2020")

def get_landcover_percent(lat, lon, radius=5000):
    point = ee.Geometry.Point(lon, lat)
    region = point.buffer(radius)

    stats = landcover.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(),
        geometry=region,
        scale=10,
        maxPixels=1e9
    ).getInfo()

    if not stats or "Map" not in stats:
        print("No data found, using safe defaults.")
        return 5.0, 80.0  # fallback values

    histogram = stats["Map"]

    # Convert keys to integers (important safety step)
    histogram = {int(k): v for k, v in histogram.items()}

    total = sum(histogram.values())

    # ESA WorldCover Classes
    # 10 = Tree cover
    # 20 = Shrubland
    # 30 = Grassland
    # 40 = Cropland
    # 50 = Built-up
    # 95 = Mangroves

    urban_pixels = histogram.get(50, 0)

    vegetation_pixels = (
        histogram.get(10, 0) +   # Tree cover
        histogram.get(20, 0) +   # Shrubland
        histogram.get(30, 0) +   # Grassland
        histogram.get(40, 0) +   # Cropland
        histogram.get(95, 0)     # Mangroves
    )

    urban_percent = (urban_pixels / total) * 100
    vegetation_percent = (vegetation_pixels / total) * 100

    return round(urban_percent, 2), round(vegetation_percent, 2)


for _, row in locations.iterrows():
    print("Processing:", row["Panchayat"])

    urban, forest = get_landcover_percent(
        row["Latitude"],
        row["Longitude"],
        radius=5000
    )

    print(f"Urban: {urban}% | Vegetation: {forest}%")

    master.loc[
        master["Panchayat"] == row["Panchayat"],
        "Urban_Percent"
    ] = urban

    master.loc[
        master["Panchayat"] == row["Panchayat"],
        "Forest_Percent"
    ] = forest

# Save updated dataset
master.to_csv("CW_RAS_master_dataset_updated.csv", index=False)

print("Done. Updated dataset saved.")
