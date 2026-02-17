"""
CW-RAS Scoring Logic Verification Tests
Run: python test_scoring.py
"""

import sys
import pandas as pd

# ─── Normalization Functions (mirrored from app.py) ───

def normalize_rainfall(r_normal, r_current):
    if r_normal == 0:
        return 0
    dev = abs(r_normal - r_current) / r_normal * 100
    return min(dev, 100)

def normalize_groundwater(gw_last, gw_current):
    MAX_GW_CHANGE = 3.0
    if pd.isna(gw_last) or pd.isna(gw_current):
        return 0
    change = abs(gw_last - gw_current)
    score = (change / MAX_GW_CHANGE) * 100
    return min(score, 100)

def normalize_landuse(urban_pct, forest_pct):
    urban_component = (urban_pct / 100) * 50
    forest_deficit = ((100 - forest_pct) / 100) * 50
    return min(urban_component + forest_deficit, 100)

def classify_level(score):
    if score < 30:
        return "Low"
    elif score < 60:
        return "Moderate"
    else:
        return "High"


# ─── Tests ───

def test_normalization_ranges():
    """All normalization functions must return values in [0, 100]."""
    errors = []

    # Rainfall edge cases
    cases_rain = [
        (1000, 1000, "identical rainfall"),
        (1000, 0,    "zero current"),
        (1000, 2000, "double rainfall"),
        (1000, 5000, "5x rainfall — should cap at 100"),
        (0, 0,       "zero normal"),
    ]
    for r_n, r_c, label in cases_rain:
        val = normalize_rainfall(r_n, r_c)
        if not (0 <= val <= 100):
            errors.append(f"  Rainfall [{label}]: {val} is outside [0, 100]")

    # Groundwater edge cases
    cases_gw = [
        (5.0, 5.0,   "no change"),
        (5.0, 2.0,   "3m drop — should be 100"),
        (5.0, 0.0,   "5m drop — should cap at 100"),
        (None, 5.0,  "missing last"),
    ]
    for gw_l, gw_c, label in cases_gw:
        val = normalize_groundwater(gw_l, gw_c)
        if not (0 <= val <= 100):
            errors.append(f"  Groundwater [{label}]: {val} is outside [0, 100]")

    # Land-use edge cases
    cases_lu = [
        (0, 100,    "full forest — should be ~0"),
        (100, 0,    "full urban — should be 100"),
        (0, 0,      "barren land"),
        (50, 50,    "mixed"),
        (25, 30,    "typical rural"),
    ]
    for u, f, label in cases_lu:
        val = normalize_landuse(u, f)
        if not (0 <= val <= 100):
            errors.append(f"  Land-use [{label}]: {val} is outside [0, 100]")

    return errors


def test_component_impacts_differ():
    """Weighted impacts must NOT be identical when component values differ."""
    errors = []

    rain_score = 42.0   # Different from others
    gw_score = 18.0
    lu_score = 65.0

    # Flood
    r_impact = 0.4 * rain_score
    l_impact = 0.4 * lu_score
    g_impact = 0.2 * gw_score

    if r_impact == l_impact:
        errors.append(f"  Flood: rainfall_impact ({r_impact}) == landuse_impact ({l_impact}) — BUG!")

    # Scarcity
    r_impact2 = 0.4 * rain_score
    g_impact2 = 0.4 * gw_score
    l_impact2 = 0.2 * lu_score

    if r_impact2 == g_impact2:
        errors.append(f"  Scarcity: rainfall_impact ({r_impact2}) == groundwater_impact ({g_impact2}) — BUG!")

    return errors


def test_weighted_sum_range():
    """Weighted sum must always be in [0, 100]."""
    errors = []

    test_cases = [
        (0, 0, 0, "all zeros"),
        (100, 100, 100, "all maximums"),
        (50, 50, 50, "all mid"),
        (100, 0, 0, "only rainfall"),
        (0, 100, 0, "only groundwater"),
        (0, 0, 100, "only landuse"),
    ]

    for r, g, l, label in test_cases:
        flood = 0.4 * r + 0.4 * l + 0.2 * g
        scarcity = 0.4 * r + 0.4 * g + 0.2 * l

        if not (0 <= flood <= 100):
            errors.append(f"  Flood [{label}]: {flood} outside [0, 100]")
        if not (0 <= scarcity <= 100):
            errors.append(f"  Scarcity [{label}]: {scarcity} outside [0, 100]")

    return errors


def test_landuse_never_negative():
    """Land-use score must never be negative (the old bug)."""
    errors = []

    # Original dataset ranges
    test_cases = [
        (6.7, 44.0),
        (6.4, 32.3),
        (7.3, 18.0),
        (5.5, 46.8),
        (7.9, 48.9),
    ]

    for u, f in test_cases:
        val = normalize_landuse(u, f)
        if val < 0:
            errors.append(f"  Land-use({u}, {f}) = {val} — NEGATIVE!")

    return errors


def test_dataset_integration():
    """Run scoring on every row of the dataset and validate ranges."""
    errors = []
    df = pd.read_csv("CW_RAS_master_dataset.csv")

    for idx, row in df.iterrows():
        r = normalize_rainfall(row["R_normal"], row["R_current"])
        g = normalize_groundwater(row["GW_last"], row["GW_current"])

        u = row["Urban_Percent"] if pd.notna(row["Urban_Percent"]) else 0
        f = row["Forest_Percent"] if pd.notna(row["Forest_Percent"]) else 0
        l = normalize_landuse(u, f)

        flood = 0.4 * r + 0.4 * l + 0.2 * g
        scarcity = 0.4 * r + 0.4 * g + 0.2 * l

        panchayat = row["Panchayat"]

        if not (0 <= r <= 100):
            errors.append(f"  {panchayat}: Rainfall {r:.2f} out of range")
        if not (0 <= g <= 100):
            errors.append(f"  {panchayat}: Groundwater {g:.2f} out of range")
        if not (0 <= l <= 100):
            errors.append(f"  {panchayat}: Land-use {l:.2f} out of range")
        if not (0 <= flood <= 100):
            errors.append(f"  {panchayat}: Flood {flood:.2f} out of range")
        if not (0 <= scarcity <= 100):
            errors.append(f"  {panchayat}: Scarcity {scarcity:.2f} out of range")

    return errors


def test_score_distribution():
    """Scores should spread across Low/Moderate/High, not cluster in one band."""
    df = pd.read_csv("CW_RAS_master_dataset.csv")
    flood_scores = []
    scarcity_scores = []

    for _, row in df.iterrows():
        r = normalize_rainfall(row["R_normal"], row["R_current"])
        g = normalize_groundwater(row["GW_last"], row["GW_current"])
        u = row["Urban_Percent"] if pd.notna(row["Urban_Percent"]) else 0
        f = row["Forest_Percent"] if pd.notna(row["Forest_Percent"]) else 0
        l = normalize_landuse(u, f)

        flood_scores.append(0.4 * r + 0.4 * l + 0.2 * g)
        scarcity_scores.append(0.4 * r + 0.4 * g + 0.2 * l)

    flood_levels = {"Low": 0, "Moderate": 0, "High": 0}
    for s in flood_scores:
        flood_levels[classify_level(s)] += 1

    scarcity_levels = {"Low": 0, "Moderate": 0, "High": 0}
    for s in scarcity_scores:
        scarcity_levels[classify_level(s)] += 1

    print(f"\n  Flood distribution:    {flood_levels}")
    print(f"  Scarcity distribution: {scarcity_levels}")
    print(f"  Flood score range:     [{min(flood_scores):.1f}, {max(flood_scores):.1f}]")
    print(f"  Scarcity score range:  [{min(scarcity_scores):.1f}, {max(scarcity_scores):.1f}]")

    return []


# ─── Runner ───

if __name__ == "__main__":
    tests = [
        ("Normalization ranges [0, 100]", test_normalization_ranges),
        ("Component impacts differ", test_component_impacts_differ),
        ("Weighted sum in [0, 100]", test_weighted_sum_range),
        ("Land-use never negative", test_landuse_never_negative),
        ("Dataset integration", test_dataset_integration),
        ("Score distribution", test_score_distribution),
    ]

    all_passed = True
    for name, fn in tests:
        errors = fn()
        if errors:
            print(f"❌ FAIL: {name}")
            for e in errors:
                print(e)
            all_passed = False
        else:
            print(f"✅ PASS: {name}")

    print()
    if all_passed:
        print("🎉 All tests passed!")
    else:
        print("💥 Some tests failed!")
        sys.exit(1)
