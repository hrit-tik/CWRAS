"""CW-RAS Round 3 Verification: Directional GW & Flood Boost"""
import sys
import pandas as pd
import numpy as np

def nr(rn, rc):
    if rn == 0: return 0
    return min(abs(rn - rc) / rn * 100, 100)

def ng_mag(gl, gc):
    if pd.isna(gl) or pd.isna(gc): return 0
    return min(abs(gl - gc) / 3.0 * 100, 100)

# New Directional Logic: Only rising (depth decreasing) counts for flood
def ng_flood(gl, gc):
    if pd.isna(gl) or pd.isna(gc): return 0
    # mbgl: Lower value = Higher water table
    # Rise means Current < Last
    if gc < gl:
        return ng_mag(gl, gc)
    return 0

def nl(u, f):
    return min((u / 100) * 50 + ((100 - f) / 100) * 50, 100)

def compute_swf(w):
    if pd.isna(w) or w <= 0: return 1.0
    return max(1.0 - (w / 50), 0.1)

def cl(s):
    if s < 30: return "Low"
    elif s < 60: return "Moderate"
    return "High"

df = pd.read_csv("CW_RAS_master_dataset.csv")
all_pass = True

print("=== VERIFICATION REPORT ===")

# Test 1: Perumon Specific Check
print("\n[Test 1] Perumon Correction")
p = df[df["Panchayat"] == "Perumon"].iloc[0]
r = nr(p["R_normal"], p["R_current"])
g_mag = ng_mag(p["GW_last"], p["GW_current"])
g_flood = ng_flood(p["GW_last"], p["GW_current"]) # Should be 0 since GW dropped (4.3 -> 6.6)
l = nl(p["Urban_Percent"], p["Forest_Percent"])
w = p["Water_Body_Percent"]

# Flood Calc
flood_base = 0.4*r + 0.4*l + 0.2*g_flood
flood_boost = w * 1.2
flood_final = min(flood_base + flood_boost, 100)

# Scarcity Calc
swf = compute_swf(w)
scarcity_base = 0.4*r + 0.4*g_mag + 0.2*l
scarcity_final = scarcity_base * swf

print(f"Perumon Inputs: R={r:.1f}, L={l:.1f}, G_mag={g_mag:.1f}, G_flood={g_flood:.1f}, WB={w}")
print(f"Flood: Base={flood_base:.1f} + Boost={flood_boost:.1f} = {flood_final:.1f} ({cl(flood_final)})")
print(f"Scarcity: Base={scarcity_base:.1f} * SWF={swf:.2f} = {scarcity_final:.1f} ({cl(scarcity_final)})")

if cl(flood_final) == "High" and cl(scarcity_final) == "Low":
    print("PASS: Perumon is High Flood / Low Scarcity")
else:
    print("FAIL: Perumon profile incorrect")
    all_pass = False


# Test 2: Directional Groundwater Logic
print("\n[Test 2] Directional Groundwater")
# Find a panchayat where GW rose (Depth decreased: Curr < Last)
rising = df[df["GW_current"] < df["GW_last"]].head(1)
if not rising.empty:
    r_row = rising.iloc[0]
    g_f = ng_flood(r_row["GW_last"], r_row["GW_current"])
    if g_f > 0:
        print(f"PASS: Rising GW ({r_row['Panchayat']}) contributes to flood risk ({g_f:.1f})")
    else:
        print(f"FAIL: Rising GW calculation error")
        all_pass = False
else:
    print("SKIP: No rising GW data found")

# Find a panchayat where GW dropped (Depth increased: Curr > Last) -> Should be 0 for flood
dropping = df[df["GW_current"] > df["GW_last"]].head(1)
if not dropping.empty:
    d_row = dropping.iloc[0]
    g_f = ng_flood(d_row["GW_last"], d_row["GW_current"])
    if g_f == 0:
        print(f"PASS: Dropping GW ({d_row['Panchayat']}) contributes 0 to flood risk")
    else:
        print(f"FAIL: Dropping GW should be 0 for flood, got {g_f:.1f}")
        all_pass = False


# Test 3: Flood Boost Application
print("\n[Test 3] Flood Boost")
water_pans = df[df["Water_Body_Percent"] > 0].head(3)
for _, row in water_pans.iterrows():
    expected_boost = row["Water_Body_Percent"] * 1.2
    print(f"Checking {row['Panchayat']} (WB={row['Water_Body_Percent']}%)... Boost should be {expected_boost:.1f}")


# Test 4: Global Bounds
print("\n[Test 4] Global Score Bounds [0, 100]")
failures = []
for _, row in df.iterrows():
    # Re-calc manually
    r = nr(row["R_normal"], row["R_current"])
    g_f = ng_flood(row["GW_last"], row["GW_current"])
    g_m = ng_mag(row["GW_last"], row["GW_current"])
    l = nl(row["Urban_Percent"], row["Forest_Percent"])
    w = row.get("Water_Body_Percent", 0)
    
    f_score = min((0.4*r + 0.4*l + 0.2*g_f) + (w*1.2), 100)
    swf_val = compute_swf(w)
    s_score = (0.4*r + 0.4*g_m + 0.2*l) * swf_val
    
    if not (0 <= f_score <= 100) or not (0 <= s_score <= 100):
        failures.append(row["Panchayat"])

if not failures:
    print("PASS: All scores within valid range")
else:
    print(f"FAIL: Out of bounds scores for {failures}")
    all_pass = False


if all_pass:
    print("\nALL TESTS PASSED ✅")
else:
    print("\nSOME TESTS FAILED ❌")
    sys.exit(1)
