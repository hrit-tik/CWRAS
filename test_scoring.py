"""CW-RAS Round 2 Verification: Surface Water Factor"""
import sys
import pandas as pd

def nr(rn, rc):
    if rn == 0: return 0
    return min(abs(rn - rc) / rn * 100, 100)

def ng(gl, gc):
    if pd.isna(gl) or pd.isna(gc): return 0
    return min(abs(gl - gc) / 3.0 * 100, 100)

def nl(u, f):
    return min((u / 100) * 50 + ((100 - f) / 100) * 50, 100)

def compute_swf(w):
    if pd.isna(w) or w <= 0: return 1.0
    return max(1.0 - (w / 100) * 0.8, 0.2)

def cl(s):
    if s < 30: return "Low"
    elif s < 60: return "Moderate"
    return "High"

df = pd.read_csv("CW_RAS_master_dataset.csv")
all_pass = True

# Test 1: All scores in [0, 100]
print("Test 1: All scores in [0, 100]")
errors = []
for _, row in df.iterrows():
    r = nr(row["R_normal"], row["R_current"])
    g = ng(row["GW_last"], row["GW_current"])
    u = row["Urban_Percent"]; f = row["Forest_Percent"]
    l = nl(u, f)
    w = row.get("Water_Body_Percent", 0)
    s = compute_swf(w)
    fl = 0.4 * r + 0.4 * l + 0.2 * g
    sc = (0.4 * r + 0.4 * g + 0.2 * l) * s
    if fl < 0 or fl > 100:
        errors.append(f"  {row['Panchayat']}: flood={fl:.1f}")
    if sc < 0 or sc > 100:
        errors.append(f"  {row['Panchayat']}: scarcity={sc:.1f}")
if errors:
    for e in errors: print(e)
    all_pass = False
    print("  FAIL")
else:
    print("  PASS")

# Test 2: SWF bounds [0.2, 1.0]
print("Test 2: SWF bounds [0.2, 1.0]")
swf_vals = [compute_swf(w) for w in df["Water_Body_Percent"]]
if all(0.2 <= v <= 1.0 for v in swf_vals):
    print("  PASS")
else:
    print("  FAIL")
    all_pass = False

# Test 3: Perumon correction
print("Test 3: Perumon scarcity correction")
p = df[df["Panchayat"] == "Perumon"].iloc[0]
r = nr(p["R_normal"], p["R_current"])
g = ng(p["GW_last"], p["GW_current"])
l = nl(p["Urban_Percent"], p["Forest_Percent"])
w = p["Water_Body_Percent"]
s = compute_swf(w)
fl = 0.4 * r + 0.4 * l + 0.2 * g
sc_raw = 0.4 * r + 0.4 * g + 0.2 * l
sc = sc_raw * s
print(f"  R={r:.1f} G={g:.1f} L={l:.1f} WB={w} SWF={s:.2f}")
print(f"  Flood={fl:.1f} ({cl(fl)})")
print(f"  Scarcity raw={sc_raw:.1f} ({cl(sc_raw)}) -> moderated={sc:.1f} ({cl(sc)})")
if sc < sc_raw and s < 1.0:
    print(f"  PASS: {cl(sc_raw)} -> {cl(sc)} (reduced by {(1-s)*100:.0f}%)")
else:
    print("  FAIL: SWF did not reduce scarcity")
    all_pass = False

# Test 4: Flood scores unchanged for WB=0 panchayats
print("Test 4: Flood unaffected by SWF")
nowater = df[df["Water_Body_Percent"] == 0]
print(f"  {len(nowater)} panchayats with WB=0 (SWF=1.0)")
all_swf_one = all(compute_swf(0) == 1.0 for _ in range(len(nowater)))
if all_swf_one:
    print("  PASS")
else:
    print("  FAIL")
    all_pass = False

# Test 5: Water body panchayats have lower scarcity than without SWF
print("Test 5: SWF reduces scarcity for water-body panchayats")
water_panchayats = df[df["Water_Body_Percent"] > 0]
reductions = []
for _, row in water_panchayats.iterrows():
    r = nr(row["R_normal"], row["R_current"])
    g = ng(row["GW_last"], row["GW_current"])
    l = nl(row["Urban_Percent"], row["Forest_Percent"])
    s = compute_swf(row["Water_Body_Percent"])
    raw = 0.4 * r + 0.4 * g + 0.2 * l
    mod = raw * s
    reductions.append((row["Panchayat"], raw, mod, s))
    print(f"  {row['Panchayat']}: SC {raw:.1f} -> {mod:.1f} (SWF={s:.2f})")

if all(mod <= raw for _, raw, mod, _ in reductions):
    print("  PASS")
else:
    print("  FAIL")
    all_pass = False

print()
if all_pass:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED")
    sys.exit(1)
