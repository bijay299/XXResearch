import os, shutil, hashlib

ROOT = "/data/bijaypandey/cuig_pilot/modern_flux_dp_uce"
OUT = os.path.join(ROOT, "rendering_panel")
os.makedirs(OUT, exist_ok=True)

CASES = {
    0: "cassette_player", 1: "cassette_player",
    10: "chain_saw", 11: "chain_saw",
    20: "church", 21: "church",
    30: "gas_pump", 31: "gas_pump",
    40: "tench", 41: "tench",
    50: "garbage_truck", 51: "garbage_truck",
    60: "english_springer", 61: "english_springer",
    70: "golf_ball", 71: "golf_ball",
    80: "parachute", 81: "parachute",
    90: "french_horn", 91: "french_horn",
}

ENDPOINTS = ["M0", "DP_A", "DP_B", "UCE_A", "UCE_B"]

hashes = []
for ep in ENDPOINTS:
    src_dir = os.path.join(ROOT, "runs", ep, "images")
    dst_dir = os.path.join(OUT, ep)
    os.makedirs(dst_dir, exist_ok=True)
    for case, obj in sorted(CASES.items()):
        fn = f"{case:03d}_{obj}_case{case:03d}_var00.png"
        src = os.path.join(src_dir, fn)
        dst = os.path.join(dst_dir, fn)
        assert os.path.isfile(src), f"missing {src}"
        shutil.copy2(src, dst)
        h = hashlib.sha256(open(dst, "rb").read()).hexdigest()
        hashes.append((ep, fn, os.path.getsize(dst), h))

with open(os.path.join(OUT, "panel_hashes.csv"), "w") as f:
    f.write("endpoint,filename,bytes,sha256\n")
    for ep, fn, sz, h in hashes:
        f.write(f"{ep},{fn},{sz},{h}\n")

print(f"copied {len(hashes)} files into {OUT}")
