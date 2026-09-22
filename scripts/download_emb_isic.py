"""Download the ISIC-sourced portion of the EMB (Early-Stage Melanoma Benchmark)
dataset via the public ISIC API. Atlas-sourced images (source == "Atlas") are
NOT handled here -- see docs/KNOWN_GAPS.md.
"""
import csv
import os
import time
import urllib.request
import json

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS_CSV = os.path.join(BASE, "data", "EMB", "_repo", "early_melanoma_benchmark_dataset_labels.csv")
OUT_DIR = os.path.join(BASE, "data", "EMB", "images")
LICENSE_LOG = os.path.join(BASE, "data", "EMB", "isic_license_log.csv")

os.makedirs(OUT_DIR, exist_ok=True)

with open(LABELS_CSV, newline="", encoding="utf-8") as f:
    rows = [r for r in csv.DictReader(f) if r["source"] == "ISIC"]

print(f"ISIC-sourced EMB images to fetch: {len(rows)}")

ok, failed = 0, []
with open(LICENSE_LOG, "w", newline="", encoding="utf-8") as logf:
    logw = csv.writer(logf)
    logw.writerow(["image", "copyright_license", "attribution"])

    for i, row in enumerate(rows):
        isic_id = row["image"]
        dest = os.path.join(OUT_DIR, f"{isic_id}.jpg")
        if os.path.exists(dest):
            ok += 1
            continue
        try:
            with urllib.request.urlopen(f"https://api.isic-archive.com/api/v2/images/{isic_id}/", timeout=15) as r:
                meta = json.load(r)
            url = meta["files"]["full"]["url"]
            lic = meta.get("copyright_license", "")
            attr = meta.get("attribution", "")
            urllib.request.urlretrieve(url, dest)
            logw.writerow([isic_id, lic, attr])
            ok += 1
        except Exception as e:
            failed.append((isic_id, str(e)))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(rows)} processed, {ok} ok, {len(failed)} failed")
            logf.flush()
        time.sleep(0.05)

print(f"Done. {ok} downloaded/present, {len(failed)} failed.")
if failed:
    with open(os.path.join(BASE, "data", "EMB", "failed_downloads.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "error"])
        w.writerows(failed)
    print("Failures logged to data/EMB/failed_downloads.csv")
