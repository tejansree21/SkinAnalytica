"""
SkinAnalytica — self_healing_agent.py
Runs BEFORE any inference. Cleans, repairs, and validates a batch of images.
"""

import os, json, shutil, hashlib, cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from base_agent import BaseAgent, BASE

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
MIN_SIZE  = 32    # px — smaller than this is unusable
MAX_SIZE  = 8192  # px — larger than this is suspicious
OUT_DIR   = os.path.join(BASE, "outputs", "self_healing")

class SelfHealingAgent(BaseAgent):
    """
    Automatically fixes a batch before inference:
      1. Removes corrupt / unreadable images
      2. Detects and removes duplicates (MD5 hash)
      3. Flags images that are too small or too large
      4. Standardises filenames (lowercase, no spaces)
      5. Detects OOD images (extreme colour distributions)
      6. Produces a repair report
    """

    def __init__(self):
        super().__init__("self_healing_agent")
        os.makedirs(OUT_DIR, exist_ok=True)

    def _run(self, image_folder: str, auto_fix: bool = True,
             output_name: str = "batch") -> dict:

        folder = Path(image_folder)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {image_folder}")

        img_files = [f for f in folder.rglob("*")
                     if f.suffix.lower() in SUPPORTED]
        self.logger.info(f"Scanning {len(img_files)} images in {image_folder}")

        results = {
            "total"       : len(img_files),
            "corrupt"     : [],
            "duplicates"  : [],
            "too_small"   : [],
            "too_large"   : [],
            "ood_flagged" : [],
            "renamed"     : [],
            "clean"       : [],
            "actions_taken": [],
        }

        hashes = {}

        def check_image(fp: Path) -> dict:
            info = {"path": str(fp), "issues": [], "action": None}

            # 1. Readability
            img = cv2.imread(str(fp))
            if img is None:
                info["issues"].append("corrupt")
                info["action"] = "removed" if auto_fix else "flagged"
                return info

            h, w = img.shape[:2]

            # 2. Size check
            if h < MIN_SIZE or w < MIN_SIZE:
                info["issues"].append(f"too_small ({w}x{h})")
            if h > MAX_SIZE or w > MAX_SIZE:
                info["issues"].append(f"too_large ({w}x{h})")

            # 3. Duplicate check (MD5)
            md5 = hashlib.md5(fp.read_bytes()).hexdigest()
            if md5 in hashes:
                info["issues"].append(f"duplicate_of:{hashes[md5]}")
            else:
                hashes[md5] = str(fp)

            # 4. OOD detection — extreme colour distributions
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            sat_mean = float(hsv[:,:,1].mean())
            val_mean = float(hsv[:,:,2].mean())
            if sat_mean < 10:
                info["issues"].append("ood:near_grayscale")
            if val_mean < 20:
                info["issues"].append("ood:near_black")
            if val_mean > 250:
                info["issues"].append("ood:near_white")

            # 5. Filename standardisation
            clean_name = fp.name.lower().replace(" ", "_")
            if clean_name != fp.name and auto_fix:
                new_path = fp.parent / clean_name
                if not new_path.exists():
                    fp.rename(new_path)
                    info["renamed_to"] = clean_name
                    info["issues"].append("renamed")

            if not info["issues"]:
                info["action"] = "clean"
            elif "corrupt" in info["issues"] and auto_fix:
                fp.unlink(missing_ok=True)
                info["action"] = "removed"
            elif any("duplicate" in i for i in info["issues"]) and auto_fix:
                fp.unlink(missing_ok=True)
                info["action"] = "removed_duplicate"
            else:
                info["action"] = "flagged"

            return info

        # Run checks in parallel
        with ThreadPoolExecutor(max_workers=4) as ex:
            all_info = list(ex.map(check_image, img_files))

        # Collate
        for info in all_info:
            issues = info.get("issues", [])
            if not issues:
                results["clean"].append(info["path"])
            else:
                for issue in issues:
                    if "corrupt"    in issue: results["corrupt"].append(info["path"])
                    if "duplicate"  in issue: results["duplicates"].append(info["path"])
                    if "too_small"  in issue: results["too_small"].append(info["path"])
                    if "too_large"  in issue: results["too_large"].append(info["path"])
                    if "ood:"       in issue: results["ood_flagged"].append(info["path"])
                    if "renamed"    in issue: results["renamed"].append(info["path"])
            if info.get("action") and info["action"] != "clean":
                results["actions_taken"].append({
                    "path"  : info["path"],
                    "action": info["action"],
                    "issues": info["issues"],
                })

        # Summary
        n_issues = (len(results["corrupt"]) + len(results["duplicates"]) +
                    len(results["too_small"]) + len(results["ood_flagged"]))
        results["clean_count"]  = len(results["clean"])
        results["issue_count"]  = n_issues
        results["ready_count"]  = results["total"] - len(results["corrupt"]) - len(results["duplicates"])

        # Save report
        report_path = os.path.join(OUT_DIR, f"{output_name}_healing_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

        self.logger.info(
            f"Complete — {results['clean_count']} clean, "
            f"{n_issues} issues found, "
            f"{results['ready_count']} ready for inference"
        )

        # Print summary
        print(f"\nSELF-HEALING REPORT: {output_name}")
        print("=" * 50)
        print(f"  Total scanned  : {results['total']}")
        print(f"  Clean          : {results['clean_count']}")
        print(f"  Corrupt        : {len(results['corrupt'])}")
        print(f"  Duplicates     : {len(results['duplicates'])}")
        print(f"  Too small      : {len(results['too_small'])}")
        print(f"  OOD flagged    : {len(results['ood_flagged'])}")
        print(f"  Renamed        : {len(results['renamed'])}")
        print(f"  Ready for AI   : {results['ready_count']}")
        print(f"  Report saved   : {report_path}")

        return results


# ── Standalone usage ──────────────────────────────────────────────
if __name__ == "__main__":
    agent  = SelfHealingAgent()
    result = agent.run(
        image_folder = r"C:\Users\tejan\OneDrive\Desktop\drive\SkinAnalytica\data\sa04_test_sample",
        auto_fix     = False,   # set True to actually delete/rename
        output_name  = "test_batch"
    )
