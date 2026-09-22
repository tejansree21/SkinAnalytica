"""
SkinAnalytica — inspect_dataset_schema.py
Run this the moment PAD-UFES-20 / SLICE-3D Permissive metadata CSVs land
locally. Prints the real column names, dtypes, and a few sample values, and
flags likely candidates for the fields we actually care about (skin tone,
age, diagnosis, anatomical site) by fuzzy name matching.

Why this exists instead of hardcoded column names: web research on both
datasets' exact schemas came back unreliable (PDFs didn't parse cleanly,
GitHub pages don't list raw CSV headers) — and this whole session has been
about the cost of silently guessing wrong on data schema/scale (the
temperature-scaling bug). Better to confirm against the real file than
assume and risk misreading e.g. which string means "melanoma".

Usage:
    python inspect_dataset_schema.py path/to/metadata.csv
"""
import sys
import pandas as pd

CANDIDATES = {
    # NOTE: PAD-UFES-20's real column is "fitspatrick" (typo in the source
    # dataset, not "fitzpatrick") — confirmed against the actual file.
    # Kept both spellings here since fuzzy-matching on the "correct"
    # spelling alone would silently miss it.
    "skin_tone": ["fitzpatrick", "fitspatrick", "fitspat", "skin_type",
                  "skin_tone", "ita", "tbp_lv_a", "tbp_lv_b", "tbp_lv_l",
                  "tbp_lv_color", "fst"],
    "age": ["age", "age_approx", "idade"],
    "diagnosis": ["diagnostic", "diagnosis", "dx", "target", "label",
                  "biopsed", "iddx"],
    "anatomical_site": ["region", "site", "anatom", "localiz", "location"],
    "patient_id": ["patient_id", "patient", "lesion_id"],
    "sex": ["sex", "gender"],
}


def fuzzy_matches(columns, keywords):
    cols_lower = {c: c.lower() for c in columns}
    return [c for c, cl in cols_lower.items() if any(k in cl for k in keywords)]


def main(path: str):
    df = pd.read_csv(path)
    print(f"File: {path}")
    print(f"Rows: {len(df):,}  Columns: {len(df.columns)}\n")

    print("=== All columns (dtype, n_unique, sample values) ===")
    for col in df.columns:
        n_unique = df[col].nunique(dropna=True)
        samples = df[col].dropna().unique()[:4]
        print(f"  {col:<35} {str(df[col].dtype):<10} n_unique={n_unique:<6} sample={list(samples)}")

    print("\n=== Likely candidates for fields we care about ===")
    for field, keywords in CANDIDATES.items():
        matches = fuzzy_matches(df.columns, keywords)
        print(f"  {field:<18} -> {matches if matches else 'NO MATCH FOUND — inspect manually above'}")

    print("\nNext step: confirm the right column name/values for each field "
          "above, then tell me so I can write the real fairness-slicing / "
          "external-validation script against the confirmed schema.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_dataset_schema.py path/to/metadata.csv")
        sys.exit(1)
    main(sys.argv[1])
