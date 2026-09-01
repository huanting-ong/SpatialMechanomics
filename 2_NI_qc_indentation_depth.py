# -*- coding: utf-8 -*-
"""
2_NI_qc_indentation_depth.py

Reports the maximum indentation depth reached by every nanoindentation curve
and flags the ones that went deeper than the Hertz fit describes.

Parameters
    MAX_ALLOWED_UM  4.4     depth above which a curve is flagged OVER. The
                            Hertz model is fitted over the first 2 um only, so
                            a deeper indentation samples material outside the
                            regime the fit assumes
    P[max]          -0.001  treated as no indentation and skipped
    NUM_WORKERS     54      parallel workers, capped at cpu_count minus 2

Actions required
    Place this script in the project root, with one folder per slice directly
    beneath it. Set ROOT_DIR if the script is stored elsewhere.
    Runs on the raw curves, so it does not need script 1 to have run first.
        py 2_NI_qc_indentation_depth.py

Input
    <root>/<slice>/     raw .txt files. P[max] is read from the header and the
                        numeric table from the line beginning "Time (s)".
                        Files named "._*" and "XY position.txt" are ignored
Output
    <root>/<slice>/max_indentation_summary.xlsx     one row per curve for that
                                                    slice, with max depth in
                                                    um and a pass or OVER call
    <root>/max_indentation_all_slices.xlsx          the same for every slice
                                                    combined
"""

from pathlib import Path
import re
from io import StringIO
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================
ROOT_DIR = Path(__file__).resolve().parent  # folder where this script is located

# Number of parallel workers (e.g., 54 on your 64-core box)
NUM_WORKERS = min(54, max(1, multiprocessing.cpu_count() - 2))

# Threshold for classification
MAX_ALLOWED_UM = 4.4

# Regex to catch common P[max] header variants (with tabs, equals, colon, etc.)
PMAX_REGEX = re.compile(
    r"P\s*\[\s*max\s*]\s*\(uN\)\s*(?:=|:|\t)\s*([-+]?\d*\.?\d+)",
    flags=re.IGNORECASE
)

def parse_file(txt_path: Path):
    """
    Return a dict with results for one .txt file.
    On any non-fatal issue, we mark 'error' and include a message instead of raising.
    """
    try:
        # Ignore macOS resource forks explicitly as a second guard
        if txt_path.name.startswith("._"):
            return {"slice": txt_path.parent.name, "file": txt_path.name, "status": "SKIPPED (macOS resource fork)"}

        if txt_path.name.endswith("XY position.txt"):
            return {"slice": txt_path.parent.name, "file": txt_path.name, "status": "SKIPPED (XY position)"}

        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        # Find table start at "Time (s)"
        table_start = None
        for i, line in enumerate(lines):
            if line.startswith("Time (s)"):
                table_start = i
                break

        if table_start is None:
            return {"slice": txt_path.parent.name, "file": txt_path.name, "status": "ERROR (no 'Time (s)' header)"}

        header_lines = lines[:table_start]
        table_text = "".join(lines[table_start:])

        # Parse P[max] from header (robust)
        pmax_val = None
        for h in header_lines:
            m = PMAX_REGEX.search(h)
            if m:
                pmax_val = float(m.group(1))
                break

        # If P[max] is exactly -0.001 uN, treat as no indentation (skip)
        if pmax_val == -0.001:
            return {
                "slice": txt_path.parent.name,
                "file": txt_path.name,
                "status": "SKIPPED (P[max] == -0.001)"
            }

        # Read numeric table
        df = pd.read_csv(StringIO(table_text), sep="\t")
        # Be lenient with column name variations
        ind_col = None
        for cand in ["Indentation (nm)", "Indentation(nm)", "Indentation  (nm)"]:
            if cand in df.columns:
                ind_col = cand
                break
        if ind_col is None:
            # try case-insensitive match as a last resort
            low_cols = {c.lower(): c for c in df.columns}
            if "indentation (nm)" in low_cols:
                ind_col = low_cols["indentation (nm)"]

        if ind_col is None:
            return {"slice": txt_path.parent.name, "file": txt_path.name, "status": "ERROR (no 'Indentation (nm)' column)"}

        # Compute max indentation in µm
        try:
            max_nm = float(pd.to_numeric(df[ind_col], errors="coerce").max())
        except Exception as e:
            return {"slice": txt_path.parent.name, "file": txt_path.name, "status": f"ERROR (bad numeric data: {e})"}

        if not pd.notna(max_nm):
            return {"slice": txt_path.parent.name, "file": txt_path.name, "status": "ERROR (all indentation NaN)"}

        max_um = max_nm / 1000.0
        flag = "OVER" if max_um > MAX_ALLOWED_UM else "pass"

        return {
            "slice": txt_path.parent.name,
            "file": txt_path.name,
            "full_path": str(txt_path),
            "pmax_uN": pmax_val,
            "max_indentation_um": round(max_um, 6),
            "classification": flag,
            "status": "OK"
        }

    except Exception as e:
        return {"slice": txt_path.parent.name, "file": txt_path.name, "status": f"ERROR ({e})"}


def main():
    if not ROOT_DIR.exists():
        print(f"[FATAL] ROOT_DIR not found: {ROOT_DIR}")
        return

    # Collect slice folders (directories only; skip hidden and obvious non-slices)
    slice_dirs = [
        d for d in ROOT_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ]
    slice_dirs.sort(key=lambda p: p.name)

    # Gather all txt files (skip '._' right here)
    all_txts = []
    for sdir in slice_dirs:
        txts = [f for f in sdir.glob("*.txt") if not f.name.startswith("._")]
        # keep the slice even if empty; we’ll still create a per-slice empty summary
        all_txts.extend(txts)

    print(f"[INFO] Found {len(all_txts)} .txt files across slices.")
    print(f"[INFO] Using up to {NUM_WORKERS} workers (of {multiprocessing.cpu_count()} cores).")

    results = []
    processed = 0

    if all_txts:
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
            futs = {ex.submit(parse_file, p): p for p in all_txts}
            for fut in as_completed(futs):
                res = fut.result()
                results.append(res)
                processed += 1
                if processed % 200 == 0 or processed == len(all_txts):
                    print(f"[PROGRESS] {processed}/{len(all_txts)} files processed.")

    # Convert to DataFrame
    df_all = pd.DataFrame(results)

    # Ensure consistent columns
    base_cols = ["slice", "file", "full_path", "pmax_uN", "max_indentation_um", "classification", "status"]
    for c in base_cols:
        if c not in df_all.columns:
            df_all[c] = pd.NA
    df_all = df_all[base_cols]

    # Write per-slice Excel
    for sdir in slice_dirs:
        sname = sdir.name
        df_slice = df_all.loc[df_all["slice"] == sname].copy()
        out_xlsx = sdir / "max_indentation_summary.xlsx"
        try:
            with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
                df_slice.to_excel(writer, index=False, sheet_name="summary")
            print(f"[OK] Wrote per-slice summary: {out_xlsx}")
        except Exception as e:
            print(f"[WARN] Could not write per-slice Excel for {sname}: {e}")

    # Write combined Excel at root
    combined_xlsx = ROOT_DIR / "max_indentation_all_slices.xlsx"
    try:
        with pd.ExcelWriter(combined_xlsx, engine="openpyxl") as writer:
            df_all.to_excel(writer, index=False, sheet_name="all_slices")
        print(f"[OK] Wrote combined summary: {combined_xlsx}")
    except Exception as e:
        print(f"[WARN] Could not write combined Excel: {e}")

    # Small textual summary
    n_ok = (df_all["status"] == "OK").sum()
    n_skipped = (df_all["status"].astype(str).str.startswith("SKIPPED")).sum()
    n_err = (df_all["status"].astype(str).str.startswith("ERROR")).sum()
    print(f"Done. OK: {n_ok}, SKIPPED: {n_skipped}, ERROR: {n_err}")

if __name__ == "__main__":
    main()
