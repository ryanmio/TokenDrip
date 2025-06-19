"""Geolocate Virginia grant descriptions with OpenAI.

This task reads `grants.csv` (requires columns: id,text) and, chunk-by-chunk,
asks GPT-4.1 to return best-guess lat/lon.  Results accumulate in
`output/results_full.csv`.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import openai
import tiktoken

# ----------------- TokenDrip contract -----------------
DEFAULT_MODEL = "gpt-4.1-2025-04-14"  # 1-M bucket
DEFAULT_MIN_CHUNK = 50_000             # fallback used by runner

# --------------------- Files ---------------------------
CSV_PATH      = Path("grants.csv")
OUTPUT_PATH   = Path("output/results_full.csv")
ID_FIELD   = "results_row_index"  # if absent, first column
DESC_FIELD = "raw_entry"          # if absent, second column
STATE_FIELDS  = ["row_id", "description", "lat", "lon", "tokens_used"]

# ------------------ Helper funcs -----------------------
enc = tiktoken.encoding_for_model("gpt-4o-mini-2024-07-18")

def token_len(text: str) -> int:
    try:
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4


def build_prompt(desc: str) -> str:
    return (
        "Given the following colonial Virginia land grant extract, return your best-guess "
        "latitude and longitude in decimal degrees as `lat, lon`. If unsure leave blank.\n\n"
        f"Grant: {desc}\n\nAnswer:" )

# ------------------ Task API ---------------------------

def init_state():
    return {"next_row": 0}


def run_chunk(budget: int, state: dict):
    """Process rows until budget exhausted. Returns tokens used."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY env var not set")

    client = openai.OpenAI(api_key=api_key)

    # Ensure output dir
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    first_write = not OUTPUT_PATH.exists()

    used_tokens_total = 0
    row_idx = state["next_row"]

    with CSV_PATH.open() as f_in, OUTPUT_PATH.open("a", newline="") as f_out:
        reader = csv.DictReader(f_in)
        # Fast-forward to current row
        rows = list(reader)
        n_rows = len(rows)
        if row_idx >= n_rows:
            print("[grants_geolocate] All rows processed ✅")
            return 0, state

        writer = csv.DictWriter(f_out, fieldnames=STATE_FIELDS)
        if first_write:
            writer.writeheader()

        # Determine actual id and description fields
        id_col = ID_FIELD if ID_FIELD in reader.fieldnames else reader.fieldnames[0]
        desc_col = DESC_FIELD if DESC_FIELD in reader.fieldnames else reader.fieldnames[1]

        while row_idx < n_rows:
            row = rows[row_idx]
            desc = row.get(desc_col, "")

            prompt = build_prompt(desc)
            est_tok = token_len(prompt) + 200  # rough response size
            if used_tokens_total + est_tok > budget:
                break  # stop before exceeding budget

            resp = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.2,
            )

            answer = resp.choices[0].message.content.strip()
            lat, lon = "", ""
            if "," in answer:
                parts = answer.split(",")
                if len(parts) >= 2:
                    lat = parts[0].strip()
                    lon = parts[1].strip()

            tokens_used = resp.usage.total_tokens
            used_tokens_total += tokens_used

            writer.writerow({
                "row_id": row.get(id_col, row_idx),
                "description": desc[:100],
                "lat": lat,
                "lon": lon,
                "tokens_used": tokens_used,
            })

            row_idx += 1

    state["next_row"] = row_idx
    print(f"[grants_geolocate] Processed up to row {row_idx-1}, used {used_tokens_total} tokens")
    return used_tokens_total, state
