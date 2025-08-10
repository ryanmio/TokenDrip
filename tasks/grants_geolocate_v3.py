"""Geolocate Virginia grant descriptions with OpenAI - Version 3 with DMS coordinates.

Reads a CSV of grants (expects a description column `raw_entry`) and, chunk-by-chunk,
asks the model to return coordinates in DMS format.

Input file is taken from env var `CSV_FILE` (defaults to `grants_to_combine/all_digitized_grants_20250809_202343.csv`).
Output is written under `output/` as `results_<input-stem>_v3.csv` (or override via `OUTPUT_FILE`).
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import openai
import tiktoken

# ----------------- TokenDrip contract -----------------
MODEL = "gpt-5-2025-08-07"            # primary model (1-M group)

# --------------------- Files ---------------------------
CSV_PATH = Path(os.getenv("CSV_FILE", "grants_to_combine/all_digitized_grants_20250809_202343.csv"))
# Allow overriding output file name via env. By default, include input stem for CI organization
OUTPUT_PATH = Path(os.getenv("OUTPUT_FILE", f"output/results_{CSV_PATH.stem}_v3.csv"))
ID_FIELD = "grant_id"  # if absent, first column
DESC_FIELD = "raw_entry"        # if absent, second column
STATE_FIELDS = ["row_id", "description", "latlon", "tokens_used"]
ROW_LIMIT = int(os.getenv("ROW_LIMIT", "5"))  # Temporary cap for test runs

# ------------------ Helper funcs -----------------------
enc = tiktoken.encoding_for_model("gpt-5-2025-08-07")

def token_len(text: str) -> int:
    try:
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4


def build_prompt(desc: str) -> str:
    return (
        "Geolocate this colonial Virginia land grant to precise latitude and longitude coordinates. You MUST output the most precise coordinates you can, with NO other text or qualifiers, even if the grant does not give you much information.\n"
        "Respond with ONLY the coordinates in this format: [DD]°[MM]'[SS].[SSSSS]\"N [DDD]°[MM]'[SS].[SSSSS]\"W\n\n"
        f"Grant: {desc}\n" )

# ------------------ Task API ---------------------------

def init_state():
    return {"next_row": 0}


def run_chunk(budget: int, state: dict, selected_model: str | None = None):
    """Process rows until budget exhausted. Returns (tokens_used, new_state).

    The runner supplies `selected_model`, which will be either `model` or `backup_model`
    depending on which quota bucket still has room.  If it's not provided (legacy call),
    we default to the primary `model` constant.
    """
    if selected_model is None:
        selected_model = MODEL

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY env var not set")

    client = openai.OpenAI(api_key=api_key)

    # Quick preflight: verify model accessibility fast, fail early if gated/unavailable
    try:
        _ = client.models.with_options(timeout=10).retrieve(MODEL)
        print(f"[grants_geolocate_v3] Preflight OK: model {MODEL} accessible")
    except Exception as e:
        print(f"[grants_geolocate_v3] Preflight FAILED for model {MODEL}: {e}")
        return 0, state

    # Ensure output dir
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    first_write = not OUTPUT_PATH.exists()

    used_tokens_total = 0
    row_idx = state["next_row"]

    with CSV_PATH.open() as f_in, OUTPUT_PATH.open("a", newline="") as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=STATE_FIELDS)
        if first_write:
            writer.writeheader()

        # Determine actual id and description fields
        id_col = ID_FIELD if ID_FIELD in reader.fieldnames else reader.fieldnames[0]
        desc_col = DESC_FIELD if DESC_FIELD in reader.fieldnames else reader.fieldnames[1]

        current_index = -1
        processed_count = 0
        for row in reader:
            current_index += 1
            if current_index < row_idx:
                continue

            desc = row.get(desc_col, "")
            prompt = build_prompt(desc)
            
            # Progress heartbeat so logs move even if API is slow
            import time
            start_ts = time.time()
            print(f"[grants_geolocate_v3] Requesting row {current_index}…")

            try:
                resp = client.chat.completions.with_options(timeout=180).create(
                    model=selected_model,
                    messages=[{"role": "user", "content": prompt}],
                )

                answer = resp.choices[0].message.content.strip()
                answer = answer.replace("\n", " ").strip()
                latlon = answer

                tokens_used = getattr(resp.usage, 'total_tokens', 0) if hasattr(resp, 'usage') else 0
                used_tokens_total += tokens_used
            except Exception as e:
                latlon = f"ERROR: {e}"
                tokens_used = 0

            writer.writerow({
                "row_id": row.get(id_col, current_index),
                "description": desc[:100],
                "latlon": latlon,
                "tokens_used": tokens_used,
            })

            dur = time.time() - start_ts
            print(f"[grants_geolocate_v3] Row {current_index} done in {dur:.1f}s, tokens {tokens_used}")

            row_idx = current_index + 1

            if used_tokens_total >= budget:
                break
            processed_count += 1
            if ROW_LIMIT and processed_count >= ROW_LIMIT:
                print(f"[grants_geolocate_v3] Test cap reached ({ROW_LIMIT} rows). Stopping early.")
                break

    state["next_row"] = row_idx
    print(f"[grants_geolocate_v3] Processed up to row {row_idx-1}, used {used_tokens_total} tokens")
    return used_tokens_total, state 



