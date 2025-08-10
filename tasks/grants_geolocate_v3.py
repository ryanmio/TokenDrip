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
import asyncio
import time

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
ROW_LIMIT = int(os.getenv("ROW_LIMIT", "25"))  # Temporary cap for test runs
CONCURRENCY = int(os.getenv("CONCURRENCY", "5"))

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
        _ = client.models.retrieve(MODEL)
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

        # Load rows into memory to enable concurrent scheduling
        rows = list(reader)
        total_rows = len(rows)
        if row_idx >= total_rows:
            print("[grants_geolocate_v3] All rows processed ✅")
            return 0, state

        async def request_row(async_client, idx: int):
            desc_local = rows[idx].get(desc_col, "")
            prompt_local = build_prompt(desc_local)
            start_local = time.time()
            print(f"[grants_geolocate_v3] Requesting row {idx}…")
            try:
                resp = await async_client.chat.completions.create(
                    model=selected_model,
                    messages=[{"role": "user", "content": prompt_local}],
                    timeout=180,
                )
                answer = resp.choices[0].message.content.strip().replace("\n", " ").strip()
                tokens = getattr(resp, 'usage', None)
                tokens_used_local = getattr(tokens, 'total_tokens', 0) if tokens else 0
                latlon_local = answer
            except Exception as e:
                latlon_local = f"ERROR: {e}"
                tokens_used_local = 0
            dur_local = time.time() - start_local
            print(f"[grants_geolocate_v3] Row {idx} finished in {dur_local:.1f}s, tokens {tokens_used_local}")
            return {
                "index": idx,
                "row_id": rows[idx].get(id_col, idx),
                "description": rows[idx].get(desc_col, "")[:100],
                "latlon": latlon_local,
                "tokens_used": tokens_used_local,
            }

        async def run_concurrent(start_i: int):
            nonlocal used_tokens_total, row_idx
            async_client = openai.AsyncOpenAI(api_key=api_key)
            semaphore = asyncio.Semaphore(CONCURRENCY)
            in_flight = set()
            buffered = {}
            next_to_schedule = start_i
            next_to_write = start_i
            written = 0

            async def bound_request(i: int):
                async with semaphore:
                    return await request_row(async_client, i)

            while True:
                # Schedule up to concurrency respecting limits
                while (
                    next_to_schedule < total_rows
                    and (not ROW_LIMIT or written + len(in_flight) < ROW_LIMIT)
                    and len(in_flight) < CONCURRENCY
                    and used_tokens_total < budget
                ):
                    task = asyncio.create_task(bound_request(next_to_schedule))
                    in_flight.add(task)
                    next_to_schedule += 1

                if not in_flight:
                    break

                done, pending = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
                in_flight = pending
                for t in done:
                    try:
                        res = t.result()
                        buffered[res["index"]] = res
                    except Exception:
                        # Skip failed task; continue
                        pass

                # Flush results in order
                while next_to_write in buffered:
                    res = buffered.pop(next_to_write)
                    writer.writerow({
                        "row_id": res["row_id"],
                        "description": res["description"],
                        "latlon": res["latlon"],
                        "tokens_used": res["tokens_used"],
                    })
                    used_tokens_total += res.get("tokens_used", 0)
                    written += 1
                    row_idx = next_to_write + 1
                    next_to_write += 1

                    if used_tokens_total >= budget or (ROW_LIMIT and written >= ROW_LIMIT):
                        # Cancel remaining in-flight tasks
                        for p in in_flight:
                            p.cancel()
                        return

        asyncio.run(run_concurrent(row_idx))

    state["next_row"] = row_idx
    print(f"[grants_geolocate_v3] Processed up to row {row_idx-1}, used {used_tokens_total} tokens")
    return used_tokens_total, state 



