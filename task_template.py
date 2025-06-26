"""Task Template - Copy this file to create new TokenDrip tasks.

Instructions:
1. Copy this file: cp task_template.py tasks/my_new_task.py
2. Update the docstring, MODEL, OUTPUT_PATH, and logic below
3. Commit: git add tasks/my_new_task.py && git commit -m "feat: add my_new_task"

This template processes a CSV file row by row, calling OpenAI API for each row.
Modify the process_row() function for your specific use case.

=== INSTRUCTIONS FOR AI MODELS ===
If someone asks you to create a TokenDrip task using this template, follow these steps:

1. COPY this entire file content as the starting point
2. UPDATE these specific parts:
   - Line ~7: Change docstring to describe the new task
   - Line ~21: Set MODEL (choose from runner.py MODEL_GROUPS)
   - Line ~22: Optionally set BACKUP_MODEL 
   - Line ~25-26: Set INPUT_CSV and OUTPUT_CSV paths
   - Line ~28-32: Update column names for the user's CSV structure
   - Line ~46-52: Modify build_prompt() for the user's specific task
   - Line ~55-93: Modify process_row() function for the user's logic
   - Line ~157: Update task name in print statements (replace "my_task")

3. PRESERVE these parts unchanged:
   - All import statements
   - The init_state() function structure
   - The run_chunk() function signature and overall structure
   - Token counting and budget management logic
   - CSV reading/writing patterns
   - Error handling structure

4. EXAMPLE transformations:
   - For sentiment analysis: build_prompt() asks for sentiment, process_row() expects sentiment response
   - For translation: build_prompt() asks for translation, process_row() expects translated text
   - For data extraction: build_prompt() asks for specific fields, process_row() parses structured response

5. ALWAYS keep the TokenDrip contract intact (MODEL, init_state, run_chunk functions)

Ask the user for: their CSV structure, what they want the AI to do with each row, and preferred model.
===================================
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import openai
import tiktoken

# ================== TokenDrip Contract (Required) ==================
MODEL = "gpt-4o-mini-2024-07-18"        # Primary model (10M bucket - cheaper)
# BACKUP_MODEL = "gpt-4o-2024-05-13"    # Optional fallback (1M bucket)

# ==================== Task Configuration ============================
INPUT_CSV = Path(os.getenv("INPUT_CSV", "input.csv"))     # Input CSV file
OUTPUT_CSV = Path("output/my_task_results.csv")           # Output CSV file

# CSV column names (adjust for your data)
ID_COLUMN = "id"           # Unique identifier column
TEXT_COLUMN = "text"       # Text to process column

# Output CSV columns
OUTPUT_COLUMNS = ["id", "input_text", "ai_response", "tokens_used"]

# ==================== Helper Functions ==============================
enc = tiktoken.encoding_for_model("gpt-4o-mini-2024-07-18")

def count_tokens(text: str) -> int:
    """Count tokens in text."""
    try:
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4  # Rough fallback


def build_prompt(text: str) -> str:
    """Build the prompt for the AI. Customize this for your task."""
    return f"""Please analyze this text and provide a brief summary:

Text: {text}

Summary:"""


def process_row(row: dict, client: openai.OpenAI, selected_model: str) -> dict:
    """Process a single CSV row. Customize this for your task.
    
    Returns a dict with keys matching OUTPUT_COLUMNS.
    """
    text = row.get(TEXT_COLUMN, "")
    if not text.strip():
        return {
            "id": row.get(ID_COLUMN, ""),
            "input_text": "",
            "ai_response": "ERROR: Empty input text",
            "tokens_used": 0
        }
    
    prompt = build_prompt(text)
    
    try:
        response = client.chat.completions.create(
            model=selected_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.1,
        )
        
        ai_response = response.choices[0].message.content.strip()
        tokens_used = response.usage.total_tokens
        
        return {
            "id": row.get(ID_COLUMN, ""),
            "input_text": text[:100] + "..." if len(text) > 100 else text,
            "ai_response": ai_response,
            "tokens_used": tokens_used
        }
        
    except Exception as e:
        return {
            "id": row.get(ID_COLUMN, ""),
            "input_text": text[:100] + "..." if len(text) > 100 else text,
            "ai_response": f"ERROR: {str(e)}",
            "tokens_used": 0
        }


# ==================== TokenDrip Interface ===========================

def init_state():
    """Initialize state for a new task."""
    return {"next_row": 0, "total_processed": 0}


def run_chunk(budget: int, state: dict, selected_model: str | None = None):
    """Process CSV rows within the token budget.
    
    Args:
        budget: Maximum tokens to use
        state: Current task state
        selected_model: Model chosen by runner (MODEL or BACKUP_MODEL)
    
    Returns:
        (tokens_used, new_state): Tuple of tokens consumed and updated state
    """
    if selected_model is None:
        selected_model = MODEL
    
    # Get OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")
    
    client = openai.OpenAI(api_key=api_key)
    
    # Ensure output directory exists
    OUTPUT_CSV.parent.mkdir(exist_ok=True)
    
    # Check if this is the first run
    is_first_run = not OUTPUT_CSV.exists()
    
    # Load CSV data
    if not INPUT_CSV.exists():
        print(f"[my_task] Input file {INPUT_CSV} not found")
        return 0, state
    
    with INPUT_CSV.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    total_rows = len(rows)
    current_row = state.get("next_row", 0)
    
    if current_row >= total_rows:
        print(f"[my_task] All {total_rows} rows processed ✅")
        return 0, state
    
    # Open output file for appending
    with OUTPUT_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        
        # Write header if first run
        if is_first_run:
            writer.writeheader()
        
        tokens_used = 0
        processed_count = 0
        
        # Process rows until budget exhausted
        while current_row < total_rows and tokens_used < budget:
            row = rows[current_row]
            
            # Process the row
            result = process_row(row, client, selected_model)
            
            # Write result
            writer.writerow(result)
            
            # Update counters
            row_tokens = result.get("tokens_used", 0)
            tokens_used += row_tokens
            processed_count += 1
            current_row += 1
            
            # Check if we should stop before next iteration
            if tokens_used >= budget:
                break
    
    # Update state
    new_state = {
        "next_row": current_row,
        "total_processed": state.get("total_processed", 0) + processed_count
    }
    
    if current_row >= total_rows:
        print(f"[my_task] Task complete! Processed {new_state['total_processed']} rows total ✅")
    else:
        print(f"[my_task] Processed {processed_count} rows (total: {new_state['total_processed']}/{total_rows}), used {tokens_used} tokens")
    
    return tokens_used, new_state 