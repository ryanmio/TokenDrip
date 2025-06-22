# TokenDrip Task Template Guide

## Creating New Tasks

When you need to start a completely fresh task (even if similar to an existing one), create a new task file instead of modifying existing ones. This preserves the resumption logic for incomplete tasks while allowing clean restarts.

## Quick Start: Copy an Existing Task

1. **Copy** an existing task file with a new name:
   ```bash
   cp tasks/grants_geolocate.py tasks/grants_geolocate_v2.py
   ```

2. **Update** the key identifiers in the new file:
   - Change the docstring to describe the new task
   - Update `OUTPUT_PATH` to a new filename (e.g., `results_full_v2.csv`)
   - Update log messages to use the new task name
   - Modify the prompt, model, or logic as needed

3. **Commit** the new task file:
   ```bash
   git add tasks/grants_geolocate_v2.py
   git commit -m "feat: Add grants_geolocate_v2 task with DMS format"
   ```

## Task File Structure

Every task file must have:

```python
# TokenDrip contract - required
MODEL = "gpt-4o-2024-05-13"  # or your preferred model
# BACKUP_MODEL = "gpt-4o-mini-2024-07-18"  # optional fallback

def init_state():
    """Return initial state dict for new tasks"""
    return {"next_row": 0}

def run_chunk(budget: int, state: dict, selected_model: str | None = None):
    """Process work within budget. Return (tokens_used, new_state)"""
    # Your task logic here
    return tokens_used, updated_state
```

## Key Points

- **State isolation**: Each task gets its own state file (`state/{task_name}.json`)
- **Output isolation**: Use unique output filenames to avoid conflicts
- **Resumption**: Tasks automatically resume from where they left off
- **Clean restarts**: New task name = fresh start, even with similar logic

## Example: Restarting with New Prompt

Instead of:
```python
# ❌ Modifying existing task - will resume from where it left off
def build_prompt(desc: str) -> str:
    return "NEW PROMPT..."  # Old runs already used old prompt
```

Do this:
```python
# ✅ Create new task file with new name
# tasks/grants_geolocate_v2.py
OUTPUT_PATH = Path("output/results_full_v2.csv")  # New output file
def build_prompt(desc: str) -> str:
    return "NEW PROMPT..."  # Fresh start with new prompt
```

This way:
- Old task stays completed with old prompt/format
- New task starts fresh with new prompt/format  
- No state conflicts or mixed results 