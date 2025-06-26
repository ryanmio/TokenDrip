# TokenDrip Task Template

## Quick Start: Copy the Python Template

The easiest way to create a new task is to copy the working Python template:

```bash
# Copy the template
cp task_template.py tasks/my_new_task.py

# Edit the new task file
# - Update the docstring
# - Set MODEL and OUTPUT_CSV path
# - Modify process_row() function for your use case

# Commit when ready
git add tasks/my_new_task.py
git commit -m "feat: add my_new_task"
```

## Required Functions

Every task needs these two functions:

- `init_state()` → returns initial state dict for new tasks
- `run_chunk(budget, state, selected_model=None)` → processes work within token budget

## Model Selection

Set your preferred models at the top of your task file:

```python
MODEL = "gpt-4o-mini-2024-07-18"        # Primary (10M bucket)
# BACKUP_MODEL = "gpt-4o-2024-05-13"    # Optional fallback (1M bucket)
```

The runner automatically chooses which model to use based on daily quota availability.

## State Management

- Each task gets its own state file: `state/{task_name}.json`
- Tasks automatically resume from where they left off
- Create a new task file (new name) for fresh starts, even with similar logic