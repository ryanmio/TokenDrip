#!/usr/bin/env python3
"""
TokenDrip - Automated OpenAI quota-aware task runner
Discovers tasks in tasks/ directory and runs them with intelligent quota management.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from importlib import import_module
from importlib.util import spec_from_file_location, module_from_spec
import tiktoken
import os  # <-- NEW


# Daily token limits by model group (usage tiers 3+, tiers 1-2 have lower limits)
# To use lower tier limits, change to: {'1m_group': 250_000, '10m_group': 2_500_000}
DAILY_LIMITS = {
    # 1M token group (250K for usage tiers 1-2)
    '1m_group': 1_000_000,
    # 10M token group (2.5M for usage tiers 1-2) 
    '10m_group': 10_000_000,
}

# --- Optional smoke-test mode -------------------------------------------------
# Export `TOKENDRIP_TEST=1` to run the whole stack with tiny 1 000-token quotas,
# useful for CI or local debugging without waiting for large counters.
TEST_MODE = os.getenv("TOKENDRIP_TEST") == "1"
if TEST_MODE:
    print("[TokenDrip] TEST_MODE=1 → using 1 000-token daily limits for both groups")
    DAILY_LIMITS['1m_group'] = 1_000
    DAILY_LIMITS['10m_group'] = 1_000
# -----------------------------------------------------------------------------

# Model mappings to quota groups
MODEL_GROUPS = {
    # 1M token group models
    'gpt-4.5-preview-2025-02-27': '1m_group',
    'gpt-4.1-2025-04-14': '1m_group', 
    'gpt-4o-2024-05-13': '1m_group',
    'gpt-4o-2024-08-06': '1m_group',
    'gpt-4o-2024-11-20': '1m_group',
    'o3-2025-04-16': '1m_group',
    'o1-preview-2024-09-12': '1m_group',
    'o1-2024-12-17': '1m_group',
    
    # 10M token group models
    'gpt-4.1-mini-2025-04-14': '10m_group',
    'gpt-4.1-nano-2025-04-14': '10m_group',
    'gpt-4o-mini-2024-07-18': '10m_group',
    'o4-mini-2025-04-16': '10m_group',
    'o1-mini-2024-09-12': '10m_group',
    'codex-mini-latest': '10m_group',
}

# Default model group for quota tracking
DEFAULT_MODEL_GROUP = '10m_group'


class TokenDripRunner:
    """Manages task discovery, state persistence, and quota tracking."""
    
    def __init__(self):
        self.tasks_dir = Path('tasks')
        self.state_dir = Path('state')
        self.state_dir.mkdir(exist_ok=True)
        self.global_state_file = self.state_dir / 'global.json'
        
    def load_global_state(self):
        """Load or initialize global quota tracking state."""
        if not self.global_state_file.exists():
            init_state = {
                'last_reset': datetime.now(timezone.utc).strftime('%Y-%m-%d'),
                'tokens_used_1m': 0,
                'tokens_used_10m': 0,
            }
            print("[TokenDrip] No existing global state found → starting fresh")
            return init_state
        
        with open(self.global_state_file, 'r') as f:
            state = json.load(f)
            print(
                "[TokenDrip] Loaded global state: last_reset="
                f"{state.get('last_reset')} | 1M used={state.get('tokens_used_1m', 0)} | "
                f"10M used={state.get('tokens_used_10m', 0)}"
            )
            return state
    
    def save_global_state(self, state):
        """Save global quota tracking state."""
        with open(self.global_state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def reset_if_new_day(self, global_state):
        """Reset token usage if it's a new UTC day."""
        now = datetime.now(timezone.utc)
        current_date = now.strftime('%Y-%m-%d')
        
        last_reset = global_state.get('last_reset')
        if last_reset != current_date:
            print(f"[TokenDrip] Resetting quota for new day: {current_date}")
            global_state['tokens_used_1m'] = 0
            global_state['tokens_used_10m'] = 0
            global_state['last_reset'] = current_date
            return True
        return False
    
    def get_model_group(self, model_name):
        """Determine which quota group a model belongs to."""
        return MODEL_GROUPS.get(model_name, DEFAULT_MODEL_GROUP)
    
    def get_remaining_quota(self, global_state, model_group=None):
        """Calculate remaining token quota for a specific model group."""
        if model_group is None:
            model_group = DEFAULT_MODEL_GROUP
            
        if model_group == '1m_group':
            used = global_state.get('tokens_used_1m', 0)
            limit = DAILY_LIMITS['1m_group']
        else:  # 10m_group
            used = global_state.get('tokens_used_10m', 0) 
            limit = DAILY_LIMITS['10m_group']
            
        return limit - used
    
    def discover_tasks(self):
        """Discover all task modules in tasks/ directory."""
        if not self.tasks_dir.exists():
            print("[TokenDrip] No tasks/ directory found")
            return []
        
        tasks = []
        for task_file in self.tasks_dir.glob('*.py'):
            if task_file.name.startswith('__'):
                continue
            tasks.append(task_file)
        
        print(f"[TokenDrip] Discovered {len(tasks)} task(s): " + ", ".join(t.stem for t in tasks))
        return tasks
    
    def load_task_module(self, task_file):
        """Load a task module from file."""
        module_name = task_file.stem
        spec = spec_from_file_location(module_name, task_file)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    
    def load_task_state(self, task_name):
        """Load task-specific state."""
        state_file = self.state_dir / f"{task_name}.json"
        if not state_file.exists():
            return None
        
        with open(state_file, 'r') as f:
            return json.load(f)
    
    def save_task_state(self, task_name, state):
        """Save task-specific state."""
        state_file = self.state_dir / f"{task_name}.json"
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def run_task(self, task_file, global_state):
        """Execute a single task with quota budgets. Returns (used_1m, used_10m)."""
        task_name = task_file.stem
        print(f"[TokenDrip] Running task: {task_name}")
        
        try:
            # Load task module
            task_module = self.load_task_module(task_file)
            
            # Check required attributes
            if not hasattr(task_module, 'run_chunk'):
                print(f"[TokenDrip] Skipping {task_name}: missing run_chunk function")
                return 0, 0
            
            # ---------------- MODEL SELECTION LOGIC (v2) ----------------
            # Accept several spelling styles for backwards compatibility, but encourage
            #    MODEL          (required) and BACKUP_MODEL (optional).
            primary_model = (
                getattr(task_module, 'MODEL', None) or
                getattr(task_module, 'PRIMARY_MODEL', None) or
                getattr(task_module, 'model', None) or
                getattr(task_module, 'preferred_model', None)
            )

            backup_model = (
                getattr(task_module, 'BACKUP_MODEL', None) or
                getattr(task_module, 'backup_model', None)
            )
             
            if not primary_model:
                print(f"[TokenDrip] Skipping {task_name}: no MODEL/PRIMARY_MODEL attribute defined on task")
                return 0, 0

            candidate_models = [primary_model]
            if backup_model:
                candidate_models.append(backup_model)

            selected_model = None
            using_group    = None
            budget         = 0

            # Iterate through primary then backup to find a model whose group still has quota
            for mdl in candidate_models:
                grp       = self.get_model_group(mdl)
                remaining = self.get_remaining_quota(global_state, grp)
                if remaining > 0:
                    selected_model = mdl
                    using_group    = grp
                    budget         = remaining
                    break

            if selected_model is None:
                print(f"[TokenDrip] Skipping {task_name}: no quota remaining for specified model(s)")
                return 0, 0

            # ---------------- END MODEL SELECTION LOGIC ----------------

            # Load or initialize task state
            task_state = self.load_task_state(task_name)
            if task_state is None:
                print(f"[TokenDrip] Task {task_name} status: NEW 🎉")
                if hasattr(task_module, 'init_state'):
                    task_state = task_module.init_state()
                else:
                    task_state = {}
            else:
                print(f"[TokenDrip] Task {task_name} status: existing, continuing work")
            
            # Run task chunk, supporting both 2-arg and 3-arg signatures for backward compatibility
            print(f"[TokenDrip] Executing {task_name} with {budget:,} tokens from {using_group} using model {selected_model}")
            try:
                import inspect
                if len(inspect.signature(task_module.run_chunk).parameters) == 3:
                    used_tokens, new_state = task_module.run_chunk(budget, task_state, selected_model)
                else:
                    used_tokens, new_state = task_module.run_chunk(budget, task_state)
            except TypeError:
                # If signature mismatch, fall back to legacy call
                used_tokens, new_state = task_module.run_chunk(budget, task_state)
            
            # Save updated state
            self.save_task_state(task_name, new_state)
            
            print(f"[TokenDrip] Task {task_name} used {used_tokens} tokens from {using_group}")
            
            # Return usage for the appropriate group
            if using_group == '1m_group':
                return used_tokens, 0
            else:
                return 0, used_tokens
            
        except Exception as e:
            print(f"[TokenDrip] Error running task {task_name}: {e}")
            return 0, 0
    
    def run(self):
        """Main execution loop."""
        print("[TokenDrip] Starting quota-aware task runner")
        
        # Load and reset global state if needed
        global_state = self.load_global_state()
        self.reset_if_new_day(global_state)
        
        # Check remaining quota for both groups
        remaining_1m = self.get_remaining_quota(global_state, '1m_group')
        remaining_10m = self.get_remaining_quota(global_state, '10m_group')
        
        print(f"[TokenDrip] Remaining quota - 1M group: {remaining_1m:,} tokens, 10M group: {remaining_10m:,} tokens")
        
        if remaining_1m <= 0 and remaining_10m <= 0:
            print("[TokenDrip] No quota remaining for today")
            return
        
        # Discover and run tasks
        tasks = self.discover_tasks()
        if not tasks:
            print("[TokenDrip] No tasks to run")
            return
        
        total_used_1m = 0
        total_used_10m = 0
        
        for task_file in tasks:
            # Check if we still have quota in either group
            remaining_1m = self.get_remaining_quota(global_state, '1m_group')
            remaining_10m = self.get_remaining_quota(global_state, '10m_group')
            
            if remaining_1m <= 0 and remaining_10m <= 0:
                print("[TokenDrip] All quotas exhausted, stopping")
                break
            
            used_1m, used_10m = self.run_task(task_file, global_state)
            total_used_1m += used_1m
            total_used_10m += used_10m
            
            # Update global state after each task
            global_state['tokens_used_1m'] += used_1m
            global_state['tokens_used_10m'] += used_10m
            self.save_global_state(global_state)
        
        remaining_1m = self.get_remaining_quota(global_state, '1m_group')
        remaining_10m = self.get_remaining_quota(global_state, '10m_group')
        
        print(f"[TokenDrip] Session complete. Used {total_used_1m:,} tokens (1M group), {total_used_10m:,} tokens (10M group)")
        print(f"[TokenDrip] Remaining quota - 1M group: {remaining_1m:,}, 10M group: {remaining_10m:,}")

        if total_used_1m == 0 and total_used_10m == 0:
            print("[TokenDrip] No new work needed – all tasks already complete or quotas exhausted 🙌")


if __name__ == '__main__':
    runner = TokenDripRunner()
    runner.run() 