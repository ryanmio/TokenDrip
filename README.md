# TokenDrip

Schedule long-running OpenAI jobs that automatically pause when daily free-token quotas (1M/10M tokens) are hit and resume at 00 UTC. Perfect for batch processing, research, and automated content generation without manual quota management.

## Quick Start

1. Fork this repository
2. Add your `OPENAI_API_KEY` as a repository secret
3. Push to trigger the workflow
4. Tasks run hourly via GitHub Actions

## Usage

Place Python task files in `tasks/` directory. Each task must implement:

```python
min_chunk_tokens = 80_000  # Minimum tokens needed per chunk
def init_state(): return {}  # Initial state
def run_chunk(budget, state): return (used_tokens, new_state)
```

Tasks are automatically discovered and executed with intelligent quota management.

## License

MIT License - see [LICENSE](LICENSE) file. 