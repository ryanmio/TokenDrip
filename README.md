# TokenDrip

Token dripping is a strategy for long OpenAI workloads: instead of blasting millions of tokens in one go, you “let the faucet trickle.” Each day you run just enough requests to stay inside the free-token bucket granted by the data-sharing incentive (1 M or 10 M tokens, depending on model group), checkpoint the partial results, and then pause. At 00 UTC the quota refills, your job wakes up, loads its last checkpoint, and continues. The cycle repeats until the task completes, giving you effectively cost-free processing for projects that would otherwise be too big for a single day’s allowance.

## Quick Start

1. Fork this repository
2. Add your `OPENAI_API_KEY` as a repository secret
3. Push to trigger the workflow
4. Tasks run daily at 00:00 UTC via GitHub Actions


## License

MIT License - see [LICENSE](LICENSE) for details.
