# TokenDrip

Token dripping is a strategy for token-intensive OpenAI workloads: instead of blasting millions of tokens in one go, you "let the faucet trickle." Each day you run just enough requests to stay inside the free-token bucket granted by the data-sharing incentive (1 M or 10 M tokens, depending on model group), checkpoint the partial results, and then pause. At 00 UTC the quota refills, your job wakes up, loads its last checkpoint, and continues. The cycle repeats until the task completes, giving you effectively cost-free processing for projects that would otherwise be too big for a single day's allowance.

## Quick Start

1. Fork this repository
2. Add your `OPENAI_API_KEY` as a repository secret
3. Push to trigger the workflow
4. Tasks run daily at 00:00 UTC via GitHub Actions

## Free 11 M Tokens/Day?  Yes!

OpenAI's data-sharing incentive gives every opted-in organization **two separate buckets that refill every day at 00:00 UTC** (see the [official help-center article](https://help.openai.com/en/articles/10306912-sharing-feedback-evaluation-and-fine-tuning-data-and-api-inputs-and-outputs-with-openai)):

| Bucket | Size | Eligible models | Typical cost if you paid for it* |
|--------|------|-----------------|------------------------------------|
| "Large-model" bucket | **1 M tokens/day** | `gpt-4o`, `gpt-4.5-preview`, `o1`, `o3`, … | ≈ **$5–$15** (GPT-4o: $5 ↔ input, $15 ↔ output per 1 M) |
| "Small-model" bucket | **10 M tokens/day** | `gpt-4o-mini`, `o1-mini`, `o3-mini`, … | ≈ **$5–$15** (GPT-4o-mini: $0.5 ↔ input, $1.5 ↔ output → $5–$15 for 10 M) |

*Numbers come from the [OpenAI price list](https://openai.com/api/pricing/) current as of May - 2025.  The exact dollar value depends on your input/output split.  The table assumes a 50∶50 blend of input and output tokens.

### What that means in dollars

• **Per day:** between **~$10 and $30** of usage that you don't pay for.<br/>
• **Per 30-day month:** between **~$300 and $900** of free compute.

Even at the conservative end of that range the free credits are huge—enough to process a full book with GPT-4o or run tens of thousands of classification calls on `o4-mini` every single day.

### Why Token Dripping matters

Your job just has to stay *inside* those limits at any moment.  By checkpointing and continuing the next day, `TokenDrip` lets long-running projects consume **hundreds of millions of tokens over weeks** while your credit-card bill stays at $0.

> Pro-tip: The buckets reset on a rolling 24-hour window per UTC day, so you can schedule heavy runs right after UTC midnight to maximize overlap with the next refill.

---

*"Typical cost if you paid for it"* uses current public pricing; always check the pricing page for the latest numbers.

## License

MIT License - see [LICENSE](LICENSE) for details.
