"""Summarize recent arXiv articles about a topic using GPT-4o.

This TokenDrip task fetches up to `MAX_RESULTS` arXiv
articles published within the last 90 days for each topic specified in the `TOPICS` list, and asks GPT-4o to produce
concise bullet-point summaries (≤ 2 sentences) for every article.

Topics list:
- Edit the `TOPICS` list near the top of this file with the queries you want.

Output CSV columns:
- id            : identifier (topic or provided id)
- topic         : original search query
- ai_response   : GPT summary (one bullet per article)
- article_count : number of articles retrieved
- tokens_used   : OpenAI tokens consumed

To run locally:
```bash
export OPENAI_API_KEY=sk-...
python runner.py  # will auto-discover and execute the task
```
"""
from __future__ import annotations

import csv
import datetime as _dt
import os
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote_plus

import openai
import tiktoken
import requests

# ================== TokenDrip Contract (Required) ==================
MODEL = "gpt-4o-2024-05-13"         # Primary model (1M-token bucket)
BACKUP_MODEL = "gpt-4o-mini-2024-07-18"  # Falls back to 10M bucket if 1M exhausted

# ==================== Task Configuration ============================
# List of topics to summarize (edit this list)
TOPICS = [
    "diffusion models",
    "graph neural networks",
    # Add or remove topics as desired
]

# Output file (you can change to .csv or .json as preferred)
OUTPUT_CSV = Path("output/arxiv_topic_summaries.csv")

# Synthetic column names to keep existing processing code paths
ID_COLUMN = "id"
TOPIC_COLUMN = "topic"

# Output CSV columns
OUTPUT_COLUMNS = [
    "id",
    "topic",
    "ai_response",
    "article_count",
    "tokens_used",
]

# arXiv API parameters
MAX_RESULTS = 10               # Limit results per topic to control prompt size
LOOKBACK_DAYS = 90             # Fetch papers from the last 90 days (~3 months)

# ==================== Helper Functions ==============================
_enc = tiktoken.encoding_for_model("gpt-4o-mini-2024-07-18")

def _count_tokens(text: str) -> int:
    try:
        return len(_enc.encode(text))
    except Exception:
        return len(text) // 4

def _fetch_arxiv_articles(query: str, max_results: int = MAX_RESULTS):
    """Return list[{title, summary}] of recent arXiv articles for *query*."""
    today = _dt.datetime.utcnow()
    start_date = today - _dt.timedelta(days=LOOKBACK_DAYS)
    date_range = f"{start_date.strftime('%Y%m%d0000')}+TO+{today.strftime('%Y%m%d2359')}"

    search_query = (
        f"all:{quote_plus(query)}+AND+submittedDate:[{date_range}]"
    )

    params = {
        "search_query": search_query,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": str(max_results),
    }

    try:
        resp = requests.get("https://export.arxiv.org/api/query", params=params, timeout=20)
        resp.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"Network error contacting arXiv API: {exc}") from exc

    # Parse Atom XML
    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError as exc:
        raise RuntimeError("Failed parsing arXiv response XML") from exc

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    articles = []
    for entry in root.findall("atom:entry", ns):
        title_el = entry.find("atom:title", ns)
        summary_el = entry.find("atom:summary", ns)
        if title_el is None or summary_el is None:
            continue
        title = " ".join(title_el.text.split())  # squash whitespace
        summary = " ".join(summary_el.text.split())
        articles.append({"title": title, "summary": summary})
    return articles

def _build_prompt(topic: str, articles: list[dict]) -> str:
    """Return the chat prompt to send to GPT."""
    if not articles:
        return textwrap.dedent(
            f"""You are a helpful assistant. No recent arXiv papers were found on the topic \"{topic}\" 
            within the last {LOOKBACK_DAYS} days. Respond with a short sentence stating that 
            no relevant papers were located."""
        ).strip()

    articles_text = "\n\n".join(
        f"Title: {a['title']}\nAbstract: {a['summary']}" for a in articles
    )

    return textwrap.dedent(
        f"""You are an expert scientific writer. Summarize the following recent arXiv papers 
        related to the topic \"{topic}\". Provide one bullet (max two sentences) per article, 
        focusing on the novel contribution. Do not mention arXiv IDs.

        Papers:\n{{articles_text}}

        Summaries:"""
    ).replace("{{articles_text}}", articles_text).strip()

# ==================== Row Processing ================================

def process_row(row: dict, client: openai.OpenAI, selected_model: str) -> dict:
    topic = (row.get(TOPIC_COLUMN) or "").strip()
    if not topic:
        return {
            "id": row.get(ID_COLUMN, ""),
            "topic": "",
            "ai_response": "ERROR: Empty topic string",
            "article_count": 0,
            "tokens_used": 0,
        }

    # Fetch papers
    try:
        articles = _fetch_arxiv_articles(topic, MAX_RESULTS)
    except Exception as exc:
        return {
            "id": row.get(ID_COLUMN, topic),
            "topic": topic,
            "ai_response": f"ERROR contacting arXiv: {exc}",
            "article_count": 0,
            "tokens_used": 0,
        }

    prompt = _build_prompt(topic, articles)

    try:
        response = client.chat.completions.create(
            model=selected_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.2,
        )
        ai_text = response.choices[0].message.content.strip()
        tokens_used = response.usage.total_tokens
    except Exception as exc:
        ai_text = f"ERROR from OpenAI: {exc}"
        tokens_used = 0

    return {
        "id": row.get(ID_COLUMN, topic),
        "topic": topic,
        "ai_response": ai_text,
        "article_count": len(articles),
        "tokens_used": tokens_used,
    }

# ==================== TokenDrip Interface ===========================

def init_state():
    return {"next_row": 0, "total_processed": 0}


def run_chunk(budget: int, state: dict, selected_model: str | None = None):
    if selected_model is None:
        selected_model = MODEL

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")

    client = openai.OpenAI(api_key=api_key)

    OUTPUT_CSV.parent.mkdir(exist_ok=True)

    is_first_run = not OUTPUT_CSV.exists()

    # Build rows directly from the in-script TOPICS list
    rows = [{TOPIC_COLUMN: t, ID_COLUMN: t} for t in TOPICS]

    total_rows = len(rows)
    current_row = state.get("next_row", 0)

    if current_row >= total_rows:
        print(f"[arxiv_topic_summary] All {total_rows} rows processed ✅")
        return 0, state

    tokens_used = 0
    processed = 0

    with OUTPUT_CSV.open("a", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=OUTPUT_COLUMNS)
        if is_first_run:
            writer.writeheader()

        while current_row < total_rows and tokens_used < budget:
            row = rows[current_row]
            result = process_row(row, client, selected_model)
            writer.writerow(result)

            tokens_used += result.get("tokens_used", 0)
            processed += 1
            current_row += 1

            if tokens_used >= budget:
                break

    new_state = {
        "next_row": current_row,
        "total_processed": state.get("total_processed", 0) + processed,
    }

    if current_row >= total_rows:
        print(
            f"[arxiv_topic_summary] Task complete! Processed {new_state['total_processed']} rows ✅"
        )
    else:
        print(
            f"[arxiv_topic_summary] Processed {processed} rows (total: {new_state['total_processed']}/{total_rows}), used {tokens_used} tokens"
        )

    return tokens_used, new_state 