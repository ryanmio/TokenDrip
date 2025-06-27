"""Token-drip summarization of recent arXiv papers for a given topic.

This task discovers thousands of recent papers on a topic, then consumes your
OpenAI quota day-by-day to generate concise, structured summaries until all
papers are processed.

How it works (daily cycle driven by TokenDrip runner):
1. On the first run it queries the arXiv API for papers matching TOPIC that
   were published in the last LOOKBACK_DAYS.  The metadata (title, abstract,
   authors, published date, arxiv_id) is cached in the task state.
2. Each day it resumes at the saved *index* and processes as many papers as
   the remaining token budget allows.
3. For every paper it asks GPT-4o-mini to return a JSON object containing:
     • ≤250-character summary of the paper's contribution/results
     • bullet list of follow-up experiments the authors suggest (if any)
     • reproducibility score 1-5 (1 = hard, 5 = trivial)
4. Results append to `output/arxiv_drip_summaries.csv`.
5. When all papers are processed the task stops; if more quota remains it does
   nothing until new papers appear (you can bump LOOKBACK_DAYS or change TOPIC).

Editables:
• TOPIC – search query string.
• CATEGORY – optional arXiv primary category code (e.g. "cs.AI").
• TOP_N_TOTAL – cap on number of papers to queue.
• MODEL – change if you prefer a different model.
"""
from __future__ import annotations

import csv
import json
import os
import textwrap
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict

import openai
import requests
import tiktoken
import io
import time
import pypdf

# ================== TokenDrip Contract (Required) ==================
MODEL = "gpt-4o-mini-2024-07-18"   # 10M-token bucket (cheaper, larger quota)
BACKUP_MODEL = "gpt-4o-2024-05-13" # 1M-token bucket fallback

# ==================== Task Configuration ============================
TOPIC = "openai"                     # keyword(s) to match in title/abstract/etc.
CATEGORY = "cs.AI"                   # primary arXiv category filter (None to disable)
LOOKBACK_DAYS = 90                   # recent window
TOP_N_TOTAL = 3000                   # cap total papers queued
PAGE_SIZE = 200                      # arXiv page size when fetching list
MIN_START_BUDGET = 1_000  # you can raise/lower as needed

OUTPUT_CSV = Path("output/arxiv_drip_summaries.csv")
OUTPUT_COLUMNS = [
    "arxiv_id",
    "title",
    "authors",
    "published",
    "summary_250",
    "follow_ups",
    "reproducibility",
    "novelty",
    "reproduce_how",
    "tokens_used",
]

# ==================== Helper functions ==============================
_enc = tiktoken.encoding_for_model("gpt-4o-mini-2024-07-18")

def _count_tokens(text: str) -> int:
    try:
        return len(_enc.encode(text))
    except Exception:
        return len(text) // 4

def _fetch_papers_from_arxiv(topic: str) -> List[Dict]:
    """Fetch metadata for up to TOP_N_TOTAL papers matching *topic*."""
    papers: List[Dict] = []
    today = datetime.utcnow()
    start_date = today - timedelta(days=LOOKBACK_DAYS)
    keyword_part = f"all:\"{topic}\"" if topic else "all:*"
    parts = [keyword_part]
    if CATEGORY:
        parts.insert(0, f"cat:{CATEGORY}")
    base_query = " AND ".join(parts)

    start_index = 0
    while len(papers) < TOP_N_TOTAL:
        params = {
            "search_query": base_query,
            "start": str(start_index),
            "max_results": str(PAGE_SIZE),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        resp = requests.get("https://export.arxiv.org/api/query", params=params, timeout=30)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)
        if not entries:
            break
        for entry in entries:
            title_el = entry.find("atom:title", ns)
            abst_el = entry.find("atom:summary", ns)
            id_el = entry.find("atom:id", ns)
            pub_el = entry.find("atom:published", ns)
            if title_el is None or abst_el is None or id_el is None:
                continue
            arxiv_id = id_el.text.rsplit("/", 1)[-1]
            # Authors list
            authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)]
            # Filter by date
            published_str = pub_el.text.split("T")[0] if pub_el is not None else ""
            try:
                pub_dt = datetime.fromisoformat(published_str)
            except Exception:
                pub_dt = None
            if pub_dt and (today - pub_dt).days > LOOKBACK_DAYS:
                continue  # older than lookback window

            papers.append({
                "arxiv_id": arxiv_id,
                "title": " ".join(title_el.text.split()),
                "abstract": " ".join(abst_el.text.split()),
                "authors": "; ".join(authors),
                "published": published_str,
            })
            if len(papers) >= TOP_N_TOTAL:
                break
        if len(entries) < PAGE_SIZE:
            break
        start_index += PAGE_SIZE
    return papers

def _truncate(text: str, max_chars: int = 5000) -> str:
    return text[:max_chars]

def _build_prompt(paper: Dict) -> str:
    info = textwrap.dedent(
        f"""Title: {paper['title']}
        Authors: {paper['authors']}
        Abstract: {_truncate(paper['abstract'], 4000)}"""
    ).strip()
    return textwrap.dedent(
        f"""You are an expert scientific reviewer.
        Based on the information provided, output a JSON dictionary with exactly these keys:
          summary – ≤250 characters describing the main contribution/results.
          follow_up_experiments – array of bullet strings listing any follow-up experiments or future work the authors suggest. Return empty array if none.
          reproducibility – integer 1-5 rating how easy it would be to reproduce the main experiment (5 = very easy, 1 = very hard).
          novelty – integer 1-5 rating how novel or groundbreaking the paper is (5 = very novel).
          reproduce_how – short (≤200 char) suggestion describing what reproducing this work would entail (e.g., required dataset, hardware, code repo).
        Only output JSON – no commentary.

        Paper information:
        {info}
        JSON:"""
    ).strip()

# =============== Processing single paper ============================

def _process_paper(paper: Dict, client: openai.OpenAI, model: str):
    # ---------- Download and extract full PDF text ----------
    pdf_url = f"https://arxiv.org/pdf/{paper['arxiv_id']}.pdf"
    try:
        pdf_bytes = requests.get(pdf_url, timeout=30).content
        # Polite delay for arXiv (avoid >30 req/min)
        time.sleep(2)
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:
        return {
            "summary_250": f"ERROR downloading PDF – {exc}",
            "follow_ups": [],
            "reproducibility": "",
            "novelty": "",
            "reproduce_how": "",
            "tokens_used": 0,
        }

    # Truncate to fit 128k context window (leave headroom for prompt / response)
    MAX_INPUT_TOKENS = 110_000
    enc = tiktoken.encoding_for_model(model)
    tok_ids = enc.encode(full_text)
    if len(tok_ids) > MAX_INPUT_TOKENS:
        tok_ids = tok_ids[:MAX_INPUT_TOKENS]
        full_text = enc.decode(tok_ids)

    # ---------- Single-call summarization ----------
    prompt_body = _truncate(full_text, 200_000)  # safety cap in chars
    prompt = (
        "You are an expert scientific reviewer. Given the full text of an academic paper below, "
        "output a JSON object with exactly these keys:\n"
        "  summary – ≤250 characters overall summary\n"
        "  follow_up_experiments – bullet array of author-suggested future work (empty if none)\n"
        "  reproducibility – 1-5 how easy to reproduce (5 easiest)\n"
        "  novelty – 1-5 how novel/breakthrough (5 very novel)\n"
        "  reproduce_how – ≤200 characters describing what reproducing would entail\n\n"
        "Paper:\n" + prompt_body + "\n\nJSON:"
    )

    try:
        res = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.2,
        )
    except Exception as exc:
        return {
            "summary_250": f"ERROR – {exc}",
            "follow_ups": [],
            "reproducibility": "",
            "novelty": "",
            "reproduce_how": "",
            "tokens_used": 0,
        }

    content = res.choices[0].message.content.strip()
    total_tokens = res.usage.total_tokens

    # ---------- Parse JSON ----------
    json_start = content.find("{")
    json_end = content.rfind("}")
    parsed = {}
    if json_start != -1 and json_end != -1:
        try:
            parsed = json.loads(content[json_start:json_end + 1])
        except json.JSONDecodeError:
            pass

    return {
        "summary_250": parsed.get("summary", content),
        "follow_ups": parsed.get("follow_up_experiments", []),
        "reproducibility": parsed.get("reproducibility", ""),
        "novelty": parsed.get("novelty", ""),
        "reproduce_how": parsed.get("reproduce_how", ""),
        "tokens_used": total_tokens,
    }

# ==================== TokenDrip interface ===========================

def init_state():
    return {
        "index": 0,        # next paper to process
        "papers": [],      # cached metadata list
    }


def run_chunk(budget: int, state: dict, selected_model: str | None = None):
    if selected_model is None:
        selected_model = MODEL

    # Skip if budget below threshold
    if budget < MIN_START_BUDGET:
        print(
            f"[arxiv_drip] Remaining budget {budget} < MIN_START_BUDGET {MIN_START_BUDGET} – skipping until next day"
        )
        return 0, state

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")
    client = openai.OpenAI(api_key=api_key)

    # Discover papers on first run
    if not state.get("papers"):
        print("[arxiv_drip] Discovering papers…")
        state["papers"] = _fetch_papers_from_arxiv(TOPIC)
        print(f"[arxiv_drip] Queued {len(state['papers'])} papers for summarization")
        state["index"] = 0

    papers = state["papers"]
    idx = state.get("index", 0)
    total = len(papers)
    if idx >= total:
        print("[arxiv_drip] All papers processed ✅")
        return 0, state

    OUTPUT_CSV.parent.mkdir(exist_ok=True)
    first_run = not OUTPUT_CSV.exists()
    tokens_used = 0
    processed = 0

    with OUTPUT_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        if first_run:
            writer.writeheader()
        while idx < total and tokens_used < budget:
            paper = papers[idx]
            result = _process_paper(paper, client, selected_model)
            tokens_used += result["tokens_used"]

            row = {
                "arxiv_id": paper["arxiv_id"],
                "title": paper["title"],
                "authors": paper["authors"],
                "published": paper["published"],
                "summary_250": result["summary_250"],
                "follow_ups": " | ".join(result["follow_ups"]) if isinstance(result["follow_ups"], list) else result["follow_ups"],
                "reproducibility": result["reproducibility"],
                "novelty": result["novelty"],
                "reproduce_how": result["reproduce_how"],
                "tokens_used": result["tokens_used"],
            }
            writer.writerow(row)
            idx += 1
            processed += 1
            if tokens_used >= budget:
                break

    state["index"] = idx
    msg = (
        f"[arxiv_drip] Processed {processed} paper(s); index {idx}/{total}; used {tokens_used} tokens"
    )
    print(msg)

    return tokens_used, state 