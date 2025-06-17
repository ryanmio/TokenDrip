"""
Sample TokenDrip task: Wikipedia page summarizer
Demonstrates the task API by incrementally processing Wikipedia pages.
"""

import json
import re
import urllib.request
import urllib.parse
from pathlib import Path
import tiktoken
import openai


# Minimum tokens needed per execution chunk
min_chunk_tokens = 80_000

# Preferred model (uses 10M token group)
preferred_model = 'gpt-4o-mini-2024-07-18'


def init_state():
    """Initialize task state for first run."""
    return {
        "urls_done": [],
        "urls_queue": [
            "https://en.wikipedia.org/wiki/Artificial_intelligence",
            "https://en.wikipedia.org/wiki/Machine_learning", 
            "https://en.wikipedia.org/wiki/Deep_learning",
            "https://en.wikipedia.org/wiki/Natural_language_processing",
            "https://en.wikipedia.org/wiki/Computer_vision",
        ],
        "summaries": []
    }


def fetch_wikipedia_content(url):
    """Fetch and clean Wikipedia page content."""
    try:
        # Get page content
        with urllib.request.urlopen(url) as response:
            html = response.read().decode('utf-8')
        
        # Basic text extraction (remove HTML tags)
        text = re.sub(r'<[^>]+>', '', html)
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Take first portion to avoid overwhelming content
        return text[:8000]  # Limit to reasonable size
        
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None


def estimate_tokens(text):
    """Estimate token count for text."""
    try:
        encoding = tiktoken.encoding_for_model(preferred_model)
        return len(encoding.encode(text))
    except:
        # Fallback estimation
        return len(text) // 4


def run_chunk(budget, state):
    """
    Process one Wikipedia page with the given token budget.
    Returns (tokens_used, updated_state).
    """
    # Check if we have pages to process
    if not state.get("urls_queue", []):
        print("Sample task: All Wikipedia pages processed!")
        return 0, state
    
    # Get next URL to process
    url = state["urls_queue"][0]
    print(f"Sample task: Processing {url}")
    
    # Fetch content
    content = fetch_wikipedia_content(url)
    if not content:
        # Skip failed pages
        new_state = state.copy()
        new_state["urls_queue"] = state["urls_queue"][1:]
        new_state["urls_done"].append(url)
        return 0, new_state
    
    # Estimate tokens needed for this request
    prompt = f"""Summarize this Wikipedia page content in 2-3 concise paragraphs:

{content}

Focus on the key concepts and main ideas."""
    
    estimated_tokens = estimate_tokens(prompt) + 300  # Add buffer for response
    
    if estimated_tokens > budget:
        print(f"Sample task: Need {estimated_tokens} tokens, but budget is {budget}")
        return 0, state
    
    try:
        # Call OpenAI API using current model
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=preferred_model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.3
        )
        
        summary = response.choices[0].message.content
        tokens_used = response.usage.total_tokens
        
        # Update state
        new_state = state.copy()
        new_state["urls_queue"] = state["urls_queue"][1:]
        new_state["urls_done"].append(url)
        new_state["summaries"].append({
            "url": url,
            "summary": summary,
            "tokens_used": tokens_used
        })
        
        print(f"Sample task: Summarized {url} using {tokens_used} tokens")
        
        # Save summary to file for reference
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        summary_file = output_dir / "wikipedia_summaries.json"
        with open(summary_file, 'w') as f:
            json.dump(new_state["summaries"], f, indent=2)
        
        return tokens_used, new_state
        
    except Exception as e:
        print(f"Sample task: OpenAI API error: {e}")
        # Move to done list even on error to avoid infinite retry
        new_state = state.copy()
        new_state["urls_queue"] = state["urls_queue"][1:]
        new_state["urls_done"].append(url)
        return 0, new_state 