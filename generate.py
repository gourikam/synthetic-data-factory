"""
generate.py  —  Phase 1: Synthetic Data Generation
====================================================
WHAT THIS FILE DOES:
  Reads seed prompts from prompts.txt, calls the Groq API (free)
  multiple times per prompt with high temperature, parses the
  returned Q&A pairs, and streams them into raw.jsonl.

CONCEPTS TAUGHT:
  - Async API calls with asyncio + httpx
  - Prompt engineering for structured output
  - JSONL format for ML datasets
  - Basic retry logic for flaky API responses
  - Progress tracking with tqdm

RUN:
  pip install groq asyncio tqdm
  export GROQ_API_KEY=your_key_here
  python generate.py
"""

import os
import json
import asyncio
import re
import time
from pathlib import Path
from groq import AsyncGroq          # Groq's official async client
from tqdm.asyncio import tqdm       # progress bar that works with async
from dotenv import load_dotenv
load_dotenv()   # reads .env file and loads vars into os.environ automatically

# ─────────────────────────────────────────────
# CONFIG  —  tweak these numbers as needed
# ─────────────────────────────────────────────

GROQ_MODEL      = "llama-3.3-70b-versatile"   # free on Groq; fast and capable
CALLS_PER_SEED  = 15                  # 18 seeds × 15 = 270 raw pairs
TEMPERATURE     = 0.9                 # high = more varied outputs
MAX_TOKENS      = 600                 # enough for a thorough answer
CONCURRENCY     = 2                   # max parallel API calls at once
                                      # (Groq free tier limit: ~30 req/min)

PROMPTS_FILE    = Path("prompts.txt")
OUTPUT_FILE     = Path("data/raw.jsonl")

# ─────────────────────────────────────────────
# SYSTEM PROMPT  —  this is the key to getting
# structured output from the LLM.
#
# CONCEPT: A system prompt is a hidden instruction
# that shapes every response. Here we tell the model
# to ONLY return JSON so we can parse it reliably.
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a medical education assistant generating training data.

Your output must be ONLY a valid JSON object with exactly these two keys:
{
  "question": "a clear, realistic medical question",
  "answer": "a thorough, factual, 3-6 sentence answer"
}

Rules:
- The question should sound like a real patient or student asking
- The answer must be accurate, specific, and educational
- Do NOT add disclaimers like "consult a doctor" — keep it factual
- Do NOT output anything outside the JSON object
- Do NOT wrap in markdown code blocks
"""

# ─────────────────────────────────────────────
# LOAD SEED PROMPTS
#
# CONCEPT: We skip comment lines (starting with #)
# and blank lines. Each remaining line is one seed
# prompt that will be sent to the LLM.
# ─────────────────────────────────────────────

def load_prompts(filepath: Path) -> list[str]:
    prompts = []
    current = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("#") or line == "":
                if current:
                    prompts.append(" ".join(current))
                    current = []
            else:
                current.append(line)

    if current:                        # catch last prompt if no trailing newline
        prompts.append(" ".join(current))

    return prompts


# ─────────────────────────────────────────────
# PARSE LLM RESPONSE
#
# CONCEPT: Even with a strict system prompt, LLMs
# sometimes wrap JSON in markdown (```json ... ```)
# or add extra text. We strip that out before
# calling json.loads(). This is called "output
# parsing" and is a core skill in LLM engineering.
# ─────────────────────────────────────────────

def parse_qa(raw_text: str) -> dict | None:
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?", "", raw_text).strip()

    try:
        data = json.loads(text)
        # Validate structure — must have both keys with non-empty strings
        if (
            isinstance(data, dict)
            and "question" in data
            and "answer" in data
            and len(data["question"].strip()) > 20
            and len(data["answer"].strip()) > 50
        ):
            return data
    except json.JSONDecodeError:
        pass        # will return None → caller skips this sample

    return None


# ─────────────────────────────────────────────
# SINGLE API CALL WITH RETRY
#
# CONCEPT: Network calls can fail due to rate limits,
# timeouts, or transient errors. A simple retry loop
# with exponential backoff (wait 2s, then 4s, then 8s)
# is standard practice in any production pipeline.
# ─────────────────────────────────────────────

async def call_groq(client: AsyncGroq, prompt: str, retries: int = 4) -> dict | None:
    for attempt in range(retries):
        try:
            response = await client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt}
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            raw_text = response.choices[0].message.content
            return parse_qa(raw_text)

        except Exception as e:
            error_str = str(e)

            # Don't retry unrecoverable errors
            if "model_decommissioned" in error_str or "invalid_request_error" in error_str:
                print(f"\n  [fatal] {e}")
                return None

            # For rate limits, read the wait time from the error message
            if "rate_limit_exceeded" in error_str:
                # Extract "Please try again in Xs" from the message
                match = re.search(r"try again in ([0-9.]+)s", error_str)
                wait = float(match.group(1)) + 0.5 if match else 3.0
                # No print — rate limits are expected, don't clutter output
                await asyncio.sleep(wait)
                continue

            # For other transient errors, exponential backoff
            wait = 2 ** attempt
            print(f"\n  [retry {attempt+1}] Transient error: {e}. Waiting {wait}s...")
            await asyncio.sleep(wait)

    return None


# ─────────────────────────────────────────────
# MAIN GENERATION LOOP
#
# CONCEPT: asyncio.Semaphore limits concurrency —
# it's like a ticket system. Only CONCURRENCY tasks
# can hold a ticket at once. When one finishes, it
# releases its ticket for the next task.
# ─────────────────────────────────────────────

async def generate_all(prompts: list[str], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client    = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
    semaphore = asyncio.Semaphore(CONCURRENCY)

    total_tasks    = len(prompts) * CALLS_PER_SEED
    saved_count    = 0
    skipped_count  = 0

    async def bounded_call(prompt: str, seed_idx: int):
        async with semaphore:           # acquire ticket
            result = await call_groq(client, prompt)
            await asyncio.sleep(0.2)   # small delay to stay under rate limit
            return result, seed_idx

    # Build all tasks upfront
    # Each seed prompt is called CALLS_PER_SEED times
    tasks = [
        bounded_call(prompt, idx)
        for idx, prompt in enumerate(prompts)
        for _ in range(CALLS_PER_SEED)
    ]

    print(f"\nGenerating {total_tasks} Q&A pairs from {len(prompts)} seed prompts...")
    print(f"Model: {GROQ_MODEL} | Temp: {TEMPERATURE} | Concurrency: {CONCURRENCY}\n")

    # Open output file — write one JSON line per successful pair
    with open(output_path, "w") as out_file:
        for coro in tqdm(asyncio.as_completed(tasks), total=total_tasks):
            result, seed_idx = await coro

            if result is not None:
                # Enrich with metadata before saving
                # CONCEPT: metadata like topic_id and seed_index lets you
                # group samples by topic in Phase 4 evaluation
                result["seed_index"] = seed_idx
                result["model"]      = GROQ_MODEL

                out_file.write(json.dumps(result) + "\n")
                saved_count += 1
            else:
                skipped_count += 1

    print(f"\n{'─'*40}")
    print(f"  Saved  : {saved_count} Q&A pairs  →  {output_path}")
    print(f"  Skipped: {skipped_count} malformed responses")
    print(f"  Success rate: {saved_count/total_tasks*100:.1f}%")
    print(f"{'─'*40}\n")
    print("Next step: run  python filter.py  to clean the dataset.")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY environment variable not set.")
        print("Get your free key at: https://console.groq.com")
        exit(1)

    prompts = load_prompts(PROMPTS_FILE)
    print(f"Loaded {len(prompts)} seed prompts from {PROMPTS_FILE}")

    start = time.time()
    asyncio.run(generate_all(prompts, OUTPUT_FILE))
    print(f"Total time: {time.time() - start:.1f}s")