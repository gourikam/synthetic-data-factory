"""
filter.py  —  Phase 2: Quality Filtering
=========================================
WHAT THIS FILE DOES:
  Reads raw.jsonl, runs two filtering stages:
    Stage 1 — fast rule-based checks (length, structure, duplicates)
    Stage 2 — LLM-as-judge scores each pair 1-5, keeps only score >= 4
  Saves survivors to data/clean.jsonl

CONCEPTS TAUGHT:
  - Rule-based vs model-based filtering
  - LLM-as-judge evaluation pattern
  - Deduplication with string similarity
  - Dataset statistics and reporting

RUN:
  python filter.py
"""

import os
import json
import asyncio
import re
from pathlib import Path
from groq import AsyncGroq
from tqdm.asyncio import tqdm
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

GROQ_MODEL      = "llama-3.3-70b-versatile"
CONCURRENCY     = 2
MIN_SCORE       = 3         # keep pairs scored 4 or 5 out of 5
TEMPERATURE     = 0.0        # judge should be deterministic, not creative

INPUT_FILE      = Path("data/raw.jsonl")
OUTPUT_FILE     = Path("data/clean.jsonl")
STATS_FILE      = Path("data/filter_stats.json")

# ─────────────────────────────────────────────
# JUDGE PROMPT
#
# CONCEPT: The judge prompt is carefully designed
# to return ONLY a single digit. This makes parsing
# trivial and reliable. Notice we give the judge
# explicit criteria — this is called a "rubric"
# and makes scoring consistent across calls.
# ─────────────────────────────────────────────

JUDGE_PROMPT = """You are reviewing a medical Q&A training sample. Decide if it should be KEPT or DISCARDED for training a medical AI assistant.

KEEP if ALL of these are true:
- The question is a realistic thing a patient or student would ask
- The answer is factually reasonable and at least 2 sentences
- The answer actually addresses the question

DISCARD if ANY of these are true:
- The answer is vague, generic, or just says "consult a doctor"
- The question and answer don't match
- The answer is clearly wrong or nonsensical

Respond with ONLY the word KEEP or DISCARD. Nothing else.

QUESTION: {question}
ANSWER: {answer}"""


# ─────────────────────────────────────────────
# STAGE 1 — RULE-BASED FILTERING
#
# CONCEPT: Rules are cheap and fast. Run them
# first to eliminate obvious garbage before
# spending API calls on the LLM judge.
# ─────────────────────────────────────────────

def passes_rules(pair: dict) -> tuple[bool, str]:
    """Returns (passed, reason_if_failed)"""
    q = pair.get("question", "").strip()
    a = pair.get("answer", "").strip()

    if len(q) < 20:
        return False, "question too short"
    if len(a) < 80:
        return False, "answer too short"
    if len(a) > 2000:
        return False, "answer suspiciously long"
    if q.lower() == a.lower():
        return False, "question equals answer"
    if not q.endswith("?") and len(q) < 50:
        return False, "question not a question"
    if sum(1 for c in a if c.isupper()) / max(len(a), 1) > 0.5:
        return False, "answer is mostly uppercase"

    return True, ""


# ─────────────────────────────────────────────
# DEDUPLICATION
#
# CONCEPT: Two questions are "duplicates" if they
# share more than 80% of their words. We use a
# simple Jaccard similarity — intersection over
# union of word sets. Fast, no libraries needed.
# ─────────────────────────────────────────────

def jaccard_similarity(a: str, b: str) -> float:
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def deduplicate(pairs: list[dict], threshold: float = 0.8) -> list[dict]:
    kept = []
    for pair in pairs:
        q = pair["question"]
        is_duplicate = any(
            jaccard_similarity(q, kept_pair["question"]) > threshold
            for kept_pair in kept
        )
        if not is_duplicate:
            kept.append(pair)
    return kept


# ─────────────────────────────────────────────
# STAGE 2 — LLM-AS-JUDGE
#
# CONCEPT: We ask the LLM to score each pair.
# Temperature=0 means the model gives the same
# score every time for the same input (deterministic).
# This is important for reproducibility in evaluation.
# ─────────────────────────────────────────────

async def judge_pair(client: AsyncGroq, pair: dict, semaphore: asyncio.Semaphore) -> tuple[dict, int]:
    async with semaphore:
        prompt = JUDGE_PROMPT.format(
            question=pair["question"],
            answer=pair["answer"]
        )
        try:
            response = await client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=5,       # we only need one digit back
            )
            raw = response.choices[0].message.content.strip().upper()
            # Convert binary decision to score: KEEP=4, DISCARD=2
            score = 4 if "KEEP" in raw else 2
            await asyncio.sleep(0.3)
            return pair, score

        except Exception as e:
            # If judge fails, give it a neutral score of 3 (won't pass threshold)
            error_str = str(e)
            if "rate_limit_exceeded" in error_str:
                match = re.search(r"try again in ([0-9.]+)s", error_str)
                wait = float(match.group(1)) + 0.5 if match else 3.0
                await asyncio.sleep(wait)
            return pair, 3


# ─────────────────────────────────────────────
# MAIN FILTER PIPELINE
# ─────────────────────────────────────────────

async def run_filter():
    # ── Load raw dataset ──
    raw_pairs = []
    with open(INPUT_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    raw_pairs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    print(f"\nLoaded {len(raw_pairs)} raw pairs from {INPUT_FILE}")
    print("─" * 40)

    # ── Stage 1: Rule-based filtering ──
    print("\nStage 1 — Rule-based filtering...")
    passed_rules = []
    rule_failures = {}

    for pair in raw_pairs:
        passed, reason = passes_rules(pair)
        if passed:
            passed_rules.append(pair)
        else:
            rule_failures[reason] = rule_failures.get(reason, 0) + 1

    print(f"  Passed : {len(passed_rules)}")
    print(f"  Dropped: {len(raw_pairs) - len(passed_rules)}")
    if rule_failures:
        for reason, count in sorted(rule_failures.items(), key=lambda x: -x[1]):
            print(f"    - {reason}: {count}")

    # ── Deduplication ──
    print("\nDeduplication...")
    deduped = deduplicate(passed_rules)
    print(f"  Before: {len(passed_rules)}  →  After: {len(deduped)}  (removed {len(passed_rules) - len(deduped)} duplicates)")

    # ── Stage 2: LLM-as-judge ──
    print(f"\nStage 2 — LLM-as-judge scoring (threshold: {MIN_SCORE}/5)...")
    print(f"  Scoring {len(deduped)} pairs — this will take a few minutes...\n")

    client    = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
    semaphore = asyncio.Semaphore(CONCURRENCY)

    tasks = [judge_pair(client, pair, semaphore) for pair in deduped]

    scores      = []
    clean_pairs = []

    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        pair, score = await coro
        pair["quality_score"] = score      # attach score as metadata
        scores.append(score)
        if score >= MIN_SCORE:
            clean_pairs.append(pair)

    # ── Save clean dataset ──
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        for pair in clean_pairs:
            f.write(json.dumps(pair) + "\n")

    # ── Save stats for the website dashboard later ──
    stats = {
        "raw_count"        : len(raw_pairs),
        "after_rules"      : len(passed_rules),
        "after_dedup"      : len(deduped),
        "clean_count"      : len(clean_pairs),
        "score_distribution": {
            str(s): scores.count(s) for s in range(1, 6)
        },
        "avg_score"        : round(sum(scores) / len(scores), 2) if scores else 0,
        "retention_rate"   : round(len(clean_pairs) / len(raw_pairs) * 100, 1),
    }
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)

    # ── Final report ──
    print(f"\n{'─'*40}")
    print(f"  Raw pairs        : {stats['raw_count']}")
    print(f"  After rules      : {stats['after_rules']}")
    print(f"  After dedup      : {stats['after_dedup']}")
    print(f"  Clean (score≥{MIN_SCORE}) : {stats['clean_count']}")
    print(f"  Retention rate   : {stats['retention_rate']}%")
    print(f"  Avg quality score: {stats['avg_score']}/5")
    print(f"  Score breakdown  : {stats['score_distribution']}")
    print(f"{'─'*40}")
    print(f"\n  Saved → {OUTPUT_FILE}")
    print(f"  Stats → {STATS_FILE}  (used by the website dashboard)")
    print(f"\nNext step: open Google Colab and run train.py")


if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY not set. Make sure your .env file exists.")
        exit(1)

    if not INPUT_FILE.exists():
        print(f"ERROR: {INPUT_FILE} not found. Run generate.py first.")
        exit(1)

    asyncio.run(run_filter())