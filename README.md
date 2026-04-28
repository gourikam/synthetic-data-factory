# 🏥 MedQA — Synthetic Data Factory

> Fine-tuned Phi-3.5-mini on synthetic medical Q&A data, achieving **118% ROUGE-L improvement** over the base model.

**[🌐 Live Demo](https://synthetic-data-factory-eight.vercel.app)** · **[Model on HuggingFace](https://huggingface.co/gourikam/phi3-medical-qa)**

---

## What This Project Does

Builds a complete pipeline that:
1. **Generates** synthetic medical Q&A pairs using Llama-3.3-70B (Groq API)
2. **Filters** them using rule-based checks + Jaccard deduplication
3. **Fine-tunes** Phi-3.5-mini-instruct using LoRA on a free T4 GPU (Google Colab)
4. **Evaluates** the fine-tuned model against the base model using ROUGE-L
5. **Deploys** a live demo website showing the pipeline and chat interface

---

## Results

| Metric | Base Phi-3.5 | Fine-tuned MedQA |
|--------|-------------|-----------------|
| ROUGE-L | 0.235 | 0.513 |
| Improvement | — | **+118%** |
| Test questions won | — | **15/15** |

---

## Pipeline Architecture

    Seed Prompts (18)
          ↓
    Data Generation — Llama-3.3-70B via Groq (186 raw pairs)
          ↓
    Quality Filter — rules + Jaccard dedup (141 clean pairs)
          ↓
    Fine-Tuning — LoRA r=16 on Phi-3.5-mini (Colab T4, free)
          ↓
    Evaluation — ROUGE-L benchmark vs base model
          ↓
    Live Website — deployed on Vercel

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Data generation | Groq API (Llama-3.3-70B), Python asyncio |
| Quality filtering | Jaccard similarity, rule-based validation |
| Fine-tuning | HuggingFace Transformers, PEFT, Unsloth, LoRA |
| Training hardware | Google Colab T4 GPU (free) |
| Evaluation | ROUGE-L via rouge-score |
| Frontend | HTML, Tailwind CSS, vanilla JS |
| Deployment | Vercel (free) |
| Model hosting | HuggingFace Hub (free) |

---

## Project Structure

    synthetic-data-factory/
    ├── generate.py          # Phase 1 — synthetic data generation
    ├── filter.py            # Phase 2 — quality filtering + deduplication
    ├── prompts.txt          # 18 seed prompts across 6 medical categories
    ├── .gitignore
    ├── data/
    │   ├── filter_stats.json
    │   └── eval_results.json
    └── website/
        └── index.html       # Live demo website
---

## Key Concepts Demonstrated

- **Synthetic data generation** — using a large model to teach a small model
- **LoRA fine-tuning** — parameter-efficient training (only 0.78% of params trained)
- **4-bit quantization** — fitting a 3.8B model in free T4 GPU memory
- **LLM evaluation** — ROUGE-L benchmarking with held-out test set
- **Data quality** — deduplication, rule-based filtering, LLM-as-judge
- **Full-stack deployment** — from Colab notebook to live public URL

---

## Setup

```bash
# Clone
git clone https://github.com/gourikam/synthetic-data-factory
cd synthetic-data-factory

# Install dependencies
python -m venv venv
source venv/bin/activate
python -m pip install groq tqdm httpx python-dotenv

# Set your Groq API key
echo "GROQ_API_KEY=gsk_your_key" > .env

# Run the pipeline
python generate.py   # generates data/raw.jsonl
python filter.py     # generates data/clean.jsonl
# Then open Colab for fine-tuning — see train.ipynb
```

---

## Medical Topics Covered

Chest pain · Shortness of breath · Headache red flags · ER vs clinic decisions ·
OTC painkillers · Antibiotic resistance · Insulin types · Type 2 diabetes ·
Hypothyroidism · GERD · CBC interpretation · Creatinine · HbA1c · Thyroid panel ·
Lipid panel · Mental health referrals · Cancer screening · Heart health diet

---

*Built for educational purposes. Not a substitute for professional medical advice.*
