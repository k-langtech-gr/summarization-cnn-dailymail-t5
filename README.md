# Summarization (CNN/DailyMail) – Baselines + T5 Fine-Tuning

This repository contains a compact, reproducible pipeline for English news summarization experiments on the CNN/DailyMail dataset.

## What is included
- **Lead-2 baseline**
- **Classical ML baseline** (TF-IDF + Logistic Regression)
- **Transformer fine-tuning**: **T5-small** for abstractive summarization
- **Evaluation** using **ROUGE-2**

## What is NOT included (for safety/licensing)
- The original dataset files are not uploaded.
- Model checkpoints are not uploaded.
- No API keys or private credentials are used.

## Method overview
### Baseline 1: Lead-2
A strong extractive heuristic baseline that selects the first two sentences of the article.

### Baseline 2: TF-IDF + Logistic Regression
A simple supervised approach using TF-IDF features to predict sentence relevance for extractive summaries.

### Model: T5-small fine-tuning
- Framework: Hugging Face Transformers + Datasets
- Task: Abstractive summarization
- Decoding: beam search
- Metric: ROUGE-2

## Reproducibility
A small sample setup is provided. To reproduce results:
1. Create a Python environment (3.10+ recommended)
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

