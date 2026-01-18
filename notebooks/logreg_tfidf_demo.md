# TF-IDF + Logistic Regression (Extractive Summarization) – Demo

This is a public-safe demo using a tiny hand-made toy fit (no dataset included).

```python
from src.tfidf_logreg_extractive import toy_fit_demo_model

model = toy_fit_demo_model()

article = (
    "Open-source models are improving quickly. "
    "The team evaluated multiple baselines. "
    "Coffee prices rose this week."
)

print(model.summarize(article, max_sents=2))
