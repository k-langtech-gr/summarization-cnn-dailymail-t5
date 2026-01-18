# T5 inference demo (abstractive summarization)

This repo provides inference-only summarization using a public pretrained model (`t5-small`).

## Quick demo (Python)
```python
from src.t5_abstractive import summarize

text = "Your long article text goes here..."
print(summarize(text))
