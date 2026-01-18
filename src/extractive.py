import re
from dataclasses import dataclass
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def split_sentences(text: str) -> List[str]:
    # Simple sentence splitter (good enough for demo/portfolio)
    text = re.sub(r"\s+", " ", text.strip())
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def lead_2(article_text: str) -> str:
    """Lead-2 baseline: returns the first two sentences of the article."""
    sents = split_sentences(article_text)
    return " ".join(sents[:2])


@dataclass
class ExtractiveTFIDFLogReg:
    """
    A minimal TF-IDF + Logistic Regression sentence relevance model.

    This is a portfolio/demo implementation:
    - Trains on sentence-level labels (0/1).
    - Predicts relevance probabilities for sentences.
    - Selects top-k sentences as an extractive summary.
    """
    vectorizer: TfidfVectorizer
    clf: LogisticRegression

    @staticmethod
    def default():
        return ExtractiveTFIDFLogReg(
            vectorizer=TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=1,
                max_features=20000
            ),
            clf=LogisticRegression(
                max_iter=1000,
                solver="liblinear"
            )
        )

    def fit(self, sentences: List[str], labels: List[int]) -> "ExtractiveTFIDFLogReg":
        X = self.vectorizer.fit_transform(sentences)
        self.clf.fit(X, labels)
        return self

    def predict_proba(self, sentences: List[str]) -> List[float]:
        X = self.vectorizer.transform(sentences)
        # Probability of class 1 (relevant)
        return self.clf.predict_proba(X)[:, 1].tolist()

    def summarize(self, article_text: str, top_k: int = 2) -> str:
        sents = split_sentences(article_text)
        if not sents:
            return ""
        probs = self.predict_proba(sents)
        # rank by predicted relevance
        ranked = sorted(range(len(sents)), key=lambda i: probs[i], reverse=True)[:top_k]
        # keep original order for readability
        ranked_sorted = sorted(ranked)
        return " ".join(sents[i] for i in ranked_sorted)


def make_sentence_dataset(
    examples: List[Tuple[str, List[int]]]
) -> Tuple[List[str], List[int]]:
    """
    Helper to build a sentence-level dataset.
    examples: list of (article_text, sentence_labels)
      - sentence_labels length must match number of split sentences
      - labels are 0/1 for non-relevant/relevant
    """
    all_sents: List[str] = []
    all_labels: List[int] = []

    for article, labels in examples:
        sents = split_sentences(article)
        if len(sents) != len(labels):
            raise ValueError(
                f"Label count ({len(labels)}) does not match sentence count ({len(sents)})."
            )
        all_sents.extend(sents)
        all_labels.extend(labels)

    return all_sents, all_labels
