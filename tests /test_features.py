import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def test_dataset_exists():
    data = pd.read_csv("dataset/emails.csv")

    assert not data.empty
    assert "text" in data.columns
    assert "label" in data.columns


def test_dataset_labels():
    data = pd.read_csv("dataset/emails.csv")

    labels = set(data["label"].unique())

    assert "phishing" in labels
    assert "safe" in labels


def test_tfidf_vectorization():
    data = pd.read_csv("dataset/emails.csv")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english"
    )

    features = vectorizer.fit_transform(data["text"])

    assert features.shape[0] == len(data)
    assert features.shape[1] > 0
