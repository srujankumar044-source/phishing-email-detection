import re
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def extract_url_features(text):
    urls = re.findall(r"https?://\S+|www\.\S+", text, flags=re.IGNORECASE)
    return [
        len(urls),
        len(re.findall(
            r"(login|verify|secure|update|account|password|bank|confirm)",
            text,
            flags=re.IGNORECASE,
        )),
        len(re.findall(
            r"https?://(?:\d{1,3}\.){3}\d{1,3}",
            text,
            flags=re.IGNORECASE,
        )),
        len(re.findall(r"https://", text, flags=re.IGNORECASE)),
    ]


def train():
    data = pd.read_csv("dataset/emails.csv")
    data["text"] = data["text"].fillna("")
    data["label"] = data["label"].str.strip().str.lower()

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=3000,
    )

    text_features = vectorizer.fit_transform(data["text"])
    url_features = csr_matrix(
        [extract_url_features(x) for x in data["text"]]
    )
    features = hstack([text_features, url_features])

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(features, data["label"])
    return vectorizer, model


def predict_email(text, vectorizer, model):
    text_features = vectorizer.transform([text])
    url_features = csr_matrix([extract_url_features(text)])
    features = hstack([text_features, url_features])
    return model.predict(features)[0]


if __name__ == "__main__":
    vectorizer, model = train()

    print("=== Phishing Email Detector ===")
    print("Type an email message and press Enter.")
    email = input("\nEmail: ").strip()

    if not email:
        print("No email text entered.")
    else:
        result = predict_email(email, vectorizer, model)
        print(f"\nPrediction: {result.upper()}")
