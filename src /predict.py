import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def train_model():
    # Load the dataset
    data = pd.read_csv("dataset/emails.csv")

    # Input and output
    X = data["text"]
    y = data["label"]

    # Convert email text into numerical features
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english"
    )

    X_features = vectorizer.fit_transform(X)

    # Train Logistic Regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_features, y)

    return vectorizer, model


def predict_email(email):
    # Train the model
    vectorizer, model = train_model()

    # Convert new email into TF-IDF features
    email_features = vectorizer.transform([email])

    # Make prediction
    prediction = model.predict(email_features)[0]

    return prediction


def main():
    print("=== Phishing Email Detection ===")

    email = input("\nEnter email text: ")

    prediction = predict_email(email)

    print(f"\nPrediction: {prediction.upper()}")


if __name__ == "__main__":
    main()
