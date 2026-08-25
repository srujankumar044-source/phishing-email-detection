import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main():
    # Load dataset
    data = pd.read_csv("dataset/emails.csv")

    # Separate input and output
    X = data["text"]
    y = data["label"]

    # Convert text into numerical TF-IDF features
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english"
    )

    X_features = vectorizer.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_features,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Create Logistic Regression model
    model = LogisticRegression(max_iter=1000)

    # Train the model
    model.fit(X_train, y_train)

    # Predict test data
    predictions = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)

    print("=== Phishing Email Detection Model ===")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))


if __name__ == "__main__":
    main()
