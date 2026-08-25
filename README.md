# Phishing Email Detection Model

A simple machine learning project that classifies an email as **Phishing** or **Safe** using Python and Scikit-learn.

## Features

- Trains on a small labeled email dataset.
- Uses TF-IDF to analyze email text.
- Extracts simple URL-related features:
  - number of URLs
  - number of suspicious URL patterns
  - number of IP-address URLs
  - presence of HTTPS
- Uses Logistic Regression for classification.
- Displays accuracy, classification report, and confusion matrix.
- Allows a user to test a new email from the terminal.

## Project Structure

```text
phishing-email-detection/
├── dataset/
│   └── emails.csv
├── src/
│   ├── train_model.py
│   └── predict.py
├── tests/
│   └── test_features.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Requirements

- Python 3.9 or newer
- pip

## Installation

```bash
git clone <your-github-repository-url>
cd phishing-email-detection
python -m venv venv
```

Windows:
```bash
venv\Scripts\activate
```

Linux/macOS:
```bash
source venv/bin/activate
```

Install packages:

```bash
pip install -r requirements.txt
```

## Run the Project

Train and evaluate the model:

```bash
python src/train_model.py
```

Test a new email:

```bash
python src/predict.py
```

## Example

```text
Enter email text:
Your bank account has been suspended. Verify your account at http://secure-login-example.com

Prediction: PHISHING
```

## How It Works

1. The dataset contains email text and a label.
2. TF-IDF converts the email text into numerical features.
3. URL features are extracted from the email.
4. Text and URL features are combined.
5. Logistic Regression learns the difference between phishing and safe emails.
6. The trained model predicts the class of a new email.

## Limitations

This is an educational internship project. The included dataset is small and synthetic, so it should not be treated as a production email security system. A real deployment would require a much larger, diverse, and regularly updated dataset.

## License

This project is licensed under the MIT License.
