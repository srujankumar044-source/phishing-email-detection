import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train_model import extract_url_features


def test_url_features():
    features = extract_url_features(
        "Please verify your account at http://192.168.1.10/login"
    )
    assert features[0] == 1
    assert features[1] >= 1
    assert features[2] == 1


def test_safe_url_features():
    features = extract_url_features(
        "Read the documentation at https://www.python.org/"
    )
    assert features[0] == 1
    assert features[2] == 0
    assert features[3] == 1
