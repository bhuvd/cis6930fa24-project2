import pytest
import pandas as pd
import numpy as np
from scipy.sparse import isspmatrix
from unredactor import (
    load_data,
    clean_text,
    calculate_features,
    vectorize_data,
    combine_features,
    build_model,
    predict_outcomes,
    compute_metrics
)

@pytest.fixture
def sample_data():
    data = {
        "file_type": ["training", "validation"],
        "names": ["John Doe", "Jane Smith"],
        "redacted_text": ["████ ███", "████████ █████"]
    }
    return pd.DataFrame(data)

@pytest.fixture
def test_data():
    data = {
        "id": [1, 2],
        "redacted_text": ["█████ ████", "█████████"]
    }
    return pd.DataFrame(data)

def test_load_data(sample_data):
    sample_data.to_csv("test_file.tsv", sep="\t", index=False, header=False)
    df = load_data("test_file.tsv")
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["file_type", "names", "redacted_text"]
    assert df.shape == sample_data.shape

def test_clean_text():
    data = pd.Series(["Hello, World!", "Test123"])
    cleaned = clean_text(data)
    assert list(cleaned) == ["hello world", "test123"]

def test_calculate_features(sample_data):
    features = calculate_features(sample_data)
    assert "cleaned_text" in features.columns
    assert "name_length" in features.columns
    assert "space_count" in features.columns
    assert "redacted_block_count" in features.columns
    assert "word_count" in features.columns
    assert "character_count" in features.columns
    assert "unique_word_count" in features.columns
    assert "avg_word_length" in features.columns
    assert features.shape[0] == sample_data.shape[0]

'''def test_vectorize_data(sample_data):
    features = calculate_features(sample_data)
    train_vect, val_vect, _, vectorizer = vectorize_data(features["cleaned_text"], features["cleaned_text"])
    assert isspmatrix(train_vect)
    assert isspmatrix(val_vect)
    assert train_vect.shape[1] == val_vect.shape[1]

def test_combine_features(sample_data):
    features = calculate_features(sample_data)
    train_vect, _, _, _ = vectorize_data(features["cleaned_text"], features["cleaned_text"])
    combined = combine_features(train_vect, features[['name_length']])
    assert isspmatrix(combined)
    assert combined.shape[0] == train_vect.shape[0]

def test_build_model(sample_data):
    features = calculate_features(sample_data)
    train_vect, _, _, _ = vectorize_data(features["cleaned_text"], features["cleaned_text"])
    combined_features = combine_features(train_vect, features[['name_length']])
    model = build_model(combined_features, sample_data["names"])
    assert model is not None

def test_predict_outcomes(sample_data):
    features = calculate_features(sample_data)
    train_vect, _, _, _ = vectorize_data(features["cleaned_text"], features["cleaned_text"])
    combined_features = combine_features(train_vect, features[['name_length']])
    model = build_model(combined_features, sample_data["names"])
    train_pred, val_pred = predict_outcomes(model, combined_features, combined_features)
    assert len(train_pred) == combined_features.shape[0]
    assert len(val_pred) == combined_features.shape[0]
'''
def test_compute_metrics(sample_data):
    true_labels = sample_data["names"]
    predicted_labels = ["John Doe", "Jane Smith"]
    metrics = compute_metrics(true_labels, predicted_labels, true_labels, predicted_labels)
    assert metrics["train_precision"] == 1.0
    assert metrics["val_precision"] == 1.0
    assert metrics["train_recall"] == 1.0
    assert metrics["val_recall"] == 1.0
    assert metrics["train_f1"] == 1.0
    assert metrics["val_f1"] == 1.0