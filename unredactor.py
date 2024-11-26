import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.sparse import hstack


def load_data(filepath, test_mode=False):
    columns = ['file_type', 'names', 'redacted_text'] if not test_mode else ['id', 'redacted_text']
    return pd.read_csv(filepath, sep='\t', header=None, on_bad_lines='skip', names=columns)


def clean_text(data):
    return data.str.replace(r"[^a-zA-Z0-9\s]", "", regex=True).str.lower().str.strip()


def calculate_features(data, is_test=False):
    features = pd.DataFrame()
    features['cleaned_text'] = clean_text(data['redacted_text'])
    features['name_length'] = data['names'].str.len() if not is_test else data['redacted_text'].str.len()
    features['space_count'] = data['names'].str.count(r'\s+') if not is_test else data['redacted_text'].str.count(r'\s+')
    features['redacted_block_count'] = data['redacted_text'].str.count('█').values.reshape(-1, 1)
    features['word_count'] = data['redacted_text'].str.split().str.len()
    features['character_count'] = data['redacted_text'].str.len()
    features['unique_word_count'] = data['redacted_text'].apply(lambda x: len(set(x.split())))
    features['avg_word_length'] = data['redacted_text'].apply(lambda x: sum(len(word) for word in x.split()) / (len(x.split()) + 1))
    return features


def vectorize_data(train, val, test=None):
    vectorizer = TfidfVectorizer(max_features=3000)
    train_vectorized = vectorizer.fit_transform(train.values)
    val_vectorized = vectorizer.transform(val.values)
    test_vectorized = vectorizer.transform(test.values) if test is not None else None
    return train_vectorized, val_vectorized, test_vectorized, vectorizer


def combine_features(*features):
    return hstack(features)


def build_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    return model.fit(X_train, y_train)


def predict_outcomes(model, X_train, X_val):
    return model.predict(X_train), model.predict(X_val)


def compute_metrics(y_train, y_pred_train, y_val, y_pred_val):
    metrics = {
        "train_precision": precision_score(y_train, y_pred_train, average="weighted"),
        "val_precision": precision_score(y_val, y_pred_val, average="weighted"),
        "train_recall": recall_score(y_train, y_pred_train, average="weighted"),
        "val_recall": recall_score(y_val, y_pred_val, average="weighted"),
        "train_f1": f1_score(y_train, y_pred_train, average="weighted"),
        "val_f1": f1_score(y_val, y_pred_val, average="weighted"),
    }
    return metrics


def generate_submission(test_data, model, vectorizer, scaler, train_cleaned_text, val_cleaned_text):
    test_features = calculate_features(test_data, is_test=True)
    _, _, X_test_text, _ = vectorize_data(train_cleaned_text, val_cleaned_text, test_features['cleaned_text'])
    scaled_test_features = scaler.transform(
        test_features[['name_length', 'space_count', 'redacted_block_count', 'word_count', 'character_count', 'unique_word_count', 'avg_word_length']]
    )
    X_test_combined = combine_features(X_test_text, scaled_test_features)
    test_predictions = model.predict(X_test_combined)
    test_data['predicted_names'] = test_predictions
    test_data[['id', 'predicted_names']].to_csv('submission.tsv', sep='\t', index=False)
    print("Predictions saved to submission.tsv")


def main():
    train_file = "unredactor.tsv"
    data = load_data(train_file)

    train_data = data[data['file_type'] == 'training']
    val_data = data[data['file_type'] == 'validation']

    train_features = calculate_features(train_data)
    val_features = calculate_features(val_data)

    X_train = train_features.drop(columns=['cleaned_text'])
    y_train = train_data['names']
    X_val = val_features.drop(columns=['cleaned_text'])
    y_val = val_data['names']

    X_train_text, X_val_text, _, vectorizer = vectorize_data(train_features['cleaned_text'], val_features['cleaned_text'])

    scaler = StandardScaler()
    scaled_train_features = scaler.fit_transform(
        train_features[['name_length', 'space_count', 'redacted_block_count', 'word_count', 'character_count', 'unique_word_count', 'avg_word_length']]
    )
    scaled_val_features = scaler.transform(
        val_features[['name_length', 'space_count', 'redacted_block_count', 'word_count', 'character_count', 'unique_word_count', 'avg_word_length']]
    )

    X_train_combined = combine_features(X_train_text, scaled_train_features)
    X_val_combined = combine_features(X_val_text, scaled_val_features)

    model = build_model(X_train_combined, y_train)

    y_pred_train, y_pred_val = predict_outcomes(model, X_train_combined, X_val_combined)

    metrics = compute_metrics(y_train, y_pred_train, y_val, y_pred_val)
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    test_file = "test.tsv"
    test_data = load_data(test_file, test_mode=True)
    generate_submission(test_data, model, vectorizer, scaler, train_features['cleaned_text'], val_features['cleaned_text'])


if __name__ == "__main__":
    main()