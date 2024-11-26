# cis6930fa24-project2
## Assignment Description

This program unredacts the redacted text.

## How to Install
1. Open terminal with 'Run as administrator'
2. Install pipenv if not already installed: `pip install pipenv` 
3. In general to install dependencies: `pipenv install` -- for this project use command  `pipenv install requests`

## How to install Pytest
1. Open terminal with 'Run as administrator'
2. Install pytest if not already installed:  `pipenv install pytest`

## How to run 
1. To run the program use the command : pipenv run python unredactor.py
2. To run the test cases : pipenv run python -m pytest -v

The program runs with the following metrics , can be improved based on machine
train_precision: 0.9993702770780857
val_precision: 0.04762177640004657
train_recall: 0.998920474991004
val_recall: 0.05012224938875306

## Assumptions
The program expects a unredactor.tsv and test.tsv to work on in the folder location/

## Function Description

# funtion load_data
Loads the dataset from a file, processes it into a Pandas DataFrame, and assigns appropriate column names based on whether the file is for training/validation or testing.Handles incorrect or malformed lines. Assigns different column names for training/validation 

# funtion clean_text
Cleans the text by removing non-alphanumeric characters, converting to lowercase, and stripping leading or trailing whitespaces.

# funtion calculate_features
Generates additional engineered features for both training and testing data. 
Computes various features, such as:
name_length: Length of the names column or redacted_text for test data.
space_count: Number of spaces in the text.
redacted_block_count: Counts occurrences of the "█" block character.
word_count: Number of words in redacted_text.
character_count: Total number of characters.
unique_word_count: Unique word count in redacted_text.
avg_word_length: Average word length in redacted_text.

# funtion vectorize_data
Converts text data into numerical format using the TF-IDF vectorization technique.Initializes a TfidfVectorizer with a vocabulary size of 3000.Fits the vectorizer on the training data .Transforms the validation and test data into sparse matrices using the vectorizer.

# funtion combine_features
Combines multiple feature matrices (e.g., TF-IDF features and numerical features) into a single sparse matrix.

# funtion build_model
Trains a Random Forest classifier on the provided training features and labels.

# funtion predict_outcomes
Predicts outcomes for training and validation data using the trained model.

# funtion compute_metrics
Calculates evaluation metrics  for training and validation predictions.

# funtion generate_submission
Processes the test dataset, generates predictions using the trained model, and saves the results to a file.

# funtion main
Coordinates the overall workflow, including data loading, feature engineering, model training, evaluation, and test submission generation.

## Test Function Description

# load_data 
Verifies that the data is loaded correctly with the proper shape and column names.

# clean_text
 Ensures text cleaning removes unwanted characters and converts text to lowercase.

# calculate_features
 Validates that all engineered features are created correctly.

# compute_metrics
 Verifies the precision, recall, and F1-score calculations are correct.

## Video
https://youtu.be/LvzxZNfvsi4 