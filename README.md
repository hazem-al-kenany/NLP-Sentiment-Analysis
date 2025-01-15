# NLP Sentiment Analysis for Mental Health

This project implements a machine learning pipeline for classifying text data related to mental health. It focuses on NLP (Natural Language Processing) and utilizes text preprocessing, binary label mapping, and classification algorithms to analyze sentiments in mental health statements, helping distinguish between normal and abnormal mental health indicators.

---

## Features

### Dataset
- **Source**: Kaggle dataset "Sentiment Analysis for Mental Health".
- **Attributes**:
  - `statement`: Text data representing user statements.
  - `status`: Multi-class labels indicating the mental health status (e.g., Anxiety, Depression).
- **Labels**:
  - Mapped to binary classes:
    - `normal`: Indicates no significant mental health concerns.
    - `abnormal`: Includes labels such as Anxiety, Depression, Stress, etc.

### Text Preprocessing
- **Stopword Removal**: Eliminates common words that do not contribute to classification.
- **Lemmatization**: Converts words to their base form.
- **Punctuation and Special Characters**: Cleans non-alphanumeric symbols using regex.
- **Lowercasing**: Standardizes text format.

### Feature Extraction
- **Bag of Words**:
  - Transforms text into numerical vectors using `CountVectorizer`.
  - Captures word frequencies for classification.

### Classification Models
1. **Multinomial Naive Bayes**:
   - Efficient for text data with high-dimensional feature spaces.
   - Trained on the Bag of Words representation.
2. **Logistic Regression**:
   - Robust classifier with higher interpretability for binary classification.

### Model Evaluation
- **Metrics**:
  - Precision, Recall, F1-Score for `normal` and `abnormal` classes.
  - Overall test accuracy.
- **Comparison**:
  - Evaluates the performance of Naive Bayes and Logistic Regression.

### Predictions
- Predicts sentiment for new user statements.
- Outputs binary labels (`normal`, `abnormal`).

---

## Code Structure

### Preprocessing
- **Functions**:
  - `preprocess_text`: Cleans and tokenizes input text.
  - `map_to_binary`: Maps multi-class labels to binary (`normal`, `abnormal`).

### Feature Extraction
- **Bag of Words Representation**:
  - Converts cleaned text to sparse vectors for model training.

### Model Training and Evaluation
- **Algorithms**:
  - Multinomial Naive Bayes
  - Logistic Regression
- **Performance Reports**:
  - Displays precision, recall, F1-Score, and accuracy for each model.

### Predictions
- Predicts binary sentiment for new user-provided text data.

---

## How to Run

### Prerequisites
- Python 3.7 or higher.
- Required Libraries:
  ```bash
  pip install pandas numpy scikit-learn nltk kagglehub
