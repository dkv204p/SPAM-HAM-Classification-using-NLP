import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

try:
    # Read the CSV file with UTF-8 encoding and comma delimiter
    data = pd.read_csv('spam.csv', encoding='utf-8', delimiter=',')

    # Check the column names in the DataFrame
    print("Column Names:", data.columns)

    # Assuming the dataset has 'v1' for labels (spam or ham) and 'v2' for text messages
    if 'v2' in data.columns:
        X = data['v2']
    else:
        raise KeyError("Column 'v2' not found in DataFrame")
    if 'v1' in data.columns:
        y = data['v1']
    else:
        raise KeyError("Column 'v1' not found in DataFrame")

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature Extraction using TF-IDF
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Model Building (using Naive Bayes)
    clf = MultinomialNB()
    clf.fit(X_train_tfidf, y_train)

    # Model Evaluation
    y_pred = clf.predict(X_test_tfidf)
    
    # Print the accuracy and classification report
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))  # Set zero_division=0 to suppress warnings

except Exception as e:
    print("Error:", e)
