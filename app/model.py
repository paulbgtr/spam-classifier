from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle
import os
from sklearn.metrics import classification_report

os.makedirs('trained_models', exist_ok=True)

def prepare_data_and_train(df):
    df["v1"] = df["v1"].map({"ham": 0, "spam": 1})

    X = df["v2"]
    y = df["v1"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_vectorized, y_train)

    y_pred = model.predict(X_test_vectorized)
    print(classification_report(y_test, y_pred))

    with open("trained_models/model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    with open("trained_models/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    return vectorizer, model

if __name__ == "__main__":
    df = pd.read_csv("../data/spam.csv", encoding="ISO-8859-1")
    vectorizer, model = prepare_data_and_train(df)