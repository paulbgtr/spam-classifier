from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

with open("trained_models/model.pkl", "rb") as f:
    model = pickle.load(f)
with open("trained_models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(request: TextRequest):
    text_vectorized = vectorizer.transform([request.text])

    prediction = model.predict(text_vectorized)[0]
    probabilities = model.predict_proba(text_vectorized)[0]

    return {
        "text": request.text,
        "is_spam": bool(prediction),
        "spam_probability": float(probabilities[1]),
        "ham_probability": float(probabilities[0])
    }
