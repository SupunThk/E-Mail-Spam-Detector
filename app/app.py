from fastapi import FastAPI
import joblib

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR.parent / "Model" / "spam_model.pkl")
tfidf = joblib.load(BASE_DIR.parent / "Model" / "tfidf_vectorizer.pkl")

app = FastAPI()


@app.get("/predict")
def predict(text: str):

    text_tfidf = tfidf.transform([text])

    prediction = model.predict(text_tfidf)[0]

    return {"predict": int(prediction)}