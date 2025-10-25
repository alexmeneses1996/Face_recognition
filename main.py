from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import pandas as pd

app = FastAPI()

data = {
    "text": [
        "Win a free iPhone now",
        "Important meeting schedule",
        "Get cheap loans instantly",
        "Your invoice for last month",
        "Congratulations you won a prize",
        "Please review the attached report",
        "Earn money from home easily",
        "Your package has been shipped",
        "Free vacation offer just for you",
        "Let's catch up tomorrow at the office"
    ],
    "label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

class EmailInput(BaseModel):
    text: str

@app.post("/predict")
def predict_email(input: EmailInput):
    prediction = model.predict([input.text])[0]
    label = "spam" if prediction == 1 else "no_spam"
    return {"prediction": label}
