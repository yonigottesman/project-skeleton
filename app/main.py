from fastapi import FastAPI

from sentiment import MODEL_VERSION
from sentiment.inference import SentimentInference

app = FastAPI()

inferencer = SentimentInference()


@app.get("/predict")
def predict(tweet: str):
    prob = inferencer.inference(tweet)
    return {
        'version': MODEL_VERSION,
        'tweet': tweet,
        'sentiment': 'positive' if prob == 1 else 'negative'
    }