from fastapi import FastAPI

from sentiment.inference import SentimentInference

app = FastAPI()

inferencer = SentimentInference()


@app.get("/predict")
def read_item(tweet: str):
    prob = inferencer.inference(tweet)
    return {
        'tweet': tweet,
        'sentiment': 'positive' if prob == 1 else 'negative'
    }
