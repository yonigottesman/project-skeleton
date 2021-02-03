import urllib
from pathlib import Path
from typing import List

from joblib import load
from sklearn.pipeline import Pipeline

from sentiment import MODEL_VERSION
from sentiment.processing import tokenize


class SentimentInference(object):
    def __init__(self):
        url = 'https://sentimentmodels.s3-us-west-2.amazonaws.com'
        model_name = f'model_{MODEL_VERSION}.pkl'

        model_path = Path(model_name)

        if not model_path.exists():
            # better to get cache path in constructor
            urllib.request.urlretrieve(f'{url}/{model_name}', model_name)

        self.pipeline: Pipeline = load(model_path)

    def inference(self, tweet: str) -> int:
        return self.pipeline.predict([tweet])[0]

    def inference_batch(self, tweets: List[str]):
        return self.pipeline.predict(tweets)