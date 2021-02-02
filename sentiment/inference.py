from joblib import load
from sklearn.pipeline import Pipeline
from urllib.request import urlopen
from sentiment import MODEL_VERSION
from sentiment.processing import tokenize


class SentimentInference(object):
    def __init__(self):

        self.model_url = f'https://sentimentmodels.s3-us-west-2.amazonaws.com/model_{MODEL_VERSION}.pkl'

        self.pipeline: Pipeline = load(urlopen(self.model_url))

    def inference(self, tweet: str) -> int:
        return self.pipeline.predict([tweet])[0]
