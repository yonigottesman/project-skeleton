import argparse
from typing import Tuple

import nltk
import numpy as np
import pandas as pd
from joblib import dump
from nltk.corpus import twitter_samples
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sentiment import MODEL_VERSION
from sentiment.processing import tokenize
from sentiment import MODEL_VERSION


def download_dataset(path: str) -> None:
    nltk.download('twitter_samples', path)
    nltk.data.path.append(path)


def get_datasets(download_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    download_dataset(download_path)

    pos = pd.DataFrame({
        'X': twitter_samples.strings('positive_tweets.json'),
        'y': 1
    })
    neg = pd.DataFrame({
        'X': twitter_samples.strings('negative_tweets.json'),
        'y': 0
    })

    dataset = pd.concat([pos, neg]).sample(frac=1, random_state=42)
    trainset = dataset[:8000]
    testset = dataset[8000:]

    return trainset, testset


def eval_accuracy(pipeline: Pipeline, dataset: pd.DataFrame) -> float:
    predicted = pipeline.predict(dataset['X'])
    accuracy = np.mean(predicted == dataset['y'])
    return accuracy


def train(trainset: pd.DataFrame) -> Pipeline:
    vectorizer = TfidfVectorizer(tokenizer=tokenize,
                                 ngram_range=(1, 1),
                                 min_df=1,
                                 max_features=2500)

    classifier = LogisticRegression()

    pipeline = Pipeline([('vectorizer', vectorizer),
                         ('classifier', classifier)])

    pipeline.fit(trainset['X'], trainset['y'])
    return pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='.cache', type=str)
    parser.add_argument('--model_path',
                        default=f'model_{MODEL_VERSION}.pkl',
                        type=str)
    args = parser.parse_args()

    trainset, testset = get_datasets(args.data_path)

    pipeline = train(trainset)

    dump(pipeline, args.model_path)
    train_accuracy = eval_accuracy(pipeline, trainset)
    test_accuracy = eval_accuracy(pipeline, testset)
    print(f'train accuracy: {train_accuracy:.3f}. '
          f'test accuracy: {test_accuracy:.3f}')
