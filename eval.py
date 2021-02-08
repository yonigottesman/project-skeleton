import argparse

import numpy as np

from sentiment.inference import SentimentInference
from train import get_datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='.cache', type=str)
    args = parser.parse_args()

    trainset, testset = get_datasets(args.data_path)
    inferencer = SentimentInference()

    predicted = inferencer.inference_batch(testset['X'])
    accuracy = np.mean(predicted == testset['y'])
    print(accuracy)
