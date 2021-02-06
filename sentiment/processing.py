from typing import List

from nltk.tokenize import TweetTokenizer


def tokenize(tweet: str) -> List[str]:
    return tweet.split()
