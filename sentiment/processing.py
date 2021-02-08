from typing import List

from nltk.tokenize import TweetTokenizer


def tokenize(tweet: str) -> List[str]:
    tokenizer = TweetTokenizer(preserve_case=False,
                               strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)
    return tweet_tokens
