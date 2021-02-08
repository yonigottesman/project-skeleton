from sentiment.inference import SentimentInference


def test_inference():
    inferencer = SentimentInference()
    sentiment_pos = inferencer.inference('happy happy fun nice tweet')
    assert sentiment_pos == 1

    sentiment_neg = inferencer.inference('bad sad')
    assert sentiment_neg == 0
    
def test_github():
    assert True == False
