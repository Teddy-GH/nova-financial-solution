from textblob import TextBlob

# Function to compute sentiment polarity
def get_sentiment(text):
    return TextBlob(str(text)).sentiment.polarity

news_df['sentiment_score'] = news_df['headline'].apply(get_sentiment)