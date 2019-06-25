# https://github.com/amkurian/twitter_sentiment_challenge/blob/master/sentiment_analyzer.py

import tweepy
from textblob import TextBlob


# We need to declare the variables to store the various keys associated with the Twitter API.
consumer_key = ''
consumer_key_secret = ''
access_token = ''
access_token_secret = ''

# The next step is to create a connection with the Twitter API using tweepy with these tokens.
auth = tweepy.OAuthHandler(consumer_key, consumer_key_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# We can now search Twitter for any topic using the search method of the API.
public_tweets = api.search('FIFA')

# We can iterate the publice_tweets array, and check the sentiment of the text of each tweet based on the polarity.
for tweet in public_tweets:
	print(tweet.text)
	analysis = TextBlob(tweet.text)
	print(analysis.sentiment)
	if analysis.sentiment[0]>0:
		print ('Positive')
	else:
		print ('Negative')
print("")