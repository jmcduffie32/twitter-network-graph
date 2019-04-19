import itertools
import collections
import matplotlib.pyplot as plt
import seaborn as sns
import tweepy as tw 
import pandas as pd
import nltk
from nltk import bigrams
from nltk.corpus import stopwords
import re
import networkx as nx

import warnings
warnings.filterwarnings('ignore')

sns.set(font_scale=1.5)
sns.set_style('whitegrid')

CONSUMER_KEY = ''
CONSUMER_SECRET = ''
ACCESS_TOKEN = ''
ACCESS_TOKEN_SECRET = ''

def remove_url(txt):
  return " ".join(re.sub(r"([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", txt).split())
nltk.download('stopwords')
auth = tw.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tw.API(auth, wait_on_rate_limit=True)

search_term = '#climate+change -filter:retweets'
date_since = '2019-04-01'
filter_words = set(stopwords.words('english'))
collection_words = ['climatechange', 'climate', 'change']
for word in collection_words:
  filter_words.add(word)

result = tw.Cursor(
    api.search,
    q=search_term,
    lang='en',
    since=date_since
).items(1000)

tweets = [remove_url(tweet.text) for tweet in result]
words_in_tweet = [tweet.lower().split() for tweet in tweets]
words_in_tweet = [[word for word in tweet_words if not word in filter_words]
                  for tweet_words in words_in_tweet]
terms_bigram = [list(bigrams(tweet)) for tweet in words_in_tweet]

bigrams = list(itertools.chain(*terms_bigram))
bigram_counts = collections.Counter(bigrams)

bigrams_df = pd.DataFrame(bigram_counts.most_common(20),
                          columns=['bigram', 'count'])

d = bigrams_df.set_index('bigram').T.to_dict('records')

G = nx.Graph()

for k,v in d[0].items():
  G.add_edge(k[0], k[1], weight=(v * 10))

fix, ax = plt.subplots(figsize=(10, 8))

pos = nx.spring_layout(G, k=1)

nx.draw_networkx(G,
                 pos,
                 font_size=16,
                 width=3,
                 edge_color='grey',
                 node_color='purple',
                 with_labels=False,
                 ax=ax)

for key, value in pos.items():
  x, y = value[0] + 0.135, value[1] + 0.045
  ax.text(x,
          y,
          s=key,
          bbox=dict(facecolor='red', alpha=0.25),
          horizontalalignment='center',
          fontsize=13)

plt.show()

# words_in_tweet = list(itertools.chain(*words_in_tweet))

# word_counts = collections.Counter(words_in_tweet)
# most_common = word_counts.most_common(15)

# clean_tweets_no_urls = pd.DataFrame(most_common,
#                                     columns=['words', 'count'])

# fix, ax = plt.subplots(figsize=(8, 8))
# clean_tweets_no_urls.sort_values(by='count').plot.barh(x='words',
#                                                        y='count',
#                                                        ax=ax,
#                                                        color='purple')

# ax.set_title("Common Words Found in Tweets (All Words)")

# plt.show()

