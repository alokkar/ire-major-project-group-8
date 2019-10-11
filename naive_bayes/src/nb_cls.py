import sys,os
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV

stemmer = SnowballStemmer('english')
stem_map={}

stopW = stopwords.words('english')
emoji_pattern = re.compile("["
	 u"\U0001F600-\U0001F64F"  # emoticons
	 u"\U0001F300-\U0001F5FF"  # symbols & pictographs
	 u"\U0001F680-\U0001F6FF"  # transport & map symbols
	 u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
	 u"\U00002702-\U000027B0"
	 u"\U000024C2-\U0001F251"
	 "]+", flags=re.UNICODE)

def load_data(filename):
	n = ['id', 'text','HS','TR','AG']
	given_data = pd.read_csv(filename, sep='\t',error_bad_lines=False, names=n, usecols=['text','TR','AG'], skiprows=1)
	raw_data = given_data['text'].values
	labels_TR = list(map(int,given_data['TR'].values))
	labels_AG = list(map(int,given_data['AG'].values))

	return raw_data,labels_TR,labels_AG

def preprocess(tweet):
	# ' '.join([word for word in tweet.spilt() ])
	tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL', tweet)
	tweet = re.sub('@[^\s]+','USER', tweet)
	tweet = tweet.replace("ё", "е")
	tweet = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', tweet)
	tweet = re.sub(' +',' ', tweet)
	tweet = emoji_pattern.sub(r'', tweet)

	stemmed_text_token=[]
	tokens = tweet.split(' ')
	for token in tokens:
		if token=='':
			continue
		elif token=='USER' or token=='URL': 
			stemmed_text_token.append(token)
		# if token not in stopW:
		#Need to check performance with and without stopwords.
		else:
			a=stem_map.get(token,0)
			if a==0:
				a=stemmer.stem(token)
				stem_map[token]=a
			stemmed_text_token.append(a)
	return ' '.join(stemmed_text_token)

def classifier():
	text_clf = Pipeline([('vect', CountVectorizer()),
					 ('tfidf', TfidfTransformer()),
					 ('clf', MultinomialNB())])
	tuned_parameters = {
		'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
		'tfidf__use_idf': (True, False),
		'tfidf__norm': ('l1', 'l2'),
		'clf__alpha': [1, 1e-1, 1e-2]
	}

def main():

	filename = sys.argv[1]
	raw_data,labels_TR,labels_AG = load_data(filename)

	data = [preprocess(tweet) for tweet in raw_data]



	print(len(labels_AG))
	print(len(labels_TR))
	print(raw_data[103])
	print(data[103])

if __name__=='__main__':
	main()