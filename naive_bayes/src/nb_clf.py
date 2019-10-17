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
from sklearn.metrics import classification_report


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
	given_data = pd.read_csv(filename, sep='\t',error_bad_lines=False, names=n, usecols=['text','HS','TR','AG'], skiprows=1)
	raw_data = given_data['text'].values
	labels_HS = list(map(int,given_data['HS'].values))
	labels_TR = list(map(int,given_data['TR'].values))
	labels_AG = list(map(int,given_data['AG'].values))

	data=[]
	y_tr=[]
	y_ag=[]

	for i,val in enumerate(labels_HS):
		if val:
			data.append(raw_data[i])
			y_tr.append(labels_TR[i])
			y_ag.append(labels_AG[i])

	return data,y_tr,y_ag

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

def classifier(data,labels):
	text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,2))),
					 ('tfidf', TfidfTransformer(use_idf=False,norm='l2')),
					 ('clf', MultinomialNB(alpha=0.1))])

	x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.01, random_state=324)

	print("Training Start")
	text_clf.fit(x_train,y_train)
	print("Training Complete")

	return text_clf

def testAG(file,clf):
	raw_data,labels_TR,labels_AG = load_data(file)

	data = [preprocess(tweet) for tweet in raw_data]

	x_train, x_test, y_train, y_test = train_test_split(data, labels_AG, test_size=0.99, random_state=953)

	print(classification_report(y_test, clf.predict(x_test), digits=4))	

def testTR(file,clf):
	raw_data,labels_TR,labels_AG = load_data(file)

	data = [preprocess(tweet) for tweet in raw_data]

	x_train, x_test, y_train, y_test = train_test_split(data, labels_TR, test_size=0.99, random_state=953)

	print(classification_report(y_test, clf.predict(x_test), digits=4))	


def main():

	filename = sys.argv[1]
	raw_data,labels_TR,labels_AG = load_data(filename)

	data = [preprocess(tweet) for tweet in raw_data]

	clf_ag=classifier(data,labels_AG)
	clf_tr=classifier(data,labels_TR)

	test_file = 'dev_en.tsv'

	testAG(test_file,clf_ag)
	testTR(test_file,clf_tr)


if __name__=='__main__':
	main()