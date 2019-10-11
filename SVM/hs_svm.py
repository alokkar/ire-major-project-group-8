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
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score

stemmer = SnowballStemmer('english')
stem_map={}

stopW = stopwords.words('english')
emoji_pattern = re.compile("["
	 u"\U0001F600-\U0001F64F"  
	 u"\U0001F300-\U0001F5FF"  
	 u"\U0001F680-\U0001F6FF"  
	 u"\U0001F1E0-\U0001F1FF"  
	 u"\U00002702-\U000027B0"
	 u"\U000024C2-\U0001F251"
	 "]+", flags=re.UNICODE)

def load_data(filename):
	n = ['id', 'text','HS','TR','AG']
	given_data = pd.read_csv(filename, sep='\t',error_bad_lines=False, names=n, usecols=['text','HS','TR','AG'], skiprows=1)
	raw_data = given_data['text'].values
	labels_TR = list(map(int,given_data['TR'].values))
	labels_AG = list(map(int,given_data['AG'].values))
	labels_HS = list(map(int,given_data['HS'].values))
	return raw_data,labels_TR,labels_AG,labels_HS

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
	X_train,X_test,y_train,y_test = data[:3400],data[:383],labels[:3400],labels[:383]
	text_clf = Pipeline([('vect', CountVectorizer()),
					 ('tfidf', TfidfTransformer()),
					 ('clf', MultinomialNB())])

	tuned_parameters = {
		'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
		'tfidf__use_idf': (True, False),
		'tfidf__norm': ('l1', 'l2'),
		'clf__alpha': [1, 1e-1, 1e-2]
	}
	text_clf.fit(X_train,y_train)
	prediction = text_clf.predict(X_test)
	accuracy_score = text_clf.score(X_test,y_test)
	print("Accuracy : ",accuracy_score) #accuracy score
	print("f1_score : ",f1_score(y_test, prediction, average="macro")) #f1_score
	print("Precision : ",precision_score(y_test, prediction, average="macro")) #precision_score
	print("Recall : ",recall_score(y_test, prediction, average="macro")) #recall_score
	print("---------------Example---------------")
	print("Statement : i will kill all the immigrants")
	print(text_clf.predict(["i will kill all the immigrants"])) 
	print("Statement : i will kiss all the immigrants")
	print(text_clf.predict(["i will kiss all the immigrants"])) 


def main():

	filename = './train_en.tsv'
	raw_data,labels_TR,labels_AG,labels_HS = load_data(filename)

	data = [preprocess(tweet) for tweet in raw_data]
	X = []
	y_AG = []
	y_TR = []
	count = 0
	for index,word in enumerate(labels_HS):
		if word:
			X.append(data[index])
			y_AG.append(labels_AG[index]) 
			y_TR.append(labels_TR[index])
			count += 1

	# print(count)

	classifier(X,y_AG)

	# print(labels_AG[:100])

if __name__=='__main__':
	main()