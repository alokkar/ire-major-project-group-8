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

def classifier(data,labels):
	text_clf = Pipeline([('vect', CountVectorizer()),
					 ('tfidf', TfidfTransformer()),
					 ('clf', MultinomialNB())])
	tuned_parameters = {
		'vect__ngram_range': [(1, 1), (1, 2), (2, 2), (1, 3), (2, 3), (3, 3), (1, 4), (2, 4), (4, 4), (4, 4)],
		'tfidf__use_idf': (True, False),
		'tfidf__norm': ('l1', 'l2'),
		'clf__alpha': [1, 1e-1, 1e-2, 1e-3]
	}

	x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.01, random_state=324)

	score='f1_macro'
	clf = GridSearchCV(text_clf, tuned_parameters,cv=10,scoring=score)
	print("Training Start")
	clf.fit(x_train, y_train)
	print("Training Complete")

	return clf

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

	# classifier(data,labels_AG)

	ind_tr = [] #individual target
	grp_tr = [] #group target
	agg_ag = [] #aggressive
	nag_ag = [] #Non aggressive
	for (text,ltr,lag) in zip(data,labels_TR,labels_AG):
		if ltr==1:
			ind_tr.append(text)
		if ltr==0:
			grp_tr.append(text)
		if lag==1:
			agg_ag.append(text)
		if lag==0:
			nag_ag.append(text)

	size_tr = min(len(ind_tr),len(grp_tr))
	size_ag = min(len(agg_ag),len(nag_ag))

	f_data_tr = np.concatenate((ind_tr[:size_tr],grp_tr[:size_tr]),axis=0)
	f_data_ag = np.concatenate((agg_ag[:size_ag],nag_ag[:size_ag]),axis=0)

	label_tr = [1]*size_tr +[0]*size_tr
	label_ag = [1]*size_ag +[0]*size_ag

	clf_t1 = classifier(f_data_ag,label_ag)
	clf_t2 = classifier(f_data_tr,label_tr)

	test_file = 'dev_en.tsv'

	testAG(test_file,clf_t1)
	testTR(test_file,clf_t2)

if __name__=='__main__':
	main()