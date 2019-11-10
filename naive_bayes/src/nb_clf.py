import sys,os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score


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

	x_test = data
	y_test = labels_AG

	print(classification_report(y_test, clf.predict(x_test), digits=4))	
	return accuracy_score(y_test, clf.predict(x_test))

def testTR(file,clf):
	raw_data,labels_TR,labels_AG = load_data(file)

	data = [preprocess(tweet) for tweet in raw_data]

	x_test = data
	y_test = labels_TR

	print(classification_report(y_test, clf.predict(x_test), digits=4))
	return accuracy_score(y_test, clf.predict(x_test))

def make_plot(results_eda, results_tr, plot_title, yrange):
    x = range(1,5)
    plt.plot(x, results_eda)
    plt.plot(x, results_tr)
    plt.xticks(x, x)
    plt.xlabel("Size of Dataset")
    plt.ylabel("Accuracy")
    plt.ylim(yrange[0], yrange[1])
    plt.title(plot_title)
    plt.legend(['EDA', 'Machine Translation'])
    plt.show()

def main():
	print("=====================================================")
	print("Original Data")
	print()
	filename = '../../Dataset/train_en.tsv'
	raw_data,labels_TR,labels_AG = load_data(filename)

	data = [preprocess(tweet) for tweet in raw_data]

	clf_ag=classifier(data,labels_AG)
	clf_tr=classifier(data,labels_TR)

	test_file = '../../Dataset/dev_en.tsv'

	accuracy_ag = testAG(test_file,clf_ag)
	accuracy_tr = testTR(test_file,clf_tr)
	transl_ag = [accuracy_ag]
	transl_tr = [accuracy_tr]
	eda_ag = [accuracy_ag]
	eda_tr = [accuracy_tr]

	print("=====================================================")
	print("Data Augmentation using EDA technique")
	for degree in range(3):
		print("Dataset*{}".format(degree+2))
		filename = '../../Dataset/eda_aug/data_degree_{}.tsv'.format(degree+2)
		raw_data, labels_TR, labels_AG = load_data(filename)
		data = [preprocess(tweet) for tweet in raw_data]
		clf_ag = classifier(data, labels_AG)
		clf_tr = classifier(data, labels_TR)

		accuracy_ag = testAG(test_file, clf_ag)
		accuracy_tr = testTR(test_file, clf_tr)
		eda_ag.append(accuracy_ag)
		eda_tr.append(accuracy_tr)

	print("=====================================================")
	print("Data Augmentation using Machine Translation")
	for degree in range(3):
		print("Dataset*{}".format(degree+2))
		filename = '../../Dataset/transl_aug/data_degree_{}.tsv'.format(degree+2)
		raw_data, labels_TR, labels_AG = load_data(filename)
		data = [preprocess(tweet) for tweet in raw_data]
		clf_ag = classifier(data, labels_AG)
		clf_tr = classifier(data, labels_TR)

		accuracy_ag = testAG(test_file, clf_ag)
		accuracy_tr = testTR(test_file, clf_tr)
		transl_ag.append(accuracy_ag)
		transl_tr.append(accuracy_tr)

	make_plot(eda_ag, transl_ag, "Aggression results after data augmentation", [0.6, 0.65])
	make_plot(eda_tr, transl_tr, "Target results after data augmentation", [0.85, 0.9])


if __name__=='__main__':
	main()