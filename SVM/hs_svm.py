import sys,os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn import metrics
from sklearn.feature_extraction.text import HashingVectorizer

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
	X_train,X_test,y_train,y_test = train_test_split(data, labels, test_size=0.2, random_state=1)
	# X_train,X_test,y_train,y_test = data[:3400],data[-383:],labels[:3400],labels[-383:]
	text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))),
					 ('tfidf', TfidfTransformer(use_idf=False,norm='l2')),
					 ('clf', SGDClassifier(loss='hinge', penalty='l2',
						   alpha=1e-3, random_state=42,
						   max_iter=5, tol=None))]) # After applying grid_search on the tunable parameters and using the best values

	

	###############Using hashing vectorizer#######################
	# maxLen = len(max(X_train, key=len))
	
	# text_clf = Pipeline([('vect', HashingVectorizer(n_features=2048)),
	# 				 ('tfidf', TfidfTransformer(use_idf=False,norm='l2')),
	# 				 ('clf', SGDClassifier(loss='hinge', penalty='l2',
	# 					   alpha=1e-3, random_state=42,
	# 					   max_iter=5, tol=None))])


	parameters = {
   'vect__ngram_range': [(1, 1), (1, 2),(2,2)],
		'tfidf__use_idf': (True, False),
		'tfidf__norm': ('l1', 'l2'),
		'clf__alpha': [10,1, 1e-1, 1e-2,1e-3]
	}
	# gs_clf = GridSearchCV(text_clf, parameters, cv=5, iid=False, n_jobs=-1)
	# gs_clf = gs_clf.fit(X_train, y_train)
	# print(gs_clf.best_score_)
	# for param_name in sorted(parameters.keys()):
	# 	print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
	text_clf.fit(X_train,y_train)
	prediction = text_clf.predict(X_test)
	accuracy_score = text_clf.score(X_test,y_test)
	# print("Accuracy : ",accuracy_score) #accuracy score
	# print("f1_score : ",f1_score(y_test, prediction, average="macro")) #f1_score
	# print("Precision : ",precision_score(y_test, prediction, average="macro")) #precision_score
	# print("Recall : ",recall_score(y_test, prediction,average="macro")) #recall_score
	# print("---------------Example---------------")
	# print("Statement : i will kill all the immigrants")
	# print(text_clf.predict(["i will kill all the immigrants"])) 
	# print("Statement : i will kiss all the immigrants")
	# print(text_clf.predict(["i will kiss all the immigrants"])) 
	return text_clf

def get_vars(filename):
	raw_data,labels_TR,labels_AG,labels_HS = load_data(filename)

	data = [preprocess(tweet) for tweet in raw_data]
	X = []
	y_AG = []
	y_TR = []
	for index,word in enumerate(labels_HS):
		if word:
			X.append(data[index])
			y_AG.append(labels_AG[index]) 
			y_TR.append(labels_TR[index])
	return X, y_AG, y_TR

def test(filename, clf_AG, clf_TR):
	X, y_AG, y_TR = get_vars(filename)

	prediction_AG = clf_AG.predict(X)
	prediction_TR = clf_TR.predict(X)
	f1_ag = f1_score(y_AG, prediction_AG, average="macro")
	f1_tr = f1_score(y_TR, prediction_TR, average="macro")
	acc_ag = accuracy_score(y_AG, prediction_AG)
	acc_tr = accuracy_score(y_TR, prediction_TR)
	print("-----------Aggressive Task------------")
	print("Accuracy : ",acc_ag) #accuracy score
	print("f1_score : ",f1_ag) #f1_score
	print("Precision : ",precision_score(y_AG, prediction_AG, average="macro")) #precision_score
	print("Recall : ",recall_score(y_AG, prediction_AG,average="macro")) #recall_score
	print("------------Target Task-------------")
	print("Accuracy : ",acc_tr) #accuracy score
	print("f1_score : ",f1_tr) #f1_score
	print("Precision : ",precision_score(y_TR, prediction_TR, average="macro")) #precision_score
	print("Recall : ",recall_score(y_TR, prediction_TR,average="macro")) #recall_score
	return acc_ag, acc_tr

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

	filename = '../Dataset/train_en.tsv'
	test_file = '../Dataset/dev_en.tsv'
	X, y_AG, y_TR = get_vars(filename)
	clf_AG = classifier(X, y_AG)
	clf_TR = classifier(X, y_TR)

	print("=====================================================")
	print("Original Dataset")
	accuracy_ag, accuracy_tr = test(test_file, clf_AG, clf_TR)
	transl_ag = [accuracy_ag]
	transl_tr = [accuracy_tr]
	eda_ag = [accuracy_ag]
	eda_tr = [accuracy_tr]

	print("=====================================================")
	print("Data Augmentation using EDA technique")
	for degree in range(3):
		print("-------------------------------------------------")
		print("Dataset of size * {}".format(degree+2))
		X, y_AG, y_TR = get_vars('../Dataset/eda_aug/data_degree_{}.tsv'.format(degree+2))
		clf_AG = classifier(X, y_AG)
		clf_TR = classifier(X, y_TR)
		accuracy_ag, accuracy_tr = test(test_file, clf_AG, clf_TR)
		eda_ag.append(accuracy_ag)
		eda_tr.append(accuracy_tr)

	print("=====================================================")
	print("Data Augmentation using Machine Translation")
	for degree in range(3):
		print("-------------------------------------------------")
		print("Dataset of size * {}".format(degree+2))
		X, y_AG, y_TR = get_vars('../Dataset/transl_aug/data_degree_{}.tsv'.format(degree+2))
		clf_AG = classifier(X, y_AG)
		clf_TR = classifier(X, y_TR)
		accuracy_ag, accuracy_tr = test(test_file, clf_AG, clf_TR)
		transl_ag.append(accuracy_ag)
		transl_tr.append(accuracy_tr)

	make_plot(eda_ag, transl_ag, "Aggression results after data augmentation", [0.6, 0.63])
	make_plot(eda_tr, transl_tr, "Target results after data augmentation", [0.88, 0.9])


if __name__=='__main__':
	main()