import sys
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
import preprocessor as p

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm,colors
from scipy import linspace, meshgrid, arange, empty, concatenate, newaxis, shape


from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.linear_model import SGDClassifier,LogisticRegression
from sklearn.feature_extraction.text import HashingVectorizer



stemmer = SnowballStemmer('english')
stem_map={}
tweet_lengths=[]

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

	return data,labels_HS,y_tr,y_ag

def preprocess(tweet,em,m):
	# ' '.join([word for word in tweet.spilt() ])
	tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL', tweet)
	if m:
		tweet = re.sub('@[^\s]+','USER', tweet)
	else:
		tweet = re.sub('@[^\s]+','', tweet)
	if not em:
		tweet = emoji_pattern.sub('EMOJI', tweet)
		tweet = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', tweet)
	tweet = re.sub(' +',' ', tweet)

	stemmed_text_token=[]
	tokens = tweet.split(' ')
	tweet_lengths.append(len(tokens))
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

class TweetLengthExtractor(BaseEstimator, TransformerMixin):
	"""Takes in dataframe, extracts road name column, outputs average word length"""

	def __init__(self):
		pass

	def tweet_length(self, tweet):
		"""Helper code to compute average word length of a name"""
		return len([word for word in tweet.split(' ')])

	def transform(self, df, y=None):
		"""The workhorse of this feature extractor"""
		# print(df)
		length=len(df)
		return np.reshape(np.array([self.tweet_length(tweet) for tweet in df]),(length,1))

	def fit(self, df, y=None):
		"""Returns `self` unless something different happens in train and test"""
		return self

def svm_classifier(data,labels):
	X_train,X_test,y_train,y_test = data[:3400],data[-383:],labels[:3400],labels[-383:]
	text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))),
					 ('tfidf', TfidfTransformer(use_idf=False,norm='l2')),
					 ('clf', SGDClassifier(loss='hinge', penalty='l2',
						   alpha=1e-3, random_state=42,
						   max_iter=5, tol=None))]) # After applying grid_search on the tunable parameters and using the best values

	text_clf.fit(X_train,y_train)
	return text_clf

def svm_classifier_withLen(data,labels):
	X_train,X_test,y_train,y_test = data[:3400],data[-383:],labels[:3400],labels[-383:]
	tf_idf_pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1,2))),
					 ('tfidf', TfidfTransformer(use_idf=False,norm='l2'))
					 ])
	text_clf = Pipeline([
					('feats', FeatureUnion([
						('tf_idf_pip', tf_idf_pipeline), # can pass in either a pipeline
						('ave', TweetLengthExtractor()) # or a transformer
					])),
					 ('clf', SGDClassifier(loss='hinge', penalty='l2',
						   alpha=1e-3, random_state=42,
						   max_iter=5, tol=None))]) # After applying grid_search on the tunable parameters and using the best values

	text_clf.fit(X_train,y_train)
	return text_clf

def nb_classifier(data,labels):
	text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,3))),
					 ('tfidf', TfidfTransformer(use_idf=False,norm='l2')),
					 ('clf', MultinomialNB(alpha=0.01))])

	x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.01, random_state=324)

	print("Training Start")
	text_clf.fit(x_train,y_train)
	print("Training Complete")

	return text_clf


def nb_classifier_withLen(data,labels):
	tf_idf_pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1,3))),
					 ('tfidf', TfidfTransformer(use_idf=False,norm='l2'))
					 ])
	text_clf = Pipeline([
					('feats', FeatureUnion([
						('tf_idf_pip', tf_idf_pipeline), # can pass in either a pipeline
						('ave', TweetLengthExtractor()) # or a transformer
					])),
					 ('clf', MultinomialNB(alpha=0.01))])

	x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.01, random_state=324)

	print("Training Start")
	text_clf.fit(x_train,y_train)
	print("Training Complete")

	return text_clf


def logreg_classifier_withLen(data,labels):
	X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.01, random_state=324)
	tf_idf_pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1,2))),
					 ('tfidf', TfidfTransformer(use_idf=False,norm='l2'))
					 ])
	clf = Pipeline([
					('feats', FeatureUnion([
						('tf_idf_pip', tf_idf_pipeline), # can pass in either a pipeline
						('ave', TweetLengthExtractor()) # or a transformer
					])),
					 ('clf', LogisticRegression(solver = 'newton-cg', penalty = 'l2'))])


	clf.fit(X_train,y_train)
	return clf

def logreg_classifier(data, labels):
	X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.01, random_state=324)
	clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))),
					 ('tfidf', TfidfTransformer(use_idf=False,norm='l2')),
					 ('clf', LogisticRegression(solver = 'newton-cg', penalty = 'l2'))])


	clf.fit(X_train,y_train)
	return clf




def testAG(file,clf,em,m):
	raw_data,lh,labels_TR,labels_AG = load_data(file)

	data = [preprocess(tweet,em,m) for tweet in raw_data]

	x_train, x_test, y_train, y_test = train_test_split(data, labels_AG, test_size=0.99, random_state=953)

	print(classification_report(y_test, clf.predict(x_test), digits=4))	

def testTR(file,clf,em,m):
	raw_data,lh,labels_TR,labels_AG = load_data(file)

	data = [preprocess(tweet,em,m) for tweet in raw_data]

	x_train, x_test, y_train, y_test = train_test_split(data, labels_TR, test_size=0.99, random_state=953)

	print(classification_report(y_test, clf.predict(x_test), digits=4))	

# def error_analysis(file,clf,em,m):
# 	raw_data,lh,labels_TR,labels_AG = load_data(file)

# 	data = [preprocess(tweet,em,m) for tweet in raw_data]

# 	x_train, x_test, y_train, y_test = train_test_split(data, labels_TR, test_size=0.99, random_state=953)

# 	y_pred=clf.predict(x_test)

# 	TP = []
#     FP = []
#     TN = []
#     FN = []

#     for i in range(len(y_pred)): 
#         if y_test[i]==y_pred[i]==1:
#            TP.append(x_test[i])
#         if y_pred[i]==1 and y_test[i]!=y_pred[i]:
#            FP.append(x_test[i])
#         if y_test[i]==y_pred[i]==0:
#            TN.append(x_test[i])
#         if y_pred[i]==0 and y_test[i]!=y_pred[i]:
#            FN.append(x_test[i])
#     print(TP)
#     print(FP)
#     print(TN)
#     print(FN)

#     return

def lt_length_analysis(data,labels,clf,length):
	print("Percent correct predictions for all tweets shorter than this word length ",length)
	cor=0
	inc=0
	tot=len(data)
	tot_len=0
	for i,vec in enumerate(data):
		len_vec = len(vec.split(' '))
		if len_vec<=length:
			tot_len+=1
			if labels[i]==clf.predict(data[i:i+1]):
				cor+=1
			else:
				inc+=1
	print("Total Tweets",tot)
	print("Number of tweets less than length ",length," is: ", tot_len)
	if tot_len==0:
		print("Cannot calculate accuracy")
		return
	print("Number of tweets less than length ",length," which were correctly predicted ",cor)
	print("Number of tweets less than length ",length," which were correctly predicted ",inc)
	print("Accuracy for tweets less than given length: ", cor*100/tot_len)


def gt_length_analysis(data,labels,clf,length):
	print("Percent correct predictions for all tweets longer than this word length ",length)
	cor=0
	inc=0
	tot=len(data)
	tot_len=0
	for i,vec in enumerate(data):
		len_vec = len(vec.split(' '))
		if len_vec>=length:
			tot_len+=1
			if labels[i]==clf.predict(data[i:i+1]):
				cor+=1
			else:
				inc+=1
	print("Total Tweets",tot)
	print("Number of tweets more than length ",length," is: ", tot_len)
	if tot_len==0:
		print("Cannot calculate accuracy")
		return
	print("Number of tweets more than length ",length," which were correctly predicted ",cor)
	print("Number of tweets more than length ",length," which were incorrectly predicted ",inc)
	print("Accuracy for tweets more than given length: ", cor*100/tot_len)

def eq_length_analysis(data,labels,clf,length):
	cor=0
	inc=0
	tot=len(data)
	tot_len=0
	for i,vec in enumerate(data):
		len_vec = len(vec.split(' '))
		if len_vec==length:
			tot_len+=1
			if labels[i]==clf.predict(data[i:i+1]):
				cor+=1
			else:
				inc+=1
	if tot_len==0:
		return 0
	return cor*100/tot_len


def plot_acc_clf(data,label,clf,min_len,max_len,clf_name):
	X=np.linspace(min_len,max_len,max_len-min_len+1)
	Y=[]
	for x in X:
		y=eq_length_analysis(data,label,clf,x)
		Y.append(y)
	Y=np.array(Y)
	return X,Y
	# fig = plt.figure()
	# ax1=fig.add_subplot(1,1,1)
	# ax1.plot(X,Y)
	# ax1.set(xlabel='Length (Numbeer of words)', ylabel='Accuracy(%)',
 #       title='Relation between length and accuracy')
	# fig.savefig("len_acc_corr_"+clf_name+".png")
	# plt.show()

# def plot_all(data,labels,clf_list,min_len,max_len):


def main():
	filename = sys.argv[1]
	raw_data,labels_HS,labels_TR,labels_AG = load_data(filename)

	data_em_m = [preprocess(tweet,True,True) for tweet in raw_data]
	data_em_nom = [preprocess(tweet,True,False) for tweet in raw_data]
	data_noem_m = [preprocess(tweet,False,True) for tweet in raw_data]
	data_noem_nom = [preprocess(tweet,False,False) for tweet in raw_data]

	# print(data_em_m)
	# print(data_em_nom[:20])
	# return

	max_len=max(tweet_lengths)
	print("Max tweet length: ",max(tweet_lengths))
	avg_len=sum(tweet_lengths)/len(tweet_lengths)
	print("Avg tweet length: ",avg_len)
	min_len=min(tweet_lengths)
	print("Min tweet length: ",min(tweet_lengths))

	nb_clf_ag_em_m=nb_classifier(data_em_m,labels_AG)
	nb_clf_tr_em_m=nb_classifier(data_em_m,labels_TR)
	svm_clf_ag_em_m = svm_classifier(data_em_m,labels_AG)
	svm_clf_tr_em_m = svm_classifier(data_em_m,labels_TR)
	lr_clf_ag_em_m=logreg_classifier(data_em_m,labels_AG)
	lr_clf_tr_em_m=logreg_classifier(data_em_m,labels_TR)


	nb_clf_ag_em_nom=nb_classifier(data_em_nom,labels_AG)
	nb_clf_tr_em_nom=nb_classifier(data_em_nom,labels_TR)
	svm_clf_ag_em_nom = svm_classifier(data_em_nom,labels_AG)
	svm_clf_tr_em_nom = svm_classifier(data_em_nom,labels_TR)
	lr_clf_ag_em_nom=logreg_classifier(data_em_nom,labels_AG)
	lr_clf_tr_em_nom=logreg_classifier(data_em_nom,labels_TR)

	nb_clf_ag_noem_m=nb_classifier(data_noem_m,labels_AG)
	nb_clf_tr_noem_m=nb_classifier(data_noem_m,labels_TR)
	svm_clf_ag_noem_m = svm_classifier(data_noem_m,labels_AG)
	svm_clf_tr_noem_m = svm_classifier(data_noem_m,labels_TR)
	lr_clf_ag_noem_m=logreg_classifier(data_noem_m,labels_AG)
	lr_clf_tr_noem_m=logreg_classifier(data_noem_m,labels_TR)

	nb_clf_ag_noem_nom=nb_classifier(data_noem_nom,labels_AG)
	nb_clf_tr_noem_nom=nb_classifier(data_noem_nom,labels_TR)
	svm_clf_ag_noem_nom = svm_classifier(data_noem_nom,labels_AG)
	svm_clf_tr_noem_nom = svm_classifier(data_noem_nom,labels_TR)
	lr_clf_ag_noem_nom=logreg_classifier(data_noem_nom,labels_AG)
	lr_clf_tr_noem_nom=logreg_classifier(data_noem_nom,labels_TR)


	
	file = 'dev_en.tsv'
	test_data,labels_HS_test,labels_TR_test,labels_AG_test = load_data(file)

	print("NB with emojis with mentions")
	testAG(file,nb_clf_ag_em_m,True,True)
	testTR(file,nb_clf_tr_em_m,True,True)
	print("SVM with emojis with mentions")
	testAG(file,svm_clf_ag_em_m,True,True)
	testTR(file,svm_clf_tr_em_m,True,True)
	print("LR with emojis with mentions")
	testAG(file,lr_clf_ag_em_m,True,True)
	testTR(file,lr_clf_tr_em_m,True,True)

	print("NB with emojis without mentions")
	testAG(file,nb_clf_ag_em_nom,True,False)
	testTR(file,nb_clf_tr_em_nom,True,False)
	print("SVM with emojis without mentions")
	testAG(file,svm_clf_ag_em_nom,True,False)
	testTR(file,svm_clf_tr_em_nom,True,False)
	print("LR with emojis without mentions")
	testAG(file,lr_clf_ag_em_nom,True,False)
	testTR(file,lr_clf_tr_em_nom,True,False)

	print("NB without emojis with mentions")
	testAG(file,nb_clf_ag_noem_m,False,True)
	testTR(file,nb_clf_tr_noem_m,False,True)
	print("SVM without emojis with mentions")
	testAG(file,svm_clf_ag_noem_m,False,True)
	testTR(file,svm_clf_tr_noem_m,False,True)
	print("LR without emojis with mentions")
	testAG(file,lr_clf_ag_noem_m,False,True)
	testTR(file,lr_clf_tr_noem_m,False,True)

	print("NB without emojis without mentions")
	testAG(file,nb_clf_ag_noem_nom,False,False)
	testTR(file,nb_clf_tr_noem_nom,False,False)
	print("SVM without emojis without mentions")
	testAG(file,svm_clf_ag_noem_nom,False,False)
	testTR(file,svm_clf_tr_noem_nom,False,False)
	print("LR without emojis without mentions")
	testAG(file,lr_clf_ag_noem_nom,False,False)
	testTR(file,lr_clf_tr_noem_nom,False,False)

if __name__=='__main__':
	main()