import sys
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re


stemmer = SnowballStemmer('english')
stem_map={}
tweet_lenghts=[]

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

	return raw_data,labels_HS,labels_TR,labels_AG

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
	tweet_lenghts.append(len(tokens))
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


def main():
	filename = sys.argv[1]
	raw_data,labels_HS,labels_TR,labels_AG = load_data(filename)

	data = [preprocess(tweet) for tweet in raw_data]

	print("Max tweet length: ",max(tweet_lenghts))
	avg_len=sum(tweet_lenghts)/len(tweet_lenghts)
	print("Avg tweet length: ",avg_len)
	print("Min tweet length: ",min(tweet_lenghts))

	ltavg = [x for x in tweet_lenghts if x<=avg_len]
	print('Percent tweets with lenghts less than avg: ',(len(ltavg)*100)/len(tweet_lenghts))

	ltavg = [x for x in tweet_lenghts if x<=2*avg_len]
	print('Percent tweets with lenghts less than 2 times avg: ',(len(ltavg)*100)/len(tweet_lenghts))

	ltavg = [x for x in tweet_lenghts if x<=2.5*avg_len]
	print('Percent tweets with lenghts less than 2.5 times avg: ',(len(ltavg)*100)/len(tweet_lenghts))

	# classifier(data,labels_AG)
	hs = []
	nhs = []
	ind_tr = [] #individual target
	grp_tr = [] #group target
	agg_ag = [] #aggressive
	nag_ag = [] #Non aggressive
	for (text,hst,ltr,lag) in zip(data,labels_HS,labels_TR,labels_AG):
		if ltr==1:
			ind_tr.append(text)
		if ltr==0:
			grp_tr.append(text)
		if lag==1:
			agg_ag.append(text)
		if lag==0:
			nag_ag.append(text)
		if hst==1:
			hs.append(text)
		if hst==0:
			nhs.append(text)

	print('Number of hateful tweets: ',len(hs))
	print('Number of non-hateful tweets: ',len(nhs))
	print('Number of individual targeting tweets: ',len(ind_tr))
	print('Number of group targeting tweets: ',len(grp_tr))
	print('Number of Aggressive tweets: ',len(agg_ag))
	print('Number of Non-Aggressive tweets: ',len(nag_ag))

	print('Checking if all the aggressive tweets are hateful:')

	agg_hs = []
	nag_hs = []
	agg_nhs = []
	nag_nhs = []
	for (text,hst,ltr,lag) in zip(data,labels_HS,labels_TR,labels_AG):
		if lag==1 and hst==1:
			agg_hs.append(text)
		if lag==1 and hst==0:
			agg_nhs.append(text)
		if lag==0 and hst==1:
			nag_hs.append(text)
		if lag==0 and hst==0:
			nag_nhs.append(text)

	print('Num hateful and aggressive: ', len(agg_hs))
	print('Num not-hateful but aggressive: ', len(agg_nhs))
	print('Num hateful and not-aggressive: ', len(nag_hs))
	print('Num not-hateful and not-aggressive: ', len(nag_nhs))


	ind_hs=[]
	grp_hs=[]
	ind_nhs=[]
	grp_nhs=[]
	for (text,hst,ltr,lag) in zip(data,labels_HS,labels_TR,labels_AG):
		if ltr==1 and hst==1:
			ind_hs.append(text)
		if ltr==1 and hst==0:
			ind_nhs.append(text)
		if ltr==0 and hst==1:
			grp_hs.append(text)
		if ltr==0 and hst==0:
			grp_nhs.append(text)

	print('Num hateful and individual_targeting: ', len(ind_hs))
	print('Num not-hateful but individual_targeting: ', len(ind_nhs))
	print('Num hateful and Group_targeting: ', len(grp_hs))
	print('Num not-hateful and Group_targeting: ', len(grp_nhs))
	


if __name__=='__main__':
	main()