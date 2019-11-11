#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import string
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, PorterStemmer
from nltk.tokenize import TweetTokenizer
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, Embedding, GRU, Input, Bidirectional
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.utils import np_utils


# In[4]:


# functions for cleaning
def removeStopwords(tokens):
    stops = set(stopwords.words("english"))
    stops.update(['.',',','"',"'",'?',':',';','(',')','[',']','{','}'])
    toks = [tok for tok in tokens if not tok in stops and len(tok) >= 3]
    return toks

def removeURL(text):
    newText = re.sub('http\\S+', '', text, flags=re.MULTILINE)
    return newText

def removeNum(text):
    newText = re.sub('\\d+', '', text)
    return newText

def removeHashtags(tokens):
    toks = [ tok for tok in tokens if tok[0] != '#']

    return toks

def stemTweet(tokens):
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in tokens]
    return stemmed_words


# In[5]:


def processTweet(tweet, remove_swords = True, remove_url = True, remove_hashtags = True, remove_num = True, stem_tweet = True):
#     text = tweet.translate(string.punctuation)   -> to figure out what it does ?
    """
        Tokenize the tweet text using TweetTokenizer.
        set strip_handles = True to Twitter username handles.
        set reduce_len = True to replace repeated character sequences of length 3 or greater with sequences of length 3.
    """
    if remove_url:
        tweet = removeURL(tweet)
    twtk = TweetTokenizer(strip_handles=True, reduce_len=True)
    tokens = [w.lower() for w in twtk.tokenize(tweet) if w != "" and w is not None]
    if remove_hashtags:
        tokens = removeHashtags(tokens)
    if remove_swords:
        tokens = removeStopwords(tokens)
    if stem_tweet:
        tokens = stemTweet(tokens)
    text = " ".join(tokens)
    return text


# In[39]:


train_data = pd.read_csv('./train_en.tsv',delimiter='\t',encoding='utf-8')
train_data = train_data.loc[train_data['HS'] == 1]
test_data = pd.read_csv('./dev_en.tsv',delimiter='\t',encoding='utf-8')
test_data = test_data.loc[test_data['HS'] == 1]
cleaned = train_data['text'][:5].map(lambda x: processTweet(x))
# for i,t in enumerate(cleaned):
#     print(i,train_data['text'][i])
#     print(t)


# In[40]:


# tweets = train_data['text']
maxlen = 50
train_data['text'] = train_data['text'].map(lambda x: processTweet(x))
test_data['text'] = test_data['text'].map(lambda x: processTweet(x))

vocabulary_size = 30000
tokenizer = Tokenizer(num_words= vocabulary_size)
tokenizer.fit_on_texts(train_data['text'])
X_train = tokenizer.texts_to_sequences(train_data['text'])
X_test = tokenizer.texts_to_sequences(test_data['text'])

# print(sequences)
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

labels = train_data['AG']
Y_test = test_data['AG']
Y_train = np_utils.to_categorical(labels, len(set(labels)))
V = len(tokenizer.word_index) + 1



embeddings_index = dict()
f = open('../classification/glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
# print('Loaded %s word vectors.' % len(embeddings_index))

# print(tokenizer.word_index.items())

# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocabulary_size, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector


# '''


# In[42]:


model_glove = Sequential()
model_glove.add(Embedding(vocabulary_size, 100, input_length=50, weights=[embedding_matrix], trainable=True))
model_glove.add(Dropout(0.2))
model_glove.add(Conv1D(64, 5, activation='relu'))
model_glove.add(MaxPooling1D(pool_size=4))
model_glove.add(LSTM(200))
model_glove.add(Dense(1, activation='sigmoid'))
model_glove.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_glove.summary()


# In[43]:


model_glove.fit(X_train, labels, validation_split=0.1, epochs = 5)


# In[44]:


y_pred = (model_glove.predict(X_test) > 0.5).astype(int)


# In[45]:


from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score

print("F1.........: %f" %(f1_score(Y_test, y_pred, average="macro")))
print("Precision..: %f" %(precision_score(Y_test, y_pred, average="macro")))
print("Recall.....: %f" %(recall_score(Y_test, y_pred, average="macro")))
print("Accuracy...: %f" %(accuracy_score(Y_test, y_pred)))


# In[ ]:




