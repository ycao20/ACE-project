# -*- coding: utf-8 -*-

#import required libraries
import twint
import pandas as pd
import numpy as np
import os
from os import listdir
import nest_asyncio
nest_asyncio.apply()
import pickle
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import wordnet 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import random
from collections import OrderedDict
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Activation, Input, LSTM, Embedding, Dropout, GRU, Bidirectional
from keras.layers import Flatten, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras import regularizers
from keras.regularizers import l1
from keras.layers import BatchNormalization
from keras.models import Model
from keras import metrics
from keras.callbacks import EarlyStopping

#import all functions from text_data_preprocessing.py
from text_data_preprocessing import *

import warnings
warnings.filterwarnings("ignore")


#scrape all tweets from a user 
def scraping_user_full_tweets(username, replies = True, search_term= '', time_range=('',''), min_replies = 0):
    c = twint.Config()
    c.Username = username
    #c.Limit = limit
    c.Pandas = True
    if search_term != '':
        c.Search = search_term
    if time_range != ('',''):
        c.Since = time_range[0]
        c.Until = time_range[1]              
    if min_replies != 0:
        print('Min:', min_replies)
        c.Min_replies = min_replies
    c.Count = True
    c.Hide_output = True
    c.Filter_Retweets = False
    twint.run.Search(c)
    df = twint.storage.panda.Tweets_df
    print(len(df))
    if len(df) != 0 and not replies:
        s = df['reply_to']
        df = df[s.isin([[]])]
    
    return df


#calculate user's ACE alignment index
def calculate_RootUser_ACE(df): 
    model = keras.models.load_model('C:/Users/classification_models/ACE_CNN_MODEL')      
    x = df['clean_text'].values
    
    f = open('C:/Users/Pickle Files/t.pckl', 'rb')
    t = pickle.load(f)
    f.close()
    
    f = open('C:/Users/Pickle Files/max_len.pckl', 'rb')
    max_len = pickle.load(f)
    f.close()
    
    tweets_tok = textTokenizeModel(t, x, max_len)
    #get the predicted ACE probabilty
    y_pred = model.predict(tweets_tok, verbose = 1)
    #calculate the maximum value of ACE likeliness of the user
    MAX_ACE = float(max(y_pred))
    #calculate the top 10 percentile of ACE likeliness of the user
    per10th = np.percentile(y_pred, 90) 
    
    return MAX_ACE, per10th

#load followers, followees, and reciprocal followers of root users
data_followers = pd.read_csv('followers.csv').iloc[: , 1:]
data_followees = pd.read_csv('followees.csv').iloc[: , 1:]
data_friends = pd.read_csv('friends.csv').iloc[: , 1:]


#get the average ACE alignment index over the followers of each root user
follower_avg_ACE = [] #list storing the average ACE alignment index of the followers of each root user
for (columnname, columndata) in data_followers.iteritems():
    followers = [x for x in columndata.values if pd.isnull(x) == False]    
    #append NaN to follower_avg_ACE if the root user has no valid follower
    if len(followers) == 0:
        follower_avg_ACE.append(np.NaN)
        continue
    
    follower_ACE_list = [] #list storing the ACE alignment index of each follower
    for follower in followers:
        try:  
            dfnew = scraping_user_full_tweets(follower)
            if dfnew.shape[0] == 0:
                continue
            dfnew = dfnew[dfnew['language'] == 'en']
            if dfnew.shape[0] == 0:
                continue
            dfnew['clean_text'] = dfnew['tweet'].apply(clean_text, remove_punctuation = True, remove_stopwords = True, remove_num=True)
            dfnew['text_len'] = dfnew['clean_text'].apply(get_num_words)
            dfnew = dfnew[pd.to_numeric(dfnew['text_len']) > 5]
            if dfnew.shape[0] < 30:
                continue
            #calculate the ACE alignment index of a follower
            MAX_ACE, per10th = calculate_RootUser_ACE(dfnew)
            follower_ACE_list.append(per10th)
        except:
            continue
        
    #append 0 to follower_avg_ACE if the root user has no follower
    #whose ACE alignment index can be calculated (i.e., at least 30 tweets with more than five words after the cleaning)
    if len(follower_ACE_list) == 0:
        follower_avg_ACE.append(0)
    else:
        avg_ACE = np.mean(follower_ACE_list)
        follower_avg_ACE.append(avg_ACE)
          

#get the average ACE alignment index of the followees of the root users
followee_avg_ACE = [] #list storing the average ACE alignment index of the followees of the root users
for (columnname, columndata) in data_followees.iteritems():
    followees = [x for x in columndata.values if pd.isnull(x) == False]
    #append NaN to followee_avg_ACE if the root user has no valid followee
    if len(followees) == 0:
        followee_avg_ACE.append(np.NaN)
        continue
    
    followee_ACE_list = [] #list storing the ACE alignment index of each followee
    for followee in followees:
        try:  
            dfnew = scraping_user_full_tweets(followee)
            if dfnew.shape[0] == 0:
                continue
            dfnew = dfnew[dfnew['language'] == 'en']
            if dfnew.shape[0] == 0:
                continue
            dfnew['clean_text'] = dfnew['tweet'].apply(clean_text, remove_punctuation = True, remove_stopwords = True, remove_num=True)
            dfnew['text_len'] = dfnew['clean_text'].apply(get_num_words)
            dfnew = dfnew[pd.to_numeric(dfnew['text_len']) > 5]
            if dfnew.shape[0] < 30:
                continue
            #calculate the ACE alignment index of a followee
            MAX_ACE, per10th = calculate_RootUser_ACE(dfnew)
            followee_ACE_list.append(per10th)
        except:
            continue
    #append 0 to followee_avg_ACE if the root user has no followees  
    #whose ACE alignment index can be calculated (i.e., at least 30 tweets with more than five words after the cleaning)
    if len(followee_ACE_list) == 0:
        followee_avg_ACE.append(0)
    else:
        avg_ACE = np.mean(followee_ACE_list)
        followee_avg_ACE.append(avg_ACE)       


#The friend in the following code lines represents reciprocal follower.
#get the average ACE alignment index over the reciprocal followers of the root users
friend_avg_ACE = [] #list storing the average ACE alignment index of the reciprocal followers of the root users
for (columnname, columndata) in data_friends.iteritems():
    friends = [x for x in columndata.values if pd.isnull(x) == False]
    #append NaN to friend_avg_ACE if the root user has no valid reciprocal follower
    if len(friends) == 0:
        friend_avg_ACE.append(np.NaN)
        continue

    friend_ACE_list = [] #list storing the ACE alignment index of each reciprocal follower
    for friend in friends:
        try:  
            dfnew = scraping_user_full_tweets(friend)
            if dfnew.shape[0] == 0:
                continue
            dfnew = dfnew[dfnew['language'] == 'en']
            if dfnew.shape[0] == 0:
                continue
            dfnew['clean_text'] = dfnew['tweet'].apply(clean_text, remove_punctuation = True, remove_stopwords = True, remove_num=True)
            dfnew['text_len'] = dfnew['clean_text'].apply(get_num_words)
            dfnew = dfnew[pd.to_numeric(dfnew['text_len']) > 5]
            if dfnew.shape[0] < 30:
                continue
            #calculate the ACE alignment index of a reciprocal follower
            MAX_ACE, per10th = calculate_RootUser_ACE(dfnew)
            friend_ACE_list.append(per10th)
        except:
            continue
    #append 0 to friend_ACE_list if the root user has no reciprocal followers 
    #whose ACE alignment index can be calculated (i.e., at least 30 tweets with more than five words after the cleaning)
    if len(friend_ACE_list) == 0:
        friend_avg_ACE.append(0)
    else:
        avg_ACE = np.mean(friend_ACE_list)
        friend_avg_ACE.append(avg_ACE)       


#get the non-reciprocal followers of the root users
#The non_friend (or nonfriend) in the following code lines represents non-reciprocal follower
non_friends = pd.DataFrame(columns = list(data_followers))
for (columnname, columndata) in data_followers.iteritems():
    column_friends = data_friends[columnname]
    column_nonfriends = list((Counter(columndata)-Counter(column_friends)).elements())
    if len(column_nonfriends) < 100:
        column_nonfriends.extend([''] * (100-len(column_nonfriends)))                                 
    non_friends[columnname] = column_nonfriends    
#save non-reciprocal followers
#The name of each column (i.e., the first row) is the name of a root user.
#The column values are the list of non-reciprocal followers of the root user whose name is in the first row.
#non_friends.to_csv('nonfriends.csv')    
    
    
#get the average ACE alignment index over the non-reciprocal followers of the root users      
nonfriend_avg_ACE = [] #list storing the average ACE alignment index of the non-reciprocal followers of the root users
for (columnname, columndata) in non_friends.iteritems():
    nonfriends = [x for x in columndata.values if pd.isnull(x) == False]
    
    if len(nonfriends) == 0:
        nonfriend_avg_ACE.append(np.NaN)
        continue
    
    nonfriend_ACE_list = [] #list that save ACE alignment index of each single non-reciprocal follower
    for nonfriend in nonfriends:
        try:  
            dfnew = scraping_user_full_tweets(nonfriend)
            if dfnew.shape[0] == 0:
                continue
            dfnew = dfnew[dfnew['language'] == 'en']
            if dfnew.shape[0] == 0:
                continue
            dfnew['clean_text'] = dfnew['tweet'].apply(clean_text, remove_punctuation = True, remove_stopwords = True, remove_num=True)
            dfnew['text_len'] = dfnew['clean_text'].apply(get_num_words)
            dfnew = dfnew[pd.to_numeric(dfnew['text_len']) > 5]
            if dfnew.shape[0] < 30:
                continue
            #calculate the ACE alignment index of a single non-reciprocal follower
            MAX_ACE, per10th = calculate_RootUser_ACE(dfnew)
            nonfriend_ACE_list.append(per10th)
        except:
            continue
    #append 0 to nonfriend_ACE_list if the root user has no non-reciprocal followers
    #whose ACE alignment index can be calculated (i.e., at least 30 tweets with more than five words after the cleaning)
    if len(nonfriend_ACE_list) == 0:
        nonfriend_avg_ACE.append(0)
    else:
        avg_ACE = np.mean(nonfriend_ACE_list)
        nonfriend_avg_ACE.append(avg_ACE)    
  
#load the dataframe that contains the names of the root users   
RootUsers = pd.read_csv('rootusers.csv')
RootUsers = RootUsers[RootUsers['per10th'] > 0.5]
root = RootUsers[['username']]
#add the average ACE alignment index over the followers and that over the followees
root['avg_ace_followers'] = follower_avg_ACE
root['avg_ace_followees'] = followee_avg_ACE
#save the updated root user information
root.to_csv('rootusers_homophily.csv')

df1 = RootUsers[['username']]
df2 = RootUsers[['username']]
df1['group'] = ['ACE'] * df1.shape[0]
df1['part'] = ['reciprocal'] * df1.shape[0]
df1['avg_ACE'] = friend_avg_ACE
df2['group'] = ['ACE'] * df2.shape[0]
df2['part'] = ['non-reciprocal'] * df2.shape[0]
df2['avg_ACE'] = nonfriend_avg_ACE
df = pd.concat([df1, df2], axis = 0)
df.to_csv('rootusers_reciprocal_non-reciprocal.csv')
