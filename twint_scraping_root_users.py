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
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
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

#get all tweets meeting the conditions specified by the parameters 
def scraping_based_search_terms(search_term, replies = True, limit = 1000, time_range=('',''), min_replies = 0):
    c = twint.Config()
    c.Limit = limit
    c.Pandas = True
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
    if len(df) != 0 and not replies:
        s = df['reply_to']
        df = df[s.isin([[]])]
    return clean_df(df)

#clean the dataframe "df" obtained using scraping_based_search_terms()
def clean_df(df):
    df = df[['username', 'tweet', 'language']]
    df = df[df['language'] == 'en']
    df.drop_duplicates(subset=['tweet'])
    df.drop_duplicates(subset=['username'])
    df = df[['username']]
    return df

#search all the tweets from a user 
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
    #do not scrape retweets
    c.Filter_Retweets = False
    twint.run.Search(c)
    df = twint.storage.panda.Tweets_df
    print(len(df))
    if len(df) != 0 and not replies:
        s = df['reply_to']
        df = df[s.isin([[]])]
    return df

#sample users using keywords
#enter the keywords in search_term
df =  scraping_based_search_terms(search_term = '"my mom" OR "my dad" OR "my mother" OR "my father" OR "my guardian" AND ("abuse" OR "neglect" OR "jail" OR "prison" OR "substance use" OR "substance misuse" OR "substance abuse" OR "overdose" OR "OD" OR "drug addiction" OR "parental separation" OR "divorce")', limit = 100)


#calculate the ACE alignment index of a root user
#The input is a dataframe containing the username and clean_text.  
def calculate_RootUser_ACE(df): 
    model = keras.models.load_model('C:/Users/classification_models/ACE_CNN_MODEL')      
    x = df['clean_text'].values
    #load the tokenizer model
    f = open('C:/Users/Pickle Files/t.pckl', 'rb')
    t = pickle.load(f)
    f.close()
    #load the maximum sequence length
    f = open('C:/Users/Pickle Files/max_len.pckl', 'rb')
    max_len = pickle.load(f)
    f.close()
    #load word_index
    tweets_tok = textTokenizeModel(t, x, max_len)
    #ACE likeliness
    y_pred = model.predict(tweets_tok, verbose = 1)
    #calculate the maximum ACE likeliness
    MAX_ACE = float(max(y_pred))
    #calculate the top 10 percentile of ACE likeliness
    per10th = np.percentile(y_pred, 90) 
    
    #output is composed of:
    #1. maximum ACE likeliness of a user 
    #2. top 10 percentile of ACE likeliness
    return MAX_ACE, per10th


#calculate the sentiment score of the input text
#input is a text sentence
def sentiment_scores(sentence):
    #create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()
    #polarity_scores method of the SentimentIntensityAnalyzer
    #object gives a sentiment dictionary
    #which contains positive, negative, neutral, and compound scores.
    sentiment_dict = sid_obj.polarity_scores(sentence)
    #decide whether sentiment is either positive, negative, or neutral
    if sentiment_dict['compound'] >= 0.05 :
        return "Positive"
    elif sentiment_dict['compound'] <= - 0.05 :
        return "Negative"
    else :
        return "Neutral"


#calculate the percentage of positive, neutral, and negative tweets of a user.
#The input is a dataframe containing the username and clean_text.
def RootUser_sentiment(df):
    clean_txt = df['clean_text']
    tweet_score = []
    for tweet in clean_txt:                    
        score = sentiment_scores(tweet)  
        tweet_score.append(score)
    dict = Counter(tweet_score)
    num_posi = dict['Positive']
    num_neu = dict['Neutral']
    num_nega = dict['Negative']
    total_num = num_posi + num_neu + num_nega
    #calculate the percentage of positive/negative/neutral sentiments
    perc_positive = float(num_posi/total_num)
    perc_negative = float(num_nega/total_num)
    perc_neutral = float(num_neu/total_num)
    return perc_positive, perc_negative, perc_neutral 


#The input is a dataframe composed of a single column that contains the usernames of the root users
#scrape all tweets of the users in the input dataframe 
#calculate the ACE alignment index of the root users
#calculate the percentage of positive, neutral, and negative tweets of the users 
def get_RootUsers(df):
    RootUsers = pd.DataFrame(columns = ['username', 'max_ace', 'per10th', 'perc_positive','perc_negative', 'perc_neutral'])
    for username in df['username']:
        try:
            dfnew = scraping_user_full_tweets(username)
            if dfnew.shape[0] == 0:
                continue
            dfnew = dfnew[dfnew['language'] == 'en']
            if dfnew.shape[0] == 0:
                continue
            dfnew['clean_text'] = dfnew['tweet'].apply(clean_text, remove_punctuation = True, remove_stopwords = True, remove_num=True)
            dfnew['text_len'] = dfnew['clean_text'].apply(get_num_words)
            #we only keep users who have at least 30 tweets with more than five words after the cleaning
            dfnew = dfnew[pd.to_numeric(dfnew['text_len']) > 5]
            if dfnew.shape[0] < 30:
                continue
            dfnew = dfnew[['tweet', 'clean_text']]
            
            #calculate the maximum ACE likeliness of the root user
            #calculate the ACE alignment index of the root user
            MAX_ACE, per10th = calculate_RootUser_ACE(dfnew)
            #calculate the percentage of positive, neutral, and negative sentiments of the root user
            perc_positive, perc_negative, perc_neutral  =  RootUser_sentiment(dfnew)  
            RootUsers = RootUsers.append({'username': username, 'max_ace': MAX_ACE, 'per10th': per10th, 'perc_positive': perc_positive,'perc_negative': perc_negative, 'perc_neutral': perc_neutral}, ignore_index = True)
            #dfnew.to_csv(f'path/{username}.csv', index=False) 
        except:
            continue
    return RootUsers


#input as "df" the list of users sampled using the keywords specified above
#scrape all tweets of those users
#calculate the ACE alignment index of those users
#calculate the percentage of positive, neutral, and negative tweets of those users 
RootUsers = get_RootUsers(df)

#save the usernames and the ACE alignment index in rootusers.csv
df_RootUsers = RootUsers[['username', 'max_ace', 'per10th']]
df_RootUsers.to_csv('rootusers.csv')

#get a specific group of root users and save the results of sentiment analysis for the selected root users in sentiment.csv
#to save ACE root users, set ['per10th'] > 0.5 
root = RootUsers[RootUsers['per10th'] > 0.5] 
#to save non-ACE root users, set ['per10th'] < 0.3
#root = RootUsers[RootUsers['per10th'] < 0.3]

sentiment = root[['username', 'perc_positive', 'perc_negative', 'perc_neutral']]
sentiment.to_csv('sentiment.csv')
