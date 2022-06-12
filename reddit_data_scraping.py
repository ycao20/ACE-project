# -*- coding: utf-8 -*-

#import required libraries
from pmaw import PushshiftAPI
import pandas as pd
import numpy as np
import os
import datetime as dt

#import all functions from text_data_preprocessing
from text_data_preprocessing import *

#launch Reddit API
api = PushshiftAPI()
scrape_data = 1

#collect Reddit data from subreddit r/raisedbynarcissists
if scrape_data:
    start_date = dt.datetime(2020, 12, 25) #(year, month, day)
    end_date = dt.datetime(2022, 3, 31) #(year, month, day)
    delta = dt.timedelta(days=1)
    start_times = []
    end_times = []
    days = []
    dfs = []
    day = 1
    while start_date <= end_date:
        temp_end_date = start_date + delta
        start_epoch = int(start_date.timestamp())
        end_epoch = int(temp_end_date.timestamp())
        days.append(day)
        #get Reddit posts from subreddit r/raisedbynarcissists
        posts = api.search_submissions(subreddit="raisedbynarcissists", limit=None, after=start_epoch, before=end_epoch)
        post_list = [post for post in posts]
        post_df = pd.DataFrame(post_list)
        post_df['Day'] = day
        post_df["Start Date"] = start_date
        post_df["End Date"] = temp_end_date
        dfs.append(post_df)
        start_date = temp_end_date
        day += 1  
if scrape_data:
    agg_data = pd.concat(dfs)
    agg_data.to_csv("childhood_trauma_data.csv", index=False)


#collect Reddit data from subreddit r/internetparents
if scrape_data:
    start_date = dt.datetime(2020, 12, 25) #(year, month, day)  
    end_date = dt.datetime(2022, 3, 31) #(year, month, day) 
    delta = dt.timedelta(days=1)
    start_times = []
    end_times = []
    days = []
    dfs = []
    day = 1
    while start_date <= end_date:
        temp_end_date = start_date + delta
        start_epoch = int(start_date.timestamp())
        end_epoch = int(temp_end_date.timestamp())
        days.append(day)
        #get Reddit posts from subreddit r/internetparents
        posts = api.search_submissions(subreddit="internetparents", limit=None, after=start_epoch, before=end_epoch)
        post_list = [post for post in posts]
        post_df = pd.DataFrame(post_list)
        post_df['Day'] = day
        post_df["Start Date"] = start_date
        post_df["End Date"] = temp_end_date
        dfs.append(post_df)
        start_date = temp_end_date
        day += 1
if scrape_data:
    agg_data = pd.concat(dfs)
    agg_data.to_csv("childhood_negative_data.csv", index=False)  
 

#features in the cleaned Reddit data
features = ['Day', 'Start Date', 'End Date', 'score', 'num_comments', 'title','selftext']
    

#clean the Reddit data obtained from subreddit r/raisedbynarcissists
df_positive = pd.read_csv('childhood_trauma_data.csv')

df_positive = df_positive[features]
df_positive['has_comments'] = df_positive['num_comments'] > 0
df_positive['has_comments'] = df_positive['has_comments'].astype(int)
df_positive['title'] = df_positive['title'].astype(str)
df_positive["selftext"].fillna("EMPTY",inplace=True)
df_positive['selftext'] = df_positive['selftext'].astype(str)
df_positive = df_positive[df_positive['title'] != ""]
df_positive = df_positive[df_positive['title'].apply(is_en)]
df_positive = df_positive[df_positive['title'].apply(is_en)]
df_positive['clean_title'] = df_positive['title'].apply(clean_text, remove_punctuation = True, remove_stopwords = True, remove_num=True)
df_positive = df_positive[df_positive['clean_title'].apply(str.strip) != ""]
df_positive['clean_title_len'] = df_positive['clean_title'].apply(get_num_words)
df_positive['clean_selftext'] = df_positive['selftext'].apply(clean_text, remove_punctuation = True, remove_stopwords = True, remove_num=True)
df_positive['clean_text_len'] = df_positive['clean_title'].apply(get_num_words)

df_positive.to_csv('clean_childhood_trauma_data.csv')


#clean the Reddit data obtained from subreddit r/internetparents
df_negative = pd.read_csv('childhood_negative_data.csv')

df_negative = df_negative[features]
df_negative['has_comments'] = df_negative['num_comments'] > 0
df_negative['has_comments'] = df_negative['has_comments'].astype(int)
df_negative['title'] = df_negative['title'].astype(str)
df_negative["selftext"].fillna("EMPTY",inplace=True)
df_negative['selftext'] = df_negative['selftext'].astype(str)
df_negative = df_negative[df_negative['title'] != ""]
df_negative = df_negative[df_negative['title'].apply(is_en)]
df_negative = df_negative[df_negative['title'].apply(is_en)]
df_negative['clean_title'] = df_negative['title'].apply(clean_text, remove_punctuation = True, remove_stopwords = True, remove_num=True)
df_negative = df_negative[df_negative['clean_title'].apply(str.strip) != ""]
df_negative['clean_title_len'] = df_negative['clean_title'].apply(get_num_words)
df_negative['clean_selftext'] = df_negative['selftext'].apply(clean_text, remove_punctuation = True, remove_stopwords = True, remove_num=True)
df_negative['clean_text_len'] = df_negative['clean_title'].apply(get_num_words)

df_negative.to_csv('clean_childhood_negative_data.csv')
