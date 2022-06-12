# -*- coding: utf-8 -*-

#import required libraries
import tweepy
import pandas as pd
import os
from collections import defaultdict
import numpy as np

consumer_key = "Enter your consumer_key inside the quote"
consumer_secret = "Enter your consumer_secret inside the quote"
access_token = "Enter your access_token inside the quote"
access_token_secret = "Enter your_access_token_secret inside the quote"
bearer_token = "Enter your bearer_token inside the quote"

#authorize the consumer key and consumer secret
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
#give access to user's access key and access secret
auth.set_access_token(access_token, access_token_secret)

#must set wait_on_rate_limit_notify=True, then you will get message "Rate limit reached. Sleeping for: 856" every 15 minutes,
#indicating that scraping has reached the Twitter rate limit.
#If you do not set wait_on_rate_limit_notify=True, you will get an API error, and the code will terminate.
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)

#df includes a column 'username'
df = pd.read_csv('rootusers.csv')
df = df[df['per10th'] > 0.5]
df = df[['username']]

#collect the followers of each root user
#input is a dataframe that contains usernames
def get_followers(df): 
    #dictionary follower_dict stores the followers, and the dict keys are root users.
    #dict values are the lists of the followers
    follower_dict = defaultdict(list) 
    
    #dictionary friend_dict stores the reciprocal followers, and the dict keys are root users.
    #dict values are the lists of the reciprocal followers
    friend_dict = defaultdict(list) 
    
    num_followers_list = [] #list that stores the number of followers of each root user
    num_friends_list = [] #list that stores the number of reciprocal followers of each root user
    
    for user in df['username']:
        screen_name = user
        followers = [] #get the list of the followers of a root user
        friends = [] #get the list of the reciprocal followers of a root user
        num_followers = 0 #number of followers of a root user
        num_friends = 0 #number of reciprocal followers of a root user
        try:
            for follower in tweepy.Cursor(api.followers, screen_name).items(100):
                
                try:
                    target_screen_name = follower.screen_name
                    friendship = api.show_friendship(source_screen_name = screen_name, target_screen_name = target_screen_name)
                    
                    #check if the root user follows back the follower
                    #if True, the follower is a reciprocal follower. Note that "friend" in the code represents the reciprocal follower.
                    if friendship[0].following == True:
                        friends.append(target_screen_name)
                        num_friends += 1
                    followers.append(target_screen_name)
                    num_followers += 1

                #if the follower is invalid or a protected account, get TweepError
                except tweepy.TweepError: 
                    print(screen_name, ":", target_screen_name)
                    continue
            
            #append the list of followers to the follower_dict value
            follower_dict[user] = followers 
            #append the list of recipocal followers to the friend_dict value
            friend_dict[user] = friends 
            num_followers_list.append(num_followers) # append the number of the followers to the list
            num_friends_list.append(num_friends) # append the number of the reciprocal followers to the list
        except tweepy.TweepError:
            # if the root user is invalid or protected account, append NaN values to the following variables
            follower_dict[user] = [np.NaN] 
            friend_dict[user] = [np.NaN] 
            print(screen_name)
            num_followers_list.append(np.NaN)
            num_friends_list.append(np.NaN)
            continue
    return follower_dict, friend_dict, num_followers_list, num_friends_list


#collect the followees for each root user
def get_followees(df): 
    #followee_dict stores followees; dict keys represent the root users; dict values are the lists of followees
    followee_dict = defaultdict(list)
    
    #friend_dict stores reciprocal followees; dict keys represent root users;
    #dict values are the lists of reciprocal followees
    friend_dict = defaultdict(list)
    
    num_followees_list = [] #number of the followees of each root user
    num_friends_list = [] #number of the reciprocal followees of each root user
    for user in df['username']:
        screen_name = user
        followees = []
        friends = []
        num_followees = 0
        num_friends = 0
        try:
            for followee in tweepy.Cursor(api.friends, screen_name).items(100):
                try:
                    target_screen_name = followee.screen_name
                    friendship = api.show_friendship(source_screen_name = screen_name, target_screen_name = target_screen_name)
                    
                    #check if the root user is followed by the followee
                    if friendship[0].followed_by == True:
                        friends.append(target_screen_name)
                        num_friends += 1
                    followees.append(target_screen_name)
                    num_followees += 1

                except tweepy.TweepError: #if the followee is invalid or a protected account, get TweepError
                    print(screen_name, ":", target_screen_name)
                    continue
            followee_dict[user] = followees #append the list of followees to the followee dict
            friend_dict[user] = friends #append the list of reciprocal followees to the reciprocal followee dict
            num_followees_list.append(num_followees) #append the number of the followees to the list
            num_friends_list.append(num_friends) #append the number of the reciprocal followees to the list
        except tweepy.TweepError:
            # if the root user is invalid or a protected account, append NaN values to the following variables
            followee_dict[user] = [np.NaN]
            friend_dict[user] = [np.NaN]
            print(screen_name)
            num_followees_list.append(np.NaN)
            num_friends_list.append(np.NaN)
            continue        
    return followee_dict, friend_dict, num_followees_list, num_friends_list 
 
       
#count triangles containing each root user based on the reciprocal followers/followees of the root user. 
#We check if two neighbors of the root user are adjacent to each other by either bidirectional edges or at least one unidirectional edge.
#triangle1: count of triangles when we require bidirectional edges between neighbors of the root user
#triangle2: count of triangles when we only require a unidirectional edge between neighbors of the root user
def triangle(dict): #the argument is the dictionary of reciprocal followers/followees
    keys = []
    num_triangle1_list = []  #list that stores the number of triangles of type triangle1
    num_triangle2_list = []  #list that stores the number of of triangles of type triangle2
    for key,value in dict.items():
        follower_list = value
        num_triangle1 = 0
        num_triangle2 = 0
        for i in range(len(follower_list)-1):
            for j in range(i+1,len(follower_list)):
                source_screen_name = follower_list[i]
                target_screen_name = follower_list[j]
                
                try: 
                    friendship = api.show_friendship(source_screen_name = source_screen_name, target_screen_name = target_screen_name)
                
                    # check if two neighbors are adjacent to each other by bidirectional edges
                    if friendship[0].followed_by and friendship[0].following:
                        num_triangle1 += 1
                    # check if two neighbors are adjacent to each other by at least one unidirectional edge
                    if friendship[0].followed_by or friendship[0].following:
                        num_triangle2 += 1
                        
                except tweepy.TweepError:
                    continue
        keys.append(key)      
        num_triangle1_list.append(num_triangle1)
        num_triangle2_list.append(num_triangle2)
    return num_triangle1, num_triangle2

#get the dictionary of the list of followers and reciprocal followers of root users
follower_dict, friend_dict, num_followers_list, num_friends_list = get_followers(df)
#get the dictionary of the list of followees and reciprocal followees of root users
followee_dict, friend_dict2, num_followees_list, num_friends2_list = get_followees(df)
#count the triangles composed of the root user and two reciprocal followers
num_triangle1, num_triangle2 = triangle(friend_dict)
#count the triangles composed of the root user and two reciprocal followees
num_triangle3, num_triangle4 = triangle(friend_dict2)

#append the number of followers, reciprocal followers, followees, reciprocal followees to dataframe "df"
df['num_follower'] = num_followers_list
df['num_friends'] = num_friends_list
df['reciprocity1'] = df['num_friends']/df['num_follower']
df['poss_tri1'] = (df['num_friends']*(df['num_friends']-1))/2
#append the number of triangles and clustering coefficient of the root user to dataframe "df"
#tri1: triangle count when we require bidirectional edges between reciprocal followers
#tri2: triangle count when we only require a unidirectional edge between reciprocal followers
df['tri1'] = num_triangle1
df['tri2'] = num_triangle2
df['coef1'] = df['tri1']/df['poss_tri1']
df['coef2'] = df['tri2']/df['poss_tri1']

df['num_followee'] = num_followees_list
df['num_friends2'] = num_friends2_list
df['reciprocity2'] = df['num_friends2']/df['num_followee']
df['poss_tri2'] = (df['num_friends2']*(df['num_friends2']-1))/2
#append the number of triangles and clustering coefficient of the root user to dataframe "df"
#tri3: triangle count when we require bidirectional edges between reciprocal followees
#tri4: triangle count when we only require a unidirectional edge between reciprocal followees
df['tri3'] = num_triangle3
df['tri4'] = num_triangle4
df['coef3'] = df['tri3']/df['poss_tri2']
df['coef4'] = df['tri4']/df['poss_tri2']

#save the updated "df"
df.to_csv('rootusers_followership.csv')

#save follower_dict in "followers.csv"
#The column names (i.e., first row of the CSV file) are the names of the root users.
#The values in the column are the list of the followers of the root user whose name is shown in the first row of the column.
df_follower = pd.DataFrame()
for key,value in follower_dict.items():
    if len(value) < 100:
        value.extend([''] * (100-len(value)))
    df_follower[key] = value
df_follower.to_csv('followers.csv')

#save friend_dict in "friends.csv"
#The column names (i.e., first row of the CSV file) are the names of the root users.
#The values in the column are the list of reciprocal followers of the root user.
df_friend = pd.DataFrame()
for key,value in friend_dict.items():
    if len(value) < 100:
        value.extend([''] * (100-len(value)))
    df_friend[key] = value
df_friend.to_csv('friends.csv')

#save followee_dict in "followees.csv"
#The column names (i.e., first row of the CSV file) are the names of the root users.
#The values in the column are the list of followees of the root user.
df_followee = pd.DataFrame()
for key,value in followee_dict.items():
    if len(value) < 100:
        value.extend([''] * (100-len(value)))
    df_followee[key] = value
df_followee.to_csv('followees.csv')

#save friend_dict2 in "friends2.csv"
#The column names (i.e., first row of the CSV file) are the names of the root users.
#The values in the column are the list of reciprocal followees of the root user.
df_friend_2 = pd.DataFrame()
for key,value in friend_dict2.items():
    if len(value) < 100:
        value.extend([''] * (100-len(value)))
    df_friend_2[key] = value
df_friend_2.to_csv('friends2.csv')
