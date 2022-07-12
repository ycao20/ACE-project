# Searching for Adverse Childhood Experiences in Twitter

When you use this code, please cite the following paper:

"PAPER INFORMATION TO COME HERE"

## Usage 
  1. Download all files in the same folder.
  2. Activate tensorflow environment, and then run "reddit_data_scraping.py" in Spyder or by, for example,
     ```   
     conda activate tensorflow    
     python reddit_data_scraping.py   
     ```    
  3. Run "ace_classification_models.py" in Spyder or by, for example,
     ```    
     python ace_classification_models.py   
     ``` 
  4. Run "twint_scraping_root_users.py" in Spyder or by, for example,
     ```    
     python twint_scraping_root_users.py  
     ``` 
  5. Deactivate tensorflow environment, and then run "tweepy_get_followerships.py" in Spyder or by, for example,
     ```    
     conda deactivate
     python tweepy_get_followerships.py
     ```      
  6. Activate tensorflow environment, and then run "calculate_homophily.py" in Spyder or by, for example,
     ```    
     conda activate tensorflow
     python calculate_homophily.py 
     ```
## Software Requirement
Python 3.8+ in Anaconda Spyder environment.

## Code Functionality and Outputs  
### text_data_preprocessing.py

__Functions__: This is a helper file which includes a collection of text cleaning and preprocessing functions, all functions of this file will be used by the following files: "reddit_data_scrapping.py", "ace_classification_models.py", "twint_scraping_root_users.py", and "calculate_homophily.py".

__Required libraries__: pandas, numpy, os, pickle, collections, nltk, regex, string, random, langdetect, itertools, tensorflow, keras.

__Developers__: Suraj Rajendran, Prathic Sundararajan, Yiding Cao, Naoki Masuda. Yiding and Naoki only contributed to the comment lines.    

### reddit_data_scraping.py   

__Functions__: This code uses the Pushshift API to compile Reddit posts including the post titles and their associated attributes from subreddits r/raisedbynarcissists and r/internetparents into a dataframe. It cleans and filters the title and the main text of all Reddit posts in the two subreddits by calling functions in "text_data_preprocessing.py" in preparation for model construction. It uses many natural language processing techniques and libraries in the backend. Note that we only used the post title, not the main text of the post, in the manuscript.

__Required libraries__: pandas, numpy, os, pmaw, datetime.    

__Outputs__: This code generates the following four output files:<br /> 
  - A CSV file "childhood_trauma_data.csv", which contains posts in r/raisedbynarcissists along with the post titles and other information about each post.   
  - A CSV file "clean_childhood_trauma_data.csv", which contains cleaned titles of the posts in r/raisedbynarcissists.   
  - A CSV file "childhood_negative_data.csv", which contains posts in r/internetparents along with the post titles and other information about each post.   
  - A CSV file "clean_childhood_negative_data.csv", which contains cleaned titles of the posts in r/internetparents.

__Developers__:  Suraj Rajendran, Prathic Sundararajan, Yiding Cao, Naoki Masuda. Naoki only contributed to the comment lines.

### ace_classification_models.py    

__Functions__: Given the cleaned post titles from the two subreddits, this code creates the word embedding for the text data using GloVe. To carry out GloVe embedding, it is necessary to access the GloVe text file, which is available [here](https://nlp.stanford.edu/projects/glove/). Using this embedding scheme and text data, ace_classification_models.py builds and evaluates a convolutional neural network (CNN) model. Using the text data, this code also builds and evaluates a bidirectional encoder representation from transformers (BERT) model.    

__Requied libraries__: pandas, numpy, os, seaborn, matplotlib, pickle, nltk, tensorflow, keras, sklearn. 

__Inputs__: This code assumes two CSV files named "clean_childhood_trauma_data.csv" and "clean_childhood_negative_data.csv" as input, which should be placed in the same folder.   

__Outputs__: The tokenizer model, maximum sequence length, and word index (all necessary for proper textual embedding) are saved as pickle files. The CNN and BERT models are also saved for later use.    

__Developers__: Yiding Cao, Suraj Rajendran, Prathic Sundararajan, Naoki Masuda. Naoki only contributed to the comment lines.   
      
### twint_scraping_root_users.py

__Functions__: Using Twint, this code (1) pulls Twitter tweets and the username of the user who posted the tweet (i.e., root user) matching search keywords, (2) collects all tweets posted by the root users sampled in the last step, (3) calculates the ACE likeliness of all tweets of each root user using the saved CNN model, and (4) performs sentiment analysis on all tweets of each root user.

__Required libraries__: twint, pandas, numpy, pickle, os, collections, vaderSentiment, nest_asyncio, tensorflow, keras.

__Outputs__: This code generates the following files as output.
  - A CSV file "rootusers.csv" that contains the following columns in this order:
      1. username: username of a user.
      2. group: name of the group that the user belongs to.
      3. max_ace: maximum value of the ACE likelihood calculated from all tweets of the user.
      4. per10th: the top 10 percentile of the ACE likelihood calculated from all tweets of the user.
  - A CSV file "sentiment.csv" that contains the following columns in this order:
      1. username: username of a user.   
      2. group: name of the group that the user belongs to.      
      3. perc_positive: percentage of positive-sentiment tweets.    
      4. perc_negative: percentage of negative-sentiment tweets.    
      5. perc_neutral: percentage of neutral-sentiment tweets.     

__Developers__: Yiding Cao, Suraj Rajendran, Prathic Sundararajan, Naoki Masuda. Naoki only contributed to the comment lines.

### tweepy_get_followerships.py   

__Functions__: This code uses Twitter API to (1) collect the followers and followees of a Twitter root user, (2) check whether a follower is followed by the root user (in which case the follower is called the reciprocal follower of the root user), (3) check whether a followee follows back the root user (in which case the followee is called the reciprocal followee of the root user). Then, for every pair of reciprocal neighbors of the root user, it checks whether the bidirectional edges or at least one unidirectional edge exist between two reciprocal neighbors, and calculates the number of triangles formed around the root user.       

__Required Libraries__: tweepy, pandas, os, numpy, collections.

__Inputs__: This code assumes the CSV file named "rootusers.csv" as input, which should be placed in the same folder.

__Outputs__: This code generates the following files as output.
  - A CSV file "rootusers_followership.csv" that contains the following columns in this order:
      1. username: username of a root user.   
      2. group: name of the group that the root user belongs to.
      3. num_follower: number of the sampled followers of the root user.
      4. num_friends: number of the reciprocal followers in the set of the sampled followers.
      5. reciprocity1: reciprocity calculated based on the reciprocal followers.
      6. poss_tri1: number of possible triangles calculated based on the reciprocal followers.
      7. tri1: number of triangles when we require bidirectional edges between reciprocal followers.
      8. tri2: number of triangles when we only require a unidirectional edge between reciprocal followers.
      9. coef1: clustering coefficient calculated based on tri1.
      10. coef2: clustering coefficient calculated based on tri2.
      11. num_followee: number of the sampled followees of the root user.
      12. num_friends2: number of the reciprocal followees in the set of the sampled followees.
      13. reciprocity2: reciprocity calculated based on the reciprocal followees.
      14. poss_tri2: number of possible triangles calculated based on the reciprocal followees.
      15. tri3: number of triangles when we require bidirectional edges between reciprocal followees.
      16. tri4: number of triangles when we only require a unidirectional edge between reciprocal followees.
      17. coef3: clustering coefficient calculated based on tri3.
      18. coef4: clustering coefficient calculated based on tri4.    
  - A CSV file "followers.csv", which contains the list of followers of each root user. Each column except for the first column corresponds to a root user. The root user's name is shown in the first row. The remaining rows in each column are the followers of the root user shown in the first row of the same column.
  - A CSV file "friends.csv", which contains the list of reciprocal followers of each root user. The file format is the same as that of "followers.csv". This is also the case for the next two output CSV files.
  - A CSV file "followees.csv", which contains the list of the followees of each root user.
  - A CSV file "friends2.csv", which contains the list of reciprocal followees of each root user.

__Developers__: Yiding Cao, Naoki Masuda. Naoki only contributed to the comment lines.

### calculate_homophily.py

__Functions__: For each root user, calculate_homophily.py calculates the ACE alignment index of each follower, followee, reciprocal follower, and reciprocal followee whose ACE alignment index is defined (i.e., a user who has at least 30 tweets with more than five words after the cleaning) and calculates the average of the ACE alignment index over all such neighbors.

__Requied libraries__: twint, pandas, numpy, pickle, os, collections, nest_asyncio, tensorflow, keras.

__Inputs__: This code assumes four CSV files, "rootusers.csv", "followers.csv", "friends.csv", and "followees.csv" as input, which should be placed in the same folder.

__Outputs__: This code generates the following files as output.
- A CSV file "rootusers_homophily.csv" that contains the following columns in this order:
    1. username: username of a root user.
    2. group: name of the group that the root user belongs to.
    3. avg_ace_followers: average ACE alignment index over the sampled followers of the root user.
    4. avg_ace_followees: average ACE alignment index over the sampled followees of the root user.  
- A CSV file "rootusers_reciprocal_non-reciprocal.csv" that contains the following columns in this order:
    1. username: username of a root user.  
    2. group: name of the group that the root user belongs to. 
    3. type: type of followers, which is either ‘reciprocal’, representing reciprocal follower, or ‘non-reciprocal’, representing non-reciprocal follower. 
    4. avg_ace: average ACE alignment index over the sampled reciprocal followers (if ‘type’ = ‘reciprocal’) or over the sampled non-reciprocal followers (if ‘type’ = ‘non-reciprocal’).

__Developers__: Yiding Cao,Suraj Rajendran, Prathic Sundararajan, Naoki Masuda. Naoki only contributed to the comment lines.
