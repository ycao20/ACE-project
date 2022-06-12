# -*- coding: utf-8 -*-

#import required libraries
import pandas as pd
import numpy as np
import os
from collections import Counter
import nltk
import regex as re
import string
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from collections import OrderedDict
from itertools import compress
from itertools import chain
import random
import re
import langdetect as ld
import pycountry
from tqdm.auto import tqdm
from sklearn import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import statistics
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

#check if the input text is in English
def is_en(txt):
    if re.search('[a-zA-Z]', txt):   
        return True
    else:
        return False


#check if the input text includes digit 
def has_digit(txt):
    if any(map(str.isdigit, txt)):
        return 1
    else:
        return 0


#get the number of words in the text
def get_num_words(txt):
    return len(txt.split())


#calculate a sentiment dictionary, which is composed of a negative, neutral, positive, and compound scores.
def get_sentiment(title):
    vs = analyzer.polarity_scores(title)
    return pd.Series([vs['neg'], vs['neu'], vs['pos'], vs['compound']])


#check if the text includes any keyword
def has_keyword(title, keywords):
    if any(substring in title for substring in keywords):
        return 1
    else:
        return 0

#remove morphological affixes from words
default_stemmer = PorterStemmer()
#get the list of stopwords in English
default_stopwords = stopwords.words('english') # or any other list of your choice
#extend the list of stopwords with ascii lowercase characters
default_stopwords = list(string.ascii_lowercase) + default_stopwords
#remove 'no', 'nor', and 'not' from the list of stopwords
default_stopwords.remove('no')
default_stopwords.remove('nor')
default_stopwords.remove('not')

#randomly swap the order of words in the input text
def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words

#swap the order of two randomly selected words
def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
    return new_words

#change the specified regular expression to the replacement using re.sub()
def misc_cleaning(text):
    text = re.sub(' y ', '', text) #remove random y accent stuff scattered in the text
    text = re.sub('yyy', 'y', text)
    text = re.sub('\n', '', text)
    text = re.sub('rt', '', text)
    text = text.replace("("," ").replace(")"," ")
    text = " ".join(filter(lambda x:x[0]!='@', text.split()))
    
    text = re.sub(' f ', ' female ', text)
    text = re.sub(' m ', ' male ', text)
    text = re.sub(' narc ', ' narcissist ', text)
    text = re.sub(' bil ', ' brother in law ', text)
    text = re.sub('bd', ' bipolar disoder ', text)
    text = re.sub('bpd', ' borderline personality disorder ', text)
    text = re.sub('dae', ' does ', text)
    text = re.sub('darvo', ' deny attack reverse victim offender ', text)
    text = re.sub('donf', ' daughter of narcissist father ', text)
    text = re.sub('donm', ' daughter of narcissist mother ', text)
    text = re.sub('edad', ' enabler father ', text)
    text = re.sub('emom', ' enabler mother ', text)
    text = re.sub(' egg donor ', ' mother ', text)
    text = re.sub(' fleas ', ' frightening lasting effects of abuse ', text)
    text = re.sub('fm', ' abusers ', text)
    text = re.sub('foc', ' family of choice ', text)
    text = re.sub('fog', ' fear obligation guilt ', text)
    text = re.sub('foo', ' family of origin ', text)
    text = re.sub('gc', ' golden child ', text)
    text = re.sub('lc', ' low contact ', text)
    text = re.sub(' n ', ' narcissist ', text)
    text = re.sub('nc', ' no contact ', text)
    text = re.sub('nmil', ' narcissist mother in law ', text)
    text = re.sub('nfil', ' narcissist father in law ', text)
    text = re.sub('nmom', ' narcissist mother ', text)
    text = re.sub('nparents', ' narcissist parents ', text)
    text = re.sub('ndad', ' narcissist father ', text)
    text = re.sub('nsupply', ' narcissist energy ', text)
    text = re.sub(' sc ', ' structured contact ', text)
    text = re.sub('sg', ' scapegoat ', text)
    text = re.sub(' sil ', ' sister in law ', text)
    text = re.sub('sonm', ' son of narcissistic mother ', text)
    text = re.sub('sonf', ' son of narcissistic father ', text)
    text = re.sub(' sperm donor ', ' father ', text)
    text = re.sub(' tw ', ' trigger warning ', text)
    text = re.sub('vlc', ' very low contact ', text)
    text = re.sub('woes', ' walking on eggshells ', text)
    
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub(' +', ' ', text)
    return text

#get words and punctuation in a string of text
def tokenize_text(text):
    return [w for s in sent_tokenize(text) for w in word_tokenize(s)]

#randomly select words from a list of words
def random_deletion(words, p):
    if len(words) == 1:
        return words
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words)-1)
        return [words[rand_int]]
    return new_words

#replace synonyms
def synonym_replacement(words, n):
    new_words = words.copy()
    stop_words = default_stopwords
    random_word_list = list(set([word for word in words if word not in stop_words]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            #print("replaced", random_word, "with", synonym)
            num_replaced += 1
        if num_replaced >= n: #only replace up to n words
            break
    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')
    return new_words

#get a set of synonyms of the input word
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word): 
        for l in syn.lemmas(): 
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym) 
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

#randonly add words to an input list of words
def random_insertion(words, n):
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
    return new_words

#add one word to a list of words
def add_word(new_words):
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words)-1)]
        synonyms = get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return
    random_synonym = synonyms[0]
    random_idx = random.randint(0, len(new_words)-1)
    new_words.insert(random_idx, random_synonym)
    
#change the words in the input text to lemmas.
def lemmatizeWord(text):
    tokens = nlp(text)
    clean_text = ' '
    for token in tokens:
        clean_text = clean_text + " " + token.lemma_      
    return clean_text

#clean the input text
def clean_text(text, replace_numbers = False, remove_rare = False, remove_punctuation = False, stem_text = False, 
               remove_stopwords = False, remove_num = False , spell_check = False, 
               remove_repeat = False, random_delete=False, rand_swap=False, syn_replace=False, 
               rand_insert=False, lemmatize_word=False):
        
    text = text.lower()
    text = misc_cleaning(text) #look at function, random cleaning stuff
    #remove duplicate words
    if remove_repeat:
        sentences = sent_tokenize(text)
        sentences = list(dict.fromkeys(sentences))
        text = " ".join(sentences)
    #remove punctuations
    if remove_punctuation:
        text = "".join([(ch if ch not in string.punctuation else " ") for ch in text]).strip()
    #check spelling
    if spell_check:
        text = do_spellcheck(text)
    #replace the number by English words
    if replace_numbers:
        p = inflect.engine()
        words = word_tokenize(text)
        new_words = []
        for word in words:
                if word.isdigit():
                        new_word = p.number_to_words(word)
                        new_words.append(new_word)
                else:
                        new_words.append(word)
        text = " ".join(new_words)
    #optional: remove the rarest words in each text, currently set to remove the 10 rarest words
    if remove_rare:
        tokens = word_tokenize(text)
        freq_dist = nltk.FreqDist(tokens)
        rarewords = list(freq_dist.keys())[-10:]
        new_words = [word for word in tokens if word not in rarewords]
        text = " ".join(new_words)
    #optional: stem text using Porter Stemmer
    if stem_text:
        stemmer = default_stemmer
        tokens = tokenize_text(text)
        text = " ".join([stemmer.stem(t) for t in tokens])
    #remove stop words such as "a" and "the"
    if remove_stopwords:
        stop_words = default_stopwords
        tokens = [w for w in tokenize_text(text) if w not in stop_words]
        text = " ".join(tokens)
    #optional: remove numbers from the text
    if remove_num:
        text=text.split()
        text=[x for x in text if not x.isnumeric()]
        text= " ".join(text)
    if rand_swap:
        words = word_tokenize(text)
        words = random_swap(words, 10)
        text= " ".join(words)    
    if random_delete:
        words = word_tokenize(text)
        words = random_deletion(words, 0.1)
        text = " ".join(words)
    if syn_replace:
        words = word_tokenize(text)
        words = synonym_replacement(words, 50)
        text = " ".join(words)
    if rand_insert:
        words = word_tokenize(text)
        words = random_insertion(words, 50)
        text = " ".join(words) 
    if lemmatize_word:
        text = lemmatizeWord(text)
    text = " " + text + " "
    return text
    
#get length of words in the input text    
def lengths(x):
    length=[]
    for t in x:
        length.append(len(t))
    return length

#get the tokenizer model, padded sequences of the input text, word_index, and the maximum length of the text sequence
def textTokenize(text):    
    t = Tokenizer()
    t.fit_on_texts(text) #training phase
    word_index = t.word_index #get a map of word index
    sequences = t.texts_to_sequences(text)
    max_len=max(lengths(sequences))
    print('Found %s unique tokens' % len(word_index))
    text_tok=pad_sequences(sequences, maxlen=max_len)
    return t, text_tok, word_index, max_len

#transform the input text using the tokenizer model and the maximum length of the text sequence
def textTokenizeModel(t, text, max_len):
    sequences = t.texts_to_sequences(text)
    text_tok=pad_sequences(sequences, maxlen=max_len)
    return text_tok

#define word embedding function using GloVe
#GloVe text file is available at https://nlp.stanford.edu/projects/glove/
def word_Embed_glove(word_index):
    EMBEDDING_FILE = 'glove.840B.300d.txt'
    #convert pretrained word embedding to a dictionary
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE,encoding="utf-8"))
    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]
    nb_words = len(word_index)+1
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    total_num = 0
    embedded_num = 0
    for word, i in word_index.items():
        total_num += 1
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            embedded_num += 1
    print("Done making embedding matrix")
    print("Fraction of Words with Embeddings: " + str(embedded_num/total_num))
    return embedding_matrix

#get a word from the GloVe pre-trained word-vector database and obtain the vector representation of this word
def get_coefs(word,*arr):
    return word, np.asarray(arr, dtype='float32')
