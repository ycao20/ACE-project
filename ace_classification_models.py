# -*- coding: utf-8 -*-

#import required libraries
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import wordnet 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
import tensorflow_text as text
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Activation, Input, LSTM, Embedding, Dropout, GRU, Bidirectional
from keras.layers import Flatten, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras import regularizers
from keras.regularizers import l1
from keras.layers import BatchNormalization
from keras.models import Model
from keras import metrics
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import brier_score_loss, precision_score, recall_score, f1_score, fbeta_score, cohen_kappa_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from text_data_preprocessing import *

#load training data, which is the cleaned Reddit data
positive_data = pd.read_csv("clean_childhood_trauma_data.csv")
negative_data = pd.read_csv("clean_childhood_negative_data.csv")
positive_data = positive_data.dropna(subset=['clean_title'])
negative_data = negative_data.dropna(subset=['clean_title'])
positive_data = positive_data[pd.to_numeric(positive_data['clean_title_len']) > 5]
negative_data = negative_data[pd.to_numeric(negative_data['clean_title_len']) > 5]
positive_data['label'] = 1
negative_data['label'] = 0
sample_size = min(positive_data.shape[0], negative_data.shape[0])
#make the size of positive data set and the negative data set equal
positive_data = positive_data.sample(n = sample_size)
negative_data = negative_data.sample(n = sample_size)
data = pd.concat([positive_data, negative_data])

#x_text is the independent variable in the classification model
x_text = data['clean_title']
#y is the dependent variable in the classification model
y = data['label']

#function textTokenize is defined in text_data_preprocessing.py
#get tokens, padded sequence of texts, a map of word index, and the maxumum length of text
t, notes_tok, word_index, max_len = textTokenize(x_text.values)

#function word_Embed_glove is defined in text_data_preprocessing.py
#word embedding using Glove
embedding_matrix_glove = word_Embed_glove(word_index)

#save the tokenizer model
f = open('C:/Users/Pickle Files/t.pckl', 'wb')
pickle.dump(t, f)
f.close()
print("Saved Tokenizer")

#save the word index
f = open('C:/Users/Pickle Files/word_index.pckl', 'wb')
pickle.dump(word_index, f)
f.close()
print("Saved Word Indices")

#save the maximum sequence length
f = open('C:/Users/Pickle Files/max_len.pckl', 'wb')
pickle.dump(max_len, f)
f.close()
print("Saved Maximum Length")

#split Reddit data into the training, validation, and test data sets
X_train_text, X_test_text, y_init_train, y_test = train_test_split(notes_tok, y, test_size=0.2, random_state=42)
y_init_train = y_init_train.values
y_test = y_test.values
X_train_text, X_val_text, y_train, y_val = train_test_split(X_train_text, y_init_train, test_size=0.1, random_state=42)

#define the CNN model
def CNN_model(word_index, embedding_matrix, max_len):
    nlp_input = Input(shape=(max_len,), name='nlp_input')
    
    embed = Embedding(len(word_index)+1, embedding_matrix.shape[1], weights=[embedding_matrix], input_length=max_len, trainable=False)(nlp_input)
    cnn = Conv1D(32, 5, activation='relu')(embed)
    pooling = GlobalMaxPooling1D()(cnn)

    dens1 = Dense(10)(pooling)
    output = Dense(1, activation="sigmoid")(dens1)
    model = Model(nlp_input, output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
seed = 97
cnn_model = None
#feed word_index, embedding matrix, and maximum length of the sequence to the CNN model
cnn_model = CNN_model(word_index, embedding_matrix_glove, max_len)
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto', restore_best_weights=True) # pateince is number of epochs
callbacks_list = [earlystop]
#fit the model using training data
model_info=cnn_model.fit(X_train_text, y_train, epochs=100, batch_size=128, verbose = 1,callbacks=callbacks_list, validation_data=(X_val_text, y_val))

#run the model on the test data
#y_pred is between 0 and 1
y_pred=cnn_model.predict(X_test_text ,verbose=1)
#classification threshold is 0.5
y_pred_coded=np.where(y_pred>0.5,1,0)
#y_pred_coded is the predicted class (i.e., ACE or non-ACE)
y_pred_coded=y_pred_coded.flatten()

#model evaluation
print("confusion_matrix: ")
print(confusion_matrix(y_test, y_pred_coded))
print("Predicted Values:")
print(y_pred_coded)
print("True Values:")
print(y_test)
metric=[]
metric.append(['f1score',f1_score(y_test,y_pred_coded)])
metric.append(['precision',precision_score(y_test,y_pred_coded)])
metric.append(['recall',recall_score(y_test,y_pred_coded)])
metric.append(['accuracy',accuracy_score(y_test,y_pred_coded)])
#calculate the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred.flatten())
auc_keras = auc(fpr, tpr)
metric.append(['auc',auc_keras])
print(metric)
#plot the ROC curve
plt.figure()
lw = 2
plt.plot(fpr,tpr,color="darkorange",lw=lw,label="ROC curve (area = %0.2f)" % auc_keras,)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("CNN: Receiver operating characteristic example")
plt.legend(loc="lower right")
plt.show()

#save the CNN model
os.chdir("C:/Users/classification_models")
cnn_model.save("ACE_CNN_MODEL")


#define the BERT model
x_text = data['clean_title']
y = data['label']
#get training and test data sets
X_train_text, X_test_text, y_train, y_test = train_test_split(x_text, y, test_size=0.2, random_state=42)
y_train = y_train.values
y_test = y_test.values

#load a BERT model from TensorFlow Hub
preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

#build your own model by combining BERT with a classifier
#define BERT layers
text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
preprocessed_text = preprocessor(text_input)
outputs = encoder(preprocessed_text)
#define neural network layers
l = tf.keras.layers.Dropout(0.1, name="dropout")(outputs['pooled_output'])
l = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(l)
#construct a final model
model = tf.keras.Model(inputs=[text_input], outputs = [l])
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=METRICS)

#fit the BERT model with the training data
model.fit(X_train, y_train, epochs=10)
y_predicted = model.predict(X_test)
#y_predicted ranges between 0 and 1
y_predicted = y_predicted.flatten()
#classification threshold is 0.5
y_pred_coded=np.where(y_predicted>0.5,1,0)
#y_pred_coded is the predicted class (i.e., ACE or non-ACE)
y_pred_coded=y_pred_coded.flatten()

#model evaluation
print("confusion_matrix: ")
print(confusion_matrix(y_test, y_pred_coded))
print("Predicted Values:")
print(y_pred_coded)
print("True Values:")
print(y_test)
metric=[]
metric.append(['f1score',f1_score(y_test,y_pred_coded)])
metric.append(['precision',precision_score(y_test,y_pred_coded)])
metric.append(['recall',recall_score(y_test,y_pred_coded)])
metric.append(['accuracy',accuracy_score(y_test,y_pred_coded)])
#calculate the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_predicted.flatten())
auc_keras = auc(fpr, tpr)
metric.append(['auc',auc_keras])
print(metric)
#plot the ROC curve
plt.figure()
lw = 2
plt.plot(fpr,tpr,color="darkorange",lw=lw,label="ROC curve (area = %0.2f)" % auc_keras,)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("CNN: Receiver operating characteristic example")
plt.legend(loc="lower right")
plt.show()

#save the BERT model
import os
os.chdir("C:/Users/classification_models")
model.save("ACE_BERT_MODEL")
