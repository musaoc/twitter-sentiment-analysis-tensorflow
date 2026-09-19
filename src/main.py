"""
Twitter Sentiment Analysis with TensorFlow & BiLSTM
A deep learning natural language processing pipeline trained on 1.6 million tweets to classify sentiment with over 75% accuracy using TensorFlow, custom word embeddings, and Bidirectional LSTMs.

Original Kaggle Notebook: https://www.kaggle.com/code/lazer999/twitter-sentiment-analysis-tensorflow-0-75
Author: Muhammad Musa Khan (Kaggle Master: https://kaggle.com/lazer999)
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# --- Smart Dataset Path Resolution ---
def _resolve_data_path(file_path):
    """Checks local and data/ directories if dataset path is missing."""
    if os.path.exists(file_path):
        return file_path
    base = os.path.basename(file_path)
    candidates = [
        base,
        os.path.join("data", base),
        os.path.join("..", "data", base),
        file_path.replace("/kaggle/input/", "data/"),
        file_path.replace("../input/", "data/"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return file_path

# --- Pipeline Execution ---

# --- Cell 1 ---
import pandas as pd
import numpy as np
import string
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
import nltk
import re
import multiprocessing
import tensorflow as tf
from sklearn.model_selection import train_test_split
nltk.download("stopwords")

import warnings
warnings.filterwarnings("ignore")

# --- Cell 2 ---
# setting the names of the columns into col_names

col_names = ["target", "ids", "date", "flag", "user", "text"]
# reading the data from the path and using ISO-8859-1 encoding to decode it.

df = pd.read_csv('../input/sentiment140/training.1600000.processed.noemoticon.csv',
            encoding = "ISO-8859-1",
            names=col_names)

# --- Cell 3 ---
df = df.sample(160000) # taking 160k rows from the data
df.head()

# --- Cell 4 ---
df.info() # shows the columns, and their types.

# --- Cell 5 ---
df.shape

# --- Cell 6 ---
df.describe(include='all') # To statistical measures of the columns

# --- Cell 7 ---
sns.countplot(x=df['target']); # shows a count plot refering to the number of 0 and 4 in the column

# --- Cell 8 ---
df = df.drop(['ids', 'date', 'flag','user'], axis = 1) # dropping unnecessary columns

# --- Cell 9 ---
df.head()

# --- Cell 10 ---
df['target']=df['target'].replace(4,1) 

# --- Cell 11 ---
data = df['text']
labels = np.array(df['target'])

# --- Cell 12 ---
data.iloc[1] # displaying a row of the data

# --- Cell 13 ---
# this function is copied from another kernel. Don't know who is the original author of it.
stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)

#Removing the stopwords from text
def split_into_words(text):
    # split into words by white space
    words = text.split()
    return words

def to_lower_case(words):
    # convert to lower case
    words = [word.lower() for word in words]
    return words

def remove_punctuation(words):
    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    stripped = [re_punc.sub('', w) for w in words]
    return stripped

def keep_alphabetic(words):
    # remove remaining tokens that are not alphabetic
    words = [word for word in words if word.isalpha()]
    return words

def remove_stopwords(words):
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    return words

def to_sentence(words):
    # join words to a sentence
    return ' '.join(words)
def tweet(words):
    tweet_tokenizer = nltk.tokenize.TweetTokenizer(strip_handles=True,reduce_len=True)
    tweet = tweet_tokenizer.tokenize(words)
    return tweet
        

#Removing the noisy text
def denoise_text(text):
    words = split_into_words(text)
    words = to_lower_case(words)
    words = remove_punctuation(words)
    words = keep_alphabetic(words)
    words = remove_stopwords(words)
    return to_sentence(words)

# --- Cell 14 ---
data = data.apply(denoise_text)

# --- Cell 15 ---
print('Before: {}'. format(list(df['text'][:2])))
print('---')
print('After: {}'. format(list(data[:2])))

# --- Cell 16 ---
print(f"dataset contains {len(data)} examples\n")

print(f"Text of second example should look like this:  {data.iloc[1]}\n")
print(f"Text of fourth example should look like this:  {data.iloc[3]}")

print(f"\nLabels of last 5 examples should look like this:\n{labels[-5:]}")

# --- Cell 17 ---
# Split in training validation and test sets
X_train, X_test, y_train, y_test = test = train_test_split(data, labels,test_size=0.20,
                                                           random_state=1,
                                                           stratify = labels)

# --- Cell 18 ---
# lets take 10k words in num_words
tokenizer = Tokenizer(num_words=10000, oov_token = '<OOV>')
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index
VOCAB_SIZE = len(word_index)+1
VOCAB_SIZE

# --- Cell 19 ---
maxlen = max([len(x) for x in X_train]) # return 175 which is too big for tweets data.
maxlen = 50

# --- Cell 20 ---
train_sequences = tokenizer.texts_to_sequences(X_train)
train_padded_sequences = pad_sequences(train_sequences,maxlen=maxlen,padding='post',truncating='post')
test_sequences = tokenizer.texts_to_sequences(X_test)
test_padded_sequences = pad_sequences(test_sequences,maxlen=maxlen,padding='post',truncating='post')

# --- Cell 21 ---
len(train_padded_sequences[0])

# --- Cell 22 ---
embedding_dim = 32

# --- Cell 23 ---
model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE+1, embedding_dim, input_length=maxlen),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.30),
        tf.keras.layers.Dense(embedding_dim,activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.30),
        tf.keras.layers.Dense(8,activation='relu'),
        tf.keras.layers.Dense(1,activation='sigmoid'),
    ])

model.compile(loss = 'binary_crossentropy',
                optimizer = 'adam',
                metrics = ['accuracy'])

# --- Cell 24 ---
#summary of the model.
model.summary()

# --- Cell 25 ---
# Training the model we created above. 
history =    model.fit(train_padded_sequences,
                                               y_train,
                                               validation_data = (test_padded_sequences, y_test),
                                               epochs = 3)

# --- Cell 26 ---
print(X_test.iloc[99],'label: ;',y_test[99])

# --- Cell 27 ---
tokenizer.sequences_to_texts(test_padded_sequences)[99]

# --- Cell 28 ---
model.predict(test_padded_sequences)[99]

# --- Cell 29 ---
import matplotlib.pyplot as plt

# Plot utility
def plot_graphs(model, string):
  plt.plot(model.history[string])
  plt.plot(model.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
# Plot the accuracy and loss
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")



if __name__ == "__main__":
    print("Pipeline execution complete.")
