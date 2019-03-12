# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 19:21:21 2019

@author: ASUS
"""
import tensorflow as tf
import numpy as np


from scipy.spatial.distance import cdist


from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


import imdb2



def token_to_string(tokens):
    
    words = [ inverse_map[token] for token in tokens if token != 0 ]
    
    text = " ".join(words)
    
    return text

# never compile this cell ever again
imdb2.maybe_download_and_extract()

X_train, Y_train = imdb2.load_data(train = True)
X_test, Y_test = imdb2.load_data(train = False)

#
raw_text_total = X_train + X_test

# Tokenizer THIS IS THE PART WHERE 
# THE TEXT GETS MAPPED INTO TOKENS IN A DICT

number_of_words = 10000 # MEAN AMOUT

# THIS IS WHERE THE DICTINARY IS MADE I.E. TOKENIZER.WORD_INDEX
tokenizer = Tokenizer(number_of_words)
tokenizer.fit_on_texts(raw_text_total)

if number_of_words is None:
    number_of_words = len(tokenizer.word_index)
    
# CONVERTING ALL THE TEXTS IN THE TRAINING SET TO LIST OF TOKENS
X_train_tokens = tokenizer.texts_to_sequences(X_train)

# CONVERTING THE TEXT IN THE TEST SET TO TOKENS
X_test_tokens = tokenizer.texts_to_sequences(X_test)

# NOW WE PAD OUT NP ARRAY TO ZEROS 1ST
pad = 'pre'

num_tokens = [len(tokens) for tokens in X_train_tokens + X_test_tokens]
num_tokens = np.array(num_tokens)

max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
max_tokens = int(max_tokens)

X_train_pad = pad_sequences(X_train_tokens, maxlen = max_tokens, 
                           padding = pad, truncating = pad)

X_test_pad = pad_sequences(X_test_tokens, maxlen=max_tokens,
                           padding=pad, truncating=pad)

index = tokenizer.word_index
inverse_map = dict(zip(index.values(), 
                       index.keys()))

# THIS IS WHERE WE CREATE THE RECURRENT NEURAL NETOWRK MODEL

model = Sequential()

embedding_size = 8

model.add(Embedding(input_dim = number_of_words,
                    output_dim = embedding_size,
                    input_length = max_tokens,
                    name = 'layer_embedding'))


model.add(GRU(units=16, return_sequences=True))

model.add(GRU(units=16, return_sequences=True))

model.add(GRU(units=4))

model.add(Dense(1, activation='sigmoid'))

optimizer = Adam(lr=1e-3)

model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

model.fit(X_train_pad, Y_train,
          validation_split=0.05, epochs=3, batch_size=64)

result = model.evaluate(X_test_pad, Y_test)
#



print("Accuracy: {0:.2%}".format(result[1]))

Y_pred = model.predict(x=X_test_pad[0:1000])
Y_pred = Y_pred.T[0]

cls_pred = np.array([1.0 if p>0.5 else 0.0 for p in Y_pred])
cls_true = np.array(Y_test[0:1000])
incorrect = np.where(cls_pred != cls_true)
incorrect = incorrect[0]

idx = incorrect[0]

text1 = "This movie is fantastic! I really like it because it is so good!"
text2 = "Good movie!"
text3 = "Maybe I like this movie."
text4 = "Meh ..."
text5 = "If I were a drunk teenager then this movie might be good."
text6 = "Bad movie!"
text7 = "Not a good movie!"
text8 = "Don’t watch this under ANY circumstance; unless your black out drunk or baked out of your skull because those are the ONLY two ways you will ever enjoy this ungodly junk pile"
texts = [text1, text2, text3, text4, text5, text6, text7, text8]

tokens = tokenizer.texts_to_sequences(texts)

tokens_pad = pad_sequences(tokens, maxlen=max_tokens,
                           padding=pad, truncating=pad) 

print(model.predict(tokens_pad))

model.save("sentiment-CNN.model")











