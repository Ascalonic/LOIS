import os
import numpy as np
from gensim.models import KeyedVectors
from nltk import word_tokenize

from numpy import array
from numpy import argmax
from keras.utils import to_categorical

model = KeyedVectors.load_word2vec_format('F:\\Machine Learning\\Chatbot\\LOIS\\glove.6B.100d.txt.word2vec', binary=False)

data_dir = 'F:\\Machine Learning\\Chatbot\\LOIS\\dataset'

def get_lines(filename):
    return([line.rstrip('\n') for line in open(filename)])

def vectorize_sent(sentence):
    tokens = word_tokenize(sentence)
    vectors = np.empty((0,100), float)

    for tok in tokens:
        vectors = np.vstack([vectors, model.wv[tok.lower()]])

    print(vectors.shape)
    return(vectors)

def get_dataset():
    X_train = np.empty((32,100,0), float)
    Y_train = np.empty((0), int)
    
    files = sorted(os.listdir(data_dir))
    
    for f in files:
        lines = get_lines(data_dir + '\\' + f)
        for line in lines:
            sent_vec = vectorize_sent(line)
            result = np.zeros((32, 100))
            result[:sent_vec.shape[0], :sent_vec.shape[1]] = sent_vec
            X_train = np.dstack([X_train, result])
            Y_train = np.append(Y_train, int(f[0])-1)

    encoded = to_categorical(Y_train)
    return X_train, encoded


X_train, y_train = get_dataset()


embedding_vecor_length = 100
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)
