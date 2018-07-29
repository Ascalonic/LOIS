import os
from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.utils import to_categorical
from keras.models import load_model
import h5py
import numpy as np

from nltk import pos_tag, word_tokenize
from datetime import datetime

model = Sequential()
model = load_model('word_class.h5')

t = Tokenizer()

max_length = 32

data_dir = 'dataset'

weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']


def get_lines(filename):
    return([line.rstrip('\n') for line in open(filename)])

def get_dataset():
    docs = []
    labels = []
    files = sorted(os.listdir(data_dir))
    
    for f in files:
        lines = get_lines(data_dir + '\\' + f)
        for line in lines:
            docs.append(line)
            labels.append(int(f[0])-1)

    return docs, labels

def init_tokenizer():
    docs, labels = get_dataset()
    t.fit_on_texts(docs)
    vocab_size = len(t.word_index) + 1
    # load the whole embedding into memory
    embeddings_index = dict()
    f = open('glove.6B.100d.txt', encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))
    # create a weight matrix for words in training sentences
    embedding_matrix = zeros((vocab_size, 100))
    for word, i in t.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

def get_sent_class(sentence):  
    test_txt = array([sentence])
    test_encoded = t.texts_to_sequences(test_txt)
    test_padded = pad_sequences(test_encoded, maxlen=max_length, padding='post')
    scores = model.predict(test_padded)[0]
    scores = np.array(scores)
    return(np.argmax(scores))

def extract_nouns(sent):
    ret = []
    tagged = pos_tag(word_tokenize(sent))
    for pair in tagged:
        if pair[1] == 'NN':
            ret.append(pair[0])

    return ret

def get_date_func(nouns):
    now = datetime.now()
    reply = ''
    for noun in nouns:
        if noun=='year':
            reply += 'The year is ' + str(now.year) + '. '
        elif noun=='month':
            reply += 'Month is ' + now.strftime('%B') + '. '
        elif noun=='day':
            reply += 'It is ' + weekdays[datetime.today().weekday()] + '. '

    if reply=='':
        return('The date is ' + datetime.now().strftime('%Y-%m-%d'))
    else:
        return reply

def bot_respond(q):

    sent_class = get_sent_class(q)

    if(q=='bye'):
        return('exit', 'ok bye. Nice talking to you :)')
    
    if sent_class==0:
        return('', 'Hi there!')
    elif sent_class==1:
        return('', get_date_func(extract_nouns(q)))
    elif sent_class==2:
        return('', 'The time is ' + datetime.now().strftime('%H:%M:%S'))


init_tokenizer()

action = ''
reply = ''

while(action!='exit'):

    query = input('You:')
    action, reply = bot_respond(query)

    print('Lois : ' + reply)
