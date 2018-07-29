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
import h5py
# define documents

NUM_CLASSES = 4
MAX_WORDS = 32

data_dir = 'F:\\Machine Learning\\Chatbot\\LOIS\\dataset'


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


docs, labels = get_dataset()
labels = to_categorical(labels)

# prepare tokenizer
t = Tokenizer()
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1
# integer encode the documents
encoded_docs = t.texts_to_sequences(docs)
print(encoded_docs)
# pad documents to a max length of 32 words
max_length = 32
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)
# load the whole embedding into memory
embeddings_index = dict()
f = open('F:\\Machine Learning\\Chatbot\\LOIS\\glove.6B.100d.txt', encoding="utf8")
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
# define model
model = Sequential()
e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_length, trainable=False)
model.add(e)
model.add(Flatten())
model.add(Dense(NUM_CLASSES, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
# summarize the model
print(model.summary())
# fit the model
model.fit(padded_docs, labels, epochs=50, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))

model.save('word_class.h5')

test_txt = array(['How much gravity on the moon'])
test_encoded = t.texts_to_sequences(test_txt)
test_padded = pad_sequences(test_encoded, maxlen=max_length, padding='post')

print(model.predict(test_padded))
