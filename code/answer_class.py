import pickle
import gensim
import jieba
import numpy
import jieba.posseg as pseg
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from sklearn.model_selection import train_test_split
import gc

word_vector_size = 100
sentence_size = 40

with open("pos_dict.pkl", "rb") as infile:
    print("Load pos dict...")
    pos_dict = pickle.load(infile)
print("Load the data for the model...")
with open("x.pickle", "rb") as infile:
    X = pickle.load(infile)
with open("y.pickle", "rb") as infile:
    Y = pickle.load(infile)

train_X, test_X, train_Y, test_Y = train_test_split(X, Y)
print("Train the model...")
model = Sequential()
model.add(GRU(activation="tanh", recurrent_activation="hard_sigmoid", units=128, input_shape=(sentence_size, word_vector_size)))
model.add(Dropout(0.5))
model.add(Dense(len(pos_dict), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, batch_size=32, epochs=25, validation_split=0.1, shuffle=True)
model.save("answer_class_predict.h5")
gc.collect()