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

word_vector_size = 100
pos_tag_num = 41
sentence_size = 40
padding = [0 for i in range(word_vector_size)]
y_init = [0 for i in range(pos_tag_num)]
with open("answer_pos.pkl", "rb") as infile:
    pos_dict = pickle.load(infile)

def update_jieba():
    print("Update jieba...")
    with open('titles.pickle.1', 'rb') as tfile:
        titles = pickle.load(tfile)
        for t in titles:
            jieba.add_word(t, tag="wiki")

def normalize(y):
    sum = 0
    for i in range(len(y)):
        sum += y[i]
    for i in range(len(y)):
        y[i] = y[i] * 1.0 / sum
    
    return y


def question4model(q):
    wl = jieba.lcut(q)
    x = []
    for w in wl:
        if w in word_vector:
            x.append(word_vector[w])
        else:
            x.append(padding)
    while len(x) < sentence_size:
        x.append(padding)
    return x

def answer4model(a):
    wl = pseg.cut(a)
    y = y_init
    for w,f in wl:
        if f in pos_dict:
            y[pos_dict[f]] += 1
    return normalize(y)

with open("question.pkl", "rb") as infile:
    print("Load question set...")
    question = pickle.load(infile)
with open("wordvec.pkl", "rb") as infile:
    print("Load word2vec dic...")
    word_vector = pickle.load(infile)

update_jieba()

X = []
Y = []
for q,a in question:
    X.append(question4model(q))
    Y.append(answer4model(a))

print("Train the model...")
model = Sequential()
model.add(GRU(output_dim=128,input_dim = word_vector_size, activation='tanh', inner_activation='hard_sigmoid', input_length=sentence_size))
model.add(Dropout(0.5))
model.add(Dense(pos_tag_num, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y)
model.save("answer_class_predict.h5")
    
            

