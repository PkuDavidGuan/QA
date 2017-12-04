import pickle
import gensim
import jieba
import numpy
import jieba.posseg as pseg

word_vector_size = 100
sentence_size = 40
padding = [0 for i in range(word_vector_size)]

with open("pos_dict.pkl", "rb") as infile:
    print("Load pos dict...")
    pos_dict = pickle.load(infile)
with open("question.pkl", "rb") as infile:
    print("Load question set...")
    question = pickle.load(infile)
with open("wordvec.pkl", "rb") as infile:
    print("Load word2vec dic...")
    word_vector = pickle.load(infile)
# with open('titles.pickle.1', 'rb') as tfile:
#     print("Update jieba...")
#     titles = pickle.load(tfile)
#     for t in titles:
#         jieba.add_word(t)

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
    y = []
    for i in range(len(pos_dict)):
        y.append(0)
    for w,f in wl:
        if f in pos_dict:
            y[pos_dict[f]] += 1
    sum = 0
    for i in range(len(y)):
        sum += y[i]
    if sum != 0:
        for i in range(len(y)):
            y[i] = y[i] * 1.0 / sum 
    return y

print("Collect date for the model...")
X = []
Y = []
for q,a in question:
    X.append(question4model(q))
    Y.append(answer4model(a))

with open("x.pickle", "wb") as infile:
    pickle.dump(X, infile)
with open("y.pickle", "wb") as infile:
    pickle.dump(Y, infile)