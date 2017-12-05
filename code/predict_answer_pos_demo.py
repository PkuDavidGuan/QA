import pickle
import keras
import jieba
import jieba.posseg as pseg
import numpy as np



with open("answer_pos.pkl", "rb") as infile:
    pos = pickle.load(infile)
with open("wordvec.pkl", "rb") as infile:
    print("Load word2vec dic...")
    word_vector = pickle.load(infile)
# with open('titles.pickle.1', 'rb') as tfile:
#     titles = pickle.load(tfile)
#     for t in titles:
#         jieba.add_word(t)

model = keras.models.load_model("answer_class_predict.h5")

def predict(question, answer, top_k = 3):
    #print("----------------------------------")
    #print(question)
    sentence_size = 40
    word_vector_size = 100
    padding = [0 for i in range(word_vector_size)]
    
    wl = jieba.lcut(question)
    x = []
    for w in wl:
        if w in word_vector:
            x.append(word_vector[w])
        else:
            x.append(padding)
    while len(x) < sentence_size:
        x.append(padding)
    X = np.zeros([1, sentence_size, word_vector_size])
    X[0, :, :] = x
    
    # wl = pseg.cut(answer)
    # s = ""
    # for w, f in wl:
    #     s = s + w + ' ' + f + ' '
    # print(s)

    y = model.predict(X)
    Y = y[0, :]
    ans = {}
    for i in range(len(Y)):
        ans[pos[i]] = Y[i]
    ans = sorted(ans.items(), key=lambda d:d[1], reverse = True)
    # for i in range(top_k):
    #     print(ans[i])
    return ans[0:top_k]

with open("wdm_assignment_3_samples.txt", "r") as infile:
    while True:
        line = infile.readline()
        if not line:
            break
        line = line.strip('\n').strip('\r').split('\t')
        if len(line) != 2:
            continue
        predict(line[0], line[1])