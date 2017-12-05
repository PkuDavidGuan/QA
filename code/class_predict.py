import pickle
import keras
import jieba
import jieba.posseg as pseg
import numpy as np
import re

class answerClass:
    sentence_size = 40
    word_vector_size = 100
    padding = [0 for i in range(word_vector_size)]

    def __init__(self, word2vec = None):
        with open("answer_pos.pkl", "rb") as infile:
            print("Load possible pos tags...")
            self.pos = pickle.load(infile)
        if word2vec == None:
            self.word_vector = {}
            fv = open('vector.NegUnk900wName.dec', 'r', encoding='gbk', errors='ignore')
            line = fv.readline()
            line_split = re.split('[\s]', line)
            wordcount = int(line_split[0])
            dim = int(line_split[1])
            for i in range(0, wordcount):
                line = fv.readline()
                line_split = re.split('[\s]+', line)
                word = line_split[0]
                vector = np.array([float(val) for val in line_split[1:dim+1]])
                self.word_vector[word] = vector
        else:
            self.word_vector = word2vec
    
        print("Load rnn model...")
        self.model = keras.models.load_model("answer_class_predict.h5")

    def predict(self, question, top_k = 3):
        #print("----------------------------------")
        #print(question)
        
        wl = jieba.lcut(question)
        x = []
        for w in wl:
            if w in self.word_vector:
                x.append(self.word_vector[w])
            else:
                x.append(answerClass.padding)
        while len(x) < answerClass.sentence_size:
            x.append(answerClass.padding)
        X = np.zeros([1, answerClass.sentence_size, answerClass.word_vector_size])
        X[0, :, :] = x
        
        # wl = pseg.cut(answer)
        # s = ""
        # for w, f in wl:
        #     s = s + w + ' ' + f + ' '
        # print(s)

        y = self.model.predict(X)
        Y = y[0, :]
        ans = {}
        for i in range(len(Y)):
            ans[self.pos[i]] = Y[i]
        ans = sorted(ans.items(), key=lambda d:d[1], reverse = True)
        # for i in range(top_k):
        #     print(ans[i])
        return ans[0:top_k]

# with open('titles.pickle.1', 'rb') as tfile:
#     titles = pickle.load(tfile)
#     for t in titles:
#         jieba.add_word(t)





if __name__ == '__main__':
    with open("wdm_assignment_3_samples.txt", "r") as infile:
        ac = answerClass()
        while True:
            line = infile.readline()
            if not line:
                break
            line = line.strip('\n').strip('\r').split('\t')
            if len(line) != 2:
                continue
            print(ac.predict(line[0]))