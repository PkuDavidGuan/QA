#-*- coding: utf-8 -*-
'''
Learn the maxinum number of words in a question and the number of POS-tag in the answer set.abs
'''
import pickle
import jieba
import jieba.posseg as pseg

def update_jieba():
    print("Update jieba...")
    with open('titles.pickle.1', 'rb') as tfile:
        titles = pickle.load(tfile)
        for t in titles:
            jieba.add_word(t, tag="wiki")

def realword(f):
    flag = True
    if(f[0]=='u' or f[0]=='c' or f[0]=='x'):
        flag = False
    return flag



with open("question.pkl", "rb") as infile:
    question = pickle.load(infile)
update_jieba()

qw_len = 0
pos = ['wiki']
#known_pos = ['wiki', 'ns', 'nr', 'n', 'ng','nrfg', 'nz', 'nrt', 'm', 'c', 'x']

for q,a in question:
    wl = jieba.lcut(q)
    if(len(wl) > qw_len):
        qw_len = len(wl)

    wl = pseg.cut(str(a))
    for w, f in wl:
        if f not in pos:
            if realword(f):
                pos.append(f)

print("The maxinum number of words in the question: %d" % (qw_len))
print("The number of POS tag: %d" % (len(pos)))
print(pos)

pos_dict = {}
for i in range(len(pos)):
    pos_dict[pos[i]] = i
with open("answer_pos.pkl", "wb") as infile:
    pickle.dump(pos_dict, infile)

