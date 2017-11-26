import pickle
import numpy
from scipy import sparse
import jieba
import jieba.posseg as pseg
import re

docnum = 40
with open('vectorizer.dump','rb') as vfile:
    vectorizer = pickle.load(vfile)
    print('.')
with open('titles.pickle.1','rb') as tfile:
    titles = pickle.load(tfile)
    print('.')

def score(qlist, docid):
    tfidfscore = 0
    q = ''
    for word in qlist:
        q += word+' '
    qcorpus = [q]
    qvec = vectorizer.transform(qcorpus)
    with open('cut_documents_new/Doc'+str(docid)+'_cut.pickle', 'rb') as dfile:
        doc_list = pickle.load(dfile)
        doc = ''
        for word in doc_list:
            doc += word + ' '
        dcorpus = [doc]
        dvec = vectorizer.transform(dcorpus)
        q0 = qvec.power(2).sum()
        d0 = dvec.power(2).sum()
        tfidfscore = qvec.multiply(dvec).sum() / (q0*d0)
        #print(q0,d0,tfidfscore)
    return tfidfscore

def select_top_k_doc(question, k):
    qlist = jieba.lcut(question, cut_all=True)
    candidates_id = []
    for id in range(1, docnum + 1):
        if titles[id - 1] in qlist:
            candidates_id.append(id)
    id_score = []
    for id in candidates_id:
        id_score.append((id, score(qlist, id)))
    sorted_id_score = sorted(id_score, key=lambda x: x[1], reverse=True)
    return sorted_id_score[0:k]

print(select_top_k_doc('中医学起源于什么年代', 5))
