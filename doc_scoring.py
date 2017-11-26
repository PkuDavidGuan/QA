import pickle
import numpy
from scipy import sparse
import jieba
import jieba.posseg as pseg
import re

docnum = 970862
with open('vectorizer.dump','rb') as vfile:
    vectorizer = pickle.load(vfile)
    print('.')
with open('titles.pickle.1','rb') as tfile:
    titles = pickle.load(tfile)
    print('.')

def score(question, docid):
    titlescore = 0
    tfidfscore = 0
    qlist = jieba.lcut(question, cut_all=True)
    if titles[docid - 1] in qlist:
        titlescore = 1
    #print(titles[docid - 1])
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
    return tfidfscore*0.5 + titlescore*0.5

def select_top_k_doc(question, k):
    id_score = []
    for id in range(1, docnum+1):
        id_score.append((id, score(question, id)))
    sorted_id_score = sorted(id_score, key=lambda x: x[1])
    return sorted_id_score[0:k]
