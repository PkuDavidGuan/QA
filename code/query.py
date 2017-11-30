import pickle
import numpy
from scipy import sparse
import jieba
import jieba.posseg as pseg
import math
import re
class Query:
    def __init__(self):
        self.docnum = 970862
        with open('vectorizer.dump','rb') as vfile:
            self.vectorizer = pickle.load(vfile)
            print('.')
        with open('titles.pickle.1','rb') as tfile:
            self.titles = pickle.load(tfile)
        with open('title2id.pickle', 'rb') as tifile:
            self.title2id = pickle.load(tifile)
            print('.')
        with open('doc_vectors.pickle', 'rb') as dvfile:
            self.doc_vectors = pickle.load(dvfile)
            print('.')
        with open('vocabulary2inverted_index.pickle', 'rb') as vifile:
            self.vocabulary2inverted_index = pickle.load(vifile)
            print('.')

    def score(self, question, docid):
        qlist_pos = pseg.cut(question)
        qlist_real_words = []
        for word, flag in qlist_pos:
            if 'n' in flag or flag[0] == 'l' or flag[0] == 'i':
                qlist_real_words.append(word)
        qlist = jieba.lcut(question, cut_all=False) #????
        titlescore = 0
        tfidfscore = 0
        if self.titles[docid - 1] in qlist_real_words:
            titlescore = 1
        q = ''
        for word in qlist:
            q += word+' '
        qcorpus = [q]
        #print('.', end='')
        qvec = self.vectorizer.transform(qcorpus)
        #print('.', end='')
        dvec = self.doc_vectors[docid-1, :]
        #print('.', end='')
        q0 = math.sqrt(qvec.power(2).sum())
        d0 = math.sqrt(dvec.power(2).sum())
        tfidfscore = qvec.multiply(dvec).sum() / (q0*d0)
        #print('.', end='')
        return tfidfscore

    def select_top_k_doc_term_filter(self, question, k):
        qlist_pos = pseg.lcut(question)
        qlist_real_words = []
        for word, flag in qlist_pos:
            if 'n' in flag or flag[0] == 'l' or flag[0] == 'i':
                qlist_real_words.append(word)
        wordcount = 0
        common_doc_ids = set([])
        for word in qlist_real_words:
            if word in self.vocabulary2inverted_index:
                if wordcount == 0:
                    common_doc_ids = set(self.vocabulary2inverted_index[word])
                else:
                    common_doc_ids = common_doc_ids.intersection(set(self.vocabulary2inverted_index[word]))
                wordcount += 1
        #filt more candidates
        if len(common_doc_ids) > 100:
            qlist_real_words_new = []
            for word, flag in qlist_pos:
                if flag[0] == 't' or flag[0] == 'm' or flag[0] == 'f' \
                        or flag[0] == 'a' or flag[0] == 'd' or flag[0] == 'v':
                    qlist_real_words_new.append(word)
            for word in qlist_real_words_new:
                if word in self.vocabulary2inverted_index:
                    common_doc_ids = common_doc_ids.intersection(set(self.vocabulary2inverted_index[word]))
        #filt over
        candidates = common_doc_ids
        if len(common_doc_ids) > 1000:
            candidates = set([])
        print('length of common_doc_ids', len(common_doc_ids))
        qlist = jieba.lcut(question, cut_all=True)
        for word in qlist:
            if word in self.title2id:
                candidates.add(self.title2id[word])
        id_score = []
        for id in candidates:
            s = self.score(question, id)
            id_score.append((id, s))
        #print('')
        sorted_id_score = sorted(id_score, key=lambda x: x[1], reverse=True)
        if len(sorted_id_score)<k:
            return sorted_id_score
        return sorted_id_score[0:k]

query = Query()
with open('wdm_assignment_3_samples.txt', 'r') as qfile:
    with open('querylog.txt', 'w') as log:
        while True:
            line = qfile.readline()
            if len(line) <= 1:
                break
            split_line = line.split('\t')
            question = split_line[0]
            print(question, ':')
            log.write(question+':\n')
            top_k = query.select_top_k_doc_term_filter(question, 5)
            for id_s in top_k:
                id = id_s[0]
                s = id_s[1]
                print(id, query.titles[id - 1], s)
                log.write(str(id)+' '+str(query.titles[id - 1])+' '+str(s)+'\n')

