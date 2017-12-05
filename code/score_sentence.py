import pickle
import numpy
from scipy import sparse
import jieba
import jieba.posseg as pseg
import jieba.analyse
import math
import re
import query
import numpy as np
from scipy.spatial.distance import cosine
import sys

def load_word_vectors():
    word_vectors_filename = 'vector.NegUnk900wName.dec'
    fv = open('vector.NegUnk900wName.dec', 'r', encoding='gbk', errors='ignore')
    word2vector = {}
    line = fv.readline()
    line_split = re.split('[\s]', line)
    wordcount = int(line_split[0])
    dim = int(line_split[1])
    for i in range(0, wordcount):
        line = fv.readline()
        line_split = re.split('[\s]+', line)
        word = line_split[0]
        vector = np.array([float(val) for val in line_split[1:dim+1]])
        word2vector[word] = vector
        sys.stdout.write('Processed: {0} / Total: {1}\r'.format(i, wordcount))
        sys.stdout.flush()
    print('')
    return word2vector, dim


if __name__ == '__main__':
    with open('titles.pickle.1', 'rb') as tfile:
        titles = pickle.load(tfile)
    with open('questions.pickle', 'rb') as qf:
        questions = pickle.load(qf)
    with open('answers.pickle', 'rb') as af:
        answers = pickle.load(af)
    with open('candidate_docs.pickle', 'rb') as cf:
        candidate_docs = pickle.load(cf)
    word2vector, dim = load_word_vectors()
    print('Word vectors load.')
    if True:
        with open('score_sentence.txt', 'w') as log:
            for i in range(len(questions)):
                question = questions[i]
                answer = answers[i]
                print(question, ':')
                log.write(question+':\n')
                top_k = candidate_docs[i]
                sentence_score = []
                for id_s in top_k:
                    id = id_s[0]
                    s = id_s[1]
                    # print(id, titles[id - 1], s)
                    # log.write(str(id)+' '+str(titles[id - 1])+' '+str(s)+'\n')
                    cut_question = jieba.lcut(question)
                    q_vec = np.zeros(dim, dtype=float)
                    wordcount = 0
                    for word in cut_question:
                        if word in word2vector:
                            wordcount += 1
                            q_vec += word2vector[word]
                    if wordcount>0:
                        q_vec = q_vec / wordcount
                    with open('cut_documents_new/Doc'+str(id)+'_cut.pickle', 'rb') as docf:
                        doc_cut = pickle.load(docf)
                        doc = ''
                        for word in doc_cut:
                            doc += word
                        sentences = re.split(r'[\f\n\r\t\v。]+', doc)
                        for sentence in sentences:
                            cut_sentence = jieba.lcut(sentence)
                            s_vec = np.zeros(dim, dtype=float)
                            wordcount = 0
                            for word in cut_sentence:
                                if word in word2vector:
                                    wordcount += 1
                                    s_vec += word2vector[word]
                            if wordcount<=0:
                                continue
                            cos_sim = 1 - cosine(q_vec, s_vec)
                            match_score = 0
                            if answer in cut_sentence:
                                match_score = 1
                            score = cos_sim
                            sentence_score.append((titles[id-1], sentence, score))
                sorted_sentence_score = sorted(sentence_score, key=lambda x: x[2], reverse=True)
                sorted_sentence_score = sorted_sentence_score[:10]
                for x in sorted_sentence_score:
                    print('\t',x[0], x[1])
                    print('\t',x[2])
                    log.write('\t'+x[0]+' '+x[1]+'\n\t'+str(x[2])+'\n')
                print('')
                log.write('\n')


