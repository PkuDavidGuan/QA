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
# import class_predict
with open('stop_words.pickle', 'rb') as vfile:
    stop_words = pickle.load(vfile)
punctuation_list = u'.,。，、＇：∶；?‘’“”〝〞ˆˇ﹕︰﹔﹖﹑·¨….¸;！´？！～—ˉ｜‖＂〃｀@﹫¡¿﹏﹋﹌︴々﹟#﹩$﹠&﹪%*﹡﹢﹦﹤‐￣¯―﹨ˆ˜﹍﹎+=<­­＿_-\ˇ~﹉﹊（）〈〉‹›﹛﹜『』〖〗［］《》〔〕{}「」【】︵︷︿︹︽_﹁﹃︻︶︸﹀︺︾ˉ﹂﹄︼'
class CandidateAnsers:
    def __init__(self):
        pass
        # word_vectors_filename = 'vector.NegUnk900wName.dec'
        # self.word2vector = {}
        # with open('titles.pickle.1', 'rb') as tfile:
            # self.titles = pickle.load(tfile)
        # with open('vector.NegUnk900wName.dec', 'r', encoding='gbk', errors='ignore') as fv:
        #     line = fv.readline()
        #     line_split = re.split('[\s]', line)
        #     wordcount = int(line_split[0])
        #     self.dim = int(line_split[1])
        #     for i in range(0, wordcount):
        #         line = fv.readline()
        #         line_split = re.split('[\s]+', line)
        #         word = line_split[0]
        #         vector = np.array([float(val) for val in line_split[1:self.dim + 1]])
        #         self.word2vector[word] = vector
        #         sys.stdout.write('Processed: {0} / Total: {1}\r'.format(i, wordcount))
        #         sys.stdout.flush()
        # print('')
        # print('Word vectors load.')
        # answerClass = class_predict.answerClass(word2vec=self.word2vector)

    def generate_candidate_answers(self, ques, top_k_candidate_docs, answer):
        # answer_tag_scores = aClassifier.predict(ques)
        # answer_tag = [x[0] for x in answer_tag_scores]
        # question_tag_words = jieba.analyse.extract_tags(question, topK=5)
        cut_question = jieba.lcut(ques)
        # q_vec = np.zeros(self.dim, dtype=float)
        # wordcount = 0
        # for word in cut_question:
        #     if word in self.word2vector:
        #         wordcount += 1
        #         q_vec += self.word2vector[word]
        # if wordcount > 0:
        #     q_vec = q_vec / wordcount
        answer_in_candidate_docs = False
        sentence_score = []
        for id_s in top_k_candidate_docs:
            id = id_s[0]
            #s = id_s[1]
            # print('answer', answer)
            # print('s', s)
            # if answer in s:
            #     answer_in_candidate_docs = True
            # print(id, titles[id - 1], s)
            # log.write(str(id)+' '+str(titles[id - 1])+' '+str(s)+'\n')

            with open('cut_documents_new/Doc' + str(id) + '_cut.pickle', 'rb') as docf:
                doc_cut = pickle.load(docf)
                doc = ''
                for word in doc_cut:
                    doc += word
                if answer in doc:
                    answer_in_candidate_docs = True
                else:
                	continue
                sentences = re.split(r'[\f\n\r\t\v。]+', doc)
                for sentence in sentences:
                    score = 0.0
                    # cut_sentence = jieba.lcut(sentence)
                    # sentence_tag_words = jieba.analyse.extract_tags(sentence, topK=5)
                    # s_vec = np.zeros(self.dim, dtype=float)
                    # wordcount = 0
                    # for word in cut_sentence:
                    #     if word in self.word2vector:
                    #         wordcount += 1
                    #         s_vec += self.word2vector[word]
                    # if wordcount <= 0:
                    #     continue
                    # cos_sim = 1 - cosine(q_vec, s_vec)
                    for word in cut_question:
                        if word in stop_words:
                            continue
                        if word in punctuation_list:
                            continue
                        if word in sentence:
                            score += 1
                    # score = cos_sim
                    sentence_score.append((id, sentence, score))
        if len(sentence_score)==0:
        	sorted_sentence_score = []
        else:
        	sorted_sentence_score = sorted(sentence_score, key=lambda x: x[2], reverse=True)
        sorted_sentence_score = sorted_sentence_score[:20]
        candidate_answers = []
        for x in sorted_sentence_score:
            docid = x[0]
            sentence = x[1]
            cut_sentence_pos = pseg.lcut(sentence)
            position = 0
            cut_sentence = [word for word, flag in cut_sentence_pos]
            for answord, flag in cut_sentence_pos:
                # if flag in answer_tag:
                candidate_answers.append((docid, cut_sentence, answord, flag, position))
                position += 1
        return candidate_answers, answer_in_candidate_docs


if __name__ == '__main__':
    with open('titles.pickle.1', 'rb') as tfile:
        titles = pickle.load(tfile)
    with open('questions.raw.pickle', 'rb') as qf:
        questions = pickle.load(qf)
    with open('answers.raw.pickle', 'rb') as af:
        answers = pickle.load(af)
        # print ('answer', answers[0])
        # outfile = open('answers_1000', 'w')
        # for answer in answers[0:1000]:
        #     outfile.write(answer+'\n')
        # outfile.close()
    with open('candidate_docs.raw.pickle', 'rb') as cf:
        candidate_docs = pickle.load(cf)
    print('Word vectors load.')
    ca = CandidateAnsers()
    # answerClassifier = class_predict.answerClass(word2vec=ca.word2vector)
    with open('question_candidate_raw_out.txt', 'wb') as outfile:
        if True:
            if True:
                question_number = 0.0
                answer_in_docs = 0.0
                answer_outfile = open('answers_raw_out', 'w')
                for i in range(len(questions)):
                    if i%100==0:
                        print(i)
                    question_number += 1
                    question = questions[i]
                    answer = answers[i]
                    # answer_tag_scores = answerClassifier.predict(question)
                    # answer_tag = [x[0] for x in answer_tag_scores]
                    top_k = candidate_docs[i]
                    candidate_answers, flag = ca.generate_candidate_answers(question, top_k, answer)
                    if flag:
                        answer_in_docs += 1
                    else:
                        continue
                    answer_outfile.write(answer + '\n')
                    if len(candidate_answers)==0:
                        outfile.write(bytes('None\n', 'UTF-8'))
                    for cand in candidate_answers:
                        outfile.write(bytes(question+'|@|'+str(cand[0])+'|@|'+str(''.join(cand[1]))+'|@|'+str(cand[2])+'|@|'+str(cand[3])+'|@|'+str(cand[4])+'\n', 'UTF-8'))
                        # outfile.write(question+'|@|'+str(cand[0])+'|@|'+str(cand[1])+'|@|'+str(cand[2])+'|@|'+str(cand[3])+'\n')
                    outfile.write(bytes('\n', 'UTF-8'))
                print ('answer_in_docs rate:', answer_in_docs/question_number)
    outfile.close()
    answer_outfile.close()



