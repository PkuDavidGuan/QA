# _*_coding=utf-8_*_
import jieba
import pickle
from Model import train, test, train_model_save, train_model_import
from get_word_embedding import word_to_vector_dict
import numpy as np
import sys

punctuation_list = u'.,。，、＇：∶；?‘’“”〝〞ˆˇ﹕︰﹔﹖﹑·¨….¸;！´？！～—ˉ｜‖＂〃｀@﹫¡¿﹏﹋﹌︴々﹟#﹩$﹠&﹪%*﹡﹢﹦﹤‐￣¯―﹨ˆ˜﹍﹎+=<­­＿_-\ˇ~﹉﹊（）〈〉‹›﹛﹜『』〖〗［］《》〔〕{}「」【】︵︷︿︹︽_﹁﹃︻︶︸﹀︺︾ˉ﹂﹄︼'
stop_words = [u'什么', u'多少', u'怎么', u'几个', u'几']

def get_vector(sentences, sentences_candidate_words, question, gold_tags):
    sentences_words = []
    for sentence in sentences:
        words = jieba.lcut(sentence)
        sentences_words.append(words)
    sentences_vectors = []
    for sentence in sentences_words:
        sentence_vectors = []
        for word in sentence:
            if word in punctuation_list:
                continue
            if word in word_to_vector_dict:
                vector = word_to_vector_dict[word]
            else:
                vector = word_to_vector_dict['</s>']
            sentence_vectors.append(vector)
        sentences_vectors.append(sentence_vectors)
    question_words = jieba.lcut(question)
    sentences_candidate_words_vectors = []
    sentences_candidate_words_feature_vectores = []
    for sentence in sentences_candidate_words:
        sentence_vectors = []
        sentence_feature_vectores = []
        for word in sentence:
            if word in question_words:
                sentence_feature_vectores.append([0,1])
            else:
                sentence_feature_vectores.append([1,0])
            if word in word_to_vector_dict:
                vector = word_to_vector_dict[word]
            else:
                vector = word_to_vector_dict['</s>']
            sentence_vectors.append(vector)
        sentences_candidate_words_vectors.append(sentence_vectors)
        sentences_candidate_words_feature_vectores.append(sentence_feature_vectores)

    question_vectors = []

    for word in question_words:
        if word in punctuation_list:
            continue
        if word in stop_words:
            continue
        if word in word_to_vector_dict:
            vector = word_to_vector_dict[word]
        else:
            vector = word_to_vector_dict['</s>']
        question_vectors.append(vector)
    tag_vectors = []
    for tag in gold_tags:
        if tag==0:
            vector = [1, 0]
        else:
            vector = [0, 1]
        tag_vectors.append(vector)
    return sentences_vectors, sentences_candidate_words_vectors, [question_vectors], tag_vectors, sentences_candidate_words_feature_vectores

# load model_input.pickle
with open("model_input.pickle", "rb") as infile:
    print("Load pos dict...")
    model_input = pickle.load(infile)

# print len(model_input)
train_input = model_input[0:150]
test_input = model_input[150:]

def Train(max_iteration):
    for iter in range(max_iteration):
        print iter, ' begin:'
        log_file = open('log', 'a+')
        log_file.write('iteration ' + str(iter) + ' begin\n')
        log_file.close()
        all_count = 0.0
        right_count = 0.0
        zero_count = 0.0
        all_one_count = 0.1
        right_one_count = 0.0
        for question_instance in train_input:
            question = question_instance[0]
            answer = question_instance[2]
            sentences_words_tags = question_instance[1]
            sentences = [sentence[0] for sentence in sentences_words_tags]
            sentences_candidate_words = []
            gold_tags = []
            candidate_words_list = []
            for sentence in sentences_words_tags:
                sentence_words = []
                for i in range(len(sentence[1])):
                    word = sentence[1][i]
                    tag = sentence[2][i]
                    sentence_words.append(word)
                    gold_tags.append(tag)
                    candidate_words_list.append(word)
                sentences_candidate_words.append(sentence_words)
            sentences_input, sentences_candidate_words_input, question_input, gold_tags_input, sentences_candidate_words_feature_input \
                = get_vector(sentences, sentences_candidate_words, question, gold_tags)
            score = train(sentences_input, sentences_candidate_words_input, question_input, gold_tags_input, sentences_candidate_words_feature_input)
            result_index = np.argmax(score, -1)
            for i in range(len(result_index)):
                if gold_tags[i]==0:
                    zero_count += 1
                if gold_tags[i]==result_index[i]:
                    right_count += 1
                all_count += 1
                if gold_tags[i] == 1:
                    all_one_count += 1
                    if result_index[i] == 1:
                        right_one_count += 1
        log_file = open('log', 'a+')
        log_file.write('acc: ' + str(right_count/all_count) + '\n' + 'zero_rate: ' + str(zero_count/all_count) + '\n')
        log_file.write('one_rate: ' + str(right_one_count/all_one_count) + '\n')
        log_file.close()
        train_model_save(iter)

def Test(iter):
    savedStdout = sys.stdout  # 保存标准输出流
    with open('test_out/out_train'+str(iter)+'.txt', 'w') as file:
        sys.stdout = file  # 标准输出重定向至文件
    #     print 'This message is for file!'
    #
    # sys.stdout = savedStdout  # 恢复标准输出流
    # print 'This message is for screen!'
        train_model_import(iter)
        all_count = 0.0
        right_count = 0.0
        zero_count = 0.0
        all_one_count = 0.1
        right_one_count = 0.0
        question_number = 0.0
        complete_match = 0.0
        has_match = 0.0
        top5_complete_match = 0.0
        top5_has_match = 0.0
        for question_instance in train_input:
            question = question_instance[0]
            answer = question_instance[2]
            sentences_words_tags = question_instance[1]
            sentences = [sentence[0] for sentence in sentences_words_tags]
            sentences_candidate_words = []
            gold_tags = []
            candidate_words_list = []
            for sentence in sentences_words_tags:
                sentence_words = []
                for i in range(len(sentence[1])):
                    word = sentence[1][i]
                    tag = sentence[2][i]
                    sentence_words.append(word)
                    gold_tags.append(tag)
                    candidate_words_list.append(word)
                sentences_candidate_words.append(sentence_words)
            sentences_input, sentences_candidate_words_input, question_input, gold_tags_input, sentences_candidate_words_feature_input\
                = get_vector(sentences, sentences_candidate_words, question,  gold_tags)
            score = test(sentences_input, sentences_candidate_words_input, question_input, sentences_candidate_words_feature_input)
            # print score
            true_score = score[:, 1]
            max_index = np.argmax(true_score, 0)
            score_dict = {}
            for i in range(len(true_score)):
                score_dict[i] = true_score[i]
            sorted_score_tuple = sorted(score_dict.items(), key=lambda x:x[1], reverse=True)
            if len(sorted_score_tuple)>=5:
                top5_index =  [sorted_score_tuple[i][0] for i in range(5)]
            else:
                top5_index = [sorted_score_tuple[0][0] for i in range(5)]
            # print max_index
            print 'question: ', question.encode('utf-8')
            print 'answer: ', answer.encode('utf-8')
            print 'predict top5 answer: ',
            for i in range(5):
                print candidate_words_list[top5_index[i]].encode('utf-8'), '@',
            print '\n'
            question_number += 1
            if answer==candidate_words_list[max_index]:
                complete_match += 1
            if answer in candidate_words_list[max_index] or candidate_words_list[max_index] in answer:
                has_match += 1
            top5_complete_match_bool = False
            top5_has_match_bool = False
            for i in range(5):
                candidate_answer = candidate_words_list[top5_index[i]]
                if answer == candidate_answer:
                    top5_complete_match_bool = True
                if answer in candidate_answer or candidate_answer in answer:
                    top5_has_match_bool = True
            if top5_complete_match_bool:
                top5_complete_match += 1
            if top5_has_match_bool:
                top5_has_match += 1
            result_index = np.argmax(score, -1)
            for i in range(len(result_index)):
                if gold_tags[i] == 0:
                    zero_count += 1
                if gold_tags[i] == result_index[i]:
                    right_count += 1
                all_count += 1
                if gold_tags[i]==1:
                    all_one_count += 1
                    if result_index[i]==1:
                        right_one_count += 1
        print 'acc:', right_count / all_count
        print 'zero rate: ', zero_count / all_count
        print 'one rate: ', right_one_count/all_one_count
        print 'complete_match rate', complete_match/question_number
        print 'has_match rate', has_match/question_number
        print 'top5_complete_match rate', top5_complete_match/question_number
        print 'top5_has_match rate', top5_has_match/question_number

# Train(20)
for i in range(20):
    Test(i)