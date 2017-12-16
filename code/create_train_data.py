# _*_coding=utf-8_*_
import pickle
import jieba
# 这个程序用于从(question, docid, sentence, word, gold_answer)这样的五元组集合中,构造适合模型的输入文件
def get_model_input():
    with open('Data/titles.pickle', 'rb') as tfile:
        titles = pickle.load(tfile)
    for t in titles:
        jieba.add_word(t, tag='wiki')
    punctuation_list = u'.,。，、＇：∶；?‘’“”〝〞ˆˇ﹕︰﹔﹖﹑·¨….¸;！´？！～—ˉ｜‖＂〃｀@﹫¡¿﹏﹋﹌︴々﹟#﹩$﹠&﹪%*﹡﹢﹦﹤‐￣¯―﹨ˆ˜﹍﹎+=<­­＿_-\ˇ~﹉﹊（）〈〉‹›﹛﹜『』〖〗［］《》〔〕{}「」【】︵︷︿︹︽_﹁﹃︻︶︸﹀︺︾ˉ﹂﹄︼'
    other_stop_words = [u' ']
    with open('Data/stop_words.pickle', 'rb') as vfile:
        stop_words = pickle.load(vfile)
    with open('Data/question_sentence20_answer.pickle', 'rb') as qsafile:
        question_sentence5_answer = pickle.load(qsafile)
    total_question_number = 0.0
    completely_word_number = 0.0
    completely_sentence_number = 0.0
    result_question_sentence5_answer = []
    for question_instance in question_sentence5_answer:
        completely_sentence_flag = False
        completely_word_flag = False
        total_question_number += 1
        question = question_instance[0]
        sentences = question_instance[1]
        answer = question_instance[2]
        result_sentence_withtag = []
        for sentence in sentences:
            words = []
            tags = []
            sentence_cut = jieba.lcut(sentence)
            for i in range(len(sentence_cut)):
                word = sentence_cut[i]
                if word in punctuation_list:
                    continue
                if word in stop_words:
                    continue
                if word in other_stop_words:
                    continue
                if word in question:
                    continue
                if word in answer or answer in word:
                    # complete_label = True
                    tag = 1
                else:
                    tag = 0
                words.append((word, i))
                tags.append(tag)
            if answer in sentence_cut:
                completely_word_flag = True
                result_sentence_withtag.append((sentence, words, tags))
            if answer in sentence:
                completely_sentence_flag = True
            if len(words) == 0:
                # words.append(('</s>', -1))
                # tags.append(0)
                continue
        if completely_word_flag:
            completely_word_number += 1
            if len(result_sentence_withtag) == 0:
                continue
            result_question_sentence5_answer.append((question, result_sentence_withtag, answer))
        if completely_sentence_flag:
            completely_sentence_number += 1

    print len(result_question_sentence5_answer), len(question_sentence5_answer)
    with open('Model_input/model_train20_word_input.pickle', 'wb') as vfile:
        pickle.dump(result_question_sentence5_answer, vfile)
    print 'word: ', completely_word_number/total_question_number
    print 'sentence: ', completely_sentence_number/total_question_number
def get_stop_words():
    stop_file = open('Data/stop_words.txt', 'r')
    lines = stop_file.readlines()
    stop_word_list = []
    for line in lines:
        row = line.strip().decode('utf-8')
        stop_word_list.append(row)

    with open('Data/stop_words.pickle', 'wb') as vfile:
        pickle.dump(stop_word_list, vfile)

# get_stop_words()
get_model_input()
