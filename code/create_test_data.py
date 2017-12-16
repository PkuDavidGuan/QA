# _*_coding=utf-8_*_
import pickle


# 这个程序用于从(question, docid, sentence, word, gold_answer)这样的五元组集合中,构造适合模型的输入文件
def get_model_input():
    punctuation_list = u'.,。，、＇：∶；?‘’“”〝〞ˆˇ﹕︰﹔﹖﹑·¨….¸;！´？！～—ˉ｜‖＂〃｀@﹫¡¿﹏﹋﹌︴々﹟#﹩$﹠&﹪%*﹡﹢﹦﹤‐￣¯―﹨ˆ˜﹍﹎+=<­­＿_-\ˇ~﹉﹊（）〈〉‹›﹛﹜『』〖〗［］《》〔〕{}「」【】︵︷︿︹︽_﹁﹃︻︶︸﹀︺︾ˉ﹂﹄︼'
    other_stop_words = [u' ']
    with open('stop_words.pickle', 'rb') as vfile:
        stop_words = pickle.load(vfile)
    infile = open('question_candidate_200.txt', 'r')
    infile_answers = open('answers_200', 'r')
    lines = infile_answers.readlines()
    answers = []
    for line in lines:
        answers.append(line.strip().decode('utf-8'))
    print 'get answers over'
    infile_answers.close()
    lines = infile.readlines()
    print len(lines)
    data_instance = []  # (question, [(sentence, [words], [tags])], answers)
    question_index = 0
    instances = []
    for line in lines:
        line = line.decode('utf-8')
        if line.strip()=='':
            continue
        row = line.strip().split('|@|')
        instances.append(row)
    print 'get instances over'
    current_question = instances[0][0]
    current_question_tuple_list = []
    for instance in instances:
        if instance[0] == current_question:
            current_question_tuple_list.append(instance[2:])
        else:
            # print question_index
            data_instance.append((current_question, current_question_tuple_list, answers[question_index]))
            question_index += 1
            current_question = instance[0]
            current_question_tuple_list = []
            current_question_tuple_list.append(instance[2:])
    print 'get question over'
    result_data_instance = []
    for instance in data_instance:
        question = instance[0]
        answer = instance[2]
        current_sentence = instance[1][0][0]
        current_sentence_word_list = []
        sentence_data_instance = []
        for sentence_instance in instance[1]:
            if sentence_instance[0]==current_sentence:
                current_sentence_word_list.append(sentence_instance[1])
            else:
                sentence_data_instance.append((current_sentence, current_sentence_word_list))
                current_sentence = sentence_instance[0]
                current_sentence_word_list = []
                current_sentence_word_list.append(sentence_instance[1])
        result_data_instance.append((question, sentence_data_instance, answer))
    print 'get sentence over'
    # 打标签
    # Min = 10
    result_instances_with_tag = []
    question_number = 0.0
    completely_right = 0.0
    for instance in result_data_instance:
        question_number += 1
        if question_number%100==0:
            print question_number
        complete_label = False
        question = instance[0]
        answer = instance[2]
        sentences = []
        for sentence in instance[1]:
            sentence_text = sentence[0]
            # if answer in sentence_text:
            #     complete_label = True
                # print 'question',question
                # print 'answer', answer
                # print 'sentence', sentence_text
            words = []
            tags = []
            if answer in sentence[1]:
                for word in sentence[1]:
                    if word == answer:
                        complete_label = True
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
                    words.append(word)
                    tags.append(tag)
                if len(words)==0:
                    words.append('</s>')
                    tags.append(0)
                sentences.append((sentence_text, words, tags))
        if complete_label:
            completely_right += 1
            result_instances_with_tag.append((question, sentences, answer))
    print 'completely right rate:', completely_right/question_number
    print len(result_instances_with_tag)
    with open('model_test_input.pickle', 'wb') as vfile:
        pickle.dump(result_instances_with_tag, vfile)
def get_stop_words():
    stop_file = open('stop_words.txt', 'r')
    lines = stop_file.readlines()
    stop_word_list = []
    for line in lines:
        row = line.strip().decode('utf-8')
        stop_word_list.append(row)

    with open('stop_words.pickle', 'wb') as vfile:
        pickle.dump(stop_word_list, vfile)

# get_stop_words()
get_model_input()
