# _*_coding=utf-8_*_
# 加入attention机制,并且加入question embedding的信息
import tensorflow as tf
import os
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import LSTMStateTuple
import numpy as np

word_dim = 300
rnn_dim = 100
stddev_setting = 0.001
all_dim = 3*rnn_dim
# 输入
sentence = tf.placeholder(tf.float32, [1, None, word_dim], name='sentence_input')
sentence_word_candidate = tf.placeholder(tf.float32, [1, None, word_dim], name='candidate_input')
# sentence_word_candidate_feature = tf.placeholder(tf.float32, [1, None, 2])
question = tf.placeholder(tf.float32, [1, None, word_dim], name='question_input')
gold_tag = tf.placeholder(tf.float32, [None, 2])
# model
feature_para = tf.Variable(tf.truncated_normal(shape=[2, 100], stddev=stddev_setting))
# feature_embedding = tf.matmul(sentence_word_candidate_feature[0], feature_para)
Attention_Weight = {
    # sentence para
    'h_sentence':tf.Variable(tf.truncated_normal(shape=[1, rnn_dim], stddev=stddev_setting)),
    'W_sentence':tf.Variable(tf.truncated_normal(shape=[rnn_dim, rnn_dim], stddev=stddev_setting)),
    'b_sentence':tf.Variable(tf.truncated_normal(shape=[rnn_dim], stddev=stddev_setting)),

    # question para
    'h_question':tf.Variable(tf.truncated_normal(shape=[1, rnn_dim], stddev=stddev_setting)),
    'W_question':tf.Variable(tf.truncated_normal(shape=[rnn_dim, rnn_dim], stddev=stddev_setting)),
    'b_question':tf.Variable(tf.truncated_normal(shape=[rnn_dim], stddev=stddev_setting))
}
def get_cos_value(x, y):
    y2 = tf.transpose(y, [1,0])
    x_normal = tf.nn.l2_normalize(x, 1)
    y_normal = tf.nn.l2_normalize(y2, 0)
    return tf.matmul(x_normal, y_normal)  # [None, 1]

def do_attention(outputs, h, W, b):
    input = outputs[0]
    h_hiddens = tf.matmul(input, W) + b
    h2 =tf.matmul(h, W) + b
    cos_values = get_cos_value(h_hiddens, h2)
    cos_values = tf.transpose(cos_values, [1,0])
    return tf.matmul(cos_values, input)

# 1, sentence 过一个LSTM
sentence_lstm = tf.contrib.rnn.LSTMCell(rnn_dim)
sentence_embedding, _ = tf.nn.dynamic_rnn(sentence_lstm, sentence, dtype=tf.float32, scope='sentence')



# 2, 将候选词连接在question后面过lstm
with tf.variable_scope("question_cancat_word") as scope:
    question_lstm = tf.contrib.rnn.LSTMCell(rnn_dim)
    question_embedding, final_state = tf.nn.dynamic_rnn(question_lstm, question, dtype=tf.float32, scope='question')



with tf.variable_scope("question_cancat_word") as scope:
    scope.reuse_variables()
    sentence_word_candidate_reshape = tf.reshape(sentence_word_candidate, shape=[-1, 1, word_dim])
    final_state = tf.reshape(tf.tile(final_state, [tf.shape(sentence_word_candidate_reshape)[0], 1, 1]), [2, -1, rnn_dim])
    question_lstm = tf.contrib.rnn.LSTMCell(rnn_dim)
    final_state1 = LSTMStateTuple(final_state[0], final_state[1])
    question_word_embeddings, _ = tf.nn.dynamic_rnn(question_lstm, sentence_word_candidate_reshape,
                                                    initial_state=final_state1, scope='question')

sentence_embedding_attention = do_attention(sentence_embedding, question_embedding[:,-1,:],
                                  Attention_Weight['W_sentence'], Attention_Weight['b_sentence'])
sentence_embedding_tile = tf.reshape(tf.tile(sentence_embedding_attention, [1, tf.shape(sentence_word_candidate)[1]]),
                                     [-1, tf.shape(sentence_word_candidate)[1], rnn_dim])
question_embedding_attention = do_attention(question_embedding, sentence_embedding[:,-1,:],
                                      Attention_Weight['W_question'], Attention_Weight['b_question'])
question_embedding_tile = tf.reshape(tf.tile(question_embedding_attention, [1, tf.shape(sentence_word_candidate)[1]]),
                                     [-1, tf.shape(sentence_word_candidate)[1], rnn_dim])
# 3, 计算相似度分数 两个线性层+激活函数层

sentence_embedding_input = tf.reshape(sentence_embedding_tile, [-1, rnn_dim], name='sentence_embedding')
question_embedding_input = tf.reshape(question_embedding_tile, [-1, rnn_dim], name='question_embedding')
question_word_embedding_input = tf.reshape(question_word_embeddings, [-1, rnn_dim])
linear_input = tf.concat([sentence_embedding_input, question_embedding_input, question_word_embedding_input], 1)
# linear_input = tf.concat([sentence_embedding_input, question_embedding_input, question_word_embedding_input, feature_embedding], 1)
Weight = {
    'W1': tf.Variable(tf.truncated_normal([all_dim, all_dim], stddev=stddev_setting)),
    'b1': tf.Variable(tf.truncated_normal([all_dim], stddev=stddev_setting)),
    'W2': tf.Variable(tf.truncated_normal([all_dim, all_dim], stddev=stddev_setting)),
    'b2': tf.Variable(tf.truncated_normal([all_dim], stddev=stddev_setting)),
    'W_out': tf.Variable(tf.truncated_normal([all_dim, 2], stddev=stddev_setting)),
    'b_out': tf.Variable(tf.truncated_normal([2], stddev=stddev_setting))
}
# 3.1 layer1
layer1_out = tf.nn.relu(tf.matmul(linear_input, Weight['W1'])+Weight['b1'])
layer2_out = tf.nn.relu(tf.matmul(layer1_out, Weight['W2'])+Weight['b2'])
layer3_out = (tf.matmul(layer2_out, Weight['W_out'])+Weight['b_out'])

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=gold_tag, logits=layer3_out))
train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

# 保存模型相关，tf的运行相关的设置
saver = tf.train.Saver(max_to_keep=100)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


# train操作

def train(sentence_input, sentence_word_candidate_input, question_input, gold_tags_input, sentences_candidate_words_feature_input):
    current_position = 0
    final_predict = []
    for i in range(len(sentence_input)):
        sentence_i = sentence_input[i]
        sentence_word_candidate_i = sentence_word_candidate_input[i]
        sentences_candidate_words_feature_input_i = sentences_candidate_words_feature_input[i]
        gold_tags_i = gold_tags_input[current_position:current_position+len(sentence_word_candidate_i)]
        current_position = current_position+len(sentence_word_candidate_i)
        predict, _, train_cost = sess.run([layer3_out, train_step, cost], feed_dict={
            sentence:[sentence_i],
            sentence_word_candidate:[sentence_word_candidate_i],
            question:question_input,
            gold_tag:gold_tags_i
            })
        final_predict.append(predict)
    # print 'train_cost: ', train_cost
    final_predict = np.concatenate(final_predict, 0)
    return final_predict

def test(sentence_input, sentence_word_candidate_input, question_input, sentences_candidate_words_feature_input):
    current_position = 0
    final_predict = []
    for i in range(len(sentence_input)):
        sentence_i = sentence_input[i]
        sentence_word_candidate_i = sentence_word_candidate_input[i]
        sentences_candidate_words_feature_input_i = sentences_candidate_words_feature_input[i]
        current_position = current_position + len(sentence_word_candidate_i)
        predict = sess.run(layer3_out, feed_dict={
            sentence: [sentence_i],
            sentence_word_candidate: [sentence_word_candidate_i],
            question: question_input
        })
        final_predict.append(predict)
    # print 'train_cost: ', train_cost
    final_predict = np.concatenate(final_predict, 0)
    return final_predict

def train_model_save(Iter):
    save_path = "Model/" + "Iter" + str(Iter) + "/model.ckpt"
    if not os.path.isdir("Model/" + "Iter" + str(Iter)):
        os.mkdir("Model/" + "Iter" + str(Iter))
    saver.save(sess, save_path)
    print ("Model stored....")


def train_model_import(Iter):
    save_path = "Model/" + "Iter" + str(Iter) + "/model.ckpt"
    if not os.path.isdir("Model/" + "Iter" + str(Iter)):
        os.mkdir("Model/" + "Iter" + str(Iter))
    saver.restore(sess, save_path)
    print("Model restored.")
