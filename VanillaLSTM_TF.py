import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class VanillaLSTM_TF:
    def __init__(self, options):
        self.state_size = options['state_size']
        self.vocab_size = options['vocab_size']
        self.batch_size = options['batch_size']
        self.seq_len = options['seq_len']
        self.model_init = tf.contrib.layers.xavier_initializer
        self.n_classes = options['n_classes']
        self.learning_rate = options['lr']
        self.epochs = options['epochs']
        self.test_freq = options['test_freq']

    def init_variables(self):
        self.xs_ = tf.placeholder(shape=[None, None], dtype=tf.int32, name='x_in')
        self.ys_ = tf.placeholder(shape=[None], dtype=tf.int32, name='ys_')
        self.x_seq_lens = tf.placeholder(shape=[None], dtype=tf.int32, name='seqlens')

        self.emb = tf.get_variable('emb', [self.vocab_size, self.state_size], dtype=tf.float32)
        self.rnn_inputs = tf.nn.embedding_lookup(self.emb, self.xs_, name='rnn_input')

        # output activation variables
        self.Vo = tf.get_variable('Vo', shape=[self.state_size, self.n_classes],
                                  initializer=self.model_init())
        self.bo = tf.get_variable('bo', shape=[self.n_classes],
                                  initializer=tf.constant_initializer(0.0))

    def build_graph(self):
        lstmcell = tf.contrib.rnn.BasicLSTMCell(self.state_size, activation=tf.nn.tanh)
        self.init_state = lstmcell.zero_state(tf.shape(self.xs_)[0], tf.float32)
        rnnout, _final_states = tf.nn.dynamic_rnn(lstmcell, self.rnn_inputs, initial_state=self.init_state,
                                                  sequence_length=self.x_seq_lens)

        # calculate the output from lstm final hidden state (not cell state)
        self.state_reshaped = tf.reshape(_final_states.h, [-1, self.state_size])
        self.logits = tf.matmul(self.state_reshaped, self.Vo) + self.bo
        self.probs = tf.nn.softmax(self.logits)

        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                   labels=tf.reshape(self.ys_, [-1]))
        # self.loss = tf.reduce_mean(loss)
        self.cost = tf.reduce_mean(self.loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        # self.correct_pred = tf.equal(tf.argmax(self.probs), self.ys_, 1)
        # self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    def run_model(self, train_dt, test_dt, val_dt):
        tf.reset_default_graph()
        self.init_variables()
        self.build_graph()

        with tf.Session() as sess:
            tf.global_variables_initializer().run(session=sess)
            for e in range(0, self.epochs):
                global_cost = 0.

                num_batches = int(len(train_dt[0]) // self.batch_size)

                for ind in range(0, num_batches - 1):
                    train_data = train_dt[0][ind * self.batch_size: (ind + 1) * self.batch_size]
                    train_label = train_dt[1][ind * self.batch_size: (ind + 1) * self.batch_size]
                    train_len = train_dt[2][ind * self.batch_size: (ind + 1) * self.batch_size]
                    if train_data.shape[0] < self.batch_size:
                        continue

                    feed_d = {self.xs_: train_data, self.ys_: train_label, self.x_seq_lens: train_len}
                    _, cost = sess.run([self.optimizer, self.cost],
                                       feed_dict=feed_d)
                    '''logits, Vo, st = sess.run([self.logits, self.Vo, self.state_reshaped],
                                              # _, cost, probs self.optimizer, self.cost, self.probs
                                              feed_dict=feed_d)'''
                    # print(logits[0].shape, Vo.shape, st.shape)
                    global_cost += cost

                print('epoch %d, global cost %.4f' % (e,global_cost))

                if e % self.test_freq == 0:
                    a_f1_all, a_prec_all, a_recall_all, a_f1_pos, a_prec_pos, a_recall_pos = 0., 0., 0., 0., 0., 0.
                    n_val_batches = int(len(val_dt[0])) // self.batch_size

                    for ind in range(0, n_val_batches - 1):
                        val_data = val_dt[0][ind * self.batch_size: (ind + 1) * self.batch_size]
                        val_label = val_dt[1][ind * self.batch_size: (ind + 1) * self.batch_size]
                        val_len = val_dt[2][ind * self.batch_size: (ind + 1) * self.batch_size]

                        feed_d = {self.xs_: val_data, self.ys_: val_label, self.x_seq_lens: val_len}
                        probs = sess.run([self.probs],  # _, cost, probs self.optimizer, self.cost, self.probs
                                           feed_dict=feed_d)
                        # print(probs[0].shape, accracy.shape)
                        # print(np.argmax(probs[0], axis=1).shape)
                        prediction = np.argmax(probs[0], axis=1)
                        # print(prediction.shape, val_dt[1].shape)
                        f1_all, prec_all, recall_all, f1_pos, prec_pos, recall_pos = self.evaluate(prediction, val_label)

                        a_f1_all += f1_all
                        a_prec_all += prec_all
                        a_recall_all += recall_all

                        a_f1_pos += f1_pos
                        a_prec_pos += prec_pos
                        a_recall_pos += recall_pos

                    print('overall performance validation')
                    print("f1: %0.4f, precision: %0.4f, recall: %0.4f" % (a_f1_all / n_val_batches, a_prec_all / n_val_batches, a_recall_all / n_val_batches))
                    print('positive labels prediction')
                    print("f1: %0.4f, precision: %0.4f, recall: %0.4f" % (a_f1_pos / n_val_batches, a_prec_pos / n_val_batches, a_recall_pos / n_val_batches))


            print('### testing on test data ###')
            a_f1_all, a_prec_all, a_recall_all, a_f1_pos, a_prec_pos, a_recall_pos = 0., 0., 0., 0., 0., 0.
            n_test_batches = int(len(test_dt[0])) // self.batch_size
            for ind in range(0, n_test_batches - 1):
                test_data = test_dt[0][ind * self.batch_size: (ind + 1) * self.batch_size]
                test_label = test_dt[1][ind * self.batch_size: (ind + 1) * self.batch_size]
                test_len = test_dt[2][ind * self.batch_size: (ind + 1) * self.batch_size]

                feed_d = {self.xs_: test_data, self.ys_: test_label, self.x_seq_lens: test_len}
                probs = sess.run([self.probs],  # _, cost, probs self.optimizer, self.cost, self.probs
                                 feed_dict=feed_d)
                # print(probs[0].shape, accracy.shape)
                # print(np.argmax(probs[0], axis=1).shape)
                prediction = np.argmax(probs[0], axis=1)
                # print(prediction.shape, val_dt[1].shape)
                f1_all, prec_all, recall_all, f1_pos, prec_pos, recall_pos = self.evaluate(prediction, test_label)

                a_f1_all += f1_all
                a_prec_all += prec_all
                a_recall_all += recall_all

                a_f1_pos += f1_pos
                a_prec_pos += prec_pos
                a_recall_pos += recall_pos

            print('overall performance validation')
            print("f1: %0.4f, precision: %0.4f, recall: %0.4f" % (
            a_f1_all / n_test_batches, a_prec_all / n_test_batches, a_recall_all / n_test_batches))
            print('positive labels prediction')
            print("f1: %0.4f, precision: %0.4f, recall: %0.4f" % (
            a_f1_pos / n_test_batches, a_prec_pos / n_test_batches, a_recall_pos / n_test_batches))


    def evaluate(self, predict, true):
        f1_all = f1_score(y_true=true, y_pred=predict, average='macro')
        prec_all = precision_score(y_true=true, y_pred=predict, average='macro')
        recall_all = recall_score(y_true=true, y_pred=predict, average='macro')

        f1_pos = f1_score(y_true=true, y_pred=predict, labels=range(0, self.n_classes-1), average='macro')
        prec_pos = precision_score(y_true=true, y_pred=predict, labels=range(0, self.n_classes-1), average='macro')
        recall_pos = recall_score(y_true=true, y_pred=predict, labels=range(0, self.n_classes-1), average='macro')

        return f1_all, prec_all, recall_all, f1_pos, prec_pos, recall_pos

