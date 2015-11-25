import numpy as np
import random


class DataCalc2(object):
    def __init__(self, n_words=100, max_num=10, n_answer_tpls=3, n_simple_qa_pairs=7, percent_qa=0.0):
        self.vocab = ["w%.3d" % i for i in range(n_words)]
        self.max_num = max_num
        self.percent_qa = percent_qa

        p_w = np.zeros((len(self.vocab), ))
        for i in range(len(p_w)):
            word_id = np.random.randint(len(p_w))
            p_w[word_id] = 1.0 / (i + 1)

        p_w /= p_w.sum()

        self.vocab_p = p_w

        self.db = []
        for i in range(self.max_num):
            for y in range(self.max_num):
                self.vocab.append('%d' % (i,))
                self.vocab.append('%d' % (y,))
                self.vocab.append('%d' % (i + y))

                self.db.append((str(i), str(y), str(i + y), ))

        a_tpls = []
        for i in range(n_answer_tpls):
            gen_n_words = np.random.poisson(3) + 1
            a_tpl = self._gen_seq(gen_n_words)
            a_tpl_pos = np.random.randint(len(a_tpl))
            a_tpls.append((a_tpl, a_tpl_pos, ))

        self.a_tpls = a_tpls

        self.simple_qa_pairs = []
        for i in range(n_simple_qa_pairs):
            gen_n_words = np.random.poisson(3) + 1
            q = self._gen_seq(gen_n_words)
            gen_n_words = np.random.poisson(3) + 1
            a = self._gen_seq(gen_n_words)
            a.append('[EOS]')

            self.simple_qa_pairs.append((q, a))

    def get_vocab(self):
        return self.vocab

    def get_db(self):
        return self.db

    def gen_data(self, test_data=False):
        while True:
            if np.random.random() < self.percent_qa:
                yield random.choice(self.simple_qa_pairs)
            else:
                a = np.random.randint(1, self.max_num)
                b = np.random.randint(a)

                if test_data:
                    tmp = a
                    a = b
                    b = tmp

                n_words = np.random.poisson(3) + 1
                seq_q = self._gen_seq(n_words)

                # if a + b < 10:
                #     tpl_ndx = 0
                # elif a + b < 15:
                #     tpl_ndx = 1
                # else:
                #     tpl_ndx = 2
                tpl_ndx = 1

                a_tpl, a_tpl_pos = self.a_tpls[tpl_ndx]
                seq_a = list(a_tpl) + ['[EOS]']

                #q = "%d+%d" % (a, b, )
                rand_loc = np.random.randint(len(seq_q))
                seq_q.insert(rand_loc, str(a))
                rand_loc = np.random.randint(len(seq_q))
                seq_q.insert(rand_loc, str(b))

                a = "%d" % (a + b, )
                seq_a.insert(a_tpl_pos, a)

                res_a = seq_a
                res_q = seq_q

                yield (res_q, res_a)

    def _gen_seq(self, n_words):
        seq = []
        for i in range(n_words):
            seq.append(np.random.choice(self.vocab[:len(self.vocab_p)], p=self.vocab_p))

        return seq