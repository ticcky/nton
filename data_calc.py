import numpy as np


class DataCalc(object):
    def __init__(self, n_words=100, max_num=10):
        self.vocab = ["w%.3d" % i for i in range(n_words)]
        self.max_num = max_num

        p_w = np.zeros((len(self.vocab), ))
        for i in range(len(p_w)):
            word_id = np.random.randint(len(p_w))
            p_w[word_id] = 1.0 / (i + 1)

        p_w /= p_w.sum()

        self.vocab_p = p_w

        self.db = []
        for i in range(self.max_num):
            for y in range(self.max_num):
                self.vocab.append('%d+%d' % (i, y, ))
                self.vocab.append('%d' % (i + y))
                self.db.append(("%d+%d" % (i, y, ), "%d" % (i + y)))

    def get_vocab(self):
        return self.vocab

    def get_db(self):
        return self.db


    def gen_data(self, test_data=False, simple_answer=False, simple_question=False):
        while True:
            a = np.random.randint(1, self.max_num)
            b = np.random.randint(a)

            if test_data:
                tmp = a
                a = b
                b = tmp

            n_words = np.random.poisson(3) + 1
            seq_q = self._gen_seq(n_words)

            n_words = np.random.poisson(3) + 1
            seq_a = self._gen_seq(n_words)


            q = "%d+%d" % (a, b, )
            seq_q.insert(np.random.randint(len(seq_q)), q)

            a = "%d" % (a + b, )
            seq_a.insert(np.random.randint(len(seq_a)), a)

            if simple_answer:
                res_a = [a]
            else:
                res_a = seq_a

            if simple_question:
                res_q = [q]
            else:
                res_q = seq_q

            yield (res_q, res_a)

    def _gen_seq(self, n_words):
        seq = []
        for i in range(n_words):
            seq.append(np.random.choice(self.vocab[:len(self.vocab_p)], p=self.vocab_p))

        return seq