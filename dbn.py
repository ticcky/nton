import numpy as np
from collections import defaultdict

from nn import Block, Vars, Dot, Softmax

from vocab import Vocab
from nn.utils import timeit


class DBN(Block):
    item_pattern = 'item%.2d'
    def __init__(self, content, vocab):
        self.content = content

        self.vocab = Vocab()
        for word in vocab:
            self.vocab.add(word)

        self.map = defaultdict(list)
        self.map_rev = defaultdict(list)

        self.n = None

        index_cntr = defaultdict(int)

        for key, result in self.content:
            key_tuple = tuple(self.vocab[x] for x in key)
            key_ndx = DBN.item_pattern % index_cntr[key_tuple]
            self.vocab.add(key_ndx)
            self.map[self.vocab[result]].append(key_tuple + (self.vocab[key_ndx],))
            index_cntr[key_tuple] += 1

            if self.n is None:
                self.n = len(key_tuple) + 1
            else:
                assert len(key_tuple) + 1 == self.n

        self.map = dict(self.map)

    def get_vector(self, *words):
        res = np.zeros((len(self.vocab), ))

        for w in words:
            q_id = self.vocab[w]
            res[q_id] = 1.0

        return res

    @staticmethod
    def get_1hot_from(word, vocab):
        res = np.zeros((len(vocab), ))
        word_id = vocab[word]
        res[word_id] = 1

        return res

    def build_p(self, u):
        return np.dot(self.entries_a, u)

    def softmax(self, x):
        ex = np.exp(x)
        return ex / np.sum(ex)

    @timeit
    def forward(self, db_input):
        res = np.zeros((len(self.vocab), ))

        assert type(db_input) == tuple
        assert len(db_input) == self.n
        for x in db_input:
            assert len(res) == len(x)

        for i in self.map.keys():
            #print 'counting', i
            for dims in self.map[i]:
                val = 1.0
                for x, xdim in zip(db_input, dims):
                    val *= x[xdim]

                res[i] += val

        aux = Vars(
            db_input=db_input
        )

        return ((res, ), aux)

    @timeit
    def backward(self, aux, (dy, )):
        db_input = aux['db_input']

        ddb_input = []
        for v in db_input:
            ddb_input.append(np.zeros_like(v))

        for i in self.map.keys():
            dy_i = dy[i]

            for dims in self.map[i]:
                inp_total = 1.0

                for x_k, x_kdim in zip(db_input, dims):
                    inp_total *= x_k[x_kdim]

                for z, (x_k, dx_k, x_kdim) in enumerate(zip(db_input, ddb_input, dims)):
                    if x_k[x_kdim] > 0:
                        denom = x_k[x_kdim]
                    else:
                        denom = 1.0

                    dx_k[x_kdim] += dy_i * inp_total / denom

        return tuple(ddb_input)


