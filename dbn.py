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

        self.y_map = defaultdict(list)
        self.map_rev = defaultdict(list)

        self.n = None

        index_cntr = defaultdict(int)

        for key, result in self.content:
            key_tuple = tuple(self.vocab[x] for x in key)
            entry_ndx = index_cntr[key_tuple]
            #key_ndx = DBN.item_pattern % entry_ndx
            if entry_ndx != 0:  # TODO:
                continue
            #self.vocab.add(key_ndx)
            self.y_map[self.vocab[result]].append(key_tuple)
            index_cntr[key_tuple] += 1

            if self.n is None:
                self.n = len(key_tuple)
            else:
                assert len(key_tuple) == self.n

        self.y_map = dict(self.y_map)

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
        vals = {}

        assert type(db_input) == tuple
        assert len(db_input) == self.n
        for x in db_input:
            assert len(res) == len(x)

        for i in self.y_map.keys():
            for e, k_e in enumerate(self.y_map[i]):
                val = 1.0
                for x, xdim in zip(db_input, k_e):
                    val *= x[xdim]

                    if val == 0:
                        break

                vals[(i, e)] = val
                res[i] += val

        aux = Vars(
            db_input=db_input,
            res=res,
            vals=vals
        )

        return ((res, ), aux)

    @timeit
    def backward(self, aux, (dy, )):
        db_input = aux['db_input']
        res = aux['res']
        vals = aux['vals']

        ddb_input = []
        for v in db_input:
            ddb_input.append(np.zeros_like(v))

        for i in self.y_map.keys():
            dy_i = dy[i]

            for e, k_e in enumerate(self.y_map[i]):  # podivam se na entity ovlivnujici y_i
                val = vals[i, e]

                for j, (x_j, dx_j, ke_j) in enumerate(zip(db_input, ddb_input, k_e)):
                    if x_j[ke_j] == 0:
                        assert val == 0
                        continue

                    dx_j[ke_j] += dy_i * val / x_j[ke_j]


        return tuple(ddb_input)


