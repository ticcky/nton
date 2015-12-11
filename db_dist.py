import numpy as np
from collections import defaultdict

from nn import Block, Vars, Dot, Softmax

from vocab import Vocab
from nn.utils import timeit


class DBDist(Block):
    def __init__(self, content, input_vocab, output_vocab):
        self.content = content
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab

        self.y_map = defaultdict(list)
        self.n = None

        for key, result in self.content:
            key_tuple = tuple(self.input_vocab[x] for x in key)

            self.y_map[self.output_vocab[result]].append(key_tuple)

            if self.n is None:
                self.n = len(key_tuple)
            else:
                assert len(key_tuple) == self.n

        self.y_map = dict(self.y_map)

    def vocab_map_fn(self, x):
        return self.input_vocab[x]

    def get_vector(self, *words):
        res = np.zeros((len(self.input_vocab), ))

        for w in words:
            q_id = self.vocab_map_fn(w)
            res[q_id] = 1.0

        return res

    @staticmethod
    def get_1hot_from(word, vocab):
        res = np.zeros((len(vocab), ))
        word_id = vocab[word]
        res[word_id] = 1

        return res

    @timeit
    def forward(self, db_input):
        res = np.zeros((len(self.output_vocab), ))
        vals = {}

        assert type(db_input) == tuple
        assert len(db_input) == self.n, 'db_input: %d, n: %d' % (len(db_input), self.n, )

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


