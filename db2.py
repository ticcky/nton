import numpy as np
from collections import defaultdict

from nn import Block, Vars, Dot, Softmax

from vocab import Vocab
from nn.utils import timeit

VOCAB = """i would like some chinese food
what about indian
give me czech
i like english food
ok chong is a good place
i have taj here
go to hospoda
tavern is not bad
.
"""

class DB2(Block):
    def __init__(self, content, vocab):
        self.content = content

        self.vocab = Vocab()
        self.vocab.add('[EOS]')

        for word in vocab:
            self.vocab.add(word)

        self.map = defaultdict(list)
        self.map_rev = defaultdict(list)

        for e1, r, e2 in self.content:
            self.map[self.vocab[e2]].append((self.vocab[e1], self.vocab[r]))

        self.map = dict(self.map)

    def words_to_ids(self, words):
        res = []
        for word in words:
            res.append(self.vocab.add(word))

        return np.array(res)

    def get_vector(self, *words):
        res = np.zeros((len(self.vocab), ))

        for w in words:
            q_id = self.vocab[w]
            res[q_id] = 1.0

        return res

    def build_p(self, u):
        return np.dot(self.entries_a, u)

    def softmax(self, x):
        ex = np.exp(x)
        return ex / np.sum(ex)

    def _fwd_map(self, x, map):
        w = np.zeros_like(x)
        for i, val in enumerate(x):
            if i in map:
                w[map[i]] += val

        return w

    def forward(self, (e1, r)):
        e2 = np.zeros_like(e1)
        for i in range(len(e1)):
            if i in self.map:
                #print 'counting', i
                for e1_dim, r_dim in self.map[i]:
                    #print '   ', e1_dim, r_dim, e1[e1_dim], r[r_dim]
                    e2[i] += e1[e1_dim] * r[r_dim]


        aux = Vars(
            e1=e1,
            r=r
        )

        return ((e2, ), aux)


    def backward(self, aux, (dy, )):
        r = aux['r']
        e1 = aux['e1']

        de1 = np.zeros_like(dy)
        dr = np.zeros_like(dy)

        for i in range(len(dy)):
            if i in self.map:
                for e1_dim, r_dim in self.map[i]:
                    de1[e1_dim] += r[r_dim] * dy[i]
                    dr[r_dim] += e1[e1_dim] * dy[i]

        return (de1, dr, )


