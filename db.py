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

class DB(Block):
    # content = [
    #     ('chinese', 'chong'),
    #     ('indian', 'taj'),
    #     ('czech', 'hospoda'),
    #     ('english', 'tavern'),
    # ]

    def __init__(self, content, vocab, impl='fast'):
        self.content = content

        self.vocab = Vocab()
        self.vocab.add('[EOS]')

        for word in vocab:
            self.vocab.add(word)

        self.db_map = defaultdict(list)
        self.db_map_rev = defaultdict(list)

        entries_a = []
        for food, restaurant in self.content:
            entry = self.get_vector(food, restaurant)
            entries_a.append(entry)

        self.entries_a = np.array(entries_a)

        entries_c = []
        for food, restaurant in self.content:
            entry = self.get_vector(restaurant)
            #f_id = self.vocab[food]
            #entry[f_id] = -1
            entries_c.append(entry)

            self.db_map[self.vocab[food]].append(self.vocab[restaurant])
            self.db_map[self.vocab[restaurant]].append(self.vocab[restaurant])
            #self.db_map[self.vocab[food]].append(self.vocab[food])
            self.db_map_rev[self.vocab[restaurant]].append(self.vocab[food])
            self.db_map_rev[self.vocab[restaurant]].append(self.vocab[restaurant])

        self.db_map = dict(self.db_map)
        self.db_map_rev = dict(self.db_map_rev)

        self.entries_c = np.array(entries_c)

        if impl == 'fast':
            self.forward = self.forward_nosoft_fast
            self.backward = self.backward_nosoft_fast
        elif impl == 'normal':
            self.forward = self.forward_nosoft
            self.backward = self.backward_nosoft
        else:
            assert False, 'Unknown implementation type: %s' % impl

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

    def forward(self, (x, )):
        ((Ax, ), Ax_aux) = Dot.forward((self.entries_a, x))

        #((p1, ), p1_aux) = Softmax.forward((Ax.T, ))

        p1C = Ax * self.entries_c
        w = np.sum(p1C, axis=0)
        w_aux = p1C.shape[0]

        ((result, ), result_aux) = Softmax.forward((w, ))

        aux = Vars(
            x=x,
            w=w,
            w_aux=w_aux,
            result_aux=result_aux,
            Ax=Ax,
            Ax_aux=Ax_aux
        )

        return ((result, ), aux)

    def backward(self, aux, (dy, )):
        x = aux['x']
        w = aux['w']
        w_aux = aux['w_aux']
        result_aux = aux['result_aux']
        Ax = aux['Ax']
        Ax_aux = aux['Ax_aux']

        (dw, ) = Softmax.backward(result_aux, (dy, ))

        dp1C = np.tile(dw[np.newaxis, :], (w_aux, 1))

        dAx = (self.entries_c * dp1C).sum(axis=1, keepdims=True)

        (dA, dx) = Dot.backward(Ax_aux, (dAx, ))

        return (dx[:, 0], )

    @timeit
    def forward_nosoft(self, (x, )):
        ((Ax, ), Ax_aux) = Dot.forward((self.entries_a, x))

        #((p1, ), p1_aux) = Softmax.forward((Ax.T, ))

        p1C = Ax * self.entries_c
        w = np.sum(p1C, axis=0)
        w_aux = p1C.shape[0]

        #((result, ), result_aux) = Softmax.forward((w, ))

        aux = Vars(
            x=x,
            w=w,
            w_aux=w_aux,
            Ax=Ax,
            Ax_aux=Ax_aux
        )

        return ((w, ), aux)

    def forward_nosoft_fast(self, (x, )):
        w = np.zeros_like(x)
        for i, val in enumerate(x):
            if i in self.db_map:
                #print 'adding', self.vocab.rev(i), val, [self.vocab.rev(aa) for aa in self.db_map[i]]
                w[self.db_map[i]] += val

        aux = Vars(
            w=w
        )

        return ((w, ), aux)

    def backward_nosoft_fast(self, aux, (dy, )):
        dx = np.zeros_like(dy)
        for i, val in enumerate(dy):
            if i in self.db_map_rev:
                dx[self.db_map_rev[i]] += val

        return (dx, )

    @timeit
    def backward_nosoft(self, aux, (dy, )):
        w_aux = aux['w_aux']
        Ax_aux = aux['Ax_aux']

        dw = dy

        dp1C = np.tile(dw[np.newaxis, :], (w_aux, 1))

        dAx = (self.entries_c * dp1C).sum(axis=1, keepdims=True)

        (dA, dx) = Dot.backward(Ax_aux, (dAx, ))

        return (dx[:, 0], )

    def forward_old(self, (x, )):
        ((Ax, ), Ax_aux) = Dot.forward((self.entries_a, x))
        ((p1, ), p1_aux) = Softmax.forward((Ax.T, ))

        p1C = p1.T * self.entries_c
        w = np.sum(p1C, axis=0)

        ((result, ), result_aux) = Softmax.forward((w, ))


        aux = Vars(
            x=x,
            w=w,
            result_aux=result_aux,
            Ax=Ax,
            Ax_aux=Ax_aux,
            p1_aux=p1_aux
        )

        return ((result, ), aux)


    def backward_old(self, aux, (dy, )):
        x = aux['x']
        w = aux['w']
        result_aux = aux['result_aux']
        Ax = aux['Ax']
        Ax_aux = aux['Ax_aux']
        p1_aux = aux['p1_aux']

        (dresult, ) = Softmax.backward(result_aux, (dy, ))

        dp1C_dp1 = (self.entries_c * dresult).sum(axis=1)

        (dp1, ) = Softmax.backward(p1_aux, (dp1C_dp1, ))
        (dA, dx) = Dot.backward(Ax_aux, (dp1.T, ))

        return (dx[:, 0], )

