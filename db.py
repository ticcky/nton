import numpy as np

from nn import Block, Vars, Dot, Softmax

from vocab import Vocab

VOCAB = """i would like some chinese food
what about indian
give me czech
i like english food
ok chong is a good place
i have taj here
go to hospoda
tavern is not bad
"""

class DB(Block):
    content = [
        ('chinese', 'chong'),
        ('indian', 'taj'),
        ('czech', 'hospoda'),
        ('english', 'tavern'),
    ]

    def __init__(self):
        self.vocab = Vocab()
        self.vocab.add('#OOV')
        for word in VOCAB.split():
            self.vocab.add(word)

        for food, restaurant in self.content:
            self.vocab.add(food)
            self.vocab.add(restaurant)

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

        self.entries_c = np.array(entries_c)

    def words_to_ids(self, words):
        res = []
        for word in words:
            res.append(self.vocab.add(word))

        return res

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
        ((p1, ), p1_aux) = Softmax.forward((Ax.T, ))

        p1C = p1.T * self.entries_c
        w = np.sum(p1C, axis=0)

        ((result, ), result_aux) = Softmax.forward((w, ))

        aux = Vars(
            w=w,
            result_aux=result_aux,
            Ax=Ax,
            Ax_aux=Ax_aux,
            p1_aux=p1_aux
        )

        return ((result, ), aux)

    def backward(self, (x, ), aux, (dy, )):
        w = aux['w']
        result_aux = aux['result_aux']
        Ax = aux['Ax']
        Ax_aux = aux['Ax_aux']
        p1_aux = aux['p1_aux']

        (dresult, ) = Softmax.backward((w, ), result_aux, (dy, ))

        dp1C_dp1 = (self.entries_c * dresult).sum(axis=1)

        (dp1, ) = Softmax.backward((Ax.T, ), p1_aux, (dp1C_dp1, ))
        (dA, dx) = Dot.backward((self.entries_a, x), Ax_aux, (dp1.T, ))

        return (dx[:, 0], )

