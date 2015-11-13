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
.
"""

class DB(Block):
    # content = [
    #     ('chinese', 'chong'),
    #     ('indian', 'taj'),
    #     ('czech', 'hospoda'),
    #     ('english', 'tavern'),
    # ]

    def __init__(self, content, vocab):
        self.content = content

        self.vocab = Vocab()
        self.vocab.add('[EOS]')

        for word in vocab:
            self.vocab.add(word)

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

        self.forward = self.forward_nosoft
        self.backward = self.backward_nosoft

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

