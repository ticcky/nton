import numpy as np

from vocab import Vocab


class DB(object):
    content = [
        ('chinese', 'chong'),
        ('indian', 'taj'),
        ('czech', 'hospoda'),
        ('english', 'tavern'),
    ]

    def __init__(self):
        self.vocab = Vocab()
        self.vocab.add('#OOV')

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

    def forward(self, x):
        Ax = np.dot(self.entries_a, x)[:, np.newaxis]
        p1 = self.softmax(Ax)
        res = (p1 * self.entries_c).sum(axis=0)
        resp = self.softmax(res)

        return resp

    def backward(self, x):
        n_vocab = len(self.vocab)
        n_ents = len(self.content)

        Ax = np.dot(self.entries_a, x)[:, np.newaxis]
        p1 = self.softmax(Ax)
        alpha = (p1 * self.entries_c).sum(axis=0)

        c = self.entries_c

        res = np.zeros((n_vocab, n_vocab))
        for i in range(n_ents):
            li = np.zeros((n_ents, ))
            for j in range(n_ents):
                if i == j:
                    li[j] = p1[i] * (1 - p1[i])
                else:
                    li[j] = - p1[i] * p1[j]

            res += np.dot(np.outer(c[i], li), self.entries_a)


        j2 = np.outer(alpha, alpha)
        for i in range(n_vocab):
            j2[i, i] += alpha[i]

        res = np.dot(j2, res)

        return res

    # def forward(self, words, weights):
    #     u = self.get_vector()
    #     for word, weight in zip(words, weights):
    #         u += self.get_vector(word) * weight
    #
    #     p = self.build_p(u)
    #     psoft = self.softmax(p)
    #     o = np.sum(self.entries_c * p[:, np.newaxis], axis=0)
    #     a_hat = self.softmax(o + u)
    #
    #     #print self.vocab.rev(np.argmax(a_hat))
    #
    #     return a_hat
    #
    # def backward(self, y, dy):
    #     dy_dsoftmax = dy * y * (1 - y)
    #     dy_do = dy_dsoftmax
    #     dy_du = dy_dsoftmax
    #
    #     import ipdb; ipdb.set_trace()
    #     dy_dp = np.dot(dy_do, self.entries_c.T)




