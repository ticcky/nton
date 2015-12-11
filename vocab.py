import numpy as np


class Vocab(dict):
    oov_token = '#OOV'
    eos_token = '[EOS]'

    def __init__(self, no_oov_eos=False):
        self._rev = dict()
        self.frozen = False

        if not no_oov_eos:
            self.add(self.oov_token)
            self.add(self.eos_token)

    def __iter__(self):
        for k, v in sorted(self.items(), key=lambda x: x[1]):
            yield k

    def freeze(self):
        self.frozen = True

    def add(self, word):
        if not word in self:
            if self.frozen:
                raise KeyError

            val = len(self)
            self[word] = val
            self._rev[val] = word

        return self[word]

    def rev(self, word_id):
        return self._rev[word_id]

    def words_to_ids(self, words):
        res = []
        for word in words:
            try:
                res.append(self[word])
            except KeyError:
                res.append(self[self.oov_token])

        return np.array(res)