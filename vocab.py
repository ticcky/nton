class Vocab(dict):
    def __init__(self):
        self._rev = dict()
        self.frozen = False

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