class Vocab(dict):
    def __init__(self):
        self._rev = dict()

    def add(self, word):
        if not word in self:
            val = len(self)
            self[word] = val
            self._rev[val] = word

    def rev(self, word_id):
        return self._rev[word_id]