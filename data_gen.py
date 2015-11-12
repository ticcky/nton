"""

Database:
  - maps restaurant types to their names

"""
import numpy as np
import random

PLACEHOLDER = "[]"
question_tpls = [
        "looking for [] food",
        "i want [] food",
        "[]",
        "what restaurant serves []",
        "i need restaurant with [] food"
    ]
db = [
    ('chinese', 'cheng', ),
    ('indian', 'taj', ),
    ('oriental', 'orient')
]

answer_tpls = [
    "[] serves that",
    "it is []",
    "i found []"
    "[]"
]


class DataGenerator(object):
    @staticmethod
    def generate(n_words, n_db_entries, n_examples):
        words = ["w%.3d" % i for i in range(n_words)]
        db = [("q%.3d" % i, "q%.3d" % i) for i in range(n_db_entries)]

        p_w = np.zeros((len(words), ))
        for i in range(len(words)):
            word_id = np.random.randint(len(words))
            p_w[word_id] = 1.0 / (i + 1)

        p_w_zeros = p_w[p_w == 0.0]
        p_w_zeros += np.abs(np.random.randn(*p_w_zeros.shape)) * p_w.min()
        p_w /= p_w.sum()

        sentences = []
        for i in range(n_examples * 2):
            gen_words = np.random.poisson(5)

            sent = []
            for y in range(gen_words):
                sent.append(np.random.choice(words, p=p_w))

            sentences.append(sent)

        for s1, s2 in zip(sentences[::2], sentences[1::2]):
            q, a = np.random.choice(db)
            q_ndx = np.random.randint(len(s1))
            a_ndx = np.random.randint(len(s2))

            s1.insert(q_ndx, q)
            s2.insert(a_ndx, a)
            
        for q, a in db:
            words.append(q)
            words.append(a)







        return words, db

def main(n_words, n_db, n_examples):
    words, db = DataGenerator.generate(n_words, n_db, n_examples)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('n_words', type=int)
    parser.add_argument('n_db', type=int)
    parser.add_argument('n_examples', type=int)

    args = parser.parse_args()

    main(**vars(args))