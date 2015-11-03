"""

Database:
  - maps restaurant types to their names

"""
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
    def generate(n):
        res = []

        for i in range(n):
            q = random.choice(question_tpls)
            a = random.choice(answer_tpls)
            food, name = random.choice(db)
            q = q.replace(PLACEHOLDER, food)
            a = a.replace(PLACEHOLDER, name)

            res.append((q, a))

        return res

def main(n):
    for q, a in DataGenerator.generate(n):
        print "%s\n%s" % (q, a)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('n', type=int)

    args = parser.parse_args()

    main(**vars(args))