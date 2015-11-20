from unittest import TestCase, main
import numpy as np

from db import DB
from nn.utils import check_finite_differences
from data_calc import DataCalc

class TestDB(TestCase):
    content = [
        ('chinese', 'chong'),
        ('indian', 'taj'),
        ('czech', 'hospoda'),
        ('english', 'tavern'),
    ]
    vocab = ['chinese', 'chong', 'indian', 'taj', 'czech', 'hospoda', 'english', 'tavern']

    def test_forward(self):

        db = DB(self.content, self.vocab)
        #db.backward(np.array([0.7, 0.1, 0.1, 0.1]), np.array([1.0, 1.0, 1.0, 1.0]))

        self.assertEqual(db.vocab.rev(db.forward((db.get_vector('czech'), ))[0][0].argmax()), 'hospoda')
        self.assertEqual(db.vocab.rev(db.forward((db.get_vector('chinese'), ))[0][0].argmax()), 'chong')
        self.assertEqual(db.vocab.rev(db.forward((db.get_vector('indian'), ))[0][0].argmax()), 'taj')

        #(dczech, ) = db.backward((czech, ), aux, (np.random.randn(*db_y.shape), ))

    def test_forward_calc(self):
        data = DataCalc(max_num=10)
        db = DB(data.get_db(), data.get_vocab())

        self.assertEqual(db.vocab.rev(db.forward((db.get_vector('1+0'), ))[0][0].argmax()), '1')
        self.assertEqual(db.vocab.rev(db.forward((db.get_vector('0+0'), ))[0][0].argmax()), '0')
        self.assertEqual(db.vocab.rev(db.forward((db.get_vector('5+3'), ))[0][0].argmax()), '8')
        self.assertEqual(db.vocab.rev(db.forward((db.get_vector('1+6'), ))[0][0].argmax()), '7')
        self.assertEqual(db.vocab.rev(db.forward((db.get_vector('3+3'), ))[0][0].argmax()), '6')
        self.assertEqual(db.vocab.rev(db.forward((db.get_vector('4+0'), ))[0][0].argmax()), '4')


    def test_backward(self):
        db = DB(self.content, self.vocab, impl='normal')
        db_fast = DB(self.content, self.vocab, impl='fast')

        def gen_input():
            food = np.random.choice(['chinese', 'czech', 'english', 'indian'])
            x = db.get_vector(food)
            x += np.random.randn(*x.shape)

            return (x, )

        for i in range(100):
            x = gen_input()
            ((y1, ), aux1) = db.forward(x)
            ((y2, ), aux2) = db_fast.forward(x)

            err = np.random.randn(*y1.shape)
            (dx1, ) = db.backward(aux1, (err, ))
            (dx2, ) = db_fast.backward(aux1, (err, ))

            self.assertTrue(np.allclose(y1, y2))
            self.assertTrue(np.allclose(dx1, dx2))

        check = check_finite_differences(
            db.forward_nosoft_fast,
            db.backward_nosoft_fast,
            gen_input_fn=gen_input,
            aux_only=True
        )
        self.assertTrue(check)

        check = check_finite_differences(
            db.forward_nosoft,
            db.backward_nosoft,
            gen_input_fn=gen_input,
            aux_only=True
        )
        self.assertTrue(check)

    def test_backward_fast(self):
        db = DB(self.content, self.vocab, impl='fast')

        def gen_input():
            food = np.random.choice(['chinese', 'czech', 'english', 'indian'])
            x = db.get_vector(food)
            x += np.random.randn(*x.shape)

            return (x, )

        check = check_finite_differences(
            db.forward_nosoft_fast,
            db.backward_nosoft_fast,
            gen_input_fn=gen_input,
            aux_only=True
        )
        self.assertTrue(check)

        #print db.forward(['czech'], [1.0])
        #print db.forward(['indian'], [1.0])
        #print db.forward(['czech', 'indian'], [0.6, 0.4])
        #print db.forward(['czech', 'indian', 'chinese'], [0.1, 0.4, 0.5])
        #db.forward('indian')


if __name__ == "__main__":
    np.random.seed(0)
    main()