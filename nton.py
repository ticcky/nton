import numpy as np


class NTON(object):
    max_gen_steps = 7

    def __init__(self):
        self.emb = Embeddings()

    def forward(self, words):
        word_ids = self.words_to_ids(words)

        E = self.emb.forward(word_ids)
        H = self.input_rnn.forward(E)

        g_t = self.output_rnn.get_init()

        for i in range(self.max_gen_steps):
            alpha_t = self.attention.forward(H, g_t)
            db_t = self.db.forward(alpha_t, word_ids)

            g_t = self.output_rnn.forward_step(g_t, o_t)

            softmax_t = self.output_softmax.forward(g_t)
            switch_t = self.output_switch.forward(g_t)

            po_t = switch_t * softmax_t + (1 - switch_t) * db_t

            o_t = np.random.choice(po_t.size, p=po_t)




def main():
    NTON()



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    main(**vars(args))