import numpy as np

from nn import ParametrizedBlock, Vars, Sequential, LinearLayer, Softmax, LSTM, DBMap


class NLG(ParametrizedBlock):
    def __init__(self, vocab_len, lstm_n_cells, dbmap_mapping):
        self.lstm = LSTM(vocab_len, lstm_n_cells)
        self.h_to_o = Sequential([
            LinearLayer(lstm_n_cells, vocab_len),
            Softmax()
        ])
        self.db_map = DBMap(dbmap_mapping)

        self.parametrize_from_layers(
            [self.lstm, self.h_to_o], ["lstm", "h_to_o"]
        )

    def forward(self, inputs):
        input_iter = iter(inputs)

        s_prime = next(input_iter)
        y_in = next(input_iter)
        y_steps = next(input_iter)

        external_inputs = tuple(input_iter)

        #assert len(external_inputs) % 3 == 0, 'We want 1 tracker, db, slu for each slot.'

        h_tm1, c_tm1 = self.lstm.get_init()
        c_tm1[:] = s_prime

        o_star_t = None
        O = []
        O_aux = []
        O_star_aux = []
        H_aux = []
        o_star_t_used = []
        for y_in_t in tuple(y_in) + (None, ) * y_steps:
            if type(y_in_t) != np.ndarray:
                assert type(o_star_t) == np.ndarray
                y_in_t = o_star_t
                o_star_t_used.append(True)
            else:
                o_star_t_used.append(False)
            ((h_t, c_t), h_t_aux) = self.lstm.forward((y_in_t[np.newaxis, np.newaxis, :], h_tm1, c_tm1))
            h_t = h_t[0][0]
            c_t = c_t[0][0]
            H_aux.append(h_t_aux)

            ((o_star_t, ), o_star_t_aux) = self.h_to_o.forward((h_t, ))  # Get RNN LM result.
            O_star_aux.append(o_star_t_aux)

            ((o_t, ), o_t_aux) = self.db_map.forward((o_star_t, ) + external_inputs)

            O.append(o_t)
            O_aux.append(o_t_aux)

            h_tm1 = h_t
            c_tm1 = c_t

        O = np.array(O)
        return ((O, ), Vars(
            o_t=O_aux,
            o_star_t=O_star_aux,
            o_star_t_used=o_star_t_used,
            h_t=H_aux,
            y_in_len=len(y_in)
        ))

    def backward(self, aux, (dO, )):
        lst_dexternal_inputs = []
        lst_dy_in = []
        dh_tm1 = 0.0
        dc_tm1 = 0.0
        dy_in_t = 0.0
        for do_t, o_t_aux, o_star_t_aux, o_star_t_used, h_t_aux in reversed(zip(dO, aux['o_t'], aux['o_star_t'], aux['o_star_t_used'], aux['h_t'])):
            ddb_map_inputs = self.db_map.backward(o_t_aux, (do_t, ))
            do_star_t = ddb_map_inputs[0]
            if o_star_t_used:
                do_star_t = do_star_t + dy_in_t
            lst_dexternal_inputs.append(ddb_map_inputs[1:])

            (dh_t, ) = self.h_to_o.backward(o_star_t_aux, (do_star_t, ))
            dh_t = dh_t[np.newaxis, np.newaxis, :] + dh_tm1
            dc_t = np.zeros_like(dh_t) + dc_tm1

            (dy_in_t, dh_tm1, dc_tm1) = self.lstm.backward(h_t_aux, (dh_t, dc_t, ))
            dy_in_t = dy_in_t[0][0]
            lst_dy_in.append(dy_in_t)

        dexternal_inputs = tuple(sum(lst) for lst in zip(*lst_dexternal_inputs))

        dy_in = np.array(list(reversed(lst_dy_in[-aux['y_in_len']:])))

        return (dc_tm1[0], dy_in, None) + dexternal_inputs

