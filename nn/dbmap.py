import numpy as np

import nn
from nn import Block, Vars


class DBMap(Block):
    def __init__(self, mapping):
        """

        :param mapping: list, n-th element of mapping says which x's dimension
                does n-th db input correspond to
        :return:
        """
        self.mapping = mapping

    def forward(self, inputs):
        assert type(inputs) == tuple
        assert len(inputs) == len(self.mapping) + 1

        x = inputs[0]
        db = inputs[1:]

        res = np.zeros_like(x)

        mapped_mass = 0.0
        norm_aux_lst = []
        for x_dim, db_res in zip(self.mapping, db):
            ((db_res_norm,), db_res_norm_aux) = nn.Normalize.forward((db_res,))
            norm_aux_lst.append(db_res_norm_aux)
            res += x[x_dim] * db_res_norm

            mapped_mass += x[x_dim]

        res += (1 - mapped_mass) * x

        aux = Vars(
            n_inputs = len(inputs),
            x=x,
            db=db,
            mapped_mass=mapped_mass,
            norm_aux_lst=norm_aux_lst
        )

        return ((res, ), aux)

    def backward(self, aux, (dy, )):
        x = aux['x']
        db = aux['db']
        mapped_mass = aux['mapped_mass']

        dx = dy.copy() * (1 - mapped_mass)
        ddbs = []
        for db_res, x_dim, norm_aux in zip(db, self.mapping, aux['norm_aux_lst']):
            ddb = dy.copy()
            ddb *= x[x_dim]

            (ddb,) = nn.Normalize.backward(norm_aux, (ddb,))

            ddbs.append(ddb)

            dx[x_dim] += np.dot(dy, db_res) - np.dot(x, dy)

        return (dx, ) + tuple(ddbs)










