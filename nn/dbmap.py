import numpy as np

from nn import Block, Vars


class DBMap(Block):
    def __init__(self, mapping):
        self.mapping = mapping

    def forward(self, inputs):
        assert type(inputs) == tuple
        assert len(inputs) == len(self.mapping) + 1

        x = inputs[0]
        db = inputs[1:]

        res = np.zeros_like(x)

        mapped_mass = 0.0
        for x_dim, db_res in zip(self.mapping, db):
            res += x[x_dim] * db_res

            mapped_mass += x[x_dim]

        res += (1 - mapped_mass) * x

        aux = Vars(
            n_inputs = len(inputs),
            x=x,
            db=db,
            mapped_mass=mapped_mass
        )

        return ((res, ), aux)

    def backward(self, aux, (dy, )):
        x = aux['x']
        db = aux['db']
        mapped_mass = aux['mapped_mass']

        dx = dy.copy() * (1 - mapped_mass)
        ddbs = []
        for db_res, x_dim in zip(db, self.mapping):
            ddb = dy.copy()
            ddb *= x[x_dim]

            ddbs.append(ddb)

            dx[x_dim] += np.dot(dy, db_res) - np.dot(x, dy)

        return (dx, ) + tuple(ddbs)










