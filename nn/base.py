class Block(object):
    pass


class ParametrizedBlock(Block):
    def parametrize(self, params, grads):
        assert params.var_names == grads.var_names
        self._params = params
        self._grads = grads

    @property
    def params(self):
        return self._params

    @property
    def grads(self):
        return self._grads


class Loss(object):
    pass