import numpy as np


class Initializer(object):
    pass


class Normal(Initializer):
    """Initialize parameters from gaussian distribution."""
    def __call__(self, dims):
        return np.random.randn(*dims) * np.sqrt(2.0 / dims[-1])


class Eye(Initializer):
    """Initialize parameters as identity matrix."""
    def __call__(self, dims):
        return np.eye(*dims)


class Constant(Initializer):
    """Initialize all parameters to the given value."""
    def __init__(self, const):
        self.const = const

    def __call__(self, dims):
        return np.ones(dims) * self.const