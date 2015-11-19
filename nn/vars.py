import numpy as np


class Vars(object):
    """Hold variables of the blocks."""
    def __init__(self, *args, **kwargs):
        assert not args

        self.vars = kwargs
        self.var_names = []
        for var_name, var_val in sorted(kwargs.items(), key=lambda (n, v,): n):
            self.var_names.append(var_name)

    def __contains__(self, item):
        """Is variable with the given name contained?"""
        return item in self.vars

    def __getitem__(self, item):
        """Get variable with the given name."""
        return self.vars[item]

    def __setitem__(self, item, val):
        """Set variable with the given name to the given value."""
        self.vars[item]= val
        if not item in self.var_names:
            self.var_names.append(item)

        self.var_names = list(sorted(self.var_names))

    def __iter__(self):
        """Iterate over variable names."""
        for param_name in self.var_names:
            yield param_name

    @staticmethod
    def create_from(vars):
        data = {}
        for var in vars:
            data[var] = np.zeros_like(vars[var])

        return Vars(**data)


    def values(self):
        """Get a list of values of all variables in alphabetical order."""
        return [self.vars[param_name] for param_name in self.var_names]

    def names(self):
        """Get variable names."""
        return self.var_names

    def zero(self):
        """Set all variable values to zero."""
        for param_name in self:
            self.vars[param_name].flat[:] = 0

    def increment_by(self, params_inst, factor=1.0, clip=0.0):
        """Increment all values of variables by their value in params_inst.
        It is useful for updating variables by their gradients."""
        assert self.var_names == params_inst.var_names, "%s vs. %s" % (self.var_names, params_inst.var_names, )

        for param_name in self:
            grad = params_inst[param_name]
            if clip:
                np.clip(grad, -clip, clip, out=grad)
            self[param_name] += factor * grad

    def dump(self):
        return self.vars

    def load(self, params):
        """Load variable values from the given dictionary of param names and values."""
        for param_name, param_val in params.iteritems():
            assert param_name in self
            self[param_name].flat[:] = param_val.flat