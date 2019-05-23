import numpy as np

class PseudoGenerator():
    def __init__(self, dim, seed):
        self.dim = dim
        self.rng = np.random.RandomState()
        self.rng.seed(seed)

    def __iter__(self):
        return self

    def next(self):
        return self.rng.rand(self.dim)

    class __metaclass__(type):    
        def __str__(self):
            return "Pseudo random generator"
        default_seed = 1
