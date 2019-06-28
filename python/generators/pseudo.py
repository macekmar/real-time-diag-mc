import numpy as np

class PseudoGenerator():
    def __init__(self, dim, seed):
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState()
        self.rng.seed(seed)

    def reset(self):
        self.rng.seed(self.seed)

    def __iter__(self):
        self.reset()
        return self

    def next(self):
        return self.rng.rand(self.dim)

    class __metaclass__(type):    
        def __str__(self):
            return "Pseudo random generator"
        default_seed = 1
