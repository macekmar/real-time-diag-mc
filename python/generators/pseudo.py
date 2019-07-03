import numpy as np
import warnings

class PseudoGenerator():
    def __init__(self, dim, seed, **kwargs):
        self.dim = dim
        self.seed = seed
                
        if "m" not in kwargs:
            m = float("inf")
        else:
            m = kwargs["m"]
            del kwargs["m"]
        self.n = 2**m

        if len(kwargs) > 0:
            warnings.warn("PseudoGenerator does not except any other argument besides the dimension, seed number and number of requested numbers.")
        
        # We have to use separate generator so that it does not intefere with
        # calls to np.random.rand
        self.rng = np.random.RandomState()
        self.reset()

    def reset(self):
        """Reset the random number generator to its initial seed"""
        self.k = -1 # Too keep similar to lattice/sobol generators
        self.rng.seed(self.seed)

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        """Return the next point of the sequence or raise StopIteration."""
        if self.k < self.n - 1:
            self.k += 1
            return self.rng.rand(self.dim)
        else:
            raise StopIteration

    def next(self): 
        return self.__next__()

    class __metaclass__(type):    
        def __str__(self):
            return "Pseudo random generator"
        default_seed = 34788
