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



if __name__ == '__main__':
    n = 10000
    seed = 1
    gen = PseudoGenerator(5, seed)

    # Generate n points
    i = 0
    nums_1 = []
    for x in gen:
        nums_1.append(x)
        i += 1
        if i >= n:
            break

    np.random.rand(5,3123)
    # Another n points
    i = 0
    for x in gen:
        nums_1.append(x)
        i += 1
        if i >= n:
            break
    
    # Generate them directly
    np.random.seed(seed)
    nums_2 = np.random.rand(2*n, 5)

    # Compare
    print(np.allclose(nums_1, nums_2))