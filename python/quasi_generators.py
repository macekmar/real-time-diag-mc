import quasi_harmonic_gen as harm
import numpy as np
import sobol

class SobolGen():
    def __init__(self, dim, seed=1):
        self.dim = dim
        self.reset(seed)
        
    def reset(self, seed=1):
        self.gen = sobol.sobol_gen(self.dim, seed)
        
    def generate(self, N):
        nums = []
        for i in range(N):
            nums.append(self.gen.next())
        return np.array(nums)    

class HarmGen():
    def __init__(self, dim, seed=None):
        self.default_seed = 1<<30
        if seed is None:
            seed = self.default_seed
        self.dim = dim
        self.reset(seed)
        
    def reset(self, seed=None):
        if seed is None:
            seed = self.default_seed
        self.gen = harm.HARQuasirand(dim=self.dim, seed=seed)
        self.seed = self.gen.count
        
    def generate(self, N):
        nums = self.gen.rand_fast(N)
        self.seed = self.gen.count
        return np.array(nums, dtype=np.float64)    

class PseudoRandomGen():
    def __init__(self, dim, seed=1):
        self.dim = dim
        self.state = None        
        self.reset(seed)
        
    def reset(self, seed=1):
        np.random.seed(seed)
        self.state = np.random.get_state()
        
    def generate(self, N):
        np.random.set_state(self.state)
        nums = np.random.rand(N, self.dim)
        self.state = np.random.get_state()
        return nums