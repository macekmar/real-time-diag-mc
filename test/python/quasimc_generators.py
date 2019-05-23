import numpy as np
from ctint_keldysh.generators import *

# Test if generators are reset after reusing them -- we do not want that!


for generator in [SobolGenerator, HarmonicGenerator, PseudoGenerator]:

    gs = generator(5,1)
    x2 = []
    for _ in range(2):
        i = 0
        # The statement below calls gs.__iter__, which can reset gs
        for l in gs:
            x2.append(l)
            i += 1
            if i >= 10:
                break
    x2 = np.array(x2)

    gs = generator(5,1)
    x1 = []
    for i in range(20):
        x1.append(gs.next())
    x1 = np.array(x1)

    assert np.allclose(x1, x2)

print 'SUCCESS!'