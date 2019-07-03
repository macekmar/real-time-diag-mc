import numpy as np
from ctint_keldysh.generators import *

for generator in [SobolGenerator, HarmonicGenerator, PseudoGenerator, LatticeGenerator]:
  
    # Each time the generator is used as a python generator it has to be reset
    gs = generator(5,1)
    x = []
    for _ in range(2):
        x.append([])
        i = 0
        for l in gs:
            i += 1
            x[-1].append(l)
            if i >= 10:
                break
    x = np.array(x)

    # We can also produce numbers by a "manual" loop
    # Now, generator is not reset! [*]
    gs = generator(5,1)
    y = []
    for _ in range(2):
        y.append([])
        for i in range(10):
            y[-1].append(gs.next())
            i = 0
    y = np.array(y)

    gs = generator(5,1)
    z = []
    i = 0
    for l in gs:
        i += 1
        z.append(l)
        if i >= 20:
            break
    z = np.array(z)

    assert np.allclose(x[0], x[1])
    assert np.allclose(x[0], y[0])
    assert not np.allclose(y[0], y[1]) # [*] (comment above)
    assert np.allclose(np.vstack(y), z)


    # Test parameter m, test if we produce 2**m numbers
    gs = generator(4, 0, m=5)
    n = []
    for pt in gs:
        n.append(pt)
    assert np.array(n).shape == (2**5, 4)

    print(str(generator) + " passsed!")

print 'SUCCESS!'