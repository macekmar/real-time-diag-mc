import numpy as np
from ctint_keldysh.quasimc_post_treatment import *


# ### Test add_empty_orders

orders = [2,3,6]

shp = (len(orders), 10000, 2, 1)
kernels = np.random.rand(*shp) + 1j*np.random.rand(*shp)
new_shp = (max(orders), 10000, 2, 1)
expanded_kernels = add_empty_orders(kernels, orders)

assert expanded_kernels.shape == new_shp
assert np.allclose(np.sum(kernels), np.sum(expanded_kernels))
for io, order in enumerate(orders):
    assert np.allclose(expanded_kernels[order-1], kernels[io])


# Different shape
shp = (3, 10000, 2, 1, 10)
kernels_part = np.random.rand(*shp) + 1j*np.random.rand(*shp)
new_shp = (max(orders), 10000, 2, 1, 10)
expanded_kernels_part = add_empty_orders(kernels_part, orders)

assert expanded_kernels_part.shape == new_shp
assert np.allclose(np.sum(kernels_part), np.sum(expanded_kernels_part))
for io, order in enumerate(orders):
    assert np.allclose(expanded_kernels_part[order-1], kernels_part[io])

print "SUCCESS!"