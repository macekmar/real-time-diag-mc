from ctint_keldysh import make_g0_lattice_1d
from matplotlib import pyplot as plt
import numpy as np

p = {}
p['beta'] = 0.1
p['mu'] = -1.5
p['epsilon'] = 0.
p['hop'] = 1.
p['tmax_gf0'] = 100
p['Nt_gf0'] = 1000
p['nb_sites'] = 10
p['Nb_k_pts'] = 50

print 'Generating g0 ...'
g0_less1, g0_grea1 = make_g0_lattice_1d(**p)
print 'done.'

p['Nb_k_pts'] = 100
g0_less2, g0_grea2 = make_g0_lattice_1d(**p)

p['Nb_k_pts'] = 150
g0_less3, g0_grea3 = make_g0_lattice_1d(**p)

p['Nb_k_pts'] = 200
g0_less4, g0_grea4 = make_g0_lattice_1d(**p)

times = np.linspace(-90, 90, 500)

print g0_less1(1.)[0, 3]

g0_less1_v = np.array([g0_less1(t)[2, 8] for t in times])
g0_grea1_v = np.array([g0_grea1(t)[2, 8] for t in times])
g0_less2_v = np.array([g0_less2(t)[2, 8] for t in times])
g0_grea2_v = np.array([g0_grea2(t)[2, 8] for t in times])
g0_less3_v = np.array([g0_less3(t)[2, 8] for t in times])
g0_grea3_v = np.array([g0_grea3(t)[2, 8] for t in times])
g0_less4_v = np.array([g0_less4(t)[2, 8] for t in times])
g0_grea4_v = np.array([g0_grea4(t)[2, 8] for t in times])

fig, ax = plt.subplots(3, 1)
ax[0].plot(times, g0_less1_v.real, 'b')
ax[0].plot(times, g0_less1_v.imag, 'r')
ax[0].plot(times, g0_less2_v.real, 'g')
ax[0].plot(times, g0_less2_v.imag, 'm')
ax[1].plot(times, g0_grea1_v.real, 'b')
ax[1].plot(times, g0_grea1_v.imag, 'r')
ax[1].plot(times, g0_grea2_v.real, 'g')
ax[1].plot(times, g0_grea2_v.imag, 'm')
ax[2].plot(times, np.abs(g0_less1_v - g0_less2_v), 'b')
ax[2].plot(times, np.abs(g0_grea1_v - g0_grea2_v), 'b')
ax[2].plot(times, np.abs(g0_less2_v - g0_less3_v), 'g')
ax[2].plot(times, np.abs(g0_grea2_v - g0_grea3_v), 'g')
ax[2].plot(times, np.abs(g0_less3_v - g0_less4_v), 'r')
ax[2].plot(times, np.abs(g0_grea3_v - g0_grea4_v), 'r')
ax[2].semilogy()
plt.show()

exit()

t = 10.
cmap = plt.get_cmap('seismic_r')
vmax = max(np.max(np.abs(g0_less1(t))), np.max(np.abs(g0_grea1(t))))
fig, ax = plt.subplots(2, 2)
ax[0, 0].matshow(g0_less1(t).real, cmap=cmap, vmin=-vmax, vmax=vmax)
ax[0, 0].set_title('lesser real')
ax[0, 1].matshow(g0_less1(t).imag, cmap=cmap, vmin=-vmax, vmax=vmax)
ax[0, 1].set_title('lesser imag')
ax[1, 0].matshow(g0_grea1(t).real, cmap=cmap, vmin=-vmax, vmax=vmax)
ax[1, 0].set_title('greater real')
ax[1, 1].matshow(g0_grea1(t).imag, cmap=cmap, vmin=-vmax, vmax=vmax)
ax[1, 1].set_title('greater imag')
plt.show()

