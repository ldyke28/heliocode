import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy

bvdyke = np.array([28.814, 28.559, 27.074, 22.682, 17.352, 16.472, 16.907, 20.815, 25.730, 28.065, 28.685, 28.794])
bvizmodenov = np.array([29.832, 29.04, 22.176, 19.008, 20.328, 13.728, 15.84, 23.496, 25.344, 27.456, 29.04, 29.964])

bvdyke = bvdyke*(-1)
bvizmodenov = bvizmodenov*(-1)

times = np.array([1, 2, 3, 4, 5, 5.5, 6, 7, 8, 9, 10, 11])

fsize = 16
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot()
plt.grid()
#ax.set_yscale('log')
plt.plot(times, bvdyke, color='b', label='Current Paper Model')
plt.plot(times, bvizmodenov, color='r', label='Izmodenov et al. 2013 Model')
plt.xlim([1,11])
plt.xticks(fontsize=fsize)
plt.yticks(fontsize=fsize)
plt.xlabel("Time in Solar Cycle (yrs)", fontsize=fsize)
plt.ylabel("Radial Bulk Velocity Component $v_r$ (km s$^{-1}$)", fontsize=fsize)
plt.legend(fontsize=10)
plt.show()