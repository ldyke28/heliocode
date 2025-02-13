import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy

# Paper 2 maxima
#vxarray = np.array([-24.48, -25.98, -28.82, -31.88, -32.82, -33.06, -33.16, -33.00, -31.74, -28.5, -28.58, -32.34])
#vyarray = np.array([-4.1, 1.52, 3.7, 6.3, 6.84, 6.88, 6.86, 6.54, 4.78, 0.64, -4.32, 16.26])
fp2 = np.array([8.04*10**(-8), 1.30*10**(-7), 2.65*10**(-7), 3.85*10**(-7), 4.27*10**(-7), 4.38*10**(-7), 4.42*10**(-7), 4.34*10**(-7), 3.78*10**(-7), 2.54*10**(-7), 1.41*10**(-7), 3.30*10**(-8)])

# Paper 1 maxima
#vxarray = np.array([-28.76, -29.54, -33, -35.38, -36.2, -36.45, -36.52, -36.38, -35.26, -32.58, -31.74, -25.78])
#vyarray = np.array([1.22, 2.44, 6.92, 9.2, 9.68, 9.7, 9.7, 9.38, 7.7, 4.08, 2.94, -5.06])
fp1 = np.array([4.87*10**(-7), 5.88*10**(-7), 1.11*10**(-6), 1.56*10**(-6), 1.73*10**(-6), 1.79*10**(-6), 1.81*10**(-6), 1.79*10**(-6), 1.58*10**(-6), 1.14*10**(-6), 1.01*10**(-6), 2.02*10**(-7)])

times = np.array([-4, -3.824, -3, -2, -1, 0, 1, 2, 3, 3.824, 4, 5.5])

fsize = 16
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot()
plt.grid()
ax.set_yscale('log')
plt.plot(times, fp1, color='b', label='Dyke & Müller 2024 Model')
plt.plot(times, fp2, color='r', label='Current Paper Model')
plt.xticks(fontsize=fsize)
plt.yticks(fontsize=fsize)
plt.xlabel("Time in Solar Cycle (yrs)", fontsize=fsize)
plt.ylabel("Maximum PSD Value (cm$^{-3}$ km$^{-3}$ s$^3$)", fontsize=fsize)
plt.legend(fontsize=10)
plt.show()