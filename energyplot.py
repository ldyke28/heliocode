import numpy as np
import matplotlib.pyplot as plt
import matplotlib

vx = np.array([])
vy = np.array([])
vz = np.array([])
f = np.array([])

file = np.loadtxt("/Users/ldyke/Desktop/Dartmouth/HSResearch/Code/Kepler/Python Orbit Code/datafiles/pi_t0.txt", delimiter=',')
#file = np.loadtxt("C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/datafiles/5pi6_2p262502e8.txt", delimiter=',')

vx = file[:,0]
vy = file[:,1]
vz = file[:,2]
f = file[:,3]

scaledenergy = vx[:]**2 + vy[:]**2 + vz[:]**2

fig = plt.figure()
fig.set_figwidth(10)
fig.set_figheight(7)

#plt.scatter(scaledenergy, f)
plt.hist(scaledenergy, bins=300)
plt.xlabel("Rescaled Particle Energy at Target Point $\\frac{2E}{m}$ ($\\frac{m^2}{s^2}$)")
#plt.ylabel("Rescaled Phase Space Density (unitless)")
plt.ylabel("Counts")
plt.title("Energy Distribution for Trajectories Reaching (-1 au, 0 au, 0 au) at t = 0 years")
plt.show()