import numpy as np
import matplotlib.pyplot as plt
import matplotlib

vx = np.array([])
vy = np.array([])
vz = np.array([])
f = np.array([])

file = np.loadtxt("/Users/ldyke/Desktop/Dartmouth/HSResearch/Code/Kepler/Python Orbit Code/datafiles/5pi6_6p2e9_test.txt", delimiter=',')
#file = np.loadtxt("C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/datafiles/pi32_3p1e8.txt", delimiter=',')

#file = file[np.any(file > 1, axis=1)]

vx = file[:,0]
vy = file[:,1]
vz = file[:,2]
f = file[:,3]

fig3d = plt.figure()
fig3d.set_figwidth(10)
fig3d.set_figheight(7)
ax3d = plt.axes(projection='3d')
#scatterplot = ax3d.scatter3D(vx[:], vy[:], vz[:], c=f[:], cmap='plasma', s=.02, vmin=(.75-.243/np.e), vmax=(.75+.243*np.e))
scatterplot = ax3d.scatter3D(vx[:], vy[:], vz[:], c=f[:], cmap='plasma', s=.005)
cb = fig3d.colorbar(scatterplot)
ax3d.set_xlabel("$v_x$ at Target Point (km/s)")
ax3d.set_ylabel("$v_y$ at Target Point (km/s)")
ax3d.set_zlabel("$v_z$ at Target Point (km/s)")
#ax3d.view_init(90,270)
#ax3d.view_init(0,270)
#ax3d.view_init(0,180)
#ax3d.set_xlim([-25, 25])
#ax3d.set_ylim([-25, 25])
#ax3d.set_zlim([-25, 25])
ax3d.set_title("Phase space population at target (t $\\approx$ -1.458 years) drawn from Maxwellian at 100 au centered on vx = -26 km/s \
    \n Target at (.9952 au, .0980 au), Time Resolution Close to Target = 1500 s")
plt.show()