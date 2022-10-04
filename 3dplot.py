import numpy as np
import matplotlib.pyplot as plt
import matplotlib

vx = np.array([])
vy = np.array([])
vz = np.array([])
f = np.array([])

file = np.loadtxt("/Users/ldyke/Desktop/Dartmouth/HSResearch/Code/Kepler/Python Orbit Code/datafiles/cosexprp_31pi32_t0_center_cosexppi_test.txt", delimiter=',')

for i in range(np.shape(file)[0]):
    vx = np.append(vx, file[i,0])
    vy = np.append(vy, file[i,1])
    vz = np.append(vz, file[i,2])
    f = np.append(f, file[i,3])

fig3d = plt.figure()
fig3d.set_figwidth(10)
fig3d.set_figheight(7)
ax3d = plt.axes(projection='3d')
#scatterplot = ax3d.scatter3D(vx[:], vy[:], vz[:], c=f[:], cmap='plasma', s=.02, vmin=(.75-.243/np.e), vmax=(.75+.243*np.e))
scatterplot = ax3d.scatter3D(vx[:], vy[:], vz[:]/1000, c=f[:], cmap='plasma')
cb = fig3d.colorbar(scatterplot)
ax3d.set_xlabel("$v_x$ at Target Point (km/s)")
ax3d.set_ylabel("$v_y$ at Target Point (km/s)")
ax3d.set_zlabel("$v_z$ at Target Point (km/s)")
#ax3d.view_init(90,270)
ax3d.set_title("Phase space population at target (t = 0 years) drawn from Maxwellian at 100 au centered on vx = -26 km/s \
    \n Target at (-.9952 au, .0980 au), Time Resolution Close to Target = 1000 s")
plt.show()