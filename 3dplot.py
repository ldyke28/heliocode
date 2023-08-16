import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm

ThreeD = True
# Loading in the file to be unpacked
#file = np.loadtxt("/Users/ldyke/Desktop/Dartmouth/HSResearch/Code/Kepler/Python Orbit Code/datafiles/pi_t0.txt", delimiter=',')
file = np.loadtxt("C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/datafiles/5pi6_-4p5e7_lya_test.txt", delimiter=',')
#file2 = np.loadtxt("C:/Users/lucas/OneDrive/Documents/GitHub/heliocode/supplementaldata1.txt", delimiter=',')

#file = file[np.any(file > 1, axis=1)]

# Unpacking variables based on how they're saved in the code
vx = file[:,0]
vy = file[:,1]
vz = file[:,2]
f = file[:,3]

"""vx2 = file2[:,0]
vy2 = file2[:,1]
vz2 = file2[:,2]
f2 = file2[:,3]

vx = np.concatenate((vx, vx2))
vy = np.concatenate((vy, vy2))
vz = np.concatenate((vz, vz2))
f = np.concatenate((f, f2))"""


# Plotting data as a 3D scatter plot
if ThreeD == True:
    fig3d = plt.figure()
    fig3d.set_figwidth(10)
    fig3d.set_figheight(7)
    ax3d = plt.axes(projection='3d')
    #scatterplot = ax3d.scatter3D(vx[:], vy[:], vz[:], c=f[:], cmap='plasma', s=.02, vmin=(.75-.243/np.e), vmax=(.75+.243*np.e))
    #scatterplot = ax3d.scatter3D(vx[:], vy[:], vz[:], c=f[:], cmap='plasma', s=.001, alpha=.15)
    scatterplot = ax3d.scatter3D(vx[:], vy[:], vz[:], c=f[:], cmap='plasma', s=.1)
    cb = fig3d.colorbar(scatterplot)
    ax3d.set_xlabel("$v_x$ at Target Point (km/s)")
    ax3d.set_ylabel("$v_y$ at Target Point (km/s)")
    ax3d.set_zlabel("$v_z$ at Target Point (km/s)")
    # Can set initial viewing angles for the data
    #ax3d.view_init(90,270)
    #ax3d.view_init(0,270)
    #ax3d.view_init(0,180)
    # Can restrict the limits of the plot
    #ax3d.set_xlim([-25, 25])
    #ax3d.set_ylim([-25, 25])
    #ax3d.set_zlim([-25, 25])
    ax3d.set_title("Phase space population at target (t = 0 years) drawn from Maxwellian at 100 au centered on vx = -26 km/s \
        \n Target at (1 au, 0 au, 0 au), Time Resolution Close to Target = 1500 s")
    plt.show()
else:
    fig = plt.figure()
    fig.set_figwidth(10)
    fig.set_figheight(7)
    ax = plt.axes(projection='3d')
    scatterplot = plt.scatter(vx[:], vy[:], c=f[:], cmap='plasma')
    cb = fig.colorbar(scatterplot)
    plt.xlabel("$v_x$ at Target Point (km/s)")
    plt.ylabel("$v_y$ at Target Point (km/s)")
    plt.title("Phase space population at target (t = 0 years) drawn from Maxwellian at 100 au centered on vx = -26 km/s \
        \n Target at (1 au, 0 au), Time Resolution Close to Target = 1500 s")
    plt.show()


theta = 180
# section of code to calculate which trajectories could be observed by spacecraft - considers velocity shifts and viewing angle
vahwxy = 3.5 # half width of the total viewing angle width of the explorer probe in the x-y plane
vahwxyr = vahwxy*np.pi/180 # same width expressed in radians
vahwz = 3.5
vahwzr = vahwz*np.pi/180 # half width of total viewing angle above/below x-y plane in radians
vsc = 30000 # velocity of spacecraft in m/s
vxshifted = np.array([])
vyshifted = np.array([])
vzshifted = np.array([])
vxunshifted = np.array([])
vyunshifted = np.array([])
vzunshifted = np.array([])
trackvanglexy = np.array([])
trackvanglez = np.array([])
maxwcolorus = np.array([])
vsqshifted = np.array([])
thetarad = theta*np.pi/180 # expressing the value of theta in radians
# calculating the shift of the particle velocities into the spacecraft frame
vxshift = vx - vsc*np.cos(thetarad - np.pi/2)
vyshift = vy - vsc*np.sin(thetarad - np.pi/2)
vzshift = vz
vshifttotal = np.sqrt(vxshift**2 + vyshift**2 + vzshift**2)
vsquaredtotal = vxshift**2 + vyshift**2 + vzshift**2 # calculating total energies (v^2) associated with each trajectory in spacecraft frame
vanglexy = np.arccos(vxshift/(np.sqrt(vxshift**2 + vyshift**2))) # calculating the new azimuthal angle in which the velocity vector points for each trajectory
vanglez = np.arcsin(vzshift/vshifttotal) # same with the polar angle
for i in tqdm(range(vx.size)):
    i = vx.size - (i + 1)
    if vyshift[i] < 0:
        vanglexy[i] = 2*np.pi - vanglexy[i]
    if (thetarad + np.pi/2 - vahwxyr) < vanglexy[i] and (thetarad + np.pi/2 + vahwxyr) > vanglexy[i] \
    and -vahwzr < vanglez[i] and vahwzr > vanglez[i]:
        # appending values to the list of observable velocity shifted trajectories
        vxshifted = np.append(vxshifted, vxshift[i])
        vyshifted = np.append(vyshifted, vyshift[i])
        vzshifted = np.append(vzshifted, vzshift[i])
        vxunshifted = np.append(vxunshifted, vx[i])
        vyunshifted = np.append(vyunshifted, vy[i])
        vzunshifted = np.append(vzunshifted, vz[i])
        trackvanglexy = np.append(trackvanglexy, vanglexy[i])
        trackvanglez = np.append(trackvanglez, vanglez[i])
        maxwcolorus = np.append(maxwcolorus, f[i])
        vsqshifted = np.append(vsqshifted, vsquaredtotal[i])
    if i < vx.size - 100000:
        break

#print(vanglexy)
#print(vanglez)

print("Done sorting")
# plotting this set of trajectories
fsize = 16

fig3d = plt.figure()
fig3d.set_figwidth(10)
fig3d.set_figheight(7)
ax3d = plt.axes(projection='3d')
scatterplot = ax3d.scatter3D(vxunshifted[:], vyunshifted[:], vzunshifted[:], c=maxwcolorus[:], cmap='plasma', s=.001)
plt.rcParams.update({'font.size': fsize})
cb = fig3d.colorbar(scatterplot)
cb.set_label('Normalized Phase Space Density')
ax3d.set_xlabel("$v_x$ at Target Point (km/s)", fontsize=fsize)
ax3d.set_ylabel("$v_y$ at Target Point (km/s)", fontsize=fsize)
ax3d.set_zlabel("$v_z$ at Target Point (km/s)", fontsize=fsize)
# Can set initial viewing angles for the data
#ax3d.view_init(90,270)
#ax3d.view_init(0,270)
#ax3d.view_init(0,180)
# Can restrict the limits of the plot
#ax3d.set_xlim([-25, 25])
#ax3d.set_ylim([-25, 25])
#ax3d.set_zlim([-25, 25])
plt.show()

totalke = .5 * (1.6736*10**(-27)) * vsqshifted * 6.242*10**(18)

fig = plt.figure()
fig.set_figwidth(8)
fig.set_figheight(5)

plt.hist(totalke, bins=100, weights=maxwcolorus) # weighted by attenuated normalized phase space density
plt.xlabel("Particle Energy at Target Point in eV")
plt.ylabel("Weighted Counts")
plt.show()