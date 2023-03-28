import numpy as np
import matplotlib.pyplot as plt
import matplotlib

ThreeD = True
# Loading in the file to be unpacked
#file = np.loadtxt("/Users/ldyke/Desktop/Dartmouth/HSResearch/Code/Kepler/Python Orbit Code/datafiles/pi_t0.txt", delimiter=',')
file = np.loadtxt("C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/datafiles/pi_t0_direct.txt", delimiter=',')
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
    scatterplot = ax3d.scatter3D(vx[:], vy[:], vz[:], c=f[:], cmap='plasma', s=.001)
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