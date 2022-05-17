import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import odeint
import scipy
from mpl_toolkits import mplot3d

mode = 3

# Value for 1 au (astronomical unit) in meters
au = 1.496*10**11
msolar = 1.98847*10**30 # mass of the sun in kg
G = 6.6743*10**(-11) # value for gravitational constant in SI units

# Location of the sun in [x,y,z] - usually this will be at 0, but this makes it flexible just in case
sunpos = np.array([0,0,0])
ibexpos = np.array([.97*au, .2*au, 0])

# INITIAL CONDITIONS for both position and velocity (in SI units - m and m/s)
ttotal = 7000000000
tstep = 10000
if mode==1:
    t = np.arange(0, ttotal, tstep)
if mode==2:
    t = np.arange(6276000000, 0, -tstep)
if mode==3:
    t = np.arange(6276000000, 5000000000, -tstep)
tscale = int(.7*ttotal/tstep)
#tscale = 0


yres = au/300
zres = au/2
yics = np.arange(.205*au, .265*au, yres)
#yics = np.array([.2913*au])
zics = np.arange(1*au, 20*au, zres)
xic = 1000*au

vxres = 400
vyres = 5
vxics = np.arange(-29000, -23000, vxres)
#vyics = np.arange(-25, 0, vyres)
vyics = np.array([0])
vx0 = -26000
vy0 = 0
vz0 = 0

# Initial Conditions for orbit starting at point of interest
xstart = ibexpos[0]
ystart = ibexpos[1]
zstart = ibexpos[2]
vxstart = np.arange(-35000, -29000, 200)
vystart = np.arange(-1100, -700, 20)
vzstart = 0




def radPressure(t):
    # dummy function to model radiation pressure
    # takes the time as input and returns the radiation pressure function at that time
    #return (np.sin(2*np.pi*(t/347000000)))**2 + .5
    #return .7
    return 0


# extra radiation pressure functions for overlayed plots
def rp2(t):
    return .7

def rp3(t):
    return (np.sin(2*np.pi*(t/347000000)))**2

def rp4(t):
    return .5 + (np.sin(2*np.pi*(t/347000000)))**2

def dr_dt(x,t,rp):
    # integrating differential equation for gravitational force. x[0:2] = x,y,z and x[3:5] = vx,vy,vz
    # dx0-2 = vx, vy, and vz, dx3-5 = ax, ay, and az
    r = np.sqrt((sunpos[0]-x[0])**2 + (sunpos[1]-x[1])**2 + (sunpos[2]-x[2])**2)
    dx0 = x[3]
    dx1 = x[4]
    dx2 = x[5]
    dx3 = (msolar*G/(r**3))*(sunpos[0]-x[0])*(1-rp(t))
    dx4 = (msolar*G/(r**3))*(sunpos[1]-x[1])*(1-rp(t))
    dx5 = (msolar*G/(r**3))*(sunpos[2]-x[2])*(1-rp(t))
    return [dx0, dx1, dx2, dx3, dx4, dx5]

# 3D code
"""trajs = np.zeros((t.size,6,yics.size*zics.size))

for i in range(yics.size):
    for j in range(zics.size):
        init = [xic, yics[i-1], zics[j-1], vx0, vy0, vz0]
        trajs[:,:,(i-1)*zics.size + (j-1)] = odeint(dr_dt, init, t, args=(radPressure,))

for i in range(yics.size):
    for j in range(zics.size):
        for k in range(t.size):
            if np.sqrt((trajs[k,0,(i-1)*zics.size + (j-1)]-ibexpos[0])**2 + (trajs[k,1,(i-1)*zics.size + (j-1)]-ibexpos[1])**2
            + (trajs[k,2,(i-1)*zics.size + (j-1)]-ibexpos[2])**2) < .01*au:
                print(trajs[k,:,(i-1)*zics.size + (j-1)])
                print(k)
                print(yics[i-1])
                print(zics[j-1])
                print('------------------------')"""

# 2D code
"""trajs = np.zeros((t.size,6,yics.size))
for i in range(yics.size):
    init = [xic, yics[i-1], 0, vx0, vy0, vz0]
    trajs[:,:,i-1] = odeint(dr_dt, init, t, args=(radPressure,))

for i in range(yics.size):
    for k in range(t.size):
        if np.sqrt((trajs[k,0,i-1]-ibexpos[0])**2 + (trajs[k,1,i-1]-ibexpos[1])**2 + (trajs[k,2,i-1]-ibexpos[2])**2) < .001*au:
            print(trajs[k,:,i-1])
            print(t[k])
            print(yics[i-1])
            print('-------------------------')"""


# velocity scanning code
if mode==1:
    trajs = np.zeros((t.size,6,yics.size*vxics.size*vyics.size))
    storeyic = np.array([])
    storevxic = np.array([])
    storevyic = np.array([])
    storefinalvx = np.array([])
    storefinalvy = np.array([])
    storet = np.array([])
    for i in range(yics.size):
        for j in range(vxics.size):
            for q in range(vyics.size):
                init = [xic, yics[i-1], 0, vxics[j-1], vyics[q-1], vz0]
                trajs[:,:,(i-1)*(vxics.size * vyics.size) + (j-1)*vyics.size + (q-1)] = odeint(dr_dt, init, t, args=(rp4,))
                for k in range(t.size - tscale):
                    rnew = np.sqrt((trajs[k+tscale,0,(i-1)*(vxics.size * vyics.size) + (j-1)*vyics.size + (q-1)]-ibexpos[0])**2 
                    + (trajs[k+tscale,1,(i-1)*(vxics.size * vyics.size) + (j-1)*vyics.size + (q-1)]-ibexpos[1])**2 
                    + (trajs[k+tscale,2,(i-1)*(vxics.size * vyics.size) + (j-1)*vyics.size + (q-1)]-ibexpos[2])**2)
                    rold = np.sqrt((trajs[k+tscale-1,0,(i-1)*(vxics.size * vyics.size) + (j-1)*vyics.size + (q-1)]-ibexpos[0])**2 
                    + (trajs[k+tscale-1,1,(i-1)*(vxics.size * vyics.size) + (j-1)*vyics.size + (q-1)]-ibexpos[1])**2 
                    + (trajs[k+tscale-1,2,(i-1)*(vxics.size * vyics.size) + (j-1)*vyics.size + (q-1)]-ibexpos[2])**2)
                    thresh = .01*au
                    if rnew >= thresh and rold < thresh:
                        print(trajs[k+tscale-1,:,(i-1)*(vxics.size * vyics.size) + (j-1)*vyics.size + (q-1)])
                        print(t[k+tscale-1])
                        print(yics[i-1])
                        print(vxics[j-1])
                        print(vyics[q-1])
                        storeyic = np.append(storeyic, [yics[i-1]])
                        storevxic = np.append(storevxic, [vxics[j-1]])
                        storevyic = np.append(storevyic, [vyics[q-1]])
                        storefinalvx = np.append(storefinalvx, [trajs[k+tscale-1, 3, (i-1)*(vxics.size * vyics.size) + (j-1)*vyics.size + (q-1)]])
                        storefinalvy = np.append(storefinalvy, [trajs[k+tscale-1, 4, (i-1)*(vxics.size * vyics.size) + (j-1)*vyics.size + (q-1)]])
                        storet = np.append(storet, t[k+tscale-1])
                        print('-------------------------')

# code to check for proximity after calculating orbit
"""for i in range(yics.size):
    for j in range(vxics.size):
        for q in range(vyics.size):
            for k in range(t.size - tscale):
                rnew = np.sqrt((trajs[k+tscale,0,(i-1)*(vxics.size * vyics.size) + (j-1)*vyics.size + (q-1)]-ibexpos[0])**2 
                + (trajs[k+tscale,1,(i-1)*(vxics.size * vyics.size) + (j-1)*vyics.size + (q-1)]-ibexpos[1])**2 
                + (trajs[k+tscale,2,(i-1)*(vxics.size * vyics.size) + (j-1)*vyics.size + (q-1)]-ibexpos[2])**2)
                rold = np.sqrt((trajs[k+tscale-1,0,(i-1)*(vxics.size * vyics.size) + (j-1)*vyics.size + (q-1)]-ibexpos[0])**2 
                + (trajs[k+tscale-1,1,(i-1)*(vxics.size * vyics.size) + (j-1)*vyics.size + (q-1)]-ibexpos[1])**2 
                + (trajs[k+tscale-1,2,(i-1)*(vxics.size * vyics.size) + (j-1)*vyics.size + (q-1)]-ibexpos[2])**2)
                thresh = .01*au
                if rnew >= thresh and rold < thresh:
                    print(trajs[k+tscale-1,:,(i-1)*(vxics.size * vyics.size) + (j-1)*vyics.size + (q-1)])
                    print(t[k+tscale-1])
                    print(yics[i-1])
                    print(vxics[j-1])
                    print(vyics[q-1])
                    storeyic = np.append(storeyic, [yics[i-1]])
                    storevxic = np.append(storevxic, [vxics[j-1]])
                    storevyic = np.append(storevyic, [vyics[q-1]])
                    storefinalvx = np.append(storefinalvx, [trajs[k+tscale-1, 3, (i-1)*(vxics.size * vyics.size) + (j-1)*vyics.size + (q-1)]])
                    storefinalvy = np.append(storefinalvy, [trajs[k+tscale-1, 4, (i-1)*(vxics.size * vyics.size) + (j-1)*vyics.size + (q-1)]])
                    storet = np.append(storet, t[k+tscale-1])
                    print('-------------------------')"""

if mode==3:
    farvx = np.array([])
    farvy = np.array([])
    fart = np.array([])
    backtraj = np.zeros((t.size, 6, vxstart.size*vystart.size))
    for i in range(vxstart.size):
        for j in range(vystart.size):
            init = [xstart, ystart, zstart, vxstart[i-1], vystart[j-1], vzstart]
            backtraj[:,:,(i-1)*vystart.size + (j-1)] = odeint(dr_dt, init, t, args=(rp4,))
            for k in range(t.size):
                if backtraj[k,0,(i-1)*vystart.size + (j-1)] >= 100*au and backtraj[k-1,0,(i-1)*vystart.size + (j-1)] <= 100*au:
                    print(backtraj[k-1,:,(i-1)*vystart.size + (j-1)])
                    print(t[k-1])
                    farvx = np.append(farvx, [backtraj[k-1,3,(i-1)*vystart.size + (j-1)]])
                    farvy = np.append(farvy, [backtraj[k-1,4,(i-1)*vystart.size + (j-1)]])
                    fart = np.append(fart, [t[k-1]])


# code for plotting multiple trajectories with different radiation pressures
"""init = np.zeros((6,4))
init[0,:] = xic
init[1,:] = yics[:]
init[2,:] = 0
init[3,:] = vx0
init[4,:] = vy0
init[5,:] = vz0
trajs = np.zeros((t.size,6,yics.size))
trajs[:,:,0] = odeint(dr_dt, init[:,0], t, args=(radPressure,))
trajs[:,:,1] = odeint(dr_dt, init[:,1], t, args=(rp2,))
trajs[:,:,2] = odeint(dr_dt, init[:,2], t, args=(rp3,))
trajs[:,:,3] = odeint(dr_dt, init[:,3], t, args=(rp4,))"""


# single trajectory plotting code
if mode==2:
    init = [ibexpos[0], ibexpos[1], ibexpos[2], -32876.78, -978.381438, 0]
    singletraj = odeint(dr_dt, init, t, args=(rp4,))
    for k in range(t.size):
            if singletraj[k,0] >= 100*au:
                print(singletraj[k,:])
                print(t[k])
                break


#new = odeint(dr_dt, [1000*au, 5*au, 5*au, vx0, vy0, vz0], t)

print('Finished')
#print(new[-1,:])

if mode==2:
    zer = [0]
    fig3d = plt.figure()
    ax3d = plt.axes(projection='3d')
    ax3d.plot3D(singletraj[:,0], singletraj[:,1], singletraj[:,2], 'darkmagenta')
    #ax3d.plot3D(trajs[:,0,1], trajs[:,1,1], trajs[:,2,1], 'gold', linestyle='--')
    #ax3d.plot3D(trajs[:,0,2], trajs[:,1,2], trajs[:,2,2], 'forestgreen', linestyle=':')
    #ax3d.plot3D(trajs[:,0,3], trajs[:,1,3], trajs[:,2,3], 'firebrick', linestyle='-.')
    ax3d.scatter3D(zer,zer,zer,c='red')
    ax3d.scatter3D([.97*au],[.2*au],[0], c='springgreen')
    ax3d.set_xlim3d(left = -1*au, right = 100*au)
    ax3d.set_ylim3d(bottom = -5*au, top = 5*au)
    ax3d.set_zlim3d(bottom = -1*au, top = 1*au)
    plt.show()
if mode==1:
    attribs = np.vstack((storefinalvx, storefinalvy, storet))
    print(attribs.size)
    attribs = attribs[:, attribs[2,:].argsort()]
    vxtot = 0
    vytot = 0
    ttot = 0
    count = 0
    for i in range (storet.size):
        print(i, '|', attribs[0,i], '|', attribs[1,i], '|', attribs[2,i])
        if storefinalvy[i-1]<0:
            vxtot = vxtot + storefinalvx[i-1]
            vytot = vytot + storefinalvy[i-1]
            ttot = ttot + storet[i-1]
            count = count + 1

    vxavg = vxtot/count
    vyavg = vytot/count
    tavg = ttot/count
    print('~~~~~~~~~~~~~~~~~~~~~')
    print(vxavg, '||', vyavg, '||', tavg)

if mode==3:
    plt.scatter(farvx[:]/1000, farvy[:]/1000, c=fart[:], marker='o', cmap='viridis')
    cb = plt.colorbar()
    plt.xlabel("vx at 100 au in km/s")
    plt.ylabel("vy at 100 au in km/s")
    cb.set_label('Time at which orbit passes through 100 au (s)')
    plt.show()

"""plt.scatter(storeyic[:]/au, storevxic[:]/1000, c=storet[:], marker='o', cmap='magma')
cb = plt.colorbar()
plt.xlabel("Initial condition for y in au")
plt.ylabel("Initial condition for vx in km/s")
cb.set_label('Time of passing (s)')
plt.show()

plt.scatter(storefinalvx[:], storefinalvy[:], c=storet[:], marker='o', cmap='magma')
cb = plt.colorbar()
plt.xlabel("vx while passing the target point (m/s)")
plt.ylabel("vy while passing the target point (m/s)")
cb.set_label('Time of passing (s)')
plt.show()"""

"""fig3d = plt.figure()
ax3d = plt.axes(projection='3d')
ax3d.scatter3D(storevxic[:]/1000, storevyic[:]/1000, storeyic[:]/au, c='darkcyan')
ax3d.set_xlabel("Initial condition for vx in km/s")
ax3d.set_ylabel("Initial condition for vy in km/s")
ax3d.set_zlabel("Initial condition for y in au")
plt.show()"""