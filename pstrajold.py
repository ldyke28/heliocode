import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import odeint
import scipy
from mpl_toolkits import mplot3d

# MODE LIST
# 1 = generate a list of trajectories that come within proximity
# 2 = plot an individual trajectory traced backward from point of interest
# 3 = generate phase space diagram
mode = 3
circle = False

# Value for 1 au (astronomical unit) in meters
au = 1.496*10**11
msolar = 1.98847*10**30 # mass of the sun in kg
G = 6.6743*10**(-11) # value for gravitational constant in SI units

# Location of the sun in [x,y,z] - usually this will be at 0, but this makes it flexible just in case
# Second line is location of the point of interest in the same format (which is, generally, where we want IBEX to be)
sunpos = np.array([0,0,0])
ibexpos = np.array([-.97*au, .2*au, 0])

# INITIAL CONDITIONS for both position and velocity (in SI units - m and m/s)
ttotal = 7000000000
tstep = 10000
if mode==1:
    t = np.arange(0, ttotal, tstep)
if mode==2:
    t = np.arange(6276000000, 0, -tstep)
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
vxstart = np.arange(-51000, -15000, 1200)
vystart = np.arange(-38000, -10000, 1000)
#vxstart = np.arange(-40000, -10000, 500)
#vystart = np.arange(-2000, 2000, 200)
vzstart = 0
if mode==3:
    #startt = 5598410000
    startt = 5780000000
    t = np.arange(startt, 3000000000, -tstep)
    if circle: # if you want to initially use a circle in phase space
        vx1 = -51000
        vx2 = -41000
        vy1 = -2500
        vy2 = -1700
        vxinit = np.arange(vx1, vx2, (vx2-vx1)/50)
        vyinit = np.arange(vy1, vy2, (vy2-vy1)/50)
        vxstart = np.array([])
        vystart = np.array([])
        for i in range(vxinit.size):
            for j in range(vyinit.size):
                #if np.sqrt((vxinit[i]-(vx1+vx2)/2)**2 + (vyinit[j]-(vy1+vy2)/2)**2) <= (vy2-vy1)/2:
                #    vxstart = np.append(vxstart, [vxinit[i]])
                #    vystart = np.append(vystart, [vyinit[j]])
                if np.sqrt((i-(vxinit.size+1)/2)**2 + (j-(vyinit.size+1)/2)**2) <= vxinit.size/2:
                    vxstart = np.append(vxstart, [vxinit[i]])
                    vystart = np.append(vystart, [vyinit[j]])



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
        init = [xic, yics[i], zics[j], vx0, vy0, vz0]
        trajs[:,:,(i)*zics.size + (j)] = odeint(dr_dt, init, t, args=(radPressure,))

for i in range(yics.size):
    for j in range(zics.size):
        for k in range(t.size):
            if np.sqrt((trajs[k,0,(i)*zics.size + (j)]-ibexpos[0])**2 + (trajs[k,1,(i)*zics.size + (j)]-ibexpos[1])**2
            + (trajs[k,2,(i)*zics.size + (j)]-ibexpos[2])**2) < .01*au:
                print(trajs[k,:,(i)*zics.size + (j)])
                print(k)
                print(yics[i])
                print(zics[j])
                print('------------------------')"""

# 2D code
"""trajs = np.zeros((t.size,6,yics.size))
for i in range(yics.size):
    init = [xic, yics[i], 0, vx0, vy0, vz0]
    trajs[:,:,i] = odeint(dr_dt, init, t, args=(radPressure,))

for i in range(yics.size):
    for k in range(t.size):
        if np.sqrt((trajs[k,0,i]-ibexpos[0])**2 + (trajs[k,1,i]-ibexpos[1])**2 + (trajs[k,2,i]-ibexpos[2])**2) < .001*au:
            print(trajs[k,:,i])
            print(t[k])
            print(yics[i])
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
                init = [xic, yics[i], 0, vxics[j], vyics[q], vz0]
                trajs[:,:,(i)*(vxics.size * vyics.size) + (j)*vyics.size + (q)] = odeint(dr_dt, init, t, args=(rp3,))
                for k in range(t.size - tscale):
                    rnew = np.sqrt((trajs[k+tscale,0,(i)*(vxics.size * vyics.size) + (j)*vyics.size + (q)]-ibexpos[0])**2 
                    + (trajs[k+tscale,1,(i)*(vxics.size * vyics.size) + (j)*vyics.size + (q)]-ibexpos[1])**2 
                    + (trajs[k+tscale,2,(i)*(vxics.size * vyics.size) + (j)*vyics.size + (q)]-ibexpos[2])**2)
                    rold = np.sqrt((trajs[k+tscale-1,0,(i)*(vxics.size * vyics.size) + (j)*vyics.size + (q)]-ibexpos[0])**2 
                    + (trajs[k+tscale-1,1,(i)*(vxics.size * vyics.size) + (j)*vyics.size + (q)]-ibexpos[1])**2 
                    + (trajs[k+tscale-1,2,(i)*(vxics.size * vyics.size) + (j)*vyics.size + (q)]-ibexpos[2])**2)
                    thresh = .01*au
                    if rnew >= thresh and rold < thresh:
                        print(trajs[k+tscale-1,:,(i)*(vxics.size * vyics.size) + (j)*vyics.size + (q)])
                        print(t[k+tscale-1])
                        print(yics[i])
                        print(vxics[j])
                        print(vyics[q])
                        storeyic = np.append(storeyic, [yics[i]])
                        storevxic = np.append(storevxic, [vxics[j]])
                        storevyic = np.append(storevyic, [vyics[q]])
                        storefinalvx = np.append(storefinalvx, [trajs[k+tscale-1, 3, (i)*(vxics.size * vyics.size) + (j)*vyics.size + (q)]])
                        storefinalvy = np.append(storefinalvy, [trajs[k+tscale-1, 4, (i)*(vxics.size * vyics.size) + (j)*vyics.size + (q)]])
                        storet = np.append(storet, t[k+tscale-1])
                        print('-------------------------')

# code to check for proximity after calculating orbit
"""for i in range(yics.size):
    for j in range(vxics.size):
        for q in range(vyics.size):
            for k in range(t.size - tscale):
                rnew = np.sqrt((trajs[k+tscale,0,(i)*(vxics.size * vyics.size) + (j)*vyics.size + (q)]-ibexpos[0])**2 
                + (trajs[k+tscale,1,(i)*(vxics.size * vyics.size) + (j)*vyics.size + (q)]-ibexpos[1])**2 
                + (trajs[k+tscale,2,(i)*(vxics.size * vyics.size) + (j)*vyics.size + (q)]-ibexpos[2])**2)
                rold = np.sqrt((trajs[k+tscale-1,0,(i)*(vxics.size * vyics.size) + (j)*vyics.size + (q)]-ibexpos[0])**2 
                + (trajs[k+tscale-1,1,(i)*(vxics.size * vyics.size) + (j)*vyics.size + (q)]-ibexpos[1])**2 
                + (trajs[k+tscale-1,2,(i)*(vxics.size * vyics.size) + (j)*vyics.size + (q)]-ibexpos[2])**2)
                thresh = .01*au
                if rnew >= thresh and rold < thresh:
                    print(trajs[k+tscale-1,:,(i)*(vxics.size * vyics.size) + (j)*vyics.size + (q)])
                    print(t[k+tscale-1])
                    print(yics[i])
                    print(vxics[j])
                    print(vyics[q])
                    storeyic = np.append(storeyic, [yics[i]])
                    storevxic = np.append(storevxic, [vxics[j]])
                    storevyic = np.append(storevyic, [vyics[q]])
                    storefinalvx = np.append(storefinalvx, [trajs[k+tscale-1, 3, (i)*(vxics.size * vyics.size) + (j)*vyics.size + (q)]])
                    storefinalvy = np.append(storefinalvy, [trajs[k+tscale-1, 4, (i)*(vxics.size * vyics.size) + (j)*vyics.size + (q)]])
                    storet = np.append(storet, t[k+tscale-1])
                    print('-------------------------')"""

# code for tracking phase space at distance of x = 100 au away
if mode==3:
    farvx = np.array([])
    farvy = np.array([])
    fart = np.array([])
    maxwcolor = np.array([])
    backtraj = np.zeros((t.size, 6, vxstart.size*vystart.size))
    for i in range(vxstart.size):
        for j in range(vystart.size):
            init = [xstart, ystart, zstart, vxstart[i], vystart[j], vzstart]
            backtraj[:,:,(i)*vystart.size + (j)] = odeint(dr_dt, init, t, args=(rp3,))
            for k in range(t.size):
                if backtraj[k,0,(i)*vystart.size + (j)] >= 100*au and backtraj[k-1,0,(i)*vystart.size + (j)] <= 100*au:
                    print(backtraj[k-1,:,(i)*vystart.size + (j)])
                    print(t[k-1])
                    # radius in paper given to be 14 km/s
                    # only saving initial conditions corresponding to points that lie within this Maxwellian at x = 100 au
                    #if backtraj[k-1,3,(i)*vystart.size + (j)] <= -22000 and backtraj[k-1,3,(i)*vystart.size + (j)] >= -40000 and backtraj[k-1,4,(i)*vystart.size + (j)] <= 14000 and backtraj[k-1,4,(i)*vystart.size + (j)] >= -14000:
                    if np.sqrt((backtraj[k-1,3,(i)*vystart.size + (j)]+26000)**2 + (backtraj[k-1,4,(i)*vystart.size + (j)])**2) <= 14000:
                        farvx = np.append(farvx, [backtraj[0,3,(i)*vystart.size + (j)]])
                        farvy = np.append(farvy, [backtraj[0,4,(i)*vystart.size + (j)]])
                        fart = np.append(fart, [startt - t[k-1]])
                        maxwcolor = np.append(maxwcolor, [np.exp(-((backtraj[k-1,3,(i)*vystart.size + (j)]+26000)**2 + backtraj[k-1,4,(i)*vystart.size + (j)]**2)/(14000)**2)])
                    #farvx = np.append(farvx, [backtraj[k-1,3,(i)*vystart.size + (j)]])
                    #farvy = np.append(farvy, [backtraj[k-1,4,(i)*vystart.size + (j)]])
                    #fart = np.append(fart, [startt - t[k-1]])
                    #farvx = np.append(farvx, [backtraj[0,3,(i)*vystart.size + (j)]])
                    #farvy = np.append(farvy, [backtraj[0,4,(i)*vystart.size + (j)]])
                    #fart = np.append(fart, [startt - t[k-1]])
                    #maxwcolor = np.append(maxwcolor, [np.exp(-((backtraj[k-1,3,(i)*vystart.size + (j)]+26000)**2 + backtraj[k-1,4,(i)*vystart.size + (j)]**2)/(14000)**2)])


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
        if storefinalvy[i]<0:
            vxtot = vxtot + storefinalvx[i]
            vytot = vytot + storefinalvy[i]
            ttot = ttot + storet[i]
            count = count + 1

    vxavg = vxtot/count
    vyavg = vytot/count
    tavg = ttot/count
    print('~~~~~~~~~~~~~~~~~~~~~')
    print(vxavg, '||', vyavg, '||', tavg)

if mode==3:
    f = plt.figure()
    f.set_figwidth(9)
    f.set_figheight(6)
    plt.scatter(farvx[:]/1000, farvy[:]/1000, c=maxwcolor[:], marker='o', cmap='plasma')
    cb = plt.colorbar()
    plt.xlabel("vx at Target in km/s")
    plt.ylabel("vy at Target in km/s")
    #cb.set_label('Time at which orbit passes through 100 au (s)')
    #cb.set_label('Travel Time from 100 au to Point of Interest (s)')
    cb.set_label('f(r,v,t)')
    #plt.suptitle('Phase Space population at x = 100 au reaching initial position at t = 5700000000 s')
    plt.suptitle('Phase space population at target (t = 5.78e9 s) drawn from Maxwellian at 100 au centered on vx = -26 km/s')
    #plt.title('Target (-.97au, .2au): vx range -51500 m/s to -30500 m/s, vy range -30000 m/s to 30000 m/s')
    plt.title('Target at (-.97 au, .2 au)')
    #plt.title('Initial test distribution centered on vx = -41.5 km/s, vy = -1.4 km/s')
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