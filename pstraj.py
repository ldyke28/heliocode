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
#contourplot = True # determines whether scatter (False) or contour (True) plot

# Value for 1 au (astronomical unit) in meters
au = 1.496*10**11
msolar = 1.98847*10**30 # mass of the sun in kg
G = 6.6743*10**(-11) # value for gravitational constant in SI units
# one year in s = 3.156e7 s
# Note to self: solar maximum in April 2014
oneyear = 3.156*10**7
finalt = 6366750000
#6.36674976e9 force free for cosexprp
tstep = 10000
tstepclose = 200
tstepfar = 200000
phase = 0

# Location of the sun in [x,y,z] - usually this will be at 0, but this makes it flexible just in case
# Second line is location of the point of interest in the same format (which is, generally, where we want IBEX to be)
sunpos = np.array([0,0,0])
#ibexpos = np.array([np.cos(np.pi*finalt/oneyear + phase)*au, np.sin(np.pi*finalt/oneyear + phase)*au, 0])
ibexpos = np.array([-.866*au, .5*au, 0])

# INITIAL CONDITIONS for both position and velocity (in SI units - m and m/s)
ttotal = 7000000000
if mode==1:
    t = np.arange(0, ttotal, tstep)
if mode==2:
    t = np.arange(finalt, 4500000000, -tstep)
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
vxstart = np.arange(-50000, -17000, 350)
vystart = np.arange(-20000, 10000, 350)
#vxstart = np.arange(-45000, -15000, 300)
#vystart = np.arange(16500, 30000, 150)
#vxstart = np.arange(-25000, 25000, 500)
#vystart = np.arange(-25000, 25000, 500)
#vxstart = np.arange(0000, 10000, 50)
#vystart = np.arange(-15000, -5000, 50)
vzstart = 0
if mode==3:
    #startt = 5598410000
    startt = finalt
    lastt = 4500000000
    tmid = startt - 200000000
    tclose = np.arange(startt, tmid, -tstepclose)
    tfar = np.arange(tmid, lastt, -tstepfar)
    t = np.concatenate((tclose, tfar))
    mode3dt = startt-lastt



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

def rp5(t):
    return .5 + (np.sin(np.pi*(t/347000000)))**2

def rp6(t):
    # taken from eq. 8 in https://articles.adsabs.harvard.edu/pdf/1995A%26A...296..248R
    omegat = 2*np.pi/(3.47*10**(8))*t
    return .75 + .243*np.cos(omegat - np.pi)*np.exp(np.cos(omegat - np.pi))

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


# code for tracking phase space at distance of x = 100 au away
if mode==3:
    farvx = np.array([])
    farvy = np.array([])
    fart = np.array([])
    maxwcolor = np.array([])
    backtraj = np.zeros((t.size, 6))
    for i in range(vxstart.size):
        for j in range(vystart.size):
            init = [xstart, ystart, zstart, vxstart[i], vystart[j], vzstart]
            # calculating trajectories for each initial condition in phase space given
            backtraj[:,:] = odeint(dr_dt, init, t, args=(rp6,))
            if any(np.sqrt((backtraj[:,0]-sunpos[0])**2 + (backtraj[:,1]-sunpos[1])**2 + (backtraj[:,2]-sunpos[2])**2) <= .00465*au):
                continue
            for k in range(t.size - tclose.size):
                if backtraj[k+tclose.size,0] >= 100*au and backtraj[k-1+tclose.size,0] <= 100*au:
                    #if np.sqrt((backtraj[0:k+1,0]-sunpos[0])**2 + (backtraj[0:k+1,1]-sunpos[1])**2 + (backtraj[0:k+1,2]-sunpos[2])**2).any() <= .00465*au:
                    #    break
                    kn = k+tclose.size
                    print(backtraj[kn-1,:])
                    print(t[kn-1])
                    # radius in paper given to be 14 km/s
                    # only saving initial conditions corresponding to points that lie within this Maxwellian at x = 100 au
                    #if backtraj[k-1,3,(i)*vystart.size + (j)] <= -22000 and backtraj[k-1,3,(i)*vystart.size + (j)] >= -40000 and backtraj[k-1,4,(i)*vystart.size + (j)] <= 14000 and backtraj[k-1,4,(i)*vystart.size + (j)] >= -14000:
                    if np.sqrt((backtraj[kn-1,3]+26000)**2 + (backtraj[kn-1,4])**2 + (backtraj[kn-1,5])**2) <= 14000:
                        #btintegrand = 1/(startt-t[k+1])*np.exp((np.sqrt((sunpos[0]-backtraj[0:k+1,0])**2 + \
                        #    (sunpos[1]-backtraj[0:k+1,1])**2 + (sunpos[2]-backtraj[0:k+1,2])**2)/(100*au)-1))
                        #PIrate = 10**(-7) *(1 + .7*(np.sin(np.pi*(t[0:k+1]/347000000)))**2)
                        omt = 2*np.pi/(3.47*10**(8))*t[0:kn+1]
                        PIrate2 = 10**(-7)*(1 + .7/(np.e + 1/np.e)*(np.cos(omt - np.pi)*np.exp(np.cos(omt - np.pi)) + 1/np.e))
                        r1 = 1*au
                        #oldrad = np.sqrt((sunpos[0]-backtraj[1:k+2,0])**2 + (sunpos[1]-backtraj[1:k+2,1])**2 + (sunpos[2]-backtraj[1:k+2,2])**2)
                        currentrad = np.sqrt((sunpos[0]-backtraj[0:kn+1,0])**2 + (sunpos[1]-backtraj[0:kn+1,1])**2 + (sunpos[2]-backtraj[0:kn+1,2])**2)
                        #rvecx = (-sunpos[0]+backtraj[1:k+2,0])/oldrad
                        #rvecy= (-sunpos[1]+backtraj[1:k+2,1])/oldrad
                        #rvecz = (-sunpos[2]+backtraj[1:k+2,2])/oldrad
                        nrvecx = (-sunpos[0]+backtraj[0:kn+1,0])/currentrad
                        nrvecy= (-sunpos[1]+backtraj[0:kn+1,1])/currentrad
                        nrvecz = (-sunpos[2]+backtraj[0:kn+1,2])/currentrad
                        #currentvr = backtraj[1:k+2,3]*rvecx[0:k+1] + backtraj[1:k+2,4]*rvecy[0:k+1] + backtraj[1:k+2,5]*rvecz[0:k+1]
                        currentvr1 = backtraj[0:kn+1,3]*nrvecx[0:kn+1] + backtraj[0:kn+1,4]*nrvecy[0:kn+1] + backtraj[0:kn+1,5]*nrvecz[0:kn+1]
                        #currentv = np.sqrt(backtraj[0:k+1,3]**2 + backtraj[0:k+1,4]**2 + backtraj[0:k+1,5]**2)
                        #btintegrand2 = (1/(currentrad-oldrad))*PIrate/currentvr*(r1/currentrad)**2
                        btintegrand2 = PIrate2/currentvr1*(r1/currentrad)**2
                        #btintegrand2 = PIrate/((currentvr + currentvr1)/2)*(r1/currentrad)**2
                        attfact = scipy.integrate.simps(btintegrand2, currentrad)
                        farvx = np.append(farvx, [backtraj[0,3]])
                        farvy = np.append(farvy, [backtraj[0,4]])
                        fart = np.append(fart, [startt - t[kn-1]])
                        # calculating value of phase space density based on the value at the crossing of x = 100 au
                        maxwcolor = np.append(maxwcolor, [np.exp(-np.abs(attfact))*np.exp(-((backtraj[kn-1,3]+26000)**2 + backtraj[kn-1,4]**2)/(5327)**2)])
                        break
                    break


# single trajectory plotting code
if mode==2:
    init = [ibexpos[0], ibexpos[1], ibexpos[2], 3000, -13000, 0]
    singletraj = odeint(dr_dt, init, t, args=(rp5,))
    trackrp = np.zeros(t.size)
    for k in range(t.size):
        trackrp[k] = rp5(t[k])
        if np.sqrt((singletraj[k,0]-sunpos[0])**2 + (singletraj[k,1]-sunpos[1])**2 + (singletraj[k,2]-sunpos[2])**2) <= .00465*au:
            print("Orbit too close to sun")
        if singletraj[k,0] >= 100*au:
            print(singletraj[k-1,:])
            print(t[k-1])
            break

print('Finished')

if mode==2:
    zer = [0]
    fig3d = plt.figure()
    fig3d.set_figwidth(7)
    fig3d.set_figheight(6)
    ax3d = plt.axes(projection='3d')
    scatterplot = ax3d.scatter3D(singletraj[:,0]/au, singletraj[:,1]/au, singletraj[:,2]/au, c=trackrp[:], cmap='coolwarm', s=.02, vmin=.5, vmax=1.5)
    cb = fig3d.colorbar(scatterplot)
    cb.set_label('Value of mu')
    #ax3d.plot3D(trajs[:,0,1], trajs[:,1,1], trajs[:,2,1], 'gold', linestyle='--')
    #ax3d.plot3D(trajs[:,0,2], trajs[:,1,2], trajs[:,2,2], 'forestgreen', linestyle=':')
    #ax3d.plot3D(trajs[:,0,3], trajs[:,1,3], trajs[:,2,3], 'firebrick', linestyle='-.')
    ax3d.scatter3D(zer,zer,zer,c='orange')
    ax3d.scatter3D([ibexpos[0]/au],[ibexpos[1]/au],[ibexpos[2]/au], c='springgreen')
    ax3d.set_xlabel("x (au)")
    ax3d.set_ylabel("y (au)")
    ax3d.set_zlabel("z (au)")
    ax3d.set_xlim3d(left = -1, right = 3)
    ax3d.set_ylim3d(bottom = -.5, top = 1.5)
    ax3d.set_zlim3d(bottom = -1, top = 1)
    ax3d.view_init(90,270)
    ax3d.set_title("Individual Orbit at time t=6.33275e9 s \n Target at (.707 au, .707 au) \
        \n At target point v = (3.0 km/s, -13.0 km/s) \n Value of distribution function = 0.7421392933966723",fontsize=12)
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
    # writing data to a file - need to change each time or it will overwrite previous file
    #file = open("C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/datafiles/cosexprp_5pi6_6p36675e9_direct_cosexppi_test.txt", 'w')
    file = open("/Users/ldyke/Desktop/Dartmouth/HSResearch/Code/Kepler/Python Orbit Code/datafiles/cosexprp_5pi6_6p36675e9_direct_cosexppi_fix.txt", "w")
    for i in range(farvx.size):
        file.write(str(farvx[i]/1000) + ',' + str(farvy[i]/1000) + ',' + str(maxwcolor[i]) + '\n')
    file.close()

    # plotting a scatterplot of vx and vy at the target point, colored by the phase space density
    f = plt.figure()
    f.set_figwidth(9)
    f.set_figheight(6)
    plt.scatter(farvx[:]/1000, farvy[:]/1000, c=maxwcolor[:], marker='o', cmap='plasma')
    cb = plt.colorbar()
    #cb.set_label('Time at which orbit passes through 100 au (s)')
    #cb.set_label('Travel Time from 100 au to Point of Interest (s)')
    cb.set_label('PDF(r,v,t)')
    plt.xlabel("vx at Target in km/s")
    plt.ylabel("vy at Target in km/s")
    #plt.suptitle('Phase Space population at x = 100 au reaching initial position at t = 5700000000 s')
    plt.suptitle('Phase space population at target (t = 6.36675e9 s) drawn from Maxwellian at 100 au centered on vx = -26 km/s')
    #plt.title('Target (-.97au, .2au): vx range -51500 m/s to -30500 m/s, vy range -30000 m/s to 30000 m/s')
    plt.title('Target at (-.866 au, .5 au), Time Resolution Close to Target = 200 s')
    #plt.title('Initial test distribution centered on vx = -41.5 km/s, vy = -1.4 km/s')
    plt.show()
    

    # plotting a contour whose levels are values of the phase space density
    """f = plt.figure()
    f.set_figwidth(9)
    f.set_figheight(6)
    levels = [.001, .01, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .98, 1.0]
    plt.tricontour(farvx[:]/1000, farvy[:]/1000, maxwcolor[:], levels)
    cb = plt.colorbar()
    cb.set_label('f(r,v,t)')
    plt.xlabel("vx at Target in km/s")
    plt.ylabel("vy at Target in km/s")
    #plt.suptitle('Phase Space population at x = 100 au reaching initial position at t = 5700000000 s')
    plt.suptitle('Phase space population at target (t = 6.246e9 s) drawn from Maxwellian at 100 au centered on vx = -26 km/s')
    #plt.title('Target (-.97au, .2au): vx range -51500 m/s to -30500 m/s, vy range -30000 m/s to 30000 m/s')
    plt.title('Target at (.707 au, .707 au)')
    #plt.title('Initial test distribution centered on vx = -41.5 km/s, vy = -1.4 km/s')
    plt.show()"""