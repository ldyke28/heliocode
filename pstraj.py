import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import odeint
import scipy
from tqdm import tqdm

# add another sine function to the noise function so the fluctuations vary in magnitude

# MODE LIST
# 1 = generate a list of trajectories that come within proximity
# 2 = plot an individual trajectory traced backward from point of interest
# 3 = generate phase space diagram
mode = 3

# Value for 1 au (astronomical unit) in meters
au = 1.496*10**11
msolar = 1.98847*10**30 # mass of the sun in kg
G = 6.6743*10**(-11) # value for gravitational constant in SI units
# one year in s = 3.156e7 s (Julian year, average length of a year)
# 11 Julian years = 3.471e8 s
# Note to self: solar maximum in April 2014
oneyear = 3.15545454545*10**7

# 120749800 for first force free
# 226250200 for second force free
finalt = -45000000 # time to start backtracing
#6.36674976e9 force free for cosexprp
initialt = -50000000000
tstep = 10000 # general time resolution
tstepclose = 500 # time resolution for close regime
tstepfar = 200000 # time resolution for far regime
phase = 0 # phase for implementing rotation of target point around sun
refdist = 300 # upwind reference distance for backtraced trajectories, in au

# Location of the sun in [x,y,z] - usually this will be at 0, but this makes it flexible just in case
# Second line is location of the point of interest in the same format (which is, generally, where we want IBEX to be)
sunpos = np.array([0,0,0])
theta = 150
ibexx = np.cos(theta*np.pi/180)
ibexy = np.sin(theta*np.pi/180)
ibexpos = np.array([ibexx*au, ibexy*au, 0])
# implementation of target point that orbits around the sun
#ibexpos = np.array([np.cos(np.pi*finalt/oneyear + phase)*au, np.sin(np.pi*finalt/oneyear + phase)*au, 0])


# INITIAL CONDITIONS for both position and velocity (in SI units - m and m/s)
ttotal = 7000000000
if mode==1:
    t = np.arange(0, ttotal, tstep)
if mode==2:
    startt = finalt
    lastt = initialt
    tmid = startt - 500000000 # time at which we switch from high resolution to low resolution - a little more than half of a cycle
    tclose = np.arange(startt, tmid, -tstepclose) # high resolution time array (close regime)
    tfar = np.arange(tmid, lastt, -tstepfar) # low resolution time array (far regime)
    t = np.concatenate((tclose, tfar))
tscale = int(.7*ttotal/tstep)
#tscale = 0

if mode==1:
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

# Initial Conditions for orbit starting at point of interest for backtracing
xstart = ibexpos[0]
ystart = ibexpos[1]
zstart = ibexpos[2]

# Multiple sets of initial vx/vy conditions for convenience
# In order of how I use them - direct, indirect, center, extra one for zoomed testing
#vxstart = np.arange(-62000, 10000, 550)
#vystart = np.arange(-41000, 7000, 350)
#vxstart = np.arange(-50000, 20000, 500)
#vystart = np.arange(15000, 48000, 200)
vxstart = np.arange(-25000, 25000, 250)
vystart = np.arange(-25000, 25000, 250)
#vxstart = np.arange(5000, 10000, 25)
#vystart = np.arange(5000, 10000, 25)
vzstart = 0
if mode==3:
    startt = finalt
    lastt = initialt
    tmid = startt - 200000000 # time at which we switch from high resolution to low resolution - a little more than half of a cycle
    tclose = np.arange(startt, tmid, -tstepclose) # high resolution time array (close regime)
    tfar = np.arange(tmid, lastt, -tstepfar) # low resolution time array (far regime)
    t = np.concatenate((tclose, tfar))
    mode3dt = startt-lastt


def radPressure(t):
    # dummy function to model radiation pressure
    # takes the time as input and returns the radiation pressure function at that time
    #return (np.sin(2*np.pi*(t/347000000)))**2 + .5
    #return .7
    return 0

def LyaRP(t,v_r):
    lyafunction = 1.25*np.exp(-(v_r-55000)**2/(2*25000**2)) + 1.25*np.exp(-(v_r+55000)**2/(2*25000**2)) + .55*np.exp(-v_r**2/(2*25000**2))
    omegat = 2*np.pi/(3.47*10**(8))*t
    return (.75 + .243*np.cos(omegat - np.pi)*np.exp(np.cos(omegat - np.pi)))*lyafunction

# extra radiation pressure functions for overlayed plots
def rp2(t):
    return ((.75-.243/np.e))

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

def rpnoise(t):
    # a different form of the radiation pressure with fluctuations
    # taken from eq. 8 in https://articles.adsabs.harvard.edu/pdf/1995A%26A...296..248R
    omegat = 2*np.pi/(3.47*10**(8))*t
    omeganoiset = 2*np.pi/(2.333*10**6)*t # 2.333*10**6 s = period of 27 days (rotational period of the sun)
    flucmag = .1
    return .75 + .243*np.cos(omegat - np.pi)*np.exp(np.cos(omegat - np.pi)) + flucmag*np.sin(omeganoiset)

def rpnoisefluc(t):
    # a different form of the radiation pressure with fluctuations
    # taken from eq. 8 in https://articles.adsabs.harvard.edu/pdf/1995A%26A...296..248R
    omegat = 2*np.pi/(3.47*10**(8))*t
    omeganoiset = 2*np.pi/(2.333*10**6)*t # 2.333*10**6 s = period of 27 days (rotational period of the sun)
    omegaoverallfluct = omegat*20 # fluctuations of the noise itself
    flucmag = .1
    return .75 + .243*np.cos(omegat - np.pi)*np.exp(np.cos(omegat - np.pi)) + flucmag*np.sin(omeganoiset)*np.cos(omegaoverallfluct)

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

def Lya_dr_dt(x,t,rp):
    # integrating differential equation for gravitational force. x[0:2] = x,y,z and x[3:5] = vx,vy,vz
    # dx0-2 = vx, vy, and vz, dx3-5 = ax, ay, and az
    r = np.sqrt((sunpos[0]-x[0])**2 + (sunpos[1]-x[1])**2 + (sunpos[2]-x[2])**2)
    # calculating the component of the radial unit vector in each direction at each point in time
    nrvecx = x[0]/r
    nrvecy = x[1]/r
    nrvecz = x[2]/r
    # calculating the magnitude of v_r at each point in time
    v_r = x[3]*nrvecx + x[4]*nrvecy + x[5]*nrvecz
    dx0 = x[3]
    dx1 = x[4]
    dx2 = x[5]
    dx3 = (msolar*G/(r**3))*(sunpos[0]-x[0])*(1-rp(t,v_r))
    dx4 = (msolar*G/(r**3))*(sunpos[1]-x[1])*(1-rp(t,v_r))
    dx5 = (msolar*G/(r**3))*(sunpos[2]-x[2])*(1-rp(t,v_r))
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
    for i in tqdm(range(vxstart.size)): # displays progress bars for both loops to measure progress
        for j in tqdm(range(vystart.size)):
            init = [xstart, ystart, zstart, vxstart[i], vystart[j], vzstart]
            # calculating trajectories for each initial condition in phase space given
            backtraj[:,:] = odeint(Lya_dr_dt, init, t, args=(LyaRP,))
            if any(np.sqrt((backtraj[:,0]-sunpos[0])**2 + (backtraj[:,1]-sunpos[1])**2 + (backtraj[:,2]-sunpos[2])**2) <= .00465*au):
                # tells the code to not consider the trajectory if it at any point intersects the width of the sun
                continue
            if all(backtraj[:,0]-sunpos[0] < refdist*au):
                # forgoes the following checks if the trajectory never passes through the plane at the reference distance upwind
                continue
            for k in range(t.size - tclose.size):
                if backtraj[k+tclose.size,0] >= refdist*au and backtraj[k-1+tclose.size,0] <= refdist*au:
                    # adjusting the indexing to avoid checking in the close regime
                    kn = k+tclose.size
                    # printing phase space information as the trajectory passes through the plane at the reference distance upwind
                    #print(backtraj[kn-1,:])
                    #print(t[kn-1])
                    # radius in paper given to be 14 km/s
                    # only saving initial conditions corresponding to points that lie within this Maxwellian at reference distance
                    #if backtraj[k-1,3,(i)*vystart.size + (j)] <= -22000 and backtraj[k-1,3,(i)*vystart.size + (j)] >= -40000 and backtraj[k-1,4,(i)*vystart.size + (j)] <= 14000 and backtraj[k-1,4,(i)*vystart.size + (j)] >= -14000:
                    if np.sqrt((backtraj[kn-1,3]+26000)**2 + (backtraj[kn-1,4])**2 + (backtraj[kn-1,5])**2) <= 27000:
                        omt = 2*np.pi/(3.47*10**(8))*t[0:kn+1]
                        # function for the photoionization rate at each point in time
                        PIrate2 = 10**(-7)*(1 + .7/(np.e + 1/np.e)*(np.cos(omt - np.pi)*np.exp(np.cos(omt - np.pi)) + 1/np.e))
                        r1 = 1*au # reference radius
                        currentrad = np.sqrt((sunpos[0]-backtraj[0:kn+1,0])**2 + (sunpos[1]-backtraj[0:kn+1,1])**2 + (sunpos[2]-backtraj[0:kn+1,2])**2)
                        # calculating the component of the radial unit vector in each direction at each point in time
                        nrvecx = (-sunpos[0]+backtraj[0:kn+1,0])/currentrad
                        nrvecy = (-sunpos[1]+backtraj[0:kn+1,1])/currentrad
                        nrvecz = (-sunpos[2]+backtraj[0:kn+1,2])/currentrad
                        # calculating the magnitude of v_r at each point in time
                        currentvr = backtraj[0:kn+1,3]*nrvecx[0:kn+1] + backtraj[0:kn+1,4]*nrvecy[0:kn+1] + backtraj[0:kn+1,5]*nrvecz[0:kn+1]
                        # integrand for the photoionization losses
                        btintegrand = PIrate2/currentvr*(r1/currentrad)**2
                        # calculation of attenuation factor
                        attfact = scipy.integrate.simps(btintegrand, currentrad)
                        farvx = np.append(farvx, [backtraj[0,3]])
                        farvy = np.append(farvy, [backtraj[0,4]])
                        fart = np.append(fart, [startt - t[kn-1]])
                        # calculating value of phase space density based on the value at the crossing of x = 100 au
                        maxwcolor = np.append(maxwcolor, [np.exp(-np.abs(attfact))*np.exp(-((backtraj[kn-1,3]+26000)**2 + backtraj[kn-1,4]**2 + backtraj[kn-1,5]**2)/(10195)**2)])
                        #maxwcolor = np.append(maxwcolor, [np.exp(-((backtraj[kn-1,3]+26000)**2 + backtraj[kn-1,4]**2 + backtraj[kn-1,5]**2)/(10195)**2)])
                        break
                    break


# single trajectory plotting code
if mode==2:
    indxic = -10000
    indyic = -9750
    indzic = 00
    init = [ibexpos[0], ibexpos[1], ibexpos[2], indxic, indyic, indzic]
    print("Calculating trajectory...")
    #singletraj = odeint(dr_dt, init, t, mxstep=750, args=(rp6,))
    singletraj = odeint(Lya_dr_dt, init, t, args=(LyaRP,))
    print("Trajectory Calculated")
    #print(singletraj)
    trackrp = np.zeros(t.size)
    Ltrack = np.zeros(t.size)
    Evartrack = np.zeros(t.size)
    Etrack = np.zeros(t.size)
    #rtrack = np.zeros(t.size)
    rtrack = np.sqrt((sunpos[0]-singletraj[:,0])**2 + (sunpos[1]-singletraj[:,1])**2 + (sunpos[2]-singletraj[:,2])**2)
    # calculating the component of the radial unit vector in each direction at each point in time
    nrvecxk = singletraj[:,0]/rtrack
    nrvecyk = singletraj[:,1]/rtrack
    nrveczk = singletraj[:,2]/rtrack
    # calculating the magnitude of v_r at each point in time
    v_rtrack = singletraj[:,3]*nrvecxk[:] + singletraj[:,4]*nrvecyk[:] + singletraj[:,5]*nrveczk[:]
    for k in tqdm(range(t.size)):
        trackrp[k] = LyaRP(t[k],v_rtrack[k]) # calculating the value of the radiation pressure at each time point
        """rmag = np.sqrt((sunpos[0]-singletraj[k,0])**2 + (sunpos[1]-singletraj[k,1])**2 + (sunpos[2]-singletraj[k,2])**2)
        rtrack[k] = rmag
        vmag = np.sqrt(singletraj[k,3]**2 + singletraj[k,4]**2 + singletraj[k,5]**2)

        rvec = (singletraj[k,0:3]-sunpos)/rmag
        rxv = np.cross(singletraj[k,0:3], singletraj[k,3:6])
        Ltrack[k] = np.sqrt(rxv[0]**2 + rxv[1]**2 + rxv[2]**2)

        vdotr = rvec[0]*singletraj[k,3] + rvec[1]*singletraj[k,4] + rvec[2]*singletraj[k,5]
        Evartrack[k] = Evartrack[k-1] + (t[k]-t[k-1])*rp6(t[k])*vdotr/(rmag**2)
        Etrack[k] = (vmag**2)/2 - G*msolar/rmag - G*msolar*Evartrack[k]"""
        if np.sqrt((singletraj[k,0]-sunpos[0])**2 + (singletraj[k,1]-sunpos[1])**2 + (singletraj[k,2]-sunpos[2])**2) <= .00465*au:
            # checking if the orbit is too close to the sun
            print("Orbit too close to sun")
            psd = 0
            perihelion = min(np.sqrt((singletraj[0:k,0]-sunpos[0])**2 + (singletraj[0:k,1]-sunpos[1])**2 + (singletraj[0:k,2]-sunpos[2])**2))
            ttime = 0
            break
        if singletraj[k,0] >= refdist*au:
            print(singletraj[k-1,:])
            print(t[k-1])
            Etrack = Etrack[:k]
            rtrack = rtrack[:k]
            Ltrack = Ltrack[:k]
            rprange = trackrp[:k]
            perihelion = min(np.sqrt((singletraj[0:k,0]-sunpos[0])**2 + (singletraj[0:k,1]-sunpos[1])**2 + (singletraj[0:k,2]-sunpos[2])**2))
            

            omt = 2*np.pi/(3.47*10**(8))*t[0:k+1]
            # function for the photoionization rate at each point in time
            PIrate2 = 10**(-7)*(1 + .7/(np.e + 1/np.e)*(np.cos(omt - np.pi)*np.exp(np.cos(omt - np.pi)) + 1/np.e))
            r1 = 1*au # reference radius
            currentrad = np.sqrt((sunpos[0]-singletraj[0:k+1,0])**2 + (sunpos[1]-singletraj[0:k+1,1])**2 + (sunpos[2]-singletraj[0:k+1,2])**2)
            # calculating the component of the radial unit vector in each direction at each point in time
            nrvecx = (-sunpos[0]+singletraj[0:k+1,0])/currentrad
            nrvecy= (-sunpos[1]+singletraj[0:k+1,1])/currentrad
            nrvecz = (-sunpos[2]+singletraj[0:k+1,2])/currentrad
            # calculating the magnitude of v_r at each point in time
            currentvr = singletraj[0:k+1,3]*nrvecx[0:k+1] + singletraj[0:k+1,4]*nrvecy[0:k+1] + singletraj[0:k+1,5]*nrvecz[0:k+1]
            # integrand for the photoionization losses
            btintegrand = PIrate2/currentvr*(r1/currentrad)**2
            # calculation of attenuation factor
            attfact = scipy.integrate.simps(btintegrand, currentrad)
            psd = np.exp(-np.abs(attfact))*np.exp(-((singletraj[k-1,3]+26000)**2 + singletraj[k-1,4]**2 + singletraj[k-1,5]**2)/(5327)**2)

            print(np.sqrt((singletraj[k-1,3]+26000)**2 + (singletraj[k-1,4])**2 + (singletraj[k-1,5])**2))
            print("Perihelion distance in au is: " + str(perihelion/au))
            print("PSD value: " + str(psd))
            
            t = t[:k]
            ttime = t[0] - t[-1]
            print("Travel time from 100 au in years: " + str(ttime/oneyear))
            break
        if k == t.size-1:
            perihelion = min(np.sqrt((singletraj[0:k,0]-sunpos[0])**2 + (singletraj[0:k,1]-sunpos[1])**2 + (singletraj[0:k,2]-sunpos[2])**2))
            ttime = 0
            psd = 0
            #print(perihelion)

print('Finished')

if mode==2:
    zer = [0]
    fosize = 10
    """fig3d = plt.figure()
    fig3d.set_figwidth(7)
    fig3d.set_figheight(7)
    ax3d = plt.axes(projection='3d')
    # plotting the trajectory as a series of scatter plot points colored by radiation pressure
    scatterplot = ax3d.scatter3D(singletraj[:,0]/au, singletraj[:,1]/au, singletraj[:,2]/au, c=trackrp[:], cmap='coolwarm', s=.02, vmin=(.75-.243/np.e), vmax=(.75+.243*np.e))
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
    ax3d.set_xlim3d(left = -1.5, right = 1)
    ax3d.set_ylim3d(bottom = -1, top = 1)
    ax3d.set_zlim3d(bottom = -1, top = 1)
    ax3d.view_init(90,270)
    ax3d.set_title("Individual Orbit at time t$\\approx$" + str(round(finalt/(oneyear), 3)) + " years \n Target at (" + str(round(ibexpos[0]/au, 4)) + " au, " + str(round(ibexpos[1]/au, 4)) + " au, " + str(round(ibexpos[2]/au, 4)) + " au) \
        \n At target point v = (" + str(indxic/1000) + " km/s, " + str(indyic/1000) + " km/s, " + str(indzic/1000) + " km/s) \
        \n Value of distribution function = " + str(psd) + "\
        \n Perihelion at $\\sim$ " + str(round(perihelion/au, 5)) + " au \
        \n Travel time from x = 100 au to target $\\approx$ " + str(round(ttime/oneyear, 3)) + " years",fontsize=12)"""
    fig = plt.figure()
    fig.set_figwidth(9)
    fig.set_figheight(6)
    
    # for fluctuating force
    # plt.scatter(singletraj[:,0]/au, singletraj[:,1]/au, c=trackrp[:], cmap='coolwarm', s=.02, vmin=((.75-.243/np.e)-.1), vmax=((.75+.243*np.e)+.1), zorder=2)
    # for non-fluctuating force
    #plt.scatter(singletraj[:,0]/au, singletraj[:,1]/au, c=trackrp[:], cmap='coolwarm', s=.02, vmin=((.75-.243/np.e)), vmax=((.75+.243*np.e)), zorder=2)
    plt.scatter(singletraj[:,0]/au, singletraj[:,1]/au, c=trackrp[:], cmap='coolwarm', s=.02, vmin=min(rprange), zorder=2)
    cb = plt.colorbar()
    plt.scatter(ibexpos[0]/au, ibexpos[1]/au, c='springgreen', zorder=3)
    plt.scatter(zer, zer, c='orange', zorder=3)
    plt.grid(zorder=0)
    cb.set_label("Value of $\mu$", fontsize=fosize)
    cb.ax.tick_params(labelsize=fosize)
    plt.xlabel("x (au)", fontsize=fosize)
    plt.ylabel("y (au)", fontsize=fosize)
    plt.xlim([-2.5,15])
    plt.ylim([-1,12])
    
    plt.xticks(fontsize=fosize)
    plt.yticks(fontsize=fosize)
    plt.title("Individual Orbit at time t$\\approx$" + str(round(finalt/(oneyear), 3)) + " years, Target at (" + str(round(ibexpos[0]/au, 4)) + " au, " + str(round(ibexpos[1]/au, 4)) + " au, " + str(round(ibexpos[2]/au, 4)) + " au) \
        \n At target point v = (" + str(indxic/1000) + " km/s, " + str(indyic/1000) + " km/s, " + str(indzic/1000) + " km/s), Value of distribution function = " + str(psd) + " \
        \n Perihelion at $\\sim$ " + str(round(perihelion/au, 5)) + " au, Travel time from x = 100 au to target $\\approx$ " + str(round(ttime/oneyear, 3)) + " years", fontsize=10)
    plt.show()

    """f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    l1, = ax1.plot(t,Etrack, color='r')
    l2, = ax2.plot(t,Ltrack, color='b')
    l3, = ax3.plot(t,rtrack, color='y')
    ax1.legend((l1, l2, l3), ('Adjusted Specific Energy', 'Magnitude of Angular Momentum', 'Radius'), loc='upper left')
    f.subplots_adjust(hspace=.0)
    f.set_size_inches(8,4)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    plt.show()"""

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
    file = open("C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/datafiles/lyarp_5pi6_-4p5e7s_center_cosexppi_tclose1000.txt", 'w')
    #file = open("/Users/ldyke/Desktop/Dartmouth/HSResearch/Code/Kepler/Python Orbit Code/datafiles/p1fluccosexprp_35pi36_0y_direct_cosexppi_tclose400.txt", "w")
    for i in range(farvx.size):
        file.write(str(farvx[i]/1000) + ',' + str(farvy[i]/1000) + ',' + str(maxwcolor[i]) + '\n')
    file.close()

    # plotting a scatterplot of vx and vy at the target point, colored by the phase space density
    f = plt.figure()
    f.set_figwidth(10)
    f.set_figheight(6)
    fsize = 18
    plt.scatter(farvx[:]/1000, farvy[:]/1000, c=maxwcolor[:], marker='o', cmap='hsv')
    plt.rcParams.update({'font.size': fsize})
    cb = plt.colorbar()
    #cb.set_label('Time at which orbit passes through 100 au (s)')
    #cb.set_label('Travel Time from 100 au to Point of Interest (s)')
    cb.set_label('Normalized Phase Space Density')
    #plt.xlim([-25, 25])
    #plt.ylim([-25, 25])
    plt.xticks(fontsize=fsize)
    plt.yticks(fontsize=fsize)
    plt.xlabel("$v_x$ at Target in km/s", fontsize=fsize)
    plt.ylabel("$v_y$ at Target in km/s", fontsize=fsize)
    #plt.suptitle('Phase Space population at x = 100 au reaching initial position at t = 5700000000 s')
    #plt.suptitle('VDF at target, at t $\\approx$ ' + str(round(finalt/(oneyear), 3)) + ' years, drawn from Maxwellian at 100 au centered on $v_x$ = -26 km/s')
    #plt.title('Target (-.97au, .2au): vx range -51500 m/s to -30500 m/s, vy range -30000 m/s to 30000 m/s')
    #plt.title('Target at (' + str(round(ibexpos[0]/au, 3)) + ' au, ' + str(round(ibexpos[1]/au, 3)) + ' au), Time Resolution Close to Target = ' + str(tstepclose) + ' s')
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