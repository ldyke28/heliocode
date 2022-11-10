import numpy as np
from scipy.integrate import odeint
import scipy
from tqdm import tqdm
import warnings

warnings.filterwarnings("error", category=Warning)

file = np.loadtxt("lostpoints.txt", delimiter=',')
file2 = open("lostpoints_ex.txt", "w")

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
finalt = 000000000 # time to start backtracing
#6.36674976e9 force free for cosexprp
initialt = -2000000000
tstep = 10000 # general time resolution
tstepclose = 20000 # time resolution for close regime
tstepfar = 200000 # time resolution for far regime

# Location of the sun in [x,y,z] - usually this will be at 0, but this makes it flexible just in case
# Second line is location of the point of interest in the same format (which is, generally, where we want IBEX to be)
sunpos = np.array([0,0,0])
theta = 180
ibexx = np.cos(theta*np.pi/180)
ibexy = np.sin(theta*np.pi/180)
ibexpos = np.array([ibexx*au, ibexy*au, 0])

# Initial Conditions for orbit starting at point of interest for backtracing
xstart = ibexpos[0]
ystart = ibexpos[1]
zstart = ibexpos[2]

vxstart = file[:,0]
vystart = file[:,1]
vzstart = file[:,2]

startt = finalt
lastt = initialt
tmid = startt - 200000000 # time at which we switch from high resolution to low resolution - a little more than half of a cycle
tclose = np.arange(startt, tmid, -tstepclose) # high resolution time array (close regime)
tfar = np.arange(tmid, lastt, -tstepfar) # low resolution time array (far regime)
t = np.concatenate((tclose, tfar))
mode3dt = startt-lastt

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

farvx = np.array([])
farvy = np.array([])
farvz = np.array([])
fart = np.array([])
maxwcolor = np.array([])
sunloss = 0
directionloss = 0
for i in tqdm(range(vxstart.size)): # displays progress bar to measure progress
    init = [xstart, ystart, zstart, vxstart[i], vystart[i], vzstart[i]]
    # calculating trajectories for each initial condition in phase space given
    try:
        backtraj = odeint(dr_dt, init, t, args=(rp6,))
        if any(np.sqrt((backtraj[:,0]-sunpos[0])**2 + (backtraj[:,1]-sunpos[1])**2 + (backtraj[:,2]-sunpos[2])**2) <= .00465*au):
            # tells the code to not consider the trajectory if it at any point intersects the width of the sun
            sunloss += 1
            continue
        if all(backtraj[:,0]-sunpos[0] < 100*au):
            # forgoes the following checks if the trajectory never passes through x = 100 au
            directionloss += 1
            continue
        for k in range(t.size - tclose.size):
            if backtraj[k+tclose.size,0] >= 100*au and backtraj[k-1+tclose.size,0] <= 100*au:
                # adjusting the indexing to avoid checking in the close regime
                kn = k+tclose.size
                # printing phase space information as the trajectory passes through x = 100 au
                #print(backtraj[kn-1,:])
                #print(t[kn-1])
                # radius in paper given to be 14 km/s
                # only saving initial conditions corresponding to points that lie within this Maxwellian at x = 100 au
                #if backtraj[k-1,3,(i)*vystart.size + (j)] <= -22000 and backtraj[k-1,3,(i)*vystart.size + (j)] >= -40000 and backtraj[k-1,4,(i)*vystart.size + (j)] <= 14000 and backtraj[k-1,4,(i)*vystart.size + (j)] >= -14000:
                if np.sqrt((backtraj[kn-1,3]+26000)**2 + (backtraj[kn-1,4])**2 + (backtraj[kn-1,5])**2) <= 14000:
                    omt = 2*np.pi/(3.47*10**(8))*t[0:kn+1]
                    # function for the photoionization rate at each point in time
                    PIrate2 = 10**(-7)*(1 + .7/(np.e + 1/np.e)*(np.cos(omt - np.pi)*np.exp(np.cos(omt - np.pi)) + 1/np.e))
                    r1 = 1*au # reference radius
                    currentrad = np.sqrt((sunpos[0]-backtraj[0:kn+1,0])**2 + (sunpos[1]-backtraj[0:kn+1,1])**2 + (sunpos[2]-backtraj[0:kn+1,2])**2)
                    # calculating the component of the radial unit vector in each direction at each point in time
                    nrvecx = (-sunpos[0]+backtraj[0:kn+1,0])/currentrad
                    nrvecy= (-sunpos[1]+backtraj[0:kn+1,1])/currentrad
                    nrvecz = (-sunpos[2]+backtraj[0:kn+1,2])/currentrad
                    # calculating the magnitude of v_r at each point in time
                    currentvr = backtraj[0:kn+1,3]*nrvecx[0:kn+1] + backtraj[0:kn+1,4]*nrvecy[0:kn+1] + backtraj[0:kn+1,5]*nrvecz[0:kn+1]
                    # integrand for the photoionization losses
                    btintegrand = PIrate2/currentvr*(r1/currentrad)**2
                    # calculation of attenuation factor
                    attfact = scipy.integrate.simps(btintegrand, currentrad)
                    farvx = np.append(farvx, [backtraj[0,3]])
                    farvy = np.append(farvy, [backtraj[0,4]])
                    farvz = np.append(farvz, [backtraj[0,5]])
                    fart = np.append(fart, [startt - t[kn-1]])
                    # calculating value of phase space density based on the value at the crossing of x = 100 au
                    maxwcolor = np.append(maxwcolor, [np.exp(-np.abs(attfact))*np.exp(-((backtraj[kn-1,3]+26000)**2 + backtraj[kn-1,4]**2 + backtraj[kn-1,5]**2)/(5327)**2)])
                    break
                break
    except Warning:
        # Collects the points that seem to cause issues to be ran again with different temporal resolution
        file2.write(str(vxstart[i]) + ',' + str(vystart[i]) + ',' + str(vzstart[i]) + '\n')
    
# w for write mode (clears existing contents), a for append (adds onto end of file)
file3 = open("supplementaldata1.txt", "a")
for i in range(farvx.size):
    file3.write(str(farvx[i]/1000) + ',' + str(farvy[i]/1000) + ',' + str(farvz[i]/1000) + ',' + str(maxwcolor[i]) + '\n')
file3.close()

file2.close()
print("Finished")