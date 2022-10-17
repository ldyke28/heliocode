import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import odeint
import scipy
from mpl_toolkits import mplot3d
from tqdm import tqdm
from mpi4py import MPI

comm = MPI.COMM_WORLD


# Value for 1 au (astronomical unit) in meters
au = 1.496*10**11
msolar = 1.98847*10**30 # mass of the sun in kg
G = 6.6743*10**(-11) # value for gravitational constant in SI units
# one year in s = 3.156e7 s
# Note to self: solar maximum in April 2014
oneyear = 3.156*10**7

# 120749800 for first force free
# 226250200 for second force free
finalt = 00000000 # time to start backtracing
#6.36674976e9 force free for cosexprp
tstep = 10000 # general time resolution
tstepclose = 1000 # time resolution for close regime
tstepfar = 200000 # time resolution for far regime
phase = 0 # phase for implementing rotation of target point around sun

# Location of the sun in [x,y,z] - usually this will be at 0, but this makes it flexible just in case
# Second line is location of the point of interest in the same format (which is, generally, where we want IBEX to be)
sunpos = np.array([0,0,0])
ibexpos = np.array([-.9952*au, .0980*au, 0])
# implementation of target point that orbits around the sun
#ibexpos = np.array([np.cos(np.pi*finalt/oneyear + phase)*au, np.sin(np.pi*finalt/oneyear + phase)*au, 0])


# Initial Conditions for orbit starting at point of interest for backtracing
xstart = ibexpos[0]
ystart = ibexpos[1]
zstart = ibexpos[2]

# Multiple sets of initial vx/vy conditions for convenience
# In order of how I use them - direct, indirect, center, extra one for zoomed testing
#vxstart = np.arange(-50000, -15000, 300)
#vystart = np.arange(-25000, 10000, 300)
#vxstart = np.arange(24000, 45000, 250)
#vystart = np.arange(-2000, 6500, 150)
#vxstart = np.arange(-25000, 25000, 500)
#vystart = np.arange(-25000, 25000, 500)
vxstart = np.arange(-50000, 000, 1000)
vystart = np.arange(-40000, 40000, 2000)
vzstart = np.arange(-40000, 40000, 2000)
#vzstart = 0

startt = finalt
lastt = -2000000000
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


"""farvx = np.array([])
farvy = np.array([])
fart = np.array([])
maxwcolor = np.array([])
backtraj = np.zeros((t.size, 6))"""

# identify the total number of processes
nprocs = comm.Get_size()

# creating a shared array with the size of the maximum possible number of points that could exist
size = vxstart.size * vystart.size * vzstart.size
itemsize = MPI.FLOAT.Get_size()
if comm.Get_rank() == 0:
    nbytes = 5*size*itemsize
else:
    nbytes = 0

# creating a shared block on rank 0 and a window to it
win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=comm)

# creating arrays whose data points to shared memory
buf, itemsize = win.Shared_query(0)
assert itemsize == MPI.FLOAT.Get_size()
data = np.ndarray(buffer=buf, dtype='f', shape=(size,5))

bounds = np.zeros(nprocs, dtype=int)

for q in range(nprocs-1):
    bounds[q+1] = int(np.floor(vxstart.size/(nprocs-1)*(q+1)))

for m in range(nprocs-1):
    if comm.rank == m+1:
        vxstartn = vxstart[bounds[m]:(bounds[m+1]+1)]
        for i in range(vxstartn.size): # displays progress bars for both loops to measure progress
            for j in range(vystart.size):
                for l in range(vzstart.size):
                    init = [xstart, ystart, zstart, vxstartn[i], vystart[j], vzstart[l]]
                    # calculating trajectories for each initial condition in phase space given
                    backtraj = np.zeros((t.size, 6))
                    backtraj[:,:] = odeint(dr_dt, init, t, args=(rp6,))
                    if any(np.sqrt((backtraj[:,0]-sunpos[0])**2 + (backtraj[:,1]-sunpos[1])**2 + (backtraj[:,2]-sunpos[2])**2) <= .00465*au):
                        # tells the code to not consider the trajectory if it at any point intersects the width of the sun
                        continue
                    if all(backtraj[:,0]-sunpos[0] < 100*au):
                        # forgoes the following checks if the trajectory never passes through x = 100 au
                        continue
                    for k in range(t.size - tclose.size):
                        if backtraj[k+tclose.size,0] >= 100*au and backtraj[k-1+tclose.size,0] <= 100*au:
                            # adjusting the indexing to avoid checking in the close regime
                            kn = k+tclose.size
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
                                data[bounds[m]*vystart.size*vzstart.size + vystart.size*vzstart.size*i + vzstart.size*j + l,0] = vxstartn[i]
                                data[bounds[m]*vystart.size*vzstart.size + vystart.size*vzstart.size*i + vzstart.size*j + l,1] = vystart[j]
                                data[bounds[m]*vystart.size*vzstart.size + vystart.size*vzstart.size*i + vzstart.size*j + l,2] = vzstart[l]
                                data[bounds[m]*vystart.size*vzstart.size + vystart.size*vzstart.size*i + vzstart.size*j + l,3] = startt - t[kn-1]
                                # calculating value of phase space density based on the value at the crossing of x = 100 au
                                data[bounds[m]*vystart.size*vzstart.size + vystart.size*vzstart.size*i + vzstart.size*j + l,4] = np.exp(-np.abs(attfact))*np.exp(-((backtraj[kn-1,3]+26000)**2 + backtraj[kn-1,4]**2 + backtraj[kn-1,5]**2)/(5327)**2)
                                break
                            break
        break

comm.Barrier()

print('Finished')

# Figure out how to get rid of zero points, probably in the block below

# writing data to a file - need to change each time or it will overwrite previous file
if comm.rank == 0:
    data = data[~np.all(data == 0, axis=1)]
    file = open("C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/datafiles/cosexprp_31pi32_t0_direct_cosexppi_test.txt", 'w')
    #file = open("/Users/ldyke/Desktop/Dartmouth/HSResearch/Code/Kepler/Python Orbit Code/datafiles/cosexprp_31pi32_t0_direct_cosexppi_test.txt", "w")
    for i in range(np.size(data, 0)):
        file.write(str(data[i,0]/1000) + ',' + str(data[i,1]/1000) + ',' + str(data[i,2]/1000) + ',' + str(data[i,4]) + '\n')
    file.close()
    print('All done!')