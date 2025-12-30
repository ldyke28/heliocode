import numpy as np
from scipy.integrate import odeint
import scipy
from mpi4py import MPI
import os
import warnings
import h5py
from scipy.signal import butter, lfilter, freqz
import time
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm

#####################################################################################################################################
# RELEVANT (FIXED) VARIABLES THROUGHOUT THE CODE
#####################################################################################################################################
# Value for 1 au (astronomical unit) in meters
au = 1.496*10**11
msolar = 1.98847*10**30 # mass of the sun in kg
G = 6.6743*10**(-11) # value for gravitational constant in SI units
mH = 1.6736*10**(-27) # mass of hydrogen in kgS
# one year in s = 3.156e7 s (Julian year, average length of a year)
# 11 Julian years = 3.471e8 s
# Note to self: solar maximum in April 2014
oneyear = 3.15545454545*10**7
kB = 1.381*10**(-23) # Boltzmann consant in SI units

# 120749800 for first force free
# 226250200 for second force free
finalt = 0 # time to start backtracing
#6.36674976e9 force free for cosexprp
initialt = -50000000000 # time in the past to which the code should backtrace
tstep = 10000 # general time resolution
tstepclose = 300 # time resolution for close regime
tstepfar = 200000 # time resolution for far regime
phase = 0 # phase for implementing rotation of target point around sun
refdist = 100 # reference distance in au for the boundary surface

theta = 180 # angle with respect to upwind axis of target point
ibexrad = 1 # radial distance of target point from Sun

vsolarwindms = 400000 # solar wind speed in m/s
vsolarwindcms = vsolarwindms*100 # solar wind speed in cm/s
nsw1au = 5 # solar wind density at 1 au in cm^-3

r1 = 1*au # reference radius for ionization integration

noncalcextent = 16 # radius in km/s around the axis of exclusion where we don't want to calculate trajectories (to save computational time)

start = time.time()

#####################################################################################################################################
# ESTABLISHING INITIAL CONDITIONS FOR THE CODE
#####################################################################################################################################

# Location of the sun in [x,y,z] - usually this will be at 0, but this makes it flexible just in case
# Second line is location of the point of interest in the same format (which is, generally, where we want IBEX to be)
sunpos = np.array([0,0,0])
ibexx = ibexrad*np.cos(theta*np.pi/180)
ibexy = ibexrad*np.sin(theta*np.pi/180)
ibexpos = np.array([ibexx*au, ibexy*au, 0])
# implementation of target point that orbits around the sun
#ibexpos = np.array([np.cos(np.pi*finalt/oneyear + phase)*au, np.sin(np.pi*finalt/oneyear + phase)*au, 0])


# Initial Conditions for orbit starting at point of interest for backtracing
# Adjusting these so they lie in the ecliptic, 5.3 degrees off of the ISM flow vector
eclipticangle = 5.3 *np.pi/180
xstart = ibexpos[0]*np.cos(eclipticangle) + ibexpos[2]*np.sin(eclipticangle)
ystart = ibexpos[1]
zstart = -ibexpos[0]*np.sin(eclipticangle) + ibexpos[2]*np.cos(eclipticangle)

# Set of initial conditions in velocity space
# vx/vy initial conditions are sampled on a grid with chosen resolution
vxstart = np.arange(-80000, 80000, 7000)
vystart = np.arange(-80000, 80000, 7000)
vzstart = np.arange(-80000, 80000, 7000)

startt = finalt
lastt = initialt
countoffset = 0
tmid = startt - 200000000 # time at which we switch from high resolution to low resolution - a little more than half of a solar cycle
tclose = np.arange(startt, tmid, -tstepclose) # high resolution time array (close regime)
tfar = np.arange(tmid, lastt, -tstepfar) # low resolution time array (far regime)
t = np.concatenate((tclose, tfar))


#####################################################################################################################################
# DEFINING ANALYTIC FUNCTIONS TO USE TO CALCULATE PSD ON THE BOUNDARY
#####################################################################################################################################

# primary population defined by mix of symmetric/asymmetric kappa distributions

# rotating so x-y plane is equivalent to B-v plane
thetabv = np.arctan(29.52/24.52)
# hydrogen density at infinity according to Rahmanifard et al. 2023 (below link)
nHinf = .11

# defining relevant parameters for calculation of these distributions (taking angular scattering into account)
# form from https://iopscience.iop.org/article/10.3847/2041-8213/abf436/pdf
# parameters from https://iopscience.iop.org/article/10.3847/1538-4357/ad0be1/pdf
# 1 and 2 refer to slow/fast populations per Swaczyna et al. 2023
upar = 26.29
T1par = 7910
kappa1par = 18.7
T2par = 7213
kappa2par = np.inf

# in the paper, first perpendicular direction is y, second perpendicular direction is z
puperpy = -0.03
pT1perpy = 8670
pkappa1perpy = 86
pT2perpy = 8515
pkappa2perpy = np.inf

puperpz = 0
pT1perpz = 8280
pkappa1perpz = np.inf
pT2perpz = pT1perpz
pkappa2perpz = pkappa1perpz

# speed scales (in km/s)
pthetapar1 = np.sqrt(2*kB*T1par/mH)/1000
pthetapar2 = np.sqrt(2*kB*T2par/mH)/1000
pthetaperpy1 = np.sqrt(2*kB*pT1perpy/mH)/1000
pthetaperpy2 = np.sqrt(2*kB*pT2perpy/mH)/1000
pthetaperpz1 = np.sqrt(2*kB*pT1perpz/mH)/1000
pthetaperpz2 = np.sqrt(2*kB*pT2perpz/mH)/1000

#units?
# adjusting per equation 3.12 of https://ui.adsabs.harvard.edu/abs/2013SSRv..175..183L/abstract
# taking d = 1, kappa = kappa0 + 3/2
def psdprim(vx, vy, vz):
    vyp = vy
    vzp = vz
    vy = vyp*np.cos(thetabv) - vzp*np.sin(thetabv)
    vz = vyp*np.sin(thetabv) + vzp*np.cos(thetabv)
    vx = -vx/1000
    vy = -vy/1000
    vz = -vz/1000
    fperpy1 = np.sqrt(1/(np.pi * pthetaperpy1**2)) * (scipy.special.gamma(pkappa1perpy)/(np.sqrt(pkappa1perpy - 1.5)*scipy.special.gamma((pkappa1perpy - 0.5)))) * \
        (1 + 1/(pkappa1perpy - 1.5) * vy**2/pthetaperpy1**2)**(-pkappa1perpy)
    #fperpy2 = np.sqrt(1/(np.pi * pthetaperpy2**2)) * (scipy.special.gamma(pkappa2perpy)/(np.sqrt(pkappa2perpy - 1.5)*scipy.special.gamma((pkappa2perpy - 0.5)))) * \
    #    (1 + 1/(pkappa2perpy - 1.5) * vy**2/pthetaperpy1**2)**(-pkappa2perpy)
    fperpy2 = np.sqrt(1/(np.pi * pthetaperpy2**2)) * \
        np.exp(-vy**2/pthetaperpy2**2)
    #fperpz1 = np.sqrt(1/(np.pi * pthetaperpz1**2)) * (scipy.special.gamma(pkappa1perpz)/(np.sqrt(pkappa1perpz - 1.5)*scipy.special.gamma((pkappa1perpz - 0.5)))) * \
    #    (1 + 1/(pkappa1perpz - 1.5) * vz**2/pthetaperpz1**2)**(-pkappa1perpz)
    fperpz1 = np.sqrt(1/(np.pi * pthetaperpz1**2)) * \
        np.exp(-vz**2/pthetaperpz1**2)
    #fperpz2 = np.sqrt(1/(np.pi * pthetaperpz2**2)) * (scipy.special.gamma(pkappa2perpz)/(np.sqrt(pkappa2perpz - 1.5)*scipy.special.gamma((pkappa2perpz - 0.5)))) * \
    #    (1 + 1/(pkappa2perpz - 1.5) * vz**2/pthetaperpz2**2)**(-pkappa2perpz)
    fperpz2 = np.sqrt(1/(np.pi * pthetaperpz2**2)) * \
        np.exp(-vz**2/pthetaperpz2**2)

    # assuming symmetrical speed scale parameters in sunward/antisunward directions
    fpar1 = (pthetapar1 * np.sqrt(np.pi*(kappa1par - 1.5)) * scipy.special.gamma(kappa1par - 0.5)/scipy.special.gamma(kappa1par))**(-1) * \
        (1 + 1/(kappa1par - 1.5) * (vx - upar)**2/pthetapar1**2)**(-kappa1par)
    fpar2 = (pthetapar2 * np.sqrt(np.pi))**(-1) * \
        np.exp(-(vx - upar)**2/pthetapar2**2)
    #fpar2 = (pthetapar2 * np.sqrt(np.pi*(kappa2par - 1.5)) * scipy.special.gamma(kappa2par - 0.5)/scipy.special.gamma(kappa2par))**(-1) * \
    #    (1 + 1/(kappa2par - 1.5) * (vx - upar)**2/pthetapar2**2)**(-kappa2par)

    #print(fpar1)
    #print(fperpy1)
    #print(fperpz1)
    #print(fpar2)
    #print(fperpy2)
    #print(fperpz2)
    nHratio = 0.224
    return nHinf * nHratio * (fpar1 + fpar2) * (fperpy1 + fperpy2) * (fperpz1 + fperpz2)

# for secondaries - parallel is composition of four Maxwellians, perpendicular is kappa distributions 16.8
weightpar1 = 0.096
upar1 = 28.24
Tpar1 = 1570
weightpar2 = 0.302
upar2 = 21.16
Tpar2 = 3730
weightpar3 = 0.498
upar3 = 13.47
Tpar3 = 7950
weightpar4 = 0.104
upar4 = 9.73
Tpar4 = 13030

superpy = 16.48
sT1perpy = 12540
skappa1perpy = 18
sT2perpy = 11580
skappa2perpy = 16

superpz = 16.8
sT1perpz = 9970
skappa1perpz = 12
sT2perpz = sT1perpz
skappa2perpz = skappa1perpz

# speed scales (in km/s)
sthetaperpy1 = np.sqrt(2*kB*sT1perpy/mH)/1000
sthetaperpy2 = np.sqrt(2*kB*sT2perpy/mH)/1000
sthetaperpz1 = np.sqrt(2*kB*sT1perpz/mH)/1000
sthetaperpz2 = np.sqrt(2*kB*sT2perpz/mH)/1000

def psdsec(vx, vy, vz):
    vyp = vy
    vzp = vz
    vy = vyp*np.cos(thetabv) - vzp*np.sin(thetabv)
    vz = vyp*np.sin(thetabv) + vzp*np.cos(thetabv)
    vx = -vx/1000
    vy = -vy/1000
    vz = -vz/1000
    fpar1 = 1/(2*np.pi*kB*Tpar1/mH)**(0.5) * np.exp(-((vx - upar1)**2)*1000000/(2*np.pi*kB*Tpar1/mH))
    fpar2 = 1/(2*np.pi*kB*Tpar2/mH)**(0.5) * np.exp(-((vx - upar2)**2)*1000000/(2*np.pi*kB*Tpar2/mH))
    fpar3 = 1/(2*np.pi*kB*Tpar3/mH)**(0.5) * np.exp(-((vx - upar3)**2)*1000000/(2*np.pi*kB*Tpar3/mH))
    fpar4 = 1/(2*np.pi*kB*Tpar4/mH)**(0.5) * np.exp(-((vx - upar4)**2)*1000000/(2*np.pi*kB*Tpar4/mH))
    #fpar1 = 1
    #fpar2 = 1
    #fpar3 = 1
    #fpar4 = 1

    fperpy1 = np.sqrt(1/(np.pi * sthetaperpy1**2)) * (scipy.special.gamma(skappa1perpy)/(np.sqrt(skappa1perpy - 1.5)*scipy.special.gamma((skappa1perpy - 0.5)))) * \
        (1 + 1/(skappa1perpy - 1.5) * vy**2/sthetaperpy1**2)**(-skappa1perpy)
    fperpy2 = np.sqrt(1/(np.pi * sthetaperpy2**2)) * (scipy.special.gamma(skappa2perpy)/(np.sqrt(skappa2perpy - 1.5)*scipy.special.gamma((skappa2perpy - 0.5)))) * \
        (1 + 1/(skappa2perpy - 1.5) * vy**2/sthetaperpy1**2)**(-skappa2perpy)
    fperpz1 = np.sqrt(1/(np.pi * sthetaperpz1**2)) * (scipy.special.gamma(skappa1perpz)/(np.sqrt(skappa1perpz - 1.5)*scipy.special.gamma((skappa1perpz - 0.5)))) * \
        (1 + 1/(skappa1perpz - 1.5) * vz**2/sthetaperpz1**2)**(-skappa1perpz)
    fperpz2 = np.sqrt(1/(np.pi * sthetaperpz2**2)) * (scipy.special.gamma(skappa2perpz)/(np.sqrt(skappa2perpz - 1.5)*scipy.special.gamma((skappa2perpz - 0.5)))) * \
        (1 + 1/(skappa2perpz - 1.5) * vz**2/sthetaperpz2**2)**(-skappa2perpz)
    
    #print(fpar1)
    #print(fpar2)
    #print(fpar3)
    #print(fpar4)
    #print(fperpy1)
    #print(fperpy2)
    #print(fperpz1)
    #print(fperpz2)
    nHratio = 0.224
    return nHinf * nHratio * (weightpar1*fpar1 + weightpar2*fpar2 + weightpar3*fpar3 + weightpar4*fpar4) * (fperpy1 + fperpy2) * (fperpz1 + fperpz2)

def psdprimsec(vx, vy, vz):
    #print(psdprim(vx,vy,vz))
    #print(psdsec(vx,vy,vz))
    #return psdprim(vx, vy, vz) + psdsec(vx, vy, vz)
    return psdprim(vx, vy, vz)

#####################################################################################################################################
# CALCULATING A TEST DISTRIBUTION TO SEE WHAT IT LOOKS LIKE
#####################################################################################################################################

vxplot = np.array([])
vyplot = np.array([])
vzplot = np.array([])
testdists = np.array([])

for i in tqdm(range(vxstart.size)):
    for j in range(vystart.size):
        for k in range(vzstart.size):
            vxplot = np.append(vxplot, vxstart[i])
            vyplot = np.append(vyplot, vystart[j])
            vzplot = np.append(vzplot, vzstart[k])
            testdists = np.append(testdists, psdprimsec(vxstart[i], vystart[j], vzstart[k]))

print(max(testdists))

fig3d = plt.figure()
fig3d.set_figwidth(10)
fig3d.set_figheight(7)
ax3d = plt.axes(projection='3d')
scatterplot = ax3d.scatter3D(vxplot[:], vyplot[:], vzplot[:], c=testdists[:], cmap='rainbow', s=1, norm=matplotlib.colors.LogNorm(vmin=10**(-11), vmax=10**(-5)))
cb = fig3d.colorbar(scatterplot)
ax3d.set_xlabel("$v_x$ at Target Point (m/s)")
ax3d.set_ylabel("$v_y$ at Target Point (m/s)")
ax3d.set_zlabel("$v_z$ at Target Point (m/s)")
cb.set_label('Test PSD at Boundary')
plt.show()

# Set of initial conditions in velocity space
# vx/vy initial conditions are sampled on a grid with chosen resolution
vxstart1 = np.arange(-80000, 80000, 1000)
vystart1 = np.arange(-80000, 80000, 1000)
vxplot1 = np.array([])
vyplot1 = np.array([])
testdistsprim = np.array([])
testdistssec = np.array([])

for i in tqdm(range(vxstart1.size)):
    for j in range(vystart1.size):
        vxplot1 = np.append(vxplot1, vxstart1[i])
        vyplot1 = np.append(vyplot1, vystart1[j])
        testdistsprim = np.append(testdistsprim, psdprim(vxstart1[i], vystart1[j], 0))
        testdistssec = np.append(testdistssec, psdsec(vxstart1[i], vystart1[j], 0))

fig = plt.figure()
ax = plt.axes()
#im = ax.pcolormesh(vxplot1,vyplot1,testdistsprim, cmap='rainbow')
im = plt.scatter(vxplot1, vyplot1, c=testdistsprim, cmap='rainbow', norm=matplotlib.colors.LogNorm(vmin=10**(-11), vmax=10**(-5)))
plt.xlabel("vx")
plt.ylabel("vy")
cb = fig.colorbar(im, ax=ax)
cb.set_label('Primary PSD')
plt.show()

fig = plt.figure()
ax = plt.axes()
#im = ax.pcolormesh(vxplot1,vyplot1,testdistsprim, cmap='rainbow')
im = plt.scatter(vxplot1, vyplot1, c=testdistssec, cmap='rainbow', norm=matplotlib.colors.LogNorm(vmin=10**(-11), vmax=10**(-5)))
plt.xlabel("vx")
plt.ylabel("vy")
cb = fig.colorbar(im, ax=ax)
cb.set_label('Secondary PSD')
plt.show()