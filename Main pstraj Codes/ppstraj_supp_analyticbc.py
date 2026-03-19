import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import odeint
import scipy
import scipy.interpolate
from tqdm import tqdm
import h5py
from scipy.signal import butter, lfilter, freqz

#####################################################################################################################################
# CODE INTRODUCTION
#####################################################################################################################################

"""
This code re-calculates any points that were lost to the odeint warning in running on the cluster. The processes of the code are the
same as pstraj_timedepupdate, but the initial conditions are determined by the set of (vx, vy, vz) values as given in the lost points
file from which the code imports data.
"""

#####################################################################################################################################

inputfilename = "C:/Users/lukeb/Documents/Dartmouth/HSResearch/Cluster Runs/2026Data/lostpoints_312deg_ibexshifted_lya_analyticbc_datamu_nofilter_newpi_3500vres"
inputfile = np.loadtxt(inputfilename + ".txt", delimiter=',')
theta = 312 # angle with respect to upwind axis of target point
oneyear = 3.15545454545*10**7
# 120749800 for first force free
# 226250200 for second force free
finalt = 22/60*oneyear # time to start backtracing

# Set of initial conditions in velocity space
# vx/vy initial conditions are sampled on a grid with chosen resolution
vxstart = np.array([])
vystart = np.array([])
vzstart = np.array([])
for i in range(np.shape(inputfile)[0]):
    if not (inputfile[i,0] == 0 and inputfile[i,1] == 0 and inputfile[i,2] == 0):
        vxstart = np.append(vxstart, [inputfile[i,0]])
        vystart = np.append(vystart, [inputfile[i,1]])
        vzstart = np.append(vzstart, [inputfile[i,2]])

tstepclose = 1200 # time resolution for close regime


# Value for 1 au (astronomical unit) in meters
au = 1.496*10**11
msolar = 1.98847*10**30 # mass of the sun in kg
G = 6.6743*10**(-11) # value for gravitational constant in SI units
mH = 1.6736*10**(-27) # mass of hydrogen in kg
# one year in s = 3.156e7 s (Julian year, average length of a year)
# 11 Julian years = 3.471e8 s
# Note to self: solar maximum in April 2014
kB = 1.381*10**(-23) # Boltzmann consant in SI units

#6.36674976e9 force free for cosexprp
initialt = -50000000000 # time in the past to which the code should backtrace
tstep = 1000 # general time resolution
tstepfar = 200000 # time resolution for far regime
phase = 0 # phase for implementing rotation of target point around sun
refdist = 100

# Location of the sun in [x,y,z] - usually this will be at 0, but this makes it flexible just in case
# Second line is location of the point of interest in the same format (which is, generally, where we want IBEX to be)
sunpos = np.array([0,0,0])
ibexrad = 1 # radial distance of target point from Sun
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
    return psdprim(vx, vy, vz) + psdsec(vx, vy, vz)
    #return psdprim(vx, vy, vz)

#####################################################################################################################################
# FILTERING AND INTERPOLATING THE LASP IRRADIANCE DATA TO USE FOR THE RADIATION PRESSURE FORCE
#####################################################################################################################################


irradfile = np.loadtxt("C:/Users/lukeb/Documents/Dartmouth/HSResearch/Time Dependent Irradiance Data/complya.csv", delimiter=',')

day = irradfile[:,0]
irradiance = irradfile[:,1]

seconds = day*86400

N = 28

oldirradianceavg = np.zeros(seconds.size)
for i in range(seconds.size):
    avgnumerator = 0
    lowerbound = -(int(N/2))
    upperbound = int(N/2)
    if i + lowerbound < 0:
        lowerbound = -i
    if i + upperbound >= seconds.size:
        upperbound = seconds.size - i - 1
    for j in range(-lowerbound):
        avgnumerator += irradiance[i+lowerbound]
    avgnumerator += irradiance[i]
    for k in range(upperbound):
        avgnumerator += irradiance[i+upperbound]
    oldirradianceavg[i] = avgnumerator/(upperbound - lowerbound + 1)

irradianceavg = np.array([])
secondsnew = np.array([])
for i in range(seconds.size):
    if (i+1) % N == 0:
        irradianceavg = np.append(irradianceavg, [oldirradianceavg[i]])
        secondsnew = np.append(secondsnew, [seconds[i]])

secondsnew = secondsnew - 1.9*10**9
seconds = seconds - 1.9*10**9
secondstoyears = 1/(86400*365)

wm2toph = 6.12*10**(13)

def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


# Filter requirements.
order = 5
#fs = .000002       # sample rate, Hz
fs = 1/(86400)
cutoff = 1/(1.577*10**8)  # desired cutoff frequency of the filter, Hz
offset = np.mean(irradiance[0:10])

irradianceoffset = irradiance - offset

filterediaoffset = butter_lowpass_filter(irradianceoffset, cutoff, fs, order)

filteredia = filterediaoffset + offset

tgrid = np.meshgrid(secondsnew, indexing='ij') # order will be z, y, x for this

fifthorderoffset = 2.5*oneyear
firstorderoffset = .5*oneyear
fifthordertwocyclesagooffset = 2.5*oneyear - 22.4*oneyear

desiredoffset = fifthorderoffset

secondsoffset = seconds-desiredoffset

#loopoffset = -32.408
loopoffset = -54.80

irradianceinterp = scipy.interpolate.RegularGridInterpolator(points=[seconds-desiredoffset], values=filteredia)

#####################################################################################################################################

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


def lya_abs(t,x,y,z,vr):
    # taken from eq. 8 in https://articles.adsabs.harvard.edu/pdf/1995A%26A...296..248R
    omegat = 2*np.pi/(3.47*10**(8))*t
    # changing over into ecliptic coordinates (approximating solar north pole to be same as NEP)
    x = x*np.cos(eclipticangle) + z*np.sin(eclipticangle)
    y = y
    z = -x*np.sin(eclipticangle) + z*np.cos(eclipticangle)
    # can apply a second rotation after this if need be
    r = np.sqrt(x**2 + y**2 + z**2)
    rxy = np.sqrt(x**2 + y**2)
    # calculating the latitudinal (polar) angle in 3D space
    # since sine/cosine only covers half of the space, we have to manually check where the point is to get the right angle
    if z >= 0:
        latangle = np.pi/2 - np.arcsin(z/r)
    else:
        latangle = np.pi/2 + np.arcsin(np.abs(z)/r)
    # calculating the longitudinal (azimuthal) angle in 3D space
    if y >= 0:
        longangle = np.arccos(x/rxy)
    else:
        longangle = 2*np.pi - np.arccos(x/rxy)
    longangle = longangle - np.pi
    if longangle < 0:
        longangle = 2*np.pi + longangle
    latangled = latangle*180/np.pi
    longangled = longangle*180/np.pi

    alpha = .07 # alpha for the skew gaussian distribution
    
    # calculating parameters from IKL et al. 2022 paper: https://ui.adsabs.harvard.edu/abs/2022ApJ...926...27K/abstract
    # manually fitted based on Figure 3
    # amplitude
    if r < au:
        amp = 0
    else:
        amp = ((.59*(r/au - 12)/np.sqrt((r/au - 12)**2 + 200) + 0.38) + -0.4* \
        np.e**(-(180 - longangled)**2/50**2 - (r/au - 31)**2/15**2)*(1 + \
        scipy.special.erf(alpha*(r/au)/np.sqrt(2)))*(1 - np.e**(-(r/au)/4)))*1/.966

    # mean Doppler shift
    mds = -20*np.cos(longangle)*np.cos((latangled-100)*np.pi/180)
    # dispersion (width of the peak)
    disper = -.0006947*(r/au)**2 + .1745*(r/au) + 5.402 + \
        1.2*np.e**(-(longangled - 5)**2/50**2 - ((r/au) - 80)**2/60**2) + \
        3*np.e**(-(longangled - 180)**2/50**2 - ((r/au))**2/5**2) + \
        1*np.e**(-(longangled - 170)**2/50**2 - ((r/au) - 25)**2/200**2) + \
        .35*np.cos(((latangled - 75)*np.pi/180)*2)
    # fit exponent
    if r >= 50*au:
        fittype = 4
    else:
        fittype = 2
    # calculating equation 12 from the 2022 paper
    absval = amp*np.exp(-.5 * ((vr/1000 - mds)/disper)**fittype)

    # time dependent portion of the radiation pressure force function
    #tdependence = .85 - np.e/(np.e + 1/np.e)*.33 + .33/(np.e + 1/np.e) * np.cos(omegat - np.pi)*np.exp(np.cos(omegat - np.pi)) + addfactor
    #tdependence = .95 + .5/(np.e**2 + 1) + .5/(np.e + 1/np.e)*np.cos(omegat - np.pi)*np.exp(np.cos(omegat - np.pi))

    # flattening factor for heliolatitude dependence of radiation pressure force
    alya = .8
    Itotavg = 0.006944444
    latdep = np.sqrt(alya*np.sin(latangle)**2 + np.cos(latangle)**2)
    ttemp = t
    tbounded = False
    while not tbounded:
        if ttemp >= loopoffset*oneyear:
            tbounded = True
        else:
            ttemp = ttemp + 1.392*10**(9)
    tdependence = irradianceinterp([ttemp])[0]*latdep/Itotavg
    #print(irradianceinterp([ttemp]))
    # an added scale factor to adjust the total irradiance of the integral without changing the shape (adjusts total magnitude by a factor)
    # scalefactor should match divisor in first term of addfactor
    scalefactor = .333
    
    # parameters of function
    A_K = 6.523*(1 + 0.619*tdependence)
    m_K = 5.143*(1 - 1.081*tdependence)
    del_K = 38.008*(1 + 0.104*tdependence)
    K = 2.165*(1 - 0.301*tdependence)
    A_R = 580.37*(1 + 0.28*tdependence)
    dm = -0.344*(1 - 0.828*tdependence)
    del_R = 32.349*(1 - 0.049*tdependence)
    b_bkg = 0.035*(1 + 0.184*tdependence)
    a_bkg = 0.411**(-4) *(1 - 1.333*tdependence)

    r_E = 0.6
    r2 = 1
    F_R = A_R / (del_R * np.sqrt(2 * np.pi)) *np.exp(-(np.square((vr/1000) - (m_K + dm))) / (2*(del_R ** 2)))
    F_bkg = np.add(a_bkg*(vr/1000)*0.000001,b_bkg)
    F_K = A_K * np.power(1 + np.square((vr/1000) - m_K) / (2 * K * ((del_K) ** 2)), -K - 1)

    #(F_K-F_R+F_bkg)/((r_E/r)**2)
    #print(scalefactor*(F_K-F_R+F_bkg)/(r_E**2/(r2**2))*(1 - absval))
    return scalefactor*(F_K-F_R+F_bkg)/(r_E**2/(r2**2))*(1 - absval)

# odeint documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html


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


def Var_dr_dt(x,t,rp):
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
    radp = rp(t,x[0],x[1],x[2],v_r)
    dx3 = (msolar*G/(r**3))*(sunpos[0]-x[0])*(1-radp)
    dx4 = (msolar*G/(r**3))*(sunpos[1]-x[1])*(1-radp)
    dx5 = (msolar*G/(r**3))*(sunpos[2]-x[2])*(1-radp)
    return [dx0, dx1, dx2, dx3, dx4, dx5]
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# code for tracking phase space at distance of x = 100 au away
farvx = np.array([])
farvy = np.array([])
farvz = np.array([])
fart = np.array([])
maxwcolor = np.array([])
vrmax = np.array([])
vrmin = np.array([])
backtraj = np.zeros((t.size, 6))
for i in tqdm(range(vxstart.size)): # displays progress bars for both loops to measure progress
    init = [xstart, ystart, zstart, vxstart[i], vystart[i], vzstart[i]]
    t[::-1].sort()
    # calculating trajectories for each initial condition in phase space given
    backtraj[:,:] = odeint(Var_dr_dt, init, t, args=(lya_abs,))
    btr = np.sqrt((backtraj[:,0]-sunpos[0])**2 + (backtraj[:,1]-sunpos[1])**2 + (backtraj[:,2]-sunpos[2])**2)
    if any(btr <= .00465*au):
        # tells the code to not consider the trajectory if it at any point intersects the width of the sun
        farvx = np.append(farvx, [backtraj[0,3]])
        farvy = np.append(farvy, [backtraj[0,4]])
        farvz = np.append(farvz, [backtraj[0,5]])
        fart = np.append(fart, [0])
        # sets the value of the NPSD to -1 to indicate the trajectory isn't viable, as a special flag to investigate later
        maxwcolor = np.append(maxwcolor, [-1])
        continue
    if np.all(btr[:] < refdist*au):
        # forgoes the following checks if the trajectory never passes through the plane at the reference distance upwind
        farvx = np.append(farvx, [backtraj[0,3]])
        farvy = np.append(farvy, [backtraj[0,4]])
        farvz = np.append(farvz, [backtraj[0,5]])
        fart = np.append(fart, [0])
        # sets the value of the NPSD to 0 to indicate the trajectory isn't viable
        maxwcolor = np.append(maxwcolor, [0])
        continue
    for k in range(t.size - countoffset):
        if btr[k+countoffset] >= refdist*au and btr[k-1+countoffset] <= refdist*au:
            # adjusting the indexing to avoid checking in the close regime
            #kn = k+tclose.size
            kn = k+countoffset
            # radius in paper given to be 14 km/s
            # only saving initial conditions corresponding to points that lie within this Maxwellian at reference distance
            #if backtraj[k-1,3,(i)*vystart.size + (j)] <= -22000 and backtraj[k-1,3,(i)*vystart.size + (j)] >= -40000 and backtraj[k-1,4,(i)*vystart.size + (j)] <= 14000 and backtraj[k-1,4,(i)*vystart.size + (j)] >= -14000:
            #if np.sqrt((backtraj[kn-1,3]+26000)**2 + (backtraj[kn-1,4])**2 + (backtraj[kn-1,5])**2) <= 27000:
            # determining which distribution to use by calculating heliolongitude
            endradxy = np.sqrt((sunpos[0]-backtraj[kn+1,0])**2 + (sunpos[1]-backtraj[kn+1,1])**2)
            belowxaxis = backtraj[kn+1,1] < 0
            ymask = belowxaxis*2*np.pi
            longmask = -2*(belowxaxis-.5) # -1 if below x axis in xy plane, 1 if above
            # if y < 0, longitude = 2pi-arccos(x/r), otherwise longitude = arccos(x/r)
            endlongangle = ymask + np.arccos((backtraj[kn+1,0] - sunpos[0])/endradxy)*longmask
            endlongangle = endlongangle*180/np.pi
            # finding the initial value of the distribution function based on the interpolated distributions
            endvelcoords = [backtraj[kn+1,5],backtraj[kn+1,4],backtraj[kn+1,3]]
            initpsd = psdprimsec(endvelcoords[2], endvelcoords[1], endvelcoords[0])

            vsolarwindms = 400000 # solar wind speed in m/s
            vsolarwindcms = vsolarwindms*100 # solar wind speed in cm/s
            nsw1au = 5 # solar wind density at 1 au in cm^-3

            r1 = 1*au # reference radius
            # radial distance to the Sun at all points throughout the trajectory
            currentrad = np.sqrt((sunpos[0]-backtraj[0:kn+1,0])**2 + (sunpos[1]-backtraj[0:kn+1,1])**2 + (sunpos[2]-backtraj[0:kn+1,2])**2)

            # velocity squared at each point in time for the trajectory
            currentvsq = np.square(backtraj[0:kn+1,3]) + np.square(backtraj[0:kn+1,4]) + np.square(backtraj[0:kn+1,5])
            # thermal velocity (temperature taken from Federico's given temperature)
            vth = np.sqrt(2 * 1.381*10**(-23) * 7500 / (1.672*10**(-27)))
            # omega for the relative velocity
            omegavs = np.abs(np.sqrt(currentvsq) - vsolarwindms)/vth
            # calculating the current collision velocity of the particle and the SW
            currentvrel = vth*(np.exp(-omegavs**2)/np.sqrt(np.pi) + (omegavs + 1/(2*omegavs))*scipy.special.erf(omegavs))
            # calculating kinetic energy in keV at each point in time
            currentKE = (.5 * mH * np.square(currentvrel)) * 6.242*10**(15)

            # parameters for function to calculate charge exchange cross section
            a1 = 4.049
            a2 = 0.447
            a3 = 60.5
            # function for H-H+ charge exchange cross section, from Swaczyna et al. (2019)
            # result is in cm^2
            # https://ui.adsabs.harvard.edu/abs/2019AGUFMSH51C3344S/abstract
            cxcrosssection = ((a1 - a2*np.log(currentKE))**2 * (1 - np.exp(-a3 / currentKE))**(4.5))*10**(-16)

            #nsw = nsw1au * (r1/currentrad)**2 # assuming r^2 falloff for density (Sokol et al. 2019)

            cxirate = nsw1au * currentvrel*100 * cxcrosssection

            #print(cxirate)

            # approximate time-averaged charge exchange photoionization rate from Sokol et al. 2019
            #cxirate = 5*10**(-7)
            # omega*t for each time point in the trajectory
            omt = 2*np.pi/(3.47*10**(8))*t[0:kn+1]
            # function for the photoionization rate at each point in time
            #PIrate2 = 10**(-7)*(1 + .7/(np.e + 1/np.e)*(np.cos(omt - np.pi)*np.exp(np.cos(omt - np.pi)) + 1/np.e))
            #PIrate2 = 1.21163*10**(-7) # time average of above
            ttemp = t[0:kn+1]
            tbounded = False
            for ind2 in range(ttemp.size):
                tbounded = False
                while not tbounded:
                    if ttemp[ind2] >= loopoffset*oneyear:
                        tbounded = True
                    else:
                        ttemp[ind2] = ttemp[ind2] + 1.392*10**(9)
            PIrate2 = np.zeros(t[0:kn+1].size)
            for ind1 in range(t[0:kn+1].size):
                PIrate2[ind1] = (filteredia[np.abs(secondsoffset - ttemp[ind1]).argmin()]*wm2toph/(10**(11)))**(1.49929) / (7.25503 * 10**(7))
            
            # calculating the component of the radial unit vector in each direction at each point in time
            nrvecx = (-sunpos[0]+backtraj[0:kn+1,0])/currentrad
            nrvecy = (-sunpos[1]+backtraj[0:kn+1,1])/currentrad
            nrvecz = (-sunpos[2]+backtraj[0:kn+1,2])/currentrad
            # calculating the magnitude of v_r at each point in time
            currentvr = backtraj[0:kn+1,3]*nrvecx[0:kn+1] + backtraj[0:kn+1,4]*nrvecy[0:kn+1] + backtraj[0:kn+1,5]*nrvecz[0:kn+1]
            xecliptic = backtraj[0:kn+1,0]*np.cos(eclipticangle) + backtraj[0:kn+1,2]*np.sin(eclipticangle)
            yecliptic = backtraj[0:kn+1,1]
            zecliptic = -backtraj[0:kn+1,0]*np.sin(eclipticangle) + backtraj[0:kn+1,2]*np.cos(eclipticangle)
            cradecl = np.sqrt((sunpos[0]-xecliptic)**2 + (sunpos[1]-yecliptic)**2 + (sunpos[2]-zecliptic)**2)
            # calculation of heliographic latitude angle (polar angle)
            belowxy = zecliptic < 0
            zmask = 2*(belowxy-.5)
            latangleecl = np.pi/2 + zmask*np.arcsin(np.abs(zecliptic - sunpos[2])/currentrad[:])

            # integrand for the photoionization losses, with photoionization adjusted based on heliographic latitude angle
            btintegrand = PIrate2/currentvr*(r1/currentrad)**2*(.85*(np.sin(latangleecl))**2 + (np.cos(latangleecl))**2) + + cxirate/currentvr*(r1/currentrad)**2

            # calculation of heliographic latitude angle (polar angle)
            #belowxy = backtraj[0:kn+1,2] < 0
            #zmask = 2*(belowxy-.5)
            #latangle = np.pi/2 + zmask*np.arcsin(np.abs(backtraj[0:kn+1,2] - sunpos[2])/currentrad[:])
            
            # calculation of attenuation factor based on heliographic latitude angle
            #btintegrand = btintegrand*(.85*(np.sin(latangle))**2 + (np.cos(latangle))**2)
            # calculation of attenuation factor
            attfact = scipy.integrate.simpson(btintegrand, x=currentrad)
            # calculating the value of the phase space density after attenuation
            psdval = np.exp(-np.abs(attfact))*initpsd
            #if psdval > 10**(-11):
            # retaining variables corresponding to vx, vy, t at the target point
            farvx = np.append(farvx, [backtraj[0,3]])
            farvy = np.append(farvy, [backtraj[0,4]])
            farvz = np.append(farvz, [backtraj[0,5]])
            fart = np.append(fart, [startt - t[kn-1]])
            # calculating value of phase space density based on the value at the crossing of x = 100 au
            maxwcolor = np.append(maxwcolor, [psdval])
            #maxwcolor = np.append(maxwcolor, [initpsd])
            #maxwcolor = np.append(maxwcolor, [np.exp(-((backtraj[kn-1,3]+26000)**2 + backtraj[kn-1,4]**2 + backtraj[kn-1,5]**2)/(10195)**2)])
            break
            #break
        if k == (t.size + countoffset) - 1:
            print("fail")
            farvx = np.append(farvx, [backtraj[0,3]])
            farvy = np.append(farvy, [backtraj[0,4]])
            farvz = np.append(farvz, [backtraj[0,5]])
            fart = np.append(fart, [0])
            # sets the value of the NPSD to 0 to indicate the trajectory isn't viable
            maxwcolor = np.append(maxwcolor, [0])
                


# writing data to a file - need to change each time or it will overwrite previous file
file = open(inputfilename + "_revised.txt", 'w')
#file = open("/Users/ldyke/Desktop/Dartmouth/HSResearch/Code/Kepler/Python Orbit Code/datafiles/p1fluccosexprp_35pi36_0y_direct_cosexppi_tclose400.txt", "w")
for i in range(farvx.size):
    # writes vx, vy, and attenuated NPSD value
    file.write(str(farvx[i]/1000) + ',' + str(farvy[i]/1000) + ',' + str(farvz[i]/1000) + ',' + str(maxwcolor[i]) + '\n')
file.close()