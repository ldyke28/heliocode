import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import odeint
import scipy
import scipy.interpolate
from tqdm import tqdm
import h5py
from scipy.signal import butter, lfilter, freqz


# RELEVANT (FIXED) VARIABLES THROUGHOUT THE CODE
#####################################################################################################################################
# Value for 1 au (astronomical unit) in meters
au = 1.496*10**11
msolar = 1.98847*10**30 # mass of the sun in kg
G = 6.6743*10**(-11) # value for gravitational constant in SI units
mH = 1.6736*10**(-27) # mass of hydrogen in kg
# one year in s = 3.156e7 s (Julian year, average length of a year)
# 11 Julian years = 3.471e8 s
# Note to self: solar maximum in April 2014
oneyear = 3.15545454545*10**7

# 120749800 for first force free
# 226250200 for second force free
finalt = 0*oneyear # time to start backtracing
#6.36674976e9 force free for cosexprp
initialt = -5*10**(10) # time in the past to which the code should backtrace
tstep = 10000 # general time resolution
tstepclose = 300 # time resolution for close regime
tstepfar = 200000 # time resolution for far regime
phase = 0 # phase for implementing rotation of target point around sun
refdist = 70 # reference distance for backtraced trajectories, in au

theta = 275 # angle with respect to upwind axis of target point in degrees
ibexrad = 1 # radial distance of target point from Sun

vsolarwindms = 400000 # solar wind speed in m/s
vsolarwindcms = vsolarwindms*100 # solar wind speed in cm/s
nsw1au = 5 # solar wind density at 1 au in cm^-3

r1 = 1*au # reference radius for ionization integration

noncalcextent = 15 # radius in km/s around the axis of exclusion where we don't want to calculate trajectories (to save computational time)

# FILTERING AND INTERPOLATING THE LASP IRRADIANCE DATA TO USE FOR THE RADIATION PRESSURE FORCE
#####################################################################################################################################

irradfile = np.loadtxt("C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Time Dependent Irradiance Data/complya.csv", delimiter=',')

day = irradfile[:,0]
irradiance = irradfile[:,1]


print(np.average(irradiance))

seconds = day*86400

N = 28

oldirradianceavg = np.zeros(seconds.size)
for i in tqdm(range(seconds.size)):
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

datatimeoffset = 1.955*10**9
secondsnew = secondsnew - datatimeoffset
seconds = seconds - datatimeoffset
#seconds = seconds - np.max(seconds)
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

fifthorderoffset = 2.5*oneyear  - 5.5*oneyear#- 22.4*oneyear
firstorderoffset = .5*oneyear

irradianceinterp = scipy.interpolate.RegularGridInterpolator(points=[seconds-fifthorderoffset], values=filteredia)
#irradianceinterp = scipy.interpolate.RegularGridInterpolator(points=[seconds+29.5*oneyear], values=irradiance)

#tcheck = np.arange(-1*10**9, 1*10**8, 1*10**4)
#interpcheck = np.zeros(tcheck.size)
#for i in range(tcheck.size):
#    interpcheck[i] = irradianceinterp([tcheck[i]])

plt.plot((seconds-fifthorderoffset)*secondstoyears, filteredia*wm2toph/(10**(11)), alpha = 0.7,color='b')
#plt.plot((tcheck)*secondstoyears, interpcheck*wm2toph/(10**(11)), alpha = 0.7,color='y')
plt.plot((seconds)*secondstoyears, irradiance*wm2toph/(10**(11)), alpha=0.7, color='r')
plt.ylim([3,7.5])
plt.xlabel("Time (yrs)")
plt.ylabel("10$^{11}$ Irradiance (ph cm$^{-2}$ s$^{-1}$)")
plt.show()

#####################################################################################################################################


def lya_abs_update(t,x,y,z,vr):
    # taken from eq. 8 in https://articles.adsabs.harvard.edu/pdf/1995A%26A...296..248R
    omegat = 2*np.pi/(3.47*10**(8))*t
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
    #Itotavg = 0.007493
    latdep = np.sqrt(alya*np.sin(latangle)**2 + np.cos(latangle)**2)
    #print(latdep)
    ttemp = t
    tbounded = False
    while not tbounded:
        if ttemp >= -32.408*oneyear:
            tbounded = True
        else:
            ttemp = ttemp + 1.392*10**(9)
    tdependence = irradianceinterp([ttemp])*latdep/Itotavg
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
    a_bkg = 0.411*10**(-4) *(1 - 1.333*tdependence)
    #print(a_bkg)
    r_E = 1
    r2 = 1
    F_R = A_R / (del_R * np.sqrt(2 * np.pi)) *np.exp(-(np.square((vr/1000) - (m_K + dm))) / (2*(del_R ** 2)))
    F_bkg = a_bkg*(vr/1000) + b_bkg
    F_K = A_K * (1 + ((vr/1000) - m_K)**2 / (2 * K * ((del_K) ** 2)))**(-K-1)

    #(F_K-F_R+F_bkg)/((r_E/r)**2)
    #return scalefactor*(F_K-F_R+F_bkg)/(r_E**2/(r2**2))*(1 - absval)
    return (F_K-F_R+F_bkg)/(r_E**2/(r2**2))*(1 - absval)

t = 0
t2 = oneyear*1.1
t3 = oneyear*2.2
t4 = oneyear*3.3
t5 = oneyear*4.4
t6 = oneyear*5.5
t7 = oneyear*6.6
t8 = oneyear*7.7
t9 = oneyear*8.8
t10 = oneyear*9.9
t11 = oneyear*11

trange = np.arange(0, 11*oneyear, 11*oneyear/11)

inputvr = np.arange(-370000, 370000, 740000/1000)
profile1 = np.zeros(inputvr.size)
profile1t2 = np.zeros(inputvr.size)
profile1t3 = np.zeros(inputvr.size)
profile1t4 = np.zeros(inputvr.size)
profile1t5 = np.zeros(inputvr.size)
profile1t6 = np.zeros(inputvr.size)
profile1t7 = np.zeros(inputvr.size)
profile1t8 = np.zeros(inputvr.size)
profile1t9 = np.zeros(inputvr.size)
profile1t10 = np.zeros(inputvr.size)
profile1t11 = np.zeros(inputvr.size)
profile2 = np.zeros(inputvr.size)
profile2t2 = np.zeros(inputvr.size)
profile2t3 = np.zeros(inputvr.size)
profile3 = np.zeros(inputvr.size)
profile3t2 = np.zeros(inputvr.size)
profile3t3 = np.zeros(inputvr.size)
"""for i in range(inputvr.size):
    profile1[i] = lya_abs_update(t,1*au,0,0,inputvr[i])
    profile1t2[i] = lya_abs_update(t2,1*au,0,0,inputvr[i])
    profile1t3[i] = lya_abs_update(t3,1*au,0,0,inputvr[i])
    profile1t4[i] = lya_abs_update(t4,1*au,0,0,inputvr[i])
    profile1t5[i] = lya_abs_update(t5,1*au,0,0,inputvr[i])
    profile1t6[i] = lya_abs_update(t6,1*au,0,0,inputvr[i])
    profile1t7[i] = lya_abs_update(t7,1*au,0,0,inputvr[i])
    profile1t8[i] = lya_abs_update(t8,1*au,0,0,inputvr[i])
    profile1t9[i] = lya_abs_update(t9,1*au,0,0,inputvr[i])
    profile1t10[i] = lya_abs_update(t10,1*au,0,0,inputvr[i])
    profile1t11[i] = lya_abs_update(t11,1*au,0,0,inputvr[i])"""

for i in range(inputvr.size):
    profile1[i] = lya_abs_update(t,0.1*au,1*au,0,inputvr[i])
    profile1t2[i] = lya_abs_update(t2,0.1*au,1*au,0,inputvr[i])
    profile1t3[i] = lya_abs_update(t3,0.1*au,1*au,0,inputvr[i])
    profile1t4[i] = lya_abs_update(t4,0.1*au,1*au,0,inputvr[i])
    profile1t5[i] = lya_abs_update(t5,0.1*au,1*au,0,inputvr[i])
    profile1t6[i] = lya_abs_update(t6,0.1*au,1*au,0,inputvr[i])
    profile1t7[i] = lya_abs_update(t7,0.1*au,1*au,0,inputvr[i])
    profile1t8[i] = lya_abs_update(t8,0.1*au,1*au,0,inputvr[i])
    profile1t9[i] = lya_abs_update(t9,0.1*au,1*au,0,inputvr[i])
    profile1t10[i] = lya_abs_update(t10,0.1*au,1*au,0,inputvr[i])
    profile1t11[i] = lya_abs_update(t11,0.1*au,1*au,0,inputvr[i])
profiles = np.zeros((trange.size,inputvr.size))


fsize = 18
fig, ax = plt.subplots()
fig.set_figwidth(9)
fig.set_figheight(6)
ax.plot(inputvr/1000, profile1, label="t = -5.5 yrs")
ax.plot(inputvr/1000, profile1t2, label="t = -4.4 yrs")
ax.plot(inputvr/1000, profile1t3, label="t = -3.3 yrs")
ax.plot(inputvr/1000, profile1t4, label="t = -2.2 yrs")
ax.plot(inputvr/1000, profile1t5, label="t = -1.1 yrs")
ax.plot(inputvr/1000, profile1t6, label="t = 0 yrs")
ax.plot(inputvr/1000, profile1t7, label="t = 1.1 yrs")
ax.plot(inputvr/1000, profile1t8, label="t = 2.2 yrs")
ax.plot(inputvr/1000, profile1t9, label="t = 3.3 yrs")
ax.plot(inputvr/1000, profile1t10, label="t = 4.4 yrs")
ax.plot(inputvr/1000, profile1t11, label="t = 5.5 yrs")
ax.legend()
plt.xticks(fontsize=fsize)
plt.yticks(fontsize=fsize)
plt.ylim(bottom=0)
plt.grid()
plt.xlim(-200,200)
plt.ylim(0,2)
ax.set_xlabel("Radial Velocity Component $v_r$ (km/s)", fontsize=fsize)
ax.set_ylabel("Value of $\mu (t)$", fontsize=fsize)
#plt.title("Photoionization Rate over Time", fontsize=fsize)
plt.show()