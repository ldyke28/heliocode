import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
import scipy.interpolate
from tqdm import tqdm

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
finalt = -5*oneyear # time to start backtracing
#6.36674976e9 force free for cosexprp
initialt = -5*10**(10) # time in the past to which the code should backtrace
tstep = 10000 # general time resolution
tstepclose = 300 # time resolution for close regime
tstepfar = 200000 # time resolution for far regime
phase = 0 # phase for implementing rotation of target point around sun
refdist = 70 # upwind reference distance for backtraced trajectories, in au

file = np.loadtxt("C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Time Dependent Irradiance Data/complya.csv", delimiter=',')

day = file[:,0]
irradiance = file[:,1]

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

secondsnew = secondsnew - 1.946*10**9
seconds = seconds - 1.946*10**9
secondstoyears = 1/(86400*365)

wm2toph = 6.12*10**(13)


plt.plot(secondsnew*secondstoyears, irradianceavg*wm2toph, alpha = 0.5)
plt.plot(seconds*secondstoyears, irradiance*wm2toph, alpha=0.5)
plt.ylim([3*10**(11),7.25*10**(11)])
plt.show()

tgrid = np.meshgrid(secondsnew, indexing='ij') # order will be z, y, x for this

irradianceinterp = scipy.interpolate.RegularGridInterpolator(points=[secondsnew], values=irradianceavg)


def lya_abs(t,x,y,z,vr):
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
    tdependence = .95 + .5/(np.e**2 + 1) + .5/(np.e + 1/np.e)*np.cos(omegat - np.pi)*np.exp(np.cos(omegat - np.pi))

    # flattening factor for heliolatitude dependence of radiation pressure force
    alya = .8
    Itotavg = 0.006944444
    latdep = np.sqrt(alya*np.sin(latangle)**2 + np.cos(latangle)**2)
    ttemp = t
    tbounded = False
    while not tbounded:
        if ttemp >= -1.392*10**(9):
            tbounded = True
        else:
            ttemp = ttemp + 1.392*10**(9)
    tdependence = irradianceinterp([ttemp])*latdep/Itotavg
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
    #print(a_bkg)
    r_E = 0.6
    r2 = 1
    F_R = A_R / (del_R * np.sqrt(2 * np.pi)) *np.exp(-(np.square((vr/1000) - (m_K + dm))) / (2*(del_R ** 2)))
    F_bkg = np.add(a_bkg*(vr/1000)*0.000001,b_bkg)
    F_K = A_K * np.power(1 + np.square((vr/1000) - m_K) / (2 * K * ((del_K) ** 2)), -K - 1)

    #(F_K-F_R+F_bkg)/((r_E/r)**2)
    return scalefactor*(F_K-F_R+F_bkg)/(r_E**2/(r2**2))*(1 - absval)



t = 0
t2 = oneyear*1.1
t3 = oneyear*2.2
t4 = oneyear*3.3
t5 = oneyear*4.4
t6 = oneyear*5.5

inputvr = np.arange(-370000, 370000, 740000/1000)
profile1 = np.zeros(inputvr.size)
profile1t2 = np.zeros(inputvr.size)
profile1t3 = np.zeros(inputvr.size)
profile1t4 = np.zeros(inputvr.size)
profile1t5 = np.zeros(inputvr.size)
profile1t6 = np.zeros(inputvr.size)
profile2 = np.zeros(inputvr.size)
profile2t2 = np.zeros(inputvr.size)
profile2t3 = np.zeros(inputvr.size)
profile3 = np.zeros(inputvr.size)
profile3t2 = np.zeros(inputvr.size)
profile3t3 = np.zeros(inputvr.size)
for i in range(inputvr.size):
    profile1[i] = lya_abs(t, au, 0, 0, inputvr[i])
    profile1t2[i] = lya_abs(t2, au, 0, 0, inputvr[i])
    profile1t3[i] = lya_abs(t3, au, 0, 0, inputvr[i])
    profile1t4[i] = lya_abs(t4, au, 0, 0, inputvr[i])
    profile1t5[i] = lya_abs(t5, au, 0, 0, inputvr[i])
    profile1t6[i] = lya_abs(t6, au, 0, 0, inputvr[i])

fsize = 18
fig, ax = plt.subplots()
fig.set_figwidth(9)
fig.set_figheight(6)
ax.plot(inputvr/1000, profile1, label="t = 0 yrs")
ax.plot(inputvr/1000, profile1t2, label="t = 1.1 yrs")
ax.plot(inputvr/1000, profile1t3, label="t = 2.2 yrs")
ax.plot(inputvr/1000, profile1t4, label="t = 3.3 yrs")
ax.plot(inputvr/1000, profile1t5, label="t = 4.4 yrs")
ax.plot(inputvr/1000, profile1t6, label="t = 5.5 yrs")
ax.legend()
plt.xticks(fontsize=fsize)
plt.yticks(fontsize=fsize)
plt.ylim(bottom=0)
plt.grid()
#plt.xlim(-40,40)
ax.set_xlabel("Radial Velocity Component $v_r$ (km/s)", fontsize=fsize)
ax.set_ylabel("Value of $\mu (t)$", fontsize=fsize)
#plt.title("Photoionization Rate over Time", fontsize=fsize)
plt.show()