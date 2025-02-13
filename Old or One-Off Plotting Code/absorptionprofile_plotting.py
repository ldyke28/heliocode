import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
from tqdm import tqdm

oneyear = 3.15545454545*10**7
au = 1.496*10**11

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
    #longangle = np.pi - longangle
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

# series of times at which to plot profiles for convenience
t = 0
t2 = oneyear*1.1
t3 = oneyear*2.2
t4 = oneyear*3.3
t5 = oneyear*4.4
t6 = oneyear*5.5

# x, y, and z coordinates to allow for plotting of the profile at any target point (relevant for absorption)
xplot = 2.5*au
yplot = 0
zplot = 0

# radial distances from the Sun (also relevant for absorption)
rplot1 = 2.5*au
rplot2 = 17*au
rplot3 = 50*au
rplot4 = 70*au

# a range of times throughout the whole solar cycle
trange = np.arange(0, 11*oneyear, 11*oneyear/1000)

# range of vr values to plot across 
inputvr = np.arange(-370000, 370000, 740000/1000)
profile1 = np.zeros(inputvr.size)
profile1p2 = np.zeros(inputvr.size)
profile1p3 = np.zeros(inputvr.size)
profile1p4 = np.zeros(inputvr.size)
for i in range(inputvr.size):
    profile1[i] = lya_abs(t, -rplot1, 0, 0, inputvr[i])
    profile1p2[i] = lya_abs(t, -rplot2, 0, 0, inputvr[i])
    profile1p3[i] = lya_abs(t, -rplot3, 0, 0, inputvr[i])
    profile1p4[i] = lya_abs(t, -rplot4, 0, 0, inputvr[i])

fsize = 18
fig, ax = plt.subplots()
fig.set_figwidth(9)
fig.set_figheight(6)
ax.plot(inputvr/1000, profile1, label="r = 2.5 au")
ax.plot(inputvr/1000, profile1p2, label="r = 17 au")
ax.plot(inputvr/1000, profile1p3, label="r = 50 au")
ax.plot(inputvr/1000, profile1p4, label="r = 70 au")
ax.legend(fontsize=fsize)
plt.xticks(fontsize=fsize)
plt.yticks(fontsize=fsize)
plt.ylim(bottom=0)
plt.grid()
plt.xlim(-250,250)
ax.set_xlabel("Radial Velocity Component $v_r$ (km/s)", fontsize=fsize)
ax.set_ylabel("Value of $\mu (t)$", fontsize=fsize)
plt.title("Downwind Profile", fontsize=fsize)
plt.show()