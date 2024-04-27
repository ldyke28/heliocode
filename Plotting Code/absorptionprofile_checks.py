import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
from tqdm import tqdm

oneyear = 3.15545454545*10**7
au = 1.496*10**11

def quantity(x, y, z):

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

    print(longangle)
    print(latangled)

    alpha = .07 # alpha for the skew gaussian distribution

    amp = ((.59*(r/au - 12)/np.sqrt((r/au - 12)**2 + 200) + 0.38) + -0.4* \
    np.e**(-(longangled - 90)**2/50**2 - (r/au - 31)**2/15**2)*(1 + \
    scipy.special.erf(alpha*(r/au)/np.sqrt(2)))*(1 - np.e**(-(r/au)/4)))*1/.966

    # mean Doppler shift
    mds = 20*np.sin(longangle)*np.cos((latangled-100)*np.pi/180)
    # dispersion (width of the peak)
    disper = -.0006947*(r/au)**2 + .1745*(r/au) + 5.402 + \
        1.2*np.e**(-(longangled - 275)**2/50**2 - ((r/au) - 80)**2/60**2) + \
        3*np.e**(-(longangled - 90)**2/50**2 - ((r/au))**2/5**2) + \
        1*np.e**(-(longangled - 100)**2/50**2 - ((r/au) - 25)**2/200**2) + \
        .35*np.cos(((latangled - 75)*np.pi/180)*2)

    return amp


raddist = 50*au

longangles = np.arange(0, 2*np.pi, np.pi/20)
latangles = np.arange(-np.pi/2, np.pi/2, np.pi/40)

print(longangles)
#print(latangles)

longanglex = np.zeros(longangles.size)
longangley = np.zeros(longangles.size)
longanglez = np.zeros(longangles.size)
latanglefixed = 0
for i in range(longangles.size):
    longanglex[i] = raddist*np.cos(longangles[i])*np.cos(latanglefixed)
    longangley[i] = raddist*np.sin(longangles[i])*np.cos(latanglefixed)
    longanglez[i] = raddist*np.sin(latanglefixed)

#print(longanglex)
#print(longangley)
#print(longanglez)

profile1 = np.zeros(longangles.size)
for i in range(longangles.size):
    profile1[i] = quantity(longanglex[i], longangley[i], longanglez[i])


fsize = 18
fig, ax = plt.subplots()
fig.set_figwidth(9)
fig.set_figheight(6)
ax.plot(longangles, profile1)
#ax.legend()
plt.xticks(fontsize=fsize)
plt.yticks(fontsize=fsize)
plt.ylim(bottom=0)
plt.grid()
#plt.xlim(-100,100)
#ax.set_xlabel("Radial Velocity Component $v_r$ (km/s)", fontsize=fsize)
#ax.set_ylabel("Value of $\mu (t)$", fontsize=fsize)
#plt.title("Photoionization Rate over Time", fontsize=fsize)
plt.show()

