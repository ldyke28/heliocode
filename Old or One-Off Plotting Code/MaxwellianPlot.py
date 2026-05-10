import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import odeint
import scipy
import scipy.interpolate
from tqdm import tqdm
from scipy.signal import butter, lfilter, freqz

vxs = np.arange(-65000, 5000, 100)
vys = np.arange(-35000, 35000, 100)


def Maxwellian(vx, vy):
    return np.exp(-((vx/1000+26)**2 + (vy/1000)**2)/(10.195)**2)

vxv2 = np.zeros((vxs.size*vys.size))
vyv2 = np.zeros((vxs.size*vys.size))
npsds = np.zeros((vxs.size*vys.size))
for i in range(vxs.size):
    for j in range(vys.size):
        if Maxwellian(vxs[i], vys[j]) >= 10**(-3):
            vxv2[i*vys.size + j] = vxs[i]
            vyv2[i*vys.size + j] = vys[j]
            npsds[i*vys.size + j] = Maxwellian(vxs[i], vys[j])

f = plt.figure()
f.set_figwidth(10)
f.set_figheight(8)
fsize = 18
plt.scatter(vxv2/1000, vyv2/1000, c=npsds[:], marker='o', cmap='rainbow') # linear scale
plt.rcParams.update({'font.size': fsize})
cb = plt.colorbar()
cb.set_label('Normalized Phase Space Density')
plt.xticks(fontsize=fsize)
plt.yticks(fontsize=fsize)
plt.xlabel("$v_x$ (km/s)", fontsize=fsize)
plt.ylabel("$v_y$ (km/s)", fontsize=fsize)
plt.show()