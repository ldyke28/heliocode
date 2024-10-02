import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
from tqdm import tqdm

nH = 0.195 # hydrogen density in num/cm^3
tempH = 7500 # LISM hydrogen temperature in K
mH = 1.6736*10**(-27) # mass of hydrogen in kg
vthn = np.sqrt(2*1.381*10**(-29)*tempH/mH)

theta = 275
vsc = 30000 # velocity of spacecraft in m/s
thetarad = theta*np.pi/180 # expressing the value of theta in radians
# calculating the shift of the particle velocities into the spacecraft frame
xshiftfactor = -vsc*np.cos(thetarad + np.pi/2)
yshiftfactor = -vsc*np.sin(thetarad + np.pi/2)

vxarray = np.arange(-45000, 45000, 2000)
vyarray = np.arange(-45000, 45000, 2000)
vzarray = np.arange(-45000, 45000, 2000)

vxstore = np.array([])
vystore = np.array([])
vzstore = np.array([])
fstore = np.array([])

for i in tqdm(range(vxarray.size)):
    for j in tqdm(range(vyarray.size)):
        for k in range(vzarray.size):
            vxstore = np.append(vxstore, [vxarray[i]])
            vystore = np.append(vystore, [vyarray[j]])
            vzstore = np.append(vzstore, [vzarray[k]])
            fstore = np.append(fstore, nH*(1/(np.sqrt(np.pi)*vthn))**3*np.exp(-((vxarray[i]+26000)**2 + vyarray[j]**2 + vzarray[k]**2)/(10195)**2))

vxshift = vxstore + xshiftfactor
vyshift = vystore + yshiftfactor

vsquaredshift =  vxshift**2 + vyshift**2 

particleflux = (vsquaredshift/1000)/(mH*6.242*10**(16)) * fstore # calculating particle flux at the device (https://link.springer.com/chapter/10.1007/978-3-030-82167-8_3 chapter 3.3)
# converting to cm^-2 s^-1 ster^-1 keV^-1

fig3d = plt.figure()
fig3d.set_figwidth(10)
fig3d.set_figheight(7)
ax3d = plt.axes(projection='3d')
scatterplot = ax3d.scatter3D(vxstore[:]/1000, vystore[:]/1000, vzstore[:]/1000, c=fstore[:], cmap='rainbow', s=.1, norm=matplotlib.colors.LogNorm(vmin=10**(-11)))
cb = fig3d.colorbar(scatterplot)
ax3d.set_xlabel("$v_x$ at Target Point (km/s)")
ax3d.set_ylabel("$v_y$ at Target Point (km/s)")
ax3d.set_zlabel("$v_z$ at Target Point (km/s)")
plt.show()