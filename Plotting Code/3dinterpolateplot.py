import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm


# Loading in the file to be unpacked
#file = np.loadtxt("/Users/ldyke/Desktop/Dartmouth/HSResearch/Code/Kepler/Python Orbit Code/datafiles/pi_t0.txt", delimiter=',')
file = np.loadtxt("C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Cluster Runs/3ddata/2pi3_t0_lya_Federicodist_updatedmu.txt", delimiter=',')
#file2 = np.loadtxt("C:/Users/lucas/OneDrive/Documents/GitHub/heliocode/supplementaldata1.txt", delimiter=',')

#file = file[np.any(file > 1, axis=1)]

# Unpacking variables based on how they're saved in the code
vxinit = file[:,0]
vyinit = file[:,1]
vzinit = file[:,2]
finit = file[:,3]

for i in range(finit.size):
    if finit[i] == -1:
        # points in Sun are set to -1 - this will not work for integration, so we revert them to 0 here
        finit[i] = 0

newvz = np.array([])
for i in range(vzinit.size):
    if vzinit[i] not in newvz:
        newvz = np.append(newvz, [vzinit[i]])
print(newvz)
vzshape = newvz.size

newvy = np.array([])
for i in range(vyinit.size):
    if vyinit[i] not in newvy:
        newvy = np.append(newvy, [vyinit[i]])
print(newvy)
vyshape = newvy.size

newvx = np.array([])
for i in range(vxinit.size):
    if vxinit[i] not in newvx:
        newvx = np.append(newvx, [vxinit[i]])
print(newvx)
vxshape = newvx.size

