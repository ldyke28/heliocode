import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import odeint
import scipy
from mpl_toolkits import mplot3d

# Value for 1 au (astronomical unit) in meters
au = 1.496*10**11
# INITIAL CONDITIONS for both position and velocity (in SI units - m and m/s)
x0 = 0.97*au #1000000
y0 = .2*au
z0 = 0

vx0 = -48000
vy0 = -1000#2000000*np.sqrt(35)
vz0 = 0

msolar = 1.98847*10**30 # mass of the sun in kg
G = 6.6743*10**(-11) # value for gravitational constant in SI units

# Location of the sun in [x,y,z] - usually this will be at 0, but this makes it flexible just in case
sunpos = np.array([0,0,0])


def radPressure(t):
    # dummy function to model radiation pressure
    # takes the time as input and returns the radiation pressure function at that time
    return (np.sin(2*np.pi*(t/347000000)+np.pi/3))**2
    #return .2
    #return 0


def dr_dt(x,t):
    # integrating differential equation for gravitational force. x[0:2] = x,y,z and x[3:5] = vx,vy,vz
    # dx0-2 = vx, vy, and vz, dx3-5 = ax, ay, and az
    r = np.sqrt((sunpos[0]-x[0])**2 + (sunpos[1]-x[1])**2 + (sunpos[2]-x[2])**2)
    dx0 = x[3]
    dx1 = x[4]
    dx2 = x[5]
    dx3 = (msolar*G/(r**3))*(sunpos[0]-x[0])*(1-radPressure(t))
    dx4 = (msolar*G/(r**3))*(sunpos[1]-x[1])*(1-radPressure(t))
    dx5 = (msolar*G/(r**3))*(sunpos[2]-x[2])*(1-radPressure(t))
    return [dx0, dx1, dx2, dx3, dx4, dx5]

def dr_dtbase(x,t):
    # integrating differential equation for gravitational force. x[0:2] = x,y,z and x[3:5] = vx,vy,vz
    # dx0-2 = vx, vy, and vz, dx3-5 = ax, ay, and az
    r = np.sqrt((sunpos[0]-x[0])**2 + (sunpos[1]-x[1])**2 + (sunpos[2]-x[2])**2)
    dx0 = x[3]
    dx1 = x[4]
    dx2 = x[5]
    dx3 = (msolar*G/(r**3))*(sunpos[0]-x[0])
    dx4 = (msolar*G/(r**3))*(sunpos[1]-x[1])
    dx5 = (msolar*G/(r**3))*(sunpos[2]-x[2])
    return [dx0, dx1, dx2, dx3, dx4, dx5]

# Series of time values with a specific dt to use as input
# t0-2 meant to be merged to give higher resolution closer to the sun
# tback meant to trace particle trajectory backward in time

#t0 = np.arange(0,.10,.00001)
#t1 = np.arange(.10,.18,.00000001)
#t2 = np.arange(.18,.3,.00001)
t = np.arange(0, 10000000, 100)
#t = np.arange(0,1,.00001)
#tback = np.arange(0,-10000000,-10)
#t = np.hstack((t0,t1,t2))

init = [x0, y0, z0, vx0, vy0, vz0] # Creating an array of initial values to input into odeint operation
# Solving the differential equation at various points in time to get the position/velocity components
# Output is in the format of a timesteps x 6 dimensional array, the 6 columns being x, y, z, vx, vy, vz
new = odeint(dr_dt, init, t)
newbase = odeint(dr_dtbase, init, t)
#newback = odeint(dr_dt, init, tback)

# initializing arrays to store values of desired quantities to be calculated
Etrack = np.zeros(t.size)
Evartrack = np.zeros(t.size)
Ltrack = np.zeros(t.size)
atrack = np.zeros(t.size)
avartrack = np.zeros((t.size,3))
afulltrack = np.zeros((t.size,3))
vtrack = np.zeros(t.size)
rtrack = np.zeros(t.size)
theta = np.zeros(t.size)
L = np.zeros((t.size,3))
radptrack = np.zeros(t.size)
comptrack1 = np.zeros((t.size,3))
comptrack2 = np.zeros((t.size,3))

# Doing the same as above for the backward trace
"""Etrackb = np.zeros(tback.size)
Ltrackb = np.zeros(tback.size)
atrackb = np.zeros(tback.size)
vtrackb = np.zeros(tback.size)
rtrackb = np.zeros(tback.size)
Lb = np.zeros((tback.size,3))"""

functtrackx = np.zeros(t.size)
functtracky = np.zeros(t.size)
functtrackz = np.zeros(t.size)
for i in range(t.size):
    rmag = np.sqrt((sunpos[0]-new[i,0])**2 + (sunpos[1]-new[i,1])**2 + (sunpos[2]-new[i,2])**2) # calculation of radius
    rtrack[i] = rmag
    vmag = np.sqrt(new[i,3]**2 + new[i,4]**2 + new[i,5]**2) # calculating velocity
    vtrack[i] = vmag
    radptrack[i] = radPressure(t[i]) # calculating radiation pressure at each step

    afulltrack[i,:] = radptrack[i]*1/rtrack[i]*(new[i,3:6]) + radptrack[i]*1/rtrack[i]*(-vtrack[i]*new[i,0:3]/rtrack[i])
    comptrack1[i,:] = radptrack[i]*1/rtrack[i]*(new[i,3:6])
    comptrack2[i,:] = radptrack[i]*1/rtrack[i]*(-vtrack[i]*new[i,0:3]/rtrack[i])

    pf = 1 # prefactor for testing
    functtrackx[i] = pf*(radptrack[i])*1/rtrack[i]*(
        new[i,3] 
        - vtrack[i]*new[i,0]/(rtrack[i]))
    functtracky[i] = pf*(radptrack[i])*1/rtrack[i]*(
        new[i,4] 
        - vtrack[i]*new[i,1]/(rtrack[i]))
    functtrackz[i] = pf*(radptrack[i])*1/rtrack[i]*(
        new[i,5] 
        - vtrack[i]*new[i,2]/(rtrack[i]))
avartrack[:,0] = scipy.integrate.cumulative_trapezoid(functtrackx, t[0:i+1],initial=0)
avartrack[:,1] = scipy.integrate.cumulative_trapezoid(functtracky, t[0:i+1],initial=0)
avartrack[:,2] = scipy.integrate.cumulative_trapezoid(functtrackz, t[0:i+1],initial=0)
#afulltrack[:,:] = avartrack[:,:]


# Calculating said quantities at each step
for i in range(t.size):
    rmag = np.sqrt((sunpos[0]-new[i,0])**2 + (sunpos[1]-new[i,1])**2 + (sunpos[2]-new[i,2])**2) # calculation of radius
    rtrack[i] = rmag
    vmag = np.sqrt(new[i,3]**2 + new[i,4]**2 + new[i,5]**2) # calculating velocity
    vtrack[i] = vmag
    radptrack[i] = radPressure(t[i])

    rmagback = np.sqrt((sunpos[0]-new[i-1,0])**2 + (sunpos[1]-new[i-1,1])**2 + (sunpos[2]-new[i-1,2])**2)
    vmagback = np.sqrt(new[i-1,3]**2 + new[i-1,4]**2 + new[i-1,5]**2)

    rmagmid = np.sqrt((sunpos[0]-(new[i,0]+new[i-1,0])/2)**2 + (sunpos[1]-(new[i,1]+new[i-1,1])/2)**2 + (sunpos[2]-(new[i,2]+new[i-1,2])/2)**2)
    vmagmid = np.sqrt(((new[i,3]+new[i-1,3])/2)**2 + ((new[i,4]+new[i-1,4])/2)**2 + ((new[i,5]+new[i-1,5])/2)**2)

    #theta[i] = np.arccos(((new[i,0]-sunpos[0])*new[i,3] + (new[i,1]-sunpos[1])*new[i,4] + (new[i,2]-sunpos[2])*new[i,5]) / (rtrack[i]*vtrack[i])) # angle between radius/velocity vectors
    #tht = np.arctan(np.sqrt(new[i,0]**2 + new[i,1]**2)/new[i,2])
    #phi = np.arctan(new[i,1]/new[i,0])
    #rvec = np.array([np.sin(tht)*np.cos(phi), np.sin(tht)*np.sin(phi), np.cos(tht)]) 
    #rvec = np.array([(new[i,0]-sunpos[0])/rmag, (new[i,1]-sunpos[1])/rmag, (new[i,2]-sunpos[2])/rmag]) # expansion of radial unit vector in Cartesian coordinates
    rvec = (new[i,0:3]-sunpos)/rmag
    rxv = np.cross(new[i,0:3],new[i,3:6]) # cross product r x v
    Lmag = np.sqrt(rxv[0]**2 + rxv[1]**2 + rxv[2]**2) # calculating magnitude of specific angular momentum
    Ltrack[i] = Lmag
    
    vxl = np.cross(new[i,3:6],rxv)  # cross product v x l
    vxlmag = np.sqrt(vxl[0]**2 + vxl[1]**2 + vxl[2]**2)

    #avartrack[i,:] = avartrack[i-1,:] + (t[i]-t[i-1]) * (radPressure(t[i-1])) * ((1/rmagback)*new[i-1,3:6] - vmagback*(new[i-1,0:3]-sunpos)/(rmagback**2))
    #avartrack[i,:] = avartrack[i-1,:] + (t[i]-t[i-1])*(radPressure((t[i]+t[i-1])/2))*1/rmagmid*((new[i,3:6]+new[i-1,3:6])/2 - vmagmid*(new[i,0:3]+new[i-1,0:3])/(2*rmagmid))
    #afull = vxl - G*msolar*rvec # calculation of specific LRL vector
    #afull = vxl + G*msolar*avartrack[i,:]
    afull = vxl - G*msolar*rvec + G*msolar*avartrack[i,:]
    afulltrack[i,:] = afull
    
    atrack[i] = np.sqrt(afull[0]**2 + afull[1]**2 + afull[2]**2) # magnitude of LRL vector
    
    #Etrack[i] = (vtrack[i]**2)/2 - G*msolar/rtrack[i]
    vdotr = rvec[0]*new[i,3] + rvec[1]*new[i,4] + rvec[2]*new[i,5]
    Evartrack[i] = Evartrack[i-1] + (t[i]-t[i-1])*radPressure(t[i])*vdotr/(rmag**2)
    Etrack[i] = (vtrack[i]**2)/2 - G*msolar/rtrack[i] - G*msolar*Evartrack[i] # magnitude of specific energy


# Same process for back tracing 
"""for i in range(tback.size):
    rmag = np.sqrt((sunpos[0]-newback[i,0])**2 + (sunpos[1]-newback[i,1])**2 + (sunpos[2]-newback[i,2])**2)
    rtrackb[i] = rmag
    vmag = np.sqrt(newback[i,3]**2 + newback[i,4]**2 + newback[i,5]**2)
    vtrackb[i] = vmag
    #theta[i] = np.arccos(((newback[i,0]-sunpos[0])*newback[i,3] + (newback[i,1]-sunpos[1])*newback[i,4] + (newback[i,2]-sunpos[2])*newback[i,5]) / (rtrack[i]*vtrack[i]))
    #tht = np.arctan(np.sqrt(newback[i,0]**2 + newback[i,1]**2)/newback[i,2])
    #phi = np.arctan(newback[i,1]/newback[i,0])
    #rvec = np.array([np.sin(tht)*np.cos(phi), np.sin(tht)*np.sin(phi), np.cos(tht)])
    rvec = np.array([newback[i,0]/rmag, newback[i,1]/rmag, newback[i,2]/rmag])
    rxv = np.cross(newback[i,0:3],newback[i,3:6])
    Lmag = np.sqrt(rxv[0]**2 + rxv[1]**2 + rxv[2]**2)
    Ltrackb[i] = Lmag
    vxl = np.cross(newback[i,3:6],rxv)
    vxlmag = np.sqrt(vxl[0]**2 + vxl[1]**2 + vxl[2]**2)
    afull = vxl - G*msolar*rvec
    atrackb[i] = np.sqrt(afull[0]**2 + afull[1]**2 + afull[2]**2)
    Etrackb[i] = (vtrackb[i]**2)/2 - G*msolar/rtrackb[i]"""

#print statements to check initial/final values
print(Etrack[0])
print(Etrack[-1])
print(Ltrack[0])
print(Ltrack[-1])
print(atrack[0])
print(atrack[-1])

#plotting trajectory in 2D
#plt.plot(new[:,0], new[:,1])
#plt.show()

# plotting of calculated values at various timesteps on the same graph with a shared x axis
f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True)
l1, = ax1.plot(t,Etrack, color='r')
#l1b, = ax1.plot(tback,Etrackb, color='r', linestyle='-.')
l2, = ax2.plot(t,Ltrack, color='b')
#l2b, = ax2.plot(tback,Ltrackb, color='b', linestyle='--')
l3, = ax3.plot(t,atrack, color='y')
#l3b, = ax3.plot(tback,atrackb, color='y', linestyle=':')
l4, = ax4.plot(t,vtrack, color='m')
#l4b, = ax4.plot(tback,vtrackb, color='m', linestyle='-.')
l5, = ax5.plot(t,rtrack/au, color='c')
#l5b, = ax5.plot(tback,rtrackb/au, color='c', linestyle='--')
ax1.legend((l1, l2, l3, l4, l5), ('Adjusted Specific Energy', 'Magnitude of Angular Momentum', 'Magnitude of Eccentricity Vector', 'Magnitude of Velocity', 'Radial Distance'), loc='upper left')
f.subplots_adjust(hspace=.0)
f.set_size_inches(13,12)
plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
#plt.show()

fa, (ax1a, ax2a) = plt.subplots(2, sharex=True)
l1a, = ax1a.plot(t,afulltrack[:,0], color='r')
#l1a1, = ax1a.plot(t,comptrack1[:,0], color='b', linestyle='--')
#l1a2, = ax1a.plot(t,comptrack2[:,0], color='y', linestyle='-.')
l2a, = ax2a.plot(t,afulltrack[:,1], color='r')
#l2a1, = ax2a.plot(t,comptrack1[:,1], color='b', linestyle='--')
#l2a2, = ax2a.plot(t,comptrack2[:,1], color='y', linestyle='-.')
#l3a, = ax3a.plot(t,afulltrack[:,2], color='y')
#l3a1, = ax3a.plot(t,comptrack1[:,2], color='y', linestyle='--')
#l3a2, = ax3a.plot(t,comptrack2[:,2], color='y', linestyle='-.')
ax1a.legend((l1a, l2a), ('x component', 'y component'), loc='upper left')
fa.subplots_adjust(hspace=.0)
fa.set_size_inches(13,6)
plt.setp([a.get_xticklabels() for a in fa.axes[:-1]], visible=False)

# plotting the trajectory of the orbit in 3D, as well as the position of the sun as a red dot
"""zer = [0]
fig3d = plt.figure()
ax3d = plt.axes(projection='3d')
ax3d.plot3D(new[:,0], new[:,1], new[:,2], 'midnightblue')
ax3d.plot3D(newbase[:,0], newbase[:,1], newbase[:,2], 'deeppink', linestyle='--')
#ax3d.plot3D(newback[:,0], newback[:,1], newback[:,2], 'plum')
ax3d.scatter3D(zer,zer,zer,c='red')"""
plt.show()