import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import odeint
import scipy
from mpl_toolkits import mplot3d

au = 1.496*10**11
msolar = 1.98847*10**30 # mass of the sun in kg
G = 6.6743*10**(-11) # value for gravitational constant in SI units

mu = msolar*G

def stumpff(x):
    # definition of the Stumpff functions of orders 0 to 3 in all regimes such that errors don't arise
    if x<-1e-6:
        c0 = np.cosh(np.sqrt(-x))
        c1 = np.sinh(np.sqrt(-x))/np.sqrt(-x)   
    elif x>1e-6:
        c0 = np.cos(np.sqrt(x))
        c1 = np.sin(np.sqrt(x))/np.sqrt(x)
    elif abs(x)<=1e-6:
        c0 = 1.
        c1 = 1.
    
    if x > 1e-6:
        c2 = (1 - np.cos(np.sqrt(x))) / x
        c3 = (np.sqrt(x) - np.sin(np.sqrt(x))) / np.sqrt(x ** 3)
    elif x < -1e-6:
        c2 = (1 - np.cosh(np.sqrt(-x))) / x
        c3 = (np.sinh(np.sqrt(-x)) - np.sqrt(-x)) / np.sqrt(-x**3)
    elif abs(x) <= 1e-6:
        c2 = .5
        c3 = 1. / 6.

    return [c0,c1,c2,c3]


# a = semimajor axis of the orbit
# set of f and g functions as defined on wikipedia, all meant to take scalar quantities
def f(s, r0, a):
    return 1 - (mu/r0)*s**2*stumpff(mu/a*s**2)[2]

def g(s, r0, a, t, t0):
    return t - t0 - mu*s**3*stumpff(mu/a*s**2)[3]

def dfdt(s, r0, a, r):
    return -(mu/(r*r0))*s*stumpff(mu/a*s**2)[1]

def dgdt(s, r0, a, r):
    return 1 - mu/r * s**2*stumpff(mu/a*s**2)[2]


def posNew(r0, v0, s, t, t0, a):
    # function to update the position of the particle based on the relevant quantities and the initial position/velocity vectors
    r0mag = np.sqrt(r0[0]**2 + r0[1]**2 + r0[2]**2)
    return f(s, r0mag, a)*r0 + g(s, r0mag, a, t, t0)*v0

def updaterv(r0, v0, s, t, t0, a):
    # function that returns the updated position and velocity
    r = posNew(r0, v0, s, t, t0, a)
    r0mag = np.sqrt(r0[0]**2 + r0[1]**2 + r0[2]**2)
    rmagnew = np.sqrt(r[0]**2 + r[1]**2 + r[2]**2)
    return r, r0*dfdt(s, r0mag, a, rmagnew) + v0*dgdt(s, r0mag, a, rmagnew)


# Root finding
def sfind(s, t, t0, r0, v0, a):
    # function used to calculate the root for s for a given value of t
    cs = stumpff(mu/a*s**2)
    return r0*s*cs[1] + r0*v0*s**2*cs[2] + mu*s**3*cs[3] - (t-t0)

def findroot(t, t0, r0, v0, a):
    # function that uses above function to actually calculate and return the root
    root = scipy.optimize.root(sfind, -1, args=(t,t0,r0,v0,a))
    return root.x[0]

t = np.arange(0, 6000000000, 10000)
getr = np.zeros((t.size,3))
getv = np.zeros((t.size,3))
t0 = 0.
r0vec = np.array([1000*au, .2913*au, 0.])
r0 = np.sqrt(r0vec[0]**2 + r0vec[1]**2 + r0vec[2]**2)
v0vec = np.array([-26000, 0, 0.])
v0 = np.sqrt(v0vec[0]**2 + v0vec[1]**2 + v0vec[2]**2)
#a = 1/(2/r0 - v0**2/mu) # assuming an elliptical orbit
a = -mu/(v0**2)
#a = .00000001*au

for i in range(t.size):
    rootcalc = findroot(t[i], t0, r0, v0, a)
    getr[i,:], getv[i,:] = updaterv(r0vec, v0vec, rootcalc, t[i], t0, a)


fig3d = plt.figure()
ax3d = plt.axes(projection='3d')
ax3d.scatter3D(getr[:,0], getr[:,1], getr[:,2], 'chartreuse')
plt.show()