import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import odeint
import scipy
import scipy.interpolate
from tqdm import tqdm
import h5py
from scipy.signal import butter, lfilter, freqz

inputfilename = "C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Cluster Runs/3ddata/lostpoints_-17pi36_5p5yr_lya_Federicodist_updatedmu_2000vres"

inputfile = np.loadtxt(inputfilename + ".txt", delimiter=',')

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
finalt = 5*oneyear # time to start backtracing
#6.36674976e9 force free for cosexprp
initialt = -50000000000 # time in the past to which the code should backtrace
tstep = 10000 # general time resolution
tstepclose = 1000 # time resolution for close regime
tstepfar = 200000 # time resolution for far regime
phase = 0 # phase for implementing rotation of target point around sun
refdist = 115

# Location of the sun in [x,y,z] - usually this will be at 0, but this makes it flexible just in case
# Second line is location of the point of interest in the same format (which is, generally, where we want IBEX to be)
sunpos = np.array([0,0,0])
theta = 275 # angle with respect to upwind axis of target point
ibexrad = 1 # radial distance of target point from Sun
ibexx = ibexrad*np.cos(theta*np.pi/180)
ibexy = ibexrad*np.sin(theta*np.pi/180)
ibexpos = np.array([ibexx*au, ibexy*au, 0])
# implementation of target point that orbits around the sun
#ibexpos = np.array([np.cos(np.pi*finalt/oneyear + phase)*au, np.sin(np.pi*finalt/oneyear + phase)*au, 0])


# Initial Conditions for orbit starting at point of interest for backtracing
xstart = ibexpos[0]
ystart = ibexpos[1]
zstart = ibexpos[2]

startt = finalt
lastt = initialt
countoffset = 0
tmid = startt - 200000000 # time at which we switch from high resolution to low resolution - a little more than half of a solar cycle
tclose = np.arange(startt, tmid, -tstepclose) # high resolution time array (close regime)
tfar = np.arange(tmid, lastt, -tstepfar) # low resolution time array (far regime)
t = np.concatenate((tclose, tfar))


datafilename1 = 'C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Collaborations/FedericoVDF/VDF3D_HE013Ksw_PRB_Eclip256_R115_001_H_RegAll.h5'
datafilename2 = 'C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Collaborations/FedericoVDF/VDF3D_HE013Ksw_PRB_Eclip256_R115_002_H_RegAll.h5'
datafilename3 = 'C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Collaborations/FedericoVDF/VDF3D_HE013Ksw_PRB_Eclip256_R115_003_H_RegAll.h5'
datafilename4 = 'C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Collaborations/FedericoVDF/VDF3D_HE013Ksw_PRB_Eclip256_R115_004_H_RegAll.h5'
datafilename5 = 'C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Collaborations/FedericoVDF/VDF3D_HE013Ksw_PRB_Eclip256_R115_005_H_RegAll.h5'
datafilename6 = 'C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Collaborations/FedericoVDF/VDF3D_HE013Ksw_PRB_Eclip256_R115_006_H_RegAll.h5'
datafilename7 = 'C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Collaborations/FedericoVDF/VDF3D_HE013Ksw_PRB_Eclip256_R115_007_H_RegAll.h5'
datafilename8 = 'C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Collaborations/FedericoVDF/VDF3D_HE013Ksw_PRB_Eclip256_R115_008_H_RegAll.h5'


with h5py.File(datafilename1, "r") as f:
    # Print all root level object names (aka keys) 
    # these can be group or dataset names 
    print("Keys: %s" % f.keys())
    # get first object name/key; may or may NOT be a group
    a_group_key = list(f.keys())[0]

    # get the object type for a_group_key: usually group or dataset
    print(type(f[a_group_key])) 

    # If a_group_key is a group name, 
    # this gets the object names in the group and returns as a list
    data = list(f[a_group_key])

    # If a_group_key is a dataset name, 
    # this gets the dataset values and returns as a list
    data = list(f[a_group_key])
    # preferred methods to get dataset values:
    ds_obj = f[a_group_key]      # returns as a h5py dataset object
    ds_arr = f[a_group_key][()]  # returns as a numpy array

    print(list(f.keys()))
    dsvdf1 = f['VDF3D'] # returns h5py dataset object
    arrvdf1 = f['VDF3D'][()] # returns np.array of values
    #print(arrvdf[128,128,128])
    xloc = f['vx_grid'][()]
    yloc = f['vy_grid'][()]
    zloc = f['vz_grid'][()]

with h5py.File(datafilename2, "r") as f:
    dsvdf2 = f['VDF3D'] # returns h5py dataset object
    arrvdf2 = f['VDF3D'][()] # returns np.array of values

with h5py.File(datafilename3, "r") as f:
    dsvdf3 = f['VDF3D'] # returns h5py dataset object
    arrvdf3 = f['VDF3D'][()] # returns np.array of values

with h5py.File(datafilename4, "r") as f:
    dsvdf4 = f['VDF3D'] # returns h5py dataset object
    arrvdf4 = f['VDF3D'][()] # returns np.array of values

with h5py.File(datafilename5, "r") as f:
    dsvdf5 = f['VDF3D'] # returns h5py dataset object
    arrvdf5 = f['VDF3D'][()] # returns np.array of values

with h5py.File(datafilename6, "r") as f:
    dsvdf6 = f['VDF3D'] # returns h5py dataset object
    arrvdf6 = f['VDF3D'][()] # returns np.array of values

with h5py.File(datafilename7, "r") as f:
    dsvdf7 = f['VDF3D'] # returns h5py dataset object
    arrvdf7 = f['VDF3D'][()] # returns np.array of values

with h5py.File(datafilename8, "r") as f:
    dsvdf8 = f['VDF3D'] # returns h5py dataset object
    arrvdf8 = f['VDF3D'][()] # returns np.array of values

zgrid, ygrid, xgrid = np.meshgrid(zloc, yloc, xloc, indexing='ij') # order will be z, y, x for this

interp1 = scipy.interpolate.RegularGridInterpolator((zloc, yloc, xloc), arrvdf1)
interp2 = scipy.interpolate.RegularGridInterpolator((zloc, yloc, xloc), arrvdf2)
interp3 = scipy.interpolate.RegularGridInterpolator((zloc, yloc, xloc), arrvdf3)
interp4 = scipy.interpolate.RegularGridInterpolator((zloc, yloc, xloc), arrvdf4)
interp5 = scipy.interpolate.RegularGridInterpolator((zloc, yloc, xloc), arrvdf5)
interp6 = scipy.interpolate.RegularGridInterpolator((zloc, yloc, xloc), arrvdf6)
interp7 = scipy.interpolate.RegularGridInterpolator((zloc, yloc, xloc), arrvdf7)
interp8 = scipy.interpolate.RegularGridInterpolator((zloc, yloc, xloc), arrvdf8)

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
        np.e**(-(longangled - 90)**2/50**2 - (r - 31)**2/15**2)*(1 + \
        scipy.special.erf(alpha*(r/au)/np.sqrt(2)))*(1 - np.e**(-(r/au)/4)))*1/.966
    
    # mean Doppler shift
    mds = 20*np.sin(longangle)*np.cos((latangled-10)*np.pi/180)
    # dispersion (width of the peak)
    disper = -.0006947*(r/au)**2 + .1745*(r/au) + 5.402 + \
        1.2*np.e**(-(longangled - 275)**2/50**2 - ((r/au) - 80)**2/60**2) + \
        3*np.e**(-(longangled - 90)**2/50**2 - ((r/au))**2/5**2) + \
        1*np.e**(-(longangled - 100)**2/50**2 - ((r/au) - 25)**2/200**2) + \
        .35*np.cos(((latangled + 15)*np.pi/180)*2)
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

    # flattening factor for heliolatitude dependence of radiation pressure force
    alya = .8
    #(F_K-F_R+F_bkg)/((r_E/r)**2)
    return scalefactor*(F_K-F_R+F_bkg)/(r_E**2/(r2**2))*(1 - absval)*np.sqrt(alya*np.sin(latangle)**2 + np.cos(latangle)**2)


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
            endvelcoords = [backtraj[kn+1,5]/1000,backtraj[kn+1,4]/1000,backtraj[kn+1,3]/1000]
            if endlongangle >= 0 and endlongangle <= 45:
                anglepct = (endlongangle)/45
                initpsd = interp1(endvelcoords)*(1 - anglepct) + interp2(endvelcoords)*anglepct
            elif endlongangle > 45 and endlongangle <= 90:
                anglepct = (endlongangle - 45)/45
                initpsd = interp2(endvelcoords)*(1 - anglepct) + interp3(endvelcoords)*anglepct
            elif endlongangle > 90 and endlongangle <= 135:
                anglepct = (endlongangle - 90)/45
                initpsd = interp3(endvelcoords)*(1 - anglepct) + interp4(endvelcoords)*anglepct
            elif endlongangle > 135 and endlongangle <= 180:
                anglepct = (endlongangle - 135)/45
                initpsd = interp4(endvelcoords)*(1 - anglepct) + interp5(endvelcoords)*anglepct
            elif endlongangle > 180 and endlongangle <= 225:
                anglepct = (endlongangle - 180)/45
                initpsd = interp5(endvelcoords)*(1 - anglepct) + interp6(endvelcoords)*anglepct
            elif endlongangle > 225 and endlongangle <= 270:
                anglepct = (endlongangle - 225)/45
                initpsd = interp6(endvelcoords)*(1 - anglepct) + interp7(endvelcoords)*anglepct
            elif endlongangle > 270 and endlongangle <= 315:
                anglepct = (endlongangle - 270)/45
                initpsd = interp7(endvelcoords)*(1 - anglepct) + interp8(endvelcoords)*anglepct
            elif endlongangle > 315 and endlongangle <= 360:
                anglepct = (endlongangle - 315)/45
                initpsd = interp8(endvelcoords)*(1 - anglepct) + interp1(endvelcoords)*anglepct

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
            PIrate2 = 10**(-7)*(1 + .7/(np.e + 1/np.e)*(np.cos(omt - np.pi)*np.exp(np.cos(omt - np.pi)) + 1/np.e))
            #PIrate2 = 1.21163*10**(-7) # time average of above
            # calculating the component of the radial unit vector in each direction at each point in time
            nrvecx = (-sunpos[0]+backtraj[0:kn+1,0])/currentrad
            nrvecy = (-sunpos[1]+backtraj[0:kn+1,1])/currentrad
            nrvecz = (-sunpos[2]+backtraj[0:kn+1,2])/currentrad
            # calculating the magnitude of v_r at each point in time
            currentvr = backtraj[0:kn+1,3]*nrvecx[0:kn+1] + backtraj[0:kn+1,4]*nrvecy[0:kn+1] + backtraj[0:kn+1,5]*nrvecz[0:kn+1]
            # calculating maximum and minimum radial velocities throughout the whole trajectory
            vrmax = np.append(vrmax, max(currentvr))
            vrmin = np.append(vrmin, min(currentvr))
            # integrand for the photoionization and charge exchange ionization losses
            btintegrand = PIrate2/currentvr*(r1/currentrad)**2 + cxirate/currentvr*(r1/currentrad)**2

            # calculation of heliographic latitude angle (polar angle)
            #belowxy = backtraj[0:kn+1,2] < 0
            #zmask = 2*(belowxy-.5)
            #latangle = np.pi/2 + zmask*np.arcsin(np.abs(backtraj[0:kn+1,2] - sunpos[2])/currentrad[:])
            
            # calculation of attenuation factor based on heliographic latitude angle
            #btintegrand = btintegrand*(.85*(np.sin(latangle))**2 + (np.cos(latangle))**2)
            # calculation of attenuation factor
            attfact = scipy.integrate.simps(btintegrand, currentrad)
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