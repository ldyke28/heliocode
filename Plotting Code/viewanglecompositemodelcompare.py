import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
from tqdm import tqdm
import scipy.interpolate
import cartopy.crs as ccrs

#############################################################################################################
# RELEVANT VARIABLES/PARAMETERS
#############################################################################################################

# one year divided by 60 (6 degree shift for IBEX) - 525909.09090833333 s

nH = 0.195 # hydrogen density in num/cm^3
tempH = 7500 # LISM hydrogen temperature in K
mH = 1.6736*10**(-27) # mass of hydrogen in kg
vthn = np.sqrt(2*1.381*10**(-29)*tempH/mH) # thermal velocity of LISM H

thetavar = 293 # angle with respect to the upwind axis of the target point
vsc = 30000 # velocity of spacecraft in m/s
thetarad = thetavar*np.pi/180 # expressing the value of theta in radians
# calculating the shift of the particle velocities into the spacecraft frame
xshiftfactor = -vsc*np.cos(thetarad + np.pi/2)
yshiftfactor = -vsc*np.sin(thetarad + np.pi/2)

# defining the spacecraft bin width in degrees and radians
ibexvaw = 6
ibexvawr = ibexvaw*np.pi/180
ibexvahwr = ibexvawr/2

sampledv = 2 # width of the shells in post-processing in velocity space

def vtoeV(vel):
    # converts velocity in km/s to energy in keV
    return (.5 * 1.6736*10**(-27))/(1.602*10**(-19)) * (vel/1000)**2 / 1000

sampledE = vtoeV(sampledv)

# set of velocity magnitudes to make shells of points
# CHANGE THIS FOR DIFFERENT TARGET POINT FOR BETTER ACCURACY
testvmag = np.arange(25000, 100000, 5000)

# deciding whether the flux should be calculated in the spacecraft frame (True) or the inertial frame (False)
shiftflux = True
# deciding if the view of the Mollweide should be in the spacecraft frame (True) or the inertial frame (False)
shiftorigin = True

esas = np.array([.010, .01944, .03747, .07283]) # value for relevant ESA's high energy boundary in keV

def eVtov(esaenergy):
    # converts energy in eV to velocity in m/s
    return np.sqrt(esaenergy*1.602*10**(-19)/(.5 * 1.6736*10**(-27)))


#############################################################################################################

fname = "modelcomp2010"

file1 = np.loadtxt("C:/Users/lukeb/Documents/Dartmouth/HSResearch/Cluster Runs/2026Data/228deg_ibexshifted2010_lya_analyticbc_datamu_order5_newpi_3500vres_ibexview.txt", delimiter=',')
file2 = np.loadtxt("C:/Users/lukeb/Documents/Dartmouth/HSResearch/Cluster Runs/2026Data/234deg_ibexshifted2010_lya_analyticbc_datamu_order5_newpi_3500vres_ibexview.txt", delimiter=',')
file3 = np.loadtxt("C:/Users/lukeb/Documents/Dartmouth/HSResearch/Cluster Runs/2026Data/240deg_ibexshifted2010_lya_analyticbc_datamu_order5_newpi_3500vres_ibexview.txt", delimiter=',')
file4 = np.loadtxt("C:/Users/lukeb/Documents/Dartmouth/HSResearch/Cluster Runs/2026Data/246deg_ibexshifted2010_lya_analyticbc_datamu_order5_newpi_3500vres_ibexview.txt", delimiter=',')
file5 = np.loadtxt("C:/Users/lukeb/Documents/Dartmouth/HSResearch/Cluster Runs/2026Data/252deg_ibexshifted2010_lya_analyticbc_datamu_order5_newpi_3500vres_ibexview.txt", delimiter=',')
file6 = np.loadtxt("C:/Users/lukeb/Documents/Dartmouth/HSResearch/Cluster Runs/2026Data/258deg_ibexshifted2010_lya_analyticbc_datamu_order5_newpi_3500vres_ibexview.txt", delimiter=',')
file7 = np.loadtxt("C:/Users/lukeb/Documents/Dartmouth/HSResearch/Cluster Runs/2026Data/264deg_ibexshifted2010_lya_analyticbc_datamu_order5_newpi_3500vres_ibexview.txt", delimiter=',')
file8 = np.loadtxt("C:/Users/lukeb/Documents/Dartmouth/HSResearch/Cluster Runs/2026Data/270deg_ibexshifted2010_lya_analyticbc_datamu_order5_newpi_3500vres_ibexview.txt", delimiter=',')
file9 = np.loadtxt("C:/Users/lukeb/Documents/Dartmouth/HSResearch/Cluster Runs/2026Data/276deg_ibexshifted2010_lya_analyticbc_datamu_order5_newpi_3500vres_ibexview.txt", delimiter=',')
file10 = np.loadtxt("C:/Users/lukeb/Documents/Dartmouth/HSResearch/Cluster Runs/2026Data/282deg_ibexshifted2010_lya_analyticbc_datamu_order5_newpi_3500vres_ibexview.txt", delimiter=',')
file11 = np.loadtxt("C:/Users/lukeb/Documents/Dartmouth/HSResearch/Cluster Runs/2026Data/288deg_ibexshifted2010_lya_analyticbc_datamu_order5_newpi_3500vres_ibexview.txt", delimiter=',')
file12 = np.loadtxt("C:/Users/lukeb/Documents/Dartmouth/HSResearch/Cluster Runs/2026Data/294deg_ibexshifted2010_lya_analyticbc_datamu_order5_newpi_3500vres_ibexview.txt", delimiter=',')
file13 = np.loadtxt("C:/Users/lukeb/Documents/Dartmouth/HSResearch/Cluster Runs/2026Data/300deg_ibexshifted2010_lya_analyticbc_datamu_order5_newpi_3500vres_ibexview.txt", delimiter=',')
file14 = np.loadtxt("C:/Users/lukeb/Documents/Dartmouth/HSResearch/Cluster Runs/2026Data/306deg_ibexshifted2010_lya_analyticbc_datamu_order5_newpi_3500vres_ibexview.txt", delimiter=',')
file15 = np.loadtxt("C:/Users/lukeb/Documents/Dartmouth/HSResearch/Cluster Runs/2026Data/312deg_ibexshifted2010_lya_analyticbc_datamu_order5_newpi_3500vres_ibexview.txt", delimiter=',')

file1f = np.loadtxt("C:/Users/lukeb/Documents/Dartmouth/HSResearch/Cluster Runs/2026Data/228deg_ibexshifted2010_lya_Federicodist_datamu_3500vres_ibexview.txt", delimiter=',')
file2f = np.loadtxt("C:/Users/lukeb/Documents/Dartmouth/HSResearch/Cluster Runs/2026Data/234deg_ibexshifted2010_lya_Federicodist_datamu_3500vres_ibexview.txt", delimiter=',')
file3f = np.loadtxt("C:/Users/lukeb/Documents/Dartmouth/HSResearch/Cluster Runs/2026Data/240deg_ibexshifted2010_lya_Federicodist_datamu_3500vres_ibexview.txt", delimiter=',')
file4f = np.loadtxt("C:/Users/lukeb/Documents/Dartmouth/HSResearch/Cluster Runs/2026Data/246deg_ibexshifted2010_lya_Federicodist_datamu_3500vres_ibexview.txt", delimiter=',')
file5f = np.loadtxt("C:/Users/lukeb/Documents/Dartmouth/HSResearch/Cluster Runs/2026Data/252deg_ibexshifted2010_lya_Federicodist_datamu_3500vres_ibexview.txt", delimiter=',')
file6f = np.loadtxt("C:/Users/lukeb/Documents/Dartmouth/HSResearch/Cluster Runs/2026Data/258deg_ibexshifted2010_lya_Federicodist_datamu_3500vres_ibexview.txt", delimiter=',')
file7f = np.loadtxt("C:/Users/lukeb/Documents/Dartmouth/HSResearch/Cluster Runs/2026Data/264deg_ibexshifted2010_lya_Federicodist_datamu_3500vres_ibexview.txt", delimiter=',')
file8f = np.loadtxt("C:/Users/lukeb/Documents/Dartmouth/HSResearch/Cluster Runs/2026Data/270deg_ibexshifted2010_lya_Federicodist_datamu_3500vres_ibexview.txt", delimiter=',')
file9f = np.loadtxt("C:/Users/lukeb/Documents/Dartmouth/HSResearch/Cluster Runs/2026Data/276deg_ibexshifted2010_lya_Federicodist_datamu_3500vres_ibexview.txt", delimiter=',')
file10f = np.loadtxt("C:/Users/lukeb/Documents/Dartmouth/HSResearch/Cluster Runs/2026Data/282deg_ibexshifted2010_lya_Federicodist_datamu_3500vres_ibexview.txt", delimiter=',')
file11f = np.loadtxt("C:/Users/lukeb/Documents/Dartmouth/HSResearch/Cluster Runs/2026Data/288deg_ibexshifted2010_lya_Federicodist_datamu_3500vres_ibexview.txt", delimiter=',')
file12f = np.loadtxt("C:/Users/lukeb/Documents/Dartmouth/HSResearch/Cluster Runs/2026Data/294deg_ibexshifted2010_lya_Federicodist_datamu_3500vres_ibexview.txt", delimiter=',')
file13f = np.loadtxt("C:/Users/lukeb/Documents/Dartmouth/HSResearch/Cluster Runs/2026Data/300deg_ibexshifted2010_lya_Federicodist_datamu_3500vres_ibexview.txt", delimiter=',')
file14f = np.loadtxt("C:/Users/lukeb/Documents/Dartmouth/HSResearch/Cluster Runs/2026Data/306deg_ibexshifted2010_lya_Federicodist_datamu_3500vres_ibexview.txt", delimiter=',')
#file15f = np.loadtxt("C:/Users/lukeb/Documents/Dartmouth/HSResearch/Cluster Runs/2026Data/312deg_ibexshifted2010_lya_Federicodist_datamu_3500vres_ibexview.txt", delimiter=',')



phi = np.array([])
theta = np.array([])
flux = np.array([])
vmag = np.array([])

# unpacking data from both files
for i in range(np.shape(file1)[0]):
    phi = np.append(phi, file1[i,0])
    theta = np.append(theta, file1[i,1])
    flux = np.append(flux, file1[i,2])
    vmag = np.append(vmag, file1[i,3])

for i in range(np.shape(file2)[0]):
    phi = np.append(phi, file2[i,0])
    theta = np.append(theta, file2[i,1])
    flux = np.append(flux, file2[i,2])
    vmag = np.append(vmag, file2[i,3])

for i in range(np.shape(file3)[0]):
    phi = np.append(phi, file3[i,0])
    theta = np.append(theta, file3[i,1])
    flux = np.append(flux, file3[i,2])
    vmag = np.append(vmag, file3[i,3])

for i in range(np.shape(file4)[0]):
    phi = np.append(phi, file4[i,0])
    theta = np.append(theta, file4[i,1])
    flux = np.append(flux, file4[i,2])
    vmag = np.append(vmag, file4[i,3])

for i in range(np.shape(file5)[0]):
    phi = np.append(phi, file5[i,0])
    theta = np.append(theta, file5[i,1])
    flux = np.append(flux, file5[i,2])
    vmag = np.append(vmag, file5[i,3])

for i in range(np.shape(file6)[0]):
    phi = np.append(phi, file6[i,0])
    theta = np.append(theta, file6[i,1])
    flux = np.append(flux, file6[i,2])
    vmag = np.append(vmag, file6[i,3])

for i in range(np.shape(file7)[0]):
    phi = np.append(phi, file7[i,0])
    theta = np.append(theta, file7[i,1])
    flux = np.append(flux, file7[i,2])
    vmag = np.append(vmag, file7[i,3])

for i in range(np.shape(file8)[0]):
    phi = np.append(phi, file8[i,0])
    theta = np.append(theta, file8[i,1])
    flux = np.append(flux, file8[i,2])
    vmag = np.append(vmag, file8[i,3])

for i in range(np.shape(file9)[0]):
    phi = np.append(phi, file9[i,0])
    theta = np.append(theta, file9[i,1])
    flux = np.append(flux, file9[i,2])
    vmag = np.append(vmag, file9[i,3])

for i in range(np.shape(file10)[0]):
    phi = np.append(phi, file10[i,0])
    theta = np.append(theta, file10[i,1])
    flux = np.append(flux, file10[i,2])
    vmag = np.append(vmag, file10[i,3])

for i in range(np.shape(file11)[0]):
    phi = np.append(phi, file11[i,0])
    theta = np.append(theta, file11[i,1])
    flux = np.append(flux, file11[i,2])
    vmag = np.append(vmag, file11[i,3])

for i in range(np.shape(file12)[0]):
    phi = np.append(phi, file12[i,0])
    theta = np.append(theta, file12[i,1])
    flux = np.append(flux, file12[i,2])
    vmag = np.append(vmag, file12[i,3])

for i in range(np.shape(file13)[0]):
    phi = np.append(phi, file13[i,0])
    theta = np.append(theta, file13[i,1])
    flux = np.append(flux, file13[i,2])
    vmag = np.append(vmag, file13[i,3])

for i in range(np.shape(file14)[0]):
    phi = np.append(phi, file14[i,0])
    theta = np.append(theta, file14[i,1])
    flux = np.append(flux, file14[i,2])
    vmag = np.append(vmag, file14[i,3])

for i in range(np.shape(file15)[0]):
    phi = np.append(phi, file15[i,0])
    theta = np.append(theta, file15[i,1])
    flux = np.append(flux, file15[i,2])
    vmag = np.append(vmag, file15[i,3])

#############################################################################################################################################################
# SAME THING FOR THE SECOND FILE
#############################################################################################################################################################

phif = np.array([])
thetaf = np.array([])
fluxf = np.array([])
vmagf = np.array([])

# unpacking data from both files
for i in range(np.shape(file1f)[0]):
    phif = np.append(phif, file1f[i,0])
    thetaf = np.append(thetaf, file1f[i,1])
    fluxf = np.append(fluxf, file1f[i,2])
    vmagf = np.append(vmagf, file1f[i,3])

for i in range(np.shape(file2f)[0]):
    phif = np.append(phif, file2f[i,0])
    thetaf = np.append(thetaf, file2f[i,1])
    fluxf = np.append(fluxf, file2f[i,2])
    vmagf = np.append(vmagf, file2f[i,3])

for i in range(np.shape(file3f)[0]):
    phif = np.append(phif, file3f[i,0])
    thetaf = np.append(thetaf, file3f[i,1])
    fluxf = np.append(fluxf, file3f[i,2])
    vmagf = np.append(vmagf, file3f[i,3])

for i in range(np.shape(file4f)[0]):
    phif = np.append(phif, file4f[i,0])
    thetaf = np.append(thetaf, file4f[i,1])
    fluxf = np.append(fluxf, file4f[i,2])
    vmagf = np.append(vmagf, file4f[i,3])

for i in range(np.shape(file5f)[0]):
    phif = np.append(phif, file5f[i,0])
    thetaf = np.append(thetaf, file5f[i,1])
    fluxf = np.append(fluxf, file5f[i,2])
    vmagf = np.append(vmagf, file5f[i,3])

for i in range(np.shape(file6f)[0]):
    phif = np.append(phif, file6f[i,0])
    thetaf = np.append(thetaf, file6f[i,1])
    fluxf = np.append(fluxf, file6f[i,2])
    vmagf = np.append(vmagf, file6f[i,3])

for i in range(np.shape(file7f)[0]):
    phif = np.append(phif, file7f[i,0])
    thetaf = np.append(thetaf, file7f[i,1])
    fluxf = np.append(fluxf, file7f[i,2])
    vmagf = np.append(vmagf, file7f[i,3])

for i in range(np.shape(file8f)[0]):
    phif = np.append(phif, file8f[i,0])
    thetaf = np.append(thetaf, file8f[i,1])
    fluxf = np.append(fluxf, file8f[i,2])
    vmagf = np.append(vmagf, file8f[i,3])

for i in range(np.shape(file9f)[0]):
    phif = np.append(phif, file9f[i,0])
    thetaf = np.append(thetaf, file9f[i,1])
    fluxf = np.append(fluxf, file9f[i,2])
    vmagf = np.append(vmagf, file9f[i,3])

for i in range(np.shape(file10f)[0]):
    phif = np.append(phif, file10f[i,0])
    thetaf = np.append(thetaf, file10f[i,1])
    fluxf = np.append(fluxf, file10f[i,2])
    vmagf = np.append(vmagf, file10f[i,3])

for i in range(np.shape(file11f)[0]):
    phif = np.append(phif, file11f[i,0])
    thetaf = np.append(thetaf, file11f[i,1])
    fluxf = np.append(fluxf, file11f[i,2])
    vmagf = np.append(vmagf, file11f[i,3])

for i in range(np.shape(file12f)[0]):
    phif = np.append(phif, file12f[i,0])
    thetaf = np.append(thetaf, file12f[i,1])
    fluxf = np.append(fluxf, file12f[i,2])
    vmagf = np.append(vmagf, file12f[i,3])

for i in range(np.shape(file13f)[0]):
    phif = np.append(phif, file13f[i,0])
    thetaf = np.append(thetaf, file13f[i,1])
    fluxf = np.append(fluxf, file13f[i,2])
    vmagf = np.append(vmagf, file13f[i,3])

for i in range(np.shape(file14f)[0]):
    phif = np.append(phif, file14f[i,0])
    thetaf = np.append(thetaf, file14f[i,1])
    fluxf = np.append(fluxf, file14f[i,2])
    vmagf = np.append(vmagf, file14f[i,3])

"""for i in range(np.shape(file15f)[0]):
    phif = np.append(phif, file15f[i,0])
    thetaf = np.append(thetaf, file15f[i,1])
    fluxf = np.append(fluxf, file15f[i,2])
    vmagf = np.append(vmagf, file15f[i,3])"""

phi = -phi
#print(flux)


# defining a grid for the midpoints of the cells
adjphibg = np.array([189.7,197.4,203.5,211.2,218.9,226.9,234.7,242.4,250.0,257.0,264.7,273.1,280.7,288.2,295.7,303.1,310.5,318.1,325.6,332.8])
adjthetabg = np.array([-81.0,-75.0,-69.0,-63.0,-57.0,-51.0,-45.0,-39.0,-33.0,-27.0,-21.0,-15.0,-9.0,-3.0,3.0,9.0,15.0,21.0,27.0,33.0,39.0,45.0,51.0,57.0,63.0,69.0,75.0,81.0,86.6])

adjphibg = adjphibg - 255

adjphibg10 = np.array([157.9,165.4,173.0,180.8,188.7,196.6,204.2,211.5,219.5,227.7,235.5,243.2,250.7,258.4,266.0,273.7,281.3,288.8,296.2,303.6,310.7,317.8,325.4,332.2,339.5])
adjthetabg10 = np.array([-81.0,-75.0,-69.0,-63.0,-57.0,-51.0,-45.0,-39.0,-33.0,-27.0,-21.0,-15.0,-9.0,-3.0,3.0,9.0,15.0,21.0,27.0,33.0,39.0,45.0,51.0,57.0,63.0,69.0,75.0,81.0,86.6])

adjphibg10 = adjphibg10 - 255


# making a grid from the above
Long,Latg = np.meshgrid(adjphibg,adjthetabg)
Long10, Latg10 = np.meshgrid(adjphibg10,adjthetabg10)


# creating grid points in angle space
phibounds = np.linspace(-np.pi, np.pi, int(360/ibexvaw+1))
thetabounds = np.linspace(-np.pi/2, np.pi/2, int(180/ibexvaw+1))

# creating an array to track the PSD value at the center of the cells made by the grid points
psdtracker = np.zeros((phibounds.size-1, thetabounds.size-1))
bincounter = np.zeros((phibounds.size-1, thetabounds.size-1))
psdtrackerf = np.zeros((phibounds.size-1, thetabounds.size-1))
bincounterf = np.zeros((phibounds.size-1, thetabounds.size-1))


# converting each ESA boundary to m/s to compare with velocities
esa1lowv = eVtov(esas[0]*1000)
esa12 = eVtov(esas[1]*1000)
esa23 = eVtov(esas[2]*1000)
esa3upb = eVtov(esas[3]*1000)

geometricfactor = 1.41*10**(-5) # geometric factor in cm^2 sr for IBEX-Lo, after https://iopscience.iop.org/article/10.3847/1538-4365/adcd5c/pdf

#print(esa1lowv)
#print(esa3upb)

vmagskms = vmag*1000
vmagskmsf = vmagf*1000

phichoice = adjphibg
thetachoice = adjthetabg

# arrays to track flux values and bin counts in each ESA range
pftrackeresa1 = np.zeros((phichoice.size, thetachoice.size))
bincounteresa1 = np.zeros((phichoice.size, thetachoice.size))
pftrackeresa2 = np.zeros((phichoice.size, thetachoice.size))
bincounteresa2 = np.zeros((phichoice.size, thetachoice.size))
pftrackeresa3 = np.zeros((phichoice.size, thetachoice.size))
bincounteresa3 = np.zeros((phichoice.size, thetachoice.size))
pftrackeresa12 = np.zeros((phichoice.size, thetachoice.size))
bincounteresa12 = np.zeros((phichoice.size, thetachoice.size))
pftrackeresa22 = np.zeros((phichoice.size, thetachoice.size))
bincounteresa22 = np.zeros((phichoice.size, thetachoice.size))
pftrackeresa32 = np.zeros((phichoice.size, thetachoice.size))
bincounteresa32 = np.zeros((phichoice.size, thetachoice.size))

phi = -phi
phi = 180/np.pi*phi
print(phi)

theta = theta*180/np.pi

phif = -phif
phif = 180/np.pi*phif
print(phif)

thetaf = thetaf*180/np.pi

#adjphibg = -(adjphibg + 105)

#adjphibg = -(adjphibg + 105) - 255

print(adjphibg)
print(theta)
"""for k in tqdm(range(phi.size)):
    checker = False
    for i in range(adjphibg.size-1):
        for j in range(thetachoice.size-1):
            if adjphibg[i] <= phi[k] < adjphibg[i+1] and thetachoice[j] <= theta[k] < thetachoice[j+1]:
                if esa1lowv < vmagskms[k] < esa12:
                    #if flux[k] >= pftrackeresa1[i,j]:
                    #    pftrackeresa1[i,j] = flux[k]
                    #pftrackeresa1[i,j] += flux[k]
                    bincounteresa1[i,j] += 1
                    pftrackeresa1[i,j] += flux[k]*sampledE
                    #bincounteresa1[i,j] += vmagskms[k]
                elif esa12 < vmagskms[k] < esa23:
                    pftrackeresa2[i,j] += flux[k]
                    bincounteresa2[i,j] += 1
                elif esa23 < vmagskms[k] < esa3upb:
                    pftrackeresa3[i,j] += flux[k]
                    bincounteresa3[i,j] += 1
                psdtracker[i,j] += flux[k]
                bincounter[i,j] += 1
                checker = True
        if checker == True:
            break"""


for k in tqdm(range(phi.size)):
    checker = False
    for i in range(phichoice.size-1):
        for j in range(thetachoice.size-1):
            if phichoice[i] <= phi[k] < phichoice[i+1] and thetachoice[j] <= theta[k] < thetachoice[j+1]:
                if esa1lowv < vmagskms[k] < esa12:
                    #if flux[k] >= pftrackeresa1[i,j]:
                    #    pftrackeresa1[i,j] = flux[k]
                    #pftrackeresa1[i,j] += flux[k]
                    bincounteresa1[i,j] += 1
                    pftrackeresa1[i,j] += flux[k]
                    #bincounteresa1[i,j] += vmagskms[k]
                elif esa12 < vmagskms[k] < esa23:
                    pftrackeresa2[i,j] += flux[k]
                    bincounteresa2[i,j] += 1
                elif esa23 < vmagskms[k] < esa3upb:
                    pftrackeresa3[i,j] += flux[k]
                    bincounteresa3[i,j] += 1
                psdtracker[i,j] += flux[k]
                bincounter[i,j] += 1
                checker = True
        if checker == True:
            break


for k in tqdm(range(phif.size)):
    checker = False
    for i in range(phichoice.size-1):
        for j in range(thetachoice.size-1):
            if phichoice[i] <= phif[k] < phichoice[i+1] and thetachoice[j] <= thetaf[k] < thetachoice[j+1]:
                if esa1lowv < vmagskmsf[k] < esa12:
                    #if fluxf[k] >= pftrackeresa12[i,j]:
                    #    pftrackeresa12[i,j] = fluxf[k]
                    #pftrackeresa12[i,j] += fluxf[k]
                    bincounteresa12[i,j] += 1
                    pftrackeresa12[i,j] += fluxf[k]
                    #bincounteresa12[i,j] += vmagskmsf[k]
                elif esa12 < vmagskmsf[k] < esa23:
                    pftrackeresa22[i,j] += fluxf[k]
                    bincounteresa22[i,j] += 1
                elif esa23 < vmagskmsf[k] < esa3upb:
                    pftrackeresa32[i,j] += fluxf[k]
                    bincounteresa32[i,j] += 1
                psdtrackerf[i,j] += fluxf[k]
                bincounterf[i,j] += 1
                checker = True
        if checker == True:
            break

#Long1,Latg1 = np.meshgrid(phichoice,thetachoice)
Long1,Latg1 = np.meshgrid(phichoice,thetachoice)
# normalizing by bin count
pftrackeresa1 = pftrackeresa1/bincounteresa1
pftrackeresa2 = pftrackeresa2/bincounteresa2
pftrackeresa3 = pftrackeresa3/bincounteresa3
psdtracker = psdtracker/bincounter
pftrackeresa12 = pftrackeresa12/bincounteresa12
pftrackeresa22 = pftrackeresa22/bincounteresa22
pftrackeresa32 = pftrackeresa32/bincounteresa32
psdtrackerf = psdtrackerf/bincounterf

print(pftrackeresa1)
print("Next")
print(pftrackeresa12)

# defining a grid for the midpoints of the cells
adjphib = np.linspace(-180, 180, int(360/ibexvaw+1))
adjthetab = np.linspace(-90, 90, int(180/ibexvaw+1))

# making a grid from the above
Lon,Lat = np.meshgrid(adjphib,adjthetab)

#Long,Latg = np.meshgrid(-phichoice,thetachoice)
Long,Latg = np.meshgrid(-phichoice,thetachoice)

pftrackeresa1 = np.transpose(pftrackeresa1) # transposing the PSD value array to work with the grid
pftrackeresa2 = np.transpose(pftrackeresa2)
pftrackeresa3 = np.transpose(pftrackeresa3)
psdtracker = np.transpose(psdtracker)
pftrackeresa12 = np.transpose(pftrackeresa12) # transposing the PSD value array to work with the grid
pftrackeresa22 = np.transpose(pftrackeresa22)
pftrackeresa32 = np.transpose(pftrackeresa32)
psdtrackerf = np.transpose(psdtrackerf)

psddiff = (pftrackeresa12*(esas[1]-esas[0]))-(pftrackeresa1*(esas[1]-esas[0]))
#psddiff = galliflux102-pftrackeresa1
pgdtp = np.transpose(psddiff)

file = open("C:/Users/lukeb/Documents/Dartmouth/HSResearch/Cluster Runs/2026DataTextFiles/" + fname + ".txt", 'w')
for i in range(phichoice.size):
    for j in range(thetachoice.size):
        file.write(str(phichoice[i]) + ',' + str(thetachoice[j]) + ',' + str(pgdtp[i,j]) + '\n')
file.close()



pole_lat = 90-5.3
pole_lon = 0
cent_lon = 180
cent_displ = 0
eclipticzero = -105
rotated_pole2 = ccrs.RotatedPole( pole_latitude = pole_lat, 
                                pole_longitude = pole_lon,
                                central_rotated_longitude = cent_lon)
lonlats = [ [15,0, '-120$^{\circ}$'], [eclipticzero,45, '45$^{\circ}$'], [eclipticzero,90, '90$^{\circ}$'], [eclipticzero,-45, '-45$^{\circ}$'],
            [-15,0, '-90$^{\circ}$'], [eclipticzero,0, '0$^{\circ}$'], [165,0, '90$^{\circ}$'], [eclipticzero,15, '15$^{\circ}$'],
            [eclipticzero,30, '30$^{\circ}$'], [eclipticzero,75, '75$^{\circ}$'], [eclipticzero,60, '60$^{\circ}$'], [eclipticzero,-15, '-15$^{\circ}$'],
            [eclipticzero,-30, '-30$^{\circ}$'], [eclipticzero,-60, '-60$^{\circ}$'], [eclipticzero,-75, '-75$^{\circ}$'], [-75,0, '-30$^{\circ}$'], [-45,0, '-60$^{\circ}$'],
            [45,0, '-150$^{\circ}$'], [75,0, '180$^{\circ}$'], [105,0, '150$^{\circ}$'], [135,0, '120$^{\circ}$'], [-135,0, '30$^{\circ}$'], [-165,0, '60$^{\circ}$']]
fig = plt.figure(figsize=(8,6))
# Create plot figure and axes
ax = plt.axes(projection=ccrs.Mollweide())

# Plot the graticule
#im2 = ax.pcolormesh(Long,Latg,galliflux092, cmap='rainbow', transform=rotated_pole2, alpha=0.5)
#im = ax.pcolormesh(Long,Latg,pftrackeresa1*(esas[1]-esas[0]), cmap='rainbow', transform=rotated_pole2)
im = ax.pcolormesh(Long,Latg,psddiff, cmap='berlin', transform=rotated_pole2, vmin=-10**(5),vmax=10**(5))
ax.gridlines(crs=rotated_pole2, draw_labels=False, 
             xlocs=range(-165,165,30), 
             ylocs=range(-90,90,15)) #draw_labels=True NOT allowed
for ea in lonlats:
    ax.text(ea[0], ea[1], ea[2], fontsize=10, fontweight='ultralight', color="k", transform=rotated_pole2)

ax.set_global()

plt.xlabel("Heliolongitude Angle $\phi$")
plt.ylabel("Heliolatitude Angle θ")
cb = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.04)
cb.set_label('Intensity at Detector (cm$^{-2}$ s$^{-1}$ sr$^{-1}$)')
plt.show()

fig = plt.figure(figsize=(8,6))
# Create plot figure and axes
ax = plt.axes(projection=ccrs.Mollweide())

# Plot the graticule
im = ax.pcolormesh(Long,Latg,pftrackeresa2*(esas[2]-esas[1]), cmap='rainbow', transform=rotated_pole2)
ax.gridlines(crs=rotated_pole2, draw_labels=False, 
             xlocs=range(-165,165,30), 
             ylocs=range(-90,90,15)) #draw_labels=True NOT allowed
for ea in lonlats:
    ax.text(ea[0], ea[1], ea[2], fontsize=10, fontweight='ultralight', color="k", transform=rotated_pole2)

ax.set_global()

plt.xlabel("Heliolongitude Angle $\phi$")
plt.ylabel("Heliolatitude Angle θ")
cb = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.04)
cb.set_label('Intensity at Detector (cm$^{-2}$ s$^{-1}$ sr$^{-1}$)')
plt.show()

fig = plt.figure(figsize=(8,6))
# Create plot figure and axes
ax = plt.axes(projection=ccrs.Mollweide())

# Plot the graticule
im = ax.pcolormesh(Long,Latg,pftrackeresa3*(esas[3]-esas[2]), cmap='rainbow', transform=rotated_pole2)
ax.gridlines(crs=rotated_pole2, draw_labels=False, 
             xlocs=range(-165,165,30), 
             ylocs=range(-90,90,15)) #draw_labels=True NOT allowed
for ea in lonlats:
    ax.text(ea[0], ea[1], ea[2], fontsize=10, fontweight='ultralight', color="k", transform=rotated_pole2)

ax.set_global()

plt.xlabel("Heliolongitude Angle $\phi$")
plt.ylabel("Heliolatitude Angle θ")
cb = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.04)
cb.set_label('Intensity at Detector (cm$^{-2}$ s$^{-1}$ sr$^{-1}$)')
plt.show()