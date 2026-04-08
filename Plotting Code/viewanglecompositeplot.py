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

fname = "gallicompanalytic2010"

"""file1 = np.loadtxt("C:/Users/lukeb/Documents/Dartmouth/HSResearch/Cluster Runs/2026Data/228deg_ibexshifted2010_lya_analyticbc_datamu_order5_newpi_3500vres_ibexview.txt", delimiter=',')
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
file15 = np.loadtxt("C:/Users/lukeb/Documents/Dartmouth/HSResearch/Cluster Runs/2026Data/312deg_ibexshifted2010_lya_analyticbc_datamu_order5_newpi_3500vres_ibexview.txt", delimiter=',')"""

file1 = np.loadtxt("C:/Users/lukeb/Documents/Dartmouth/HSResearch/Cluster Runs/2026Data/228deg_ibexshifted_1monthshift_lya_Federicodist_datamu_nofilter_3500vres_ibexview.txt", delimiter=',')
file2 = np.loadtxt("C:/Users/lukeb/Documents/Dartmouth/HSResearch/Cluster Runs/2026Data/234deg_ibexshifted_1monthshift_lya_Federicodist_datamu_nofilter_3500vres_ibexview.txt", delimiter=',')
file3 = np.loadtxt("C:/Users/lukeb/Documents/Dartmouth/HSResearch/Cluster Runs/2026Data/240deg_ibexshifted_1monthshift_lya_Federicodist_datamu_nofilter_3500vres_ibexview.txt", delimiter=',')
file4 = np.loadtxt("C:/Users/lukeb/Documents/Dartmouth/HSResearch/Cluster Runs/2026Data/246deg_ibexshifted_1monthshift_lya_Federicodist_datamu_nofilter_3500vres_ibexview.txt", delimiter=',')
file5 = np.loadtxt("C:/Users/lukeb/Documents/Dartmouth/HSResearch/Cluster Runs/2026Data/252deg_ibexshifted_1monthshift_lya_Federicodist_datamu_nofilter_3500vres_ibexview.txt", delimiter=',')
file6 = np.loadtxt("C:/Users/lukeb/Documents/Dartmouth/HSResearch/Cluster Runs/2026Data/258deg_ibexshifted_1monthshift_lya_Federicodist_datamu_nofilter_3500vres_ibexview.txt", delimiter=',')
file7 = np.loadtxt("C:/Users/lukeb/Documents/Dartmouth/HSResearch/Cluster Runs/2026Data/264deg_ibexshifted_1monthshift_lya_Federicodist_datamu_nofilter_3500vres_ibexview.txt", delimiter=',')
file8 = np.loadtxt("C:/Users/lukeb/Documents/Dartmouth/HSResearch/Cluster Runs/2026Data/271deg_ibexshifted_1monthshift_lya_Federicodist_datamu_nofilter_3500vres_ibexview.txt", delimiter=',')
file9 = np.loadtxt("C:/Users/lukeb/Documents/Dartmouth/HSResearch/Cluster Runs/2026Data/276deg_ibexshifted_1monthshift_lya_Federicodist_datamu_nofilter_3500vres_ibexview.txt", delimiter=',')
file10 = np.loadtxt("C:/Users/lukeb/Documents/Dartmouth/HSResearch/Cluster Runs/2026Data/282deg_ibexshifted_1monthshift_lya_Federicodist_datamu_nofilter_3500vres_ibexview.txt", delimiter=',')
file11 = np.loadtxt("C:/Users/lukeb/Documents/Dartmouth/HSResearch/Cluster Runs/2026Data/288deg_ibexshifted_1monthshift_lya_Federicodist_datamu_nofilter_3500vres_ibexview.txt", delimiter=',')
file12 = np.loadtxt("C:/Users/lukeb/Documents/Dartmouth/HSResearch/Cluster Runs/2026Data/294deg_ibexshifted_1monthshift_lya_Federicodist_datamu_nofilter_3500vres_ibexview.txt", delimiter=',')
file13 = np.loadtxt("C:/Users/lukeb/Documents/Dartmouth/HSResearch/Cluster Runs/2026Data/300deg_ibexshifted_1monthshift_lya_Federicodist_datamu_nofilter_3500vres_ibexview.txt", delimiter=',')
file14 = np.loadtxt("C:/Users/lukeb/Documents/Dartmouth/HSResearch/Cluster Runs/2026Data/306deg_ibexshifted_1monthshift_lya_Federicodist_datamu_nofilter_3500vres_ibexview.txt", delimiter=',')
file15 = np.loadtxt("C:/Users/lukeb/Documents/Dartmouth/HSResearch/Cluster Runs/2026Data/312deg_ibexshifted_1monthshift_lya_Federicodist_datamu_nofilter_3500vres_ibexview.txt", delimiter=',')



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

phi = -phi
#print(flux)


# defining a grid for the midpoints of the cells
adjphibg = np.array([189.7,197.4,203.5,211.2,218.9,226.9,234.7,242.4,250.0,257.0,264.7,273.1,280.7,288.2,295.7,303.1,310.5,318.1,325.6,332.8])
adjthetabg = np.array([-81.0,-75.0,-69.0,-63.0,-57.0,-51.0,-45.0,-39.0,-33.0,-27.0,-21.0,-15.0,-9.0,-3.0,3.0,9.0,15.0,21.0,27.0,33.0,39.0,45.0,51.0,57.0,63.0,69.0,75.0,81.0,86.6])

adjphibg = adjphibg - 255

galliflux09 = np.array([np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan, \
np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan, \
2353.660337,591.66826,1173.268337,0.0,0.0,0.0,0.0,0.0,0.0,1080.830073,376.942274,0.0,0.0,505.872921,0.0,0.0,0.0,0.0,0.0, \
0.0,0.0,654.071327,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,82.599524,0.0,2649.084747, \
0.0,1730.656702,1326.667454,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1774.365617,0.0,0.0,0.0,0.0,0.0, \
0.0,0.0,0.0,0.0,0.0,6698.25913,0.0,205.335207,0.0,0.0,0.0,0.0,0.0,498.379598,0.0,0.0,0.0,0.0,0.0, \
147.97627,0.0,0.0,2799.469193,0.0,6732.08144,0.0,0.0,0.0,1332.624014,1164.587739,0.0,68.035349,0.0,0.0,206.498811,846.645125,0.0,0.0, \
0.0,0.0,353.168358,1669.451494,0.0,6082.634887,0.0,130.059024,0.0,2124.534064,0.0,656.396088,0.0,1450.308088,0.0,0.0,0.0,0.0,0.0, \
3720.729758,0.0,2131.525032,962.18843,0.0,3175.693697,0.0,687.885444,3310.012021,3803.664396,0.0,0.0,199.479805,26.760933,765.468527,0.0,0.0,0.0,1049.887477, \
0.0,0.0,4014.772528,2553.284492,1498.390121,5325.669497,7117.020395,5717.196801,3326.885412,4291.189615,6554.226643,3972.757233,5333.735419,0.0,300.621353,1649.213095,0.0,0.0,0.0, \
0.0,13188.018859,3664.969572,0.0,0.0,11226.132656,18170.645306,18003.806597,15547.433996,18954.09011,16294.407293,11409.679037,7991.447948,3272.956989,4204.385628,0.0,1588.326714,1675.947323,82.175675, \
0.0,0.0,0.0,0.0,16475.810327,511.92518,35479.996876,48116.39413,44227.859321,45729.914714,31200.967016,26838.475311,18197.203538,6141.435115,3283.870669,2484.126508,3500.154847,1323.59582,435.049574, \
0.0,0.0,0.0,19518.48095,0.0,0.0,33224.142014,52453.938267,83327.004119,82861.393329,64907.448227,45234.556179,31796.006976,12281.252577,6435.806106,1260.136942,81.493048,290.606147,0.0, \
2749.524555,0.0,0.0,0.0,0.0,0.0,34253.656409,86599.038238,117312.631717,116973.926251,96001.103617,66250.082473,41296.72422,25913.442824,8666.738306,4013.674102,3698.703689,0.0,0.0, \
6813.273192,12648.759884,0.0,0.0,0.0,0.0,0.0,52533.293435,119203.858291,123773.516275,93509.895344,67693.564101,44465.303457,21873.271353,9800.324494,4848.707909,1907.077713,617.030402,0.0, \
0.0,0.0,0.0,0.0,0.0,0.0,0.0,37358.798123,88314.053706,102042.24668,69438.985767,46569.259017,30660.665387,17942.449788,8884.738288,4943.077391,2485.003276,0.0,240.514828, \
0.0,0.0,0.0,0.0,10418.833216,0.0,25382.563994,34756.276479,50341.452253,59102.070814,36809.349687,29010.216297,16363.418871,10750.258848,4568.976725,3309.052203,0.0,866.299975,2554.685291, \
3123.888865,0.0,0.0,0.0,35230.397601,2098.799396,7434.856799,17489.168709,19657.084284,28387.986286,11239.225638,12391.190932,7510.369463,4368.625296,3534.079652,1608.404534,1481.172453,1612.320443,3399.629041, \
0.0,4227.388332,0.0,1449.474597,7826.165473,5460.370887,5235.005905,5626.254889,7113.849542,15900.721443,4169.695063,4101.726823,4743.392714,921.914166,0.0,392.347741,0.0,0.0,0.0, \
0.0,0.0,0.0,2343.635502,9936.595956,0.0,510.853359,233.17614,379.618672,6681.61788,1583.106313,0.0,1707.747699,54.525519,417.771948,0.0,779.198618,1354.104629,0.0, \
1083.419432,0.0,0.0,1099.568151,0.0,0.0,0.0,1161.276986,2966.142366,0.0,0.0,288.442784,1655.671534,0.0,0.0,0.0,0.0,0.0,0.0, \
0.0,104.921063,0.0,1000.354394,0.0,0.0,0.0,982.655731,0.0,1352.731835,0.0,941.126526,0.0,0.0,0.0,0.0,0.0,0.0,0.0, \
5958.539581,0.0,0.0,0.0,0.0,0.0,0.0,230.410062,727.105258,0.0,723.565278,0.0,0.0,645.997114,0.0,0.0,0.0,960.219471,0.0, \
0.0,757.162307,0.0,1958.460946,0.0,0.0,0.0,0.0,0.0,0.0,0.0,288.442784,0.0,1071.827162,0.0,0.0,0.0,0.0,0.0, \
0.0,0.0,0.0,0.0,0.0,0.0,300.341049,0.0,1668.14984,1507.102619,0.0,100.381366,0.0,0.0,0.0,0.0,0.0,0.0,0.0, \
np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan, \
np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan, \
np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan])

adjphibg10 = np.array([157.9,165.4,173.0,180.8,188.7,196.6,204.2,211.5,219.5,227.7,235.5,243.2,250.7,258.4,266.0,273.7,281.3,288.8,296.2,303.6,310.7,317.8,325.4,332.2,339.5])
adjthetabg10 = np.array([-81.0,-75.0,-69.0,-63.0,-57.0,-51.0,-45.0,-39.0,-33.0,-27.0,-21.0,-15.0,-9.0,-3.0,3.0,9.0,15.0,21.0,27.0,33.0,39.0,45.0,51.0,57.0,63.0,69.0,75.0,81.0,86.6])

adjphibg10 = adjphibg10 - 255

galliflux10 = np.array([np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan, \
np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan, \
0.0,0.0,0.0,0.0,0.0,0.0,np.nan,0.0,0.0,444.677162,0.0,0.0,0.0,0.0,1406.502735,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0, \
0.0,0.0,0.0,0.0,0.0,0.0,np.nan,0.0,10.641791,0.0,0.0,0.0,0.0,0.0,0.0,0.0,830.365589,0.0,0.0,892.096715,0.0,0.0,44.249745,0.0, \
1515.434234,0.0,0.0,0.0,0.0,0.0,np.nan,186.832258,212.715627,0.0,308.672019,0.0,0.0,0.0,0.0,0.0,280.193757,666.335608,398.739371,1570.592809,0.0,0.0,0.0,0.0, \
307.499814,0.0,0.0,0.0,0.0,1201.323603,np.nan,0.0,0.0,0.0,1142.042221,0.0,0.0,0.0,0.0,882.066773,420.31795,346.790319,780.762171,0.0,0.0,0.0,0.0,8687.699979, \
0.0,0.0,0.0,0.0,0.0,0.0,np.nan,0.0,0.0,0.0,0.0,0.0,1346.219286,0.0,2459.646671,0.0,0.0,0.0,447.414091,545.90393,0.0,0.0,0.0,504.445089, \
0.0,510.546198,3516.633318,0.0,0.0,0.0,np.nan,1303.798471,526.723626,1019.127079,0.0,0.0,0.0,0.0,0.0,0.0,1329.9328,1784.379171,1489.663735,2552.494655,0.0,0.0,0.0,1371.742102, \
0.0,0.0,2468.133054,0.0,0.0,0.0,np.nan,1257.941562,327.666537,1923.685442,0.0,14.807982,0.0,0.0,0.0,0.0,0.0,0.0,954.811169,0.0,0.0,0.0,0.0,477.93962, \
0.0,175.160366,0.0,837.819286,0.0,0.0,np.nan,2654.576285,0.0,1876.591217,0.0,3322.608317,3757.578085,2467.762684,2393.466841,2489.792282,2664.147284,509.419725,245.340254,2866.667846,0.0,0.0,0.0,5516.468238, \
2060.111131,4378.176546,0.0,1499.85276,0.0,0.0,np.nan,3345.32764,321.741422,3016.960015,6536.876502,7644.06584,11064.502487,13324.094013,8644.897066,6161.258348,4078.02804,4768.28699,3420.99697,565.224537,1198.806311,0.0,0.0,1169.138267, \
273.220734,0.0,0.0,0.0,0.0,667.912742,np.nan,26532.628402,3258.282048,20750.244384,21422.982496,28395.546037,34742.055543,36033.964857,29140.40081,14899.13552,11444.756774,5970.079344,2236.773937,3343.912933,2317.211658,728.495687,0.0,2802.283533, \
0.0,0.0,0.0,0.0,0.0,10135.645745,np.nan,37373.174424,30658.304961,42400.587003,53078.325113,72896.120926,68371.18255,76759.39751,59424.887359,45845.974253,20703.476935,12284.691745,3129.774343,4147.780613,591.584419,0.0,0.0,0.0, \
7461.868398,247.593786,0.0,0.0,0.0,17330.786003,np.nan,67595.544069,87771.702039,29127.879484,71059.525681,105322.767763,110267.84005,103834.765212,94935.058648,53096.769186,31270.733131,15387.572132,10755.102057,2879.324525,1897.161168,180.034943,1482.711094,0.0, \
2347.420321,0.0,324.344291,705.816137,0.0,10692.865699,np.nan,12544.869677,68827.250739,35201.867543,68887.806947,85106.759616,112186.560229,120596.330554,88445.25744,62876.493002,33728.645588,21574.858599,3088.04766,4191.21022,1376.098594,793.510248,666.059512,0.0, \
0.0,0.0,0.0,0.0,0.0,4786.815334,np.nan,0.0,42321.478658,0.0,22083.827776,88278.749272,90534.070484,93289.57069,91590.858476,43485.251799,28057.645907,17293.391738,7319.347656,4879.376647,1318.416476,0.0,632.978713,0.0, \
14929.765641,3698.783322,3976.599192,0.0,0.0,0.0,np.nan,0.0,0.0,0.0,24024.999136,30453.1937,47916.843495,43730.699959,36237.689646,29707.825803,17445.411373,9769.241485,0.0,5249.405239,865.835836,540.996753,0.0,0.0, \
0.0,0.0,0.0,887.422169,4675.38197,12725.252538,np.nan,0.0,3212.880594,7690.178615,17921.530757,14589.208551,20478.882408,18226.149747,14479.838995,15773.84016,7337.844426,3547.21101,2639.426226,1290.344422,618.120059,0.0,1530.564647,0.0, \
4837.842671,0.0,1053.991774,4761.341654,970.900573,2702.574078,np.nan,0.0,6666.148993,4926.302851,0.0,2881.32987,3498.894263,5534.745948,4386.937872,1105.873692,1655.267424,1332.438475,0.0,2454.065497,0.0,2477.903785,0.0,1327.492357, \
22014.621815,0.0,0.0,0.0,1772.606765,247.98313,np.nan,0.0,2448.897039,0.0,0.0,0.0,2113.21487,1842.093161,0.0,523.348839,0.0,0.0,0.0,2194.071719,0.0,0.0,0.0,0.0, \
5912.539137,0.0,0.0,0.0,0.0,1175.097423,np.nan,0.0,159.854213,0.0,714.476173,1167.251578,160.543033,503.558595,736.693385,0.0,280.193757,494.941595,965.13611,0.0,0.0,1466.141558,0.0,1001.519234, \
0.0,0.0,0.0,7471.904333,0.0,0.0,np.nan,0.0,599.163402,1047.948688,0.0,0.0,0.0,0.0,0.0,0.0,0.0,14.389362,75.754855,0.0,0.0,379.080648,0.0,2728.734289, \
15037.225386,539.062076,2417.814117,0.0,0.0,0.0,np.nan,0.0,1051.985795,0.0,529.920745,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0, \
0.0,20201.161373,0.0,5495.119102,0.0,0.0,np.nan,0.0,0.0,0.0,0.0,0.0,0.0,0.0,570.18255,0.0,1494.111768,0.0,0.0,0.0,0.0,557.54679,0.0,0.0, \
0.0,0.0,0.0,4642.57401,0.0,0.0,np.nan,0.0,0.0,0.0,0.0,1232.311701,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,784.695482,0.0,0.0, \
np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan, \
np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan, \
np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan])

galliflux092 = np.zeros((adjphibg.size, adjthetabg.size))
galliflux102 = np.zeros((adjphibg10.size, adjthetabg10.size))

# finds which cell each velocity point lies in
for i in tqdm(range(adjphibg.size-1)):
    for j in range(adjthetabg.size-1):
        galliflux092[i,j] = galliflux09[j*(adjphibg.size-1) + i]

for i in tqdm(range(adjphibg10.size-1)):
    for j in range(adjthetabg10.size-1):
        galliflux102[i,j] = galliflux10[j*(adjphibg10.size-1) + i]

galliflux092 = np.transpose(galliflux092)
galliflux102 = np.transpose(galliflux102)

# making a grid from the above
Long,Latg = np.meshgrid(adjphibg,adjthetabg)
Long10, Latg10 = np.meshgrid(adjphibg10,adjthetabg10)


# creating grid points in angle space
phibounds = np.linspace(-np.pi, np.pi, int(360/ibexvaw+1))
thetabounds = np.linspace(-np.pi/2, np.pi/2, int(180/ibexvaw+1))

# creating an array to track the PSD value at the center of the cells made by the grid points
psdtracker = np.zeros((phibounds.size-1, thetabounds.size-1))
bincounter = np.zeros((phibounds.size-1, thetabounds.size-1))



# finds which cell each velocity point lies in
"""for k in tqdm(range(phi.size)):
    checker = False
    for i in range(phibounds.size-1):
        for j in range(thetabounds.size-1):
            if phibounds[i] <= phi[k] < phibounds[i+1] and thetabounds[j] <= theta[k] < thetabounds[j+1]:
                # adding the value of the PSD to the associated value for the cell and exiting the loop
                #psdtracker[i,j] += particleflux[k]
                psdtracker[i,j] += flux[k]
                bincounter[i,j] += 1
                checker = True
        if checker == True:
            break

# dividing the total summed PSD in each bin by the number of points in that bin
psdtracker = psdtracker/bincounter"""
"""fig = plt.figure()
# plotting the cells/their values on an all-sky map using a mollweide projection
ax = fig.add_subplot(111, projection='mollweide')
arr = np.random.rand(180, 360)

# defining a grid for the midpoints of the cells
adjphib = np.linspace(-np.pi+ibexvawr, np.pi-ibexvawr, int(360/ibexvaw))
adjthetab = np.linspace(-np.pi/2+ibexvawr, np.pi/2-ibexvawr, int(180/ibexvaw))

# making a grid from the above
Lon,Lat = np.meshgrid(adjphib,adjthetab)

psdtracker = np.transpose(psdtracker) # transposing the PSD value array to work with the grid

im = ax.pcolormesh(Lon,Lat,psdtracker, cmap='rainbow')#, norm=matplotlib.colors.LogNorm(vmin=10**(1),vmax=10**(6)))
ax.plot()
plt.grid()
#im = ax.scatter(phi,theta,c=fstore, cmap='rainbow')
#plt.scatter(phi, theta, c=fstore, cmap='rainbow', s=.01)
plt.xlabel("Heliolongitude Angle $\phi$")
plt.ylabel("Heliolatitude Angle θ")
cb = fig.colorbar(im, ax=ax)
cb.set_label('Differential Flux at Detector')
plt.show()"""


# converting each ESA boundary to m/s to compare with velocities
esa1lowv = eVtov(esas[0]*1000)
esa12 = eVtov(esas[1]*1000)
esa23 = eVtov(esas[2]*1000)
esa3upb = eVtov(esas[3]*1000)

geometricfactor = 1.41*10**(-5) # geometric factor in cm^2 sr for IBEX-Lo, after https://iopscience.iop.org/article/10.3847/1538-4365/adcd5c/pdf

#print(esa1lowv)
#print(esa3upb)

vmagskms = vmag*1000

"""# arrays to track flux values and bin counts in each ESA range
pftrackeresa1 = np.zeros((phibounds.size-1, thetabounds.size-1))
bincounteresa1 = np.zeros((phibounds.size-1, thetabounds.size-1))
pftrackeresa2 = np.zeros((phibounds.size-1, thetabounds.size-1))
bincounteresa2 = np.zeros((phibounds.size-1, thetabounds.size-1))
pftrackeresa3 = np.zeros((phibounds.size-1, thetabounds.size-1))
bincounteresa3 = np.zeros((phibounds.size-1, thetabounds.size-1))

# checking which bin and in which ESA range each point falls
for k in tqdm(range(phi.size)):
    checker = False
    for i in range(phibounds.size-1):
        for j in range(thetabounds.size-1):
            if phibounds[i] <= phi[k] < phibounds[i+1] and thetabounds[j] <= theta[k] < thetabounds[j+1]:
                if esa1lowv < vmagskms[k] < esa12 and vmagskms[k] > 20:
                    pftrackeresa1[i,j] += flux[k]
                    bincounteresa1[i,j] += 1
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

# arrays to track flux values and bin counts in each ESA range
pftrackeresa1 = np.zeros((adjphibg.size, adjthetabg.size))
bincounteresa1 = np.zeros((adjphibg.size, adjthetabg.size))
pftrackeresa2 = np.zeros((adjphibg.size, adjthetabg.size))
bincounteresa2 = np.zeros((adjphibg.size, adjthetabg.size))
pftrackeresa3 = np.zeros((adjphibg.size, adjthetabg.size))
bincounteresa3 = np.zeros((adjphibg.size, adjthetabg.size))
# arrays to track flux values and bin counts in each ESA range
pftrackeresa1 = np.zeros((adjphibg10.size, adjthetabg10.size))
bincounteresa1 = np.zeros((adjphibg10.size, adjthetabg10.size))
pftrackeresa2 = np.zeros((adjphibg10.size, adjthetabg10.size))
bincounteresa2 = np.zeros((adjphibg10.size, adjthetabg10.size))
pftrackeresa3 = np.zeros((adjphibg10.size, adjthetabg10.size))
bincounteresa3 = np.zeros((adjphibg10.size, adjthetabg10.size))

phi = -phi
phi = 180/np.pi*phi
print(phi)

theta = theta*180/np.pi

#adjphibg = -(adjphibg + 105)

#adjphibg = -(adjphibg + 105) - 255

print(adjphibg)
print(theta)
"""for k in tqdm(range(phi.size)):
    checker = False
    for i in range(adjphibg.size-1):
        for j in range(adjthetabg.size-1):
            if adjphibg[i] <= phi[k] < adjphibg[i+1] and adjthetabg[j] <= theta[k] < adjthetabg[j+1]:
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
    for i in range(adjphibg10.size-1):
        for j in range(adjthetabg10.size-1):
            if adjphibg10[i] <= phi[k] < adjphibg10[i+1] and adjthetabg10[j] <= theta[k] < adjthetabg10[j+1]:
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

#Long1,Latg1 = np.meshgrid(adjphibg,adjthetabg)
Long1,Latg1 = np.meshgrid(adjphibg10,adjthetabg10)
# normalizing by bin count
pftrackeresa1 = pftrackeresa1/bincounteresa1
pftrackeresa2 = pftrackeresa2/bincounteresa2
pftrackeresa3 = pftrackeresa3/bincounteresa3
psdtracker = psdtracker/bincounter



# defining a grid for the midpoints of the cells
adjphib = np.linspace(-180, 180, int(360/ibexvaw+1))
adjthetab = np.linspace(-90, 90, int(180/ibexvaw+1))

# making a grid from the above
Lon,Lat = np.meshgrid(adjphib,adjthetab)

#Long,Latg = np.meshgrid(-adjphibg,adjthetabg)
Long,Latg = np.meshgrid(-adjphibg10,adjthetabg10)

pftrackeresa1 = np.transpose(pftrackeresa1) # transposing the PSD value array to work with the grid
pftrackeresa2 = np.transpose(pftrackeresa2)
pftrackeresa3 = np.transpose(pftrackeresa3)
psdtracker = np.transpose(psdtracker)

psdgallidiff = galliflux102-(pftrackeresa1*(esas[1]-esas[0]))
psdmaxavg = (galliflux102 + (pftrackeresa1*(esas[1]-esas[0])))/2
psdmax = np.zeros((np.size(pftrackeresa1,0), np.size(pftrackeresa1,1)))
for i in range(np.size(psdmax, 0)):
    for j in range(np.size(psdmax, 1)):    
        psdmax[i, j] = max(galliflux102[i, j], pftrackeresa1[i, j]*(esas[1]-esas[0]))
psddiffpercentage = np.abs(psdgallidiff)/psdmax
#psdgallidiff = galliflux102-pftrackeresa1
pgdtp = np.transpose(psdgallidiff)

file = open("C:/Users/lukeb/Documents/Dartmouth/HSResearch/Cluster Runs/2026DataTextFiles/" + fname + ".txt", 'w')
for i in range(adjphibg10.size):
    for j in range(adjthetabg10.size):
        file.write(str(adjphibg10[i]) + ',' + str(adjthetabg10[j]) + ',' + str(pgdtp[i,j]) + '\n')
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
im = ax.pcolormesh(Long,Latg,psdgallidiff, cmap='berlin', transform=rotated_pole2, vmin=-10**(5),vmax=10**(5))
#im = ax.pcolormesh(Long,Latg,psddiffpercentage, cmap='rainbow', transform=rotated_pole2, vmin=0,vmax=1)
ax.gridlines(crs=rotated_pole2, draw_labels=False, 
             xlocs=range(-165,165,30), 
             ylocs=range(-90,90,15)) #draw_labels=True NOT allowed
for ea in lonlats:
    ax.text(ea[0], ea[1], ea[2], fontsize=10, fontweight='ultralight', color="k", transform=rotated_pole2)

ax.set_global()

plt.xlabel("Heliolongitude Angle $\phi$")
plt.ylabel("Heliolatitude Angle θ")
cb = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.04)
cb.set_label('Difference in Intensity as Percentage of Maximum Bin Intensity')
#cb.set_label('Intensity at Detector (cm$^{-2}$ s$^{-1}$ sr$^{-1}$)')
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