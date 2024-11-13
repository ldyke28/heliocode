import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
import scipy.interpolate
from tqdm import tqdm
from scipy.signal import butter, lfilter, freqz

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
finalt = -5*oneyear # time to start backtracing
#6.36674976e9 force free for cosexprp
initialt = -5*10**(10) # time in the past to which the code should backtrace
tstep = 10000 # general time resolution
tstepclose = 300 # time resolution for close regime
tstepfar = 200000 # time resolution for far regime
phase = 0 # phase for implementing rotation of target point around sun
refdist = 70 # upwind reference distance for backtraced trajectories, in au

file = np.loadtxt("C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Time Dependent Irradiance Data/complya.csv", delimiter=',')

day = file[:,0]
irradiance = file[:,1]

seconds = day*86400

N = 28

oldirradianceavg = np.zeros(seconds.size)
for i in tqdm(range(seconds.size)):
    avgnumerator = 0
    lowerbound = -(int(N/2))
    upperbound = int(N/2)
    if i + lowerbound < 0:
        lowerbound = -i
    if i + upperbound >= seconds.size:
        upperbound = seconds.size - i - 1
    for j in range(-lowerbound):
        avgnumerator += irradiance[i+lowerbound]
    avgnumerator += irradiance[i]
    for k in range(upperbound):
        avgnumerator += irradiance[i+upperbound]
    oldirradianceavg[i] = avgnumerator/(upperbound - lowerbound + 1)

irradianceavg = np.array([])
secondsnew = np.array([])
for i in range(seconds.size):
    if (i+1) % N == 0:
        irradianceavg = np.append(irradianceavg, [oldirradianceavg[i]])
        secondsnew = np.append(secondsnew, [seconds[i]])

secondsnew = secondsnew - 1.946*10**9
seconds = seconds - 1.946*10**9
secondstoyears = 1/(86400*365)

wm2toph = 6.12*10**(13)

####################################################################################################


def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


# Filter requirements.
order = 6
fs = .000002       # sample rate, Hz
#fs = 1/(86400)
cutoff = 1/(1.577*10**8)  # desired cutoff frequency of the filter, Hz

filteredia = butter_lowpass_filter(irradiance, cutoff, fs, order)

plt.plot(seconds*secondstoyears, irradiance*wm2toph, alpha = 0.5)
plt.plot(seconds*secondstoyears, filteredia*wm2toph, alpha=0.5)
plt.ylim([3*10**(11),7.25*10**(11)])
plt.show()