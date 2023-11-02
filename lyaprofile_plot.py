import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def LyaRP(t,v_r):
    # a double (triple) Gaussian function to mimic the Lyman-alpha profile
    lyafunction = 1.25*np.exp(-(v_r/1000-55)**2/(2*25**2)) + 1.25*np.exp(-(v_r/1000+55)**2/(2*25**2)) + .55*np.exp(-(v_r/1000)**2/(2*25**2))
    omegat = 2*np.pi/(3.47*10**(8))*t
    # an added scale factor to adjust the total irradiance of the integral without changing the shape (adjusts total magnitude by a factor)
    scalefactor = 1.616
    return scalefactor*(.75 + .243*np.cos(omegat - np.pi)*np.exp(np.cos(omegat - np.pi)))*lyafunction

def LyaRP2(t,v_r):
    # My Ly-a line profile function
    #lyafunction = 1.25*np.exp(-(v_r-55000)**2/(2*25000**2)) + 1.25*np.exp(-(v_r+55000)**2/(2*25000**2)) + .55*np.exp(-v_r**2/(2*25000**2))
    # Ly-a line profile function from Tarnopolski 2007
    lyafunction = np.e**(-3.8312*10**-5*(v_r/1000)**2)*(1 + .73879* \
    np.e**(.040396*(v_r/1000) - 3.5135*10**-4*(v_r/1000)**2) + .47817* \
    np.e**(-.046841*(v_r/1000) - 3.3373*10**-4*(v_r/1000)**2))
    omegat = 2*np.pi/(3.47*10**(8))*t
    tdependence = 5.6*10**11 - np.e/(np.e + 1/np.e)*2.4*10**11 + 2.4*10**11/(np.e + 1/np.e) * np.cos(omegat - np.pi)*np.exp(np.cos(omegat - np.pi))
    #return (.75 + .243*np.cos(omegat - np.pi)*np.exp(np.cos(omegat - np.pi)))*lyafunction
    return 2.4543*10**-9*(1 + 4.5694*10**-4*tdependence)*lyafunction

# constants for following function
A_K = 6.523*(1 + 0.619)
m_K = 5.143*(1 -0.081)
del_K = 38.008*(1+0.104)
K = 2.165*(1-0.301)
A_R = 580.37*(1+0.28)
dm = -0.344*(1-0.828)
del_R = 32.349*(1-0.049)
b_bkg = 0.026*(1+0.184)
a_bkg = 0.411**(-4) *(1-1.333*0.0007)
#print(a_bkg)
r_E = 0.6
r2 = 1
def LyaRP3(t,v_r):
    #Author: E. Samoylov, H. Mueller LISM Group
    #Date: 04.18.2023
    #Purpose: To confirm the graph that EQ14 produces in
    #         Kowalska-Leszczynska's 2018 paper
    #         Evolution of the Solar Lyα Line Profile during the Solar Cycle
    #https://iopscience.iop.org/article/10.3847/1538-4357/aa9f2a/pdf
    F_R = A_R / (del_R * np.sqrt(2 * np.pi)) *np.exp(-(np.square((v_r/1000) - (m_K - dm))) / (2*(del_R ** 2)))
    F_bkg = np.add(a_bkg*(v_r/1000)*0.000001,b_bkg)
    F_K = A_K * np.power(1 + np.square((v_r/1000) - m_K) / (2 * K * ((del_K) ** 2)), -K - 1)

    omegat = 2*np.pi/(3.47*10**(8))*t
    tdependence = .85 - np.e/(np.e + 1/np.e)*.33 + .33/(np.e + 1/np.e) * np.cos(omegat - np.pi)*np.exp(np.cos(omegat - np.pi))
    # an added scale factor to adjust the total irradiance of the integral without changing the shape (adjusts total magnitude by a factor)
    scalefactor = .9089
    #(F_K-F_R+F_bkg)/((r_E/r)**2)
    return scalefactor*tdependence*(F_K-F_R+F_bkg)/(r_E/(r2**2))


t = 0
t2 = 3.47*10**8 / 4
t3 = 3.47*10**8 / 2
inputvr = np.arange(-120000, 120000, 10)
profile1 = np.zeros(inputvr.size)
profile1t2 = np.zeros(inputvr.size)
profile1t3 = np.zeros(inputvr.size)
profile2 = np.zeros(inputvr.size)
profile2t2 = np.zeros(inputvr.size)
profile2t3 = np.zeros(inputvr.size)
profile3 = np.zeros(inputvr.size)
profile3t2 = np.zeros(inputvr.size)
profile3t3 = np.zeros(inputvr.size)
for i in range(inputvr.size):
    profile1[i] = LyaRP(t,inputvr[i])
    profile1t2[i] = LyaRP(t2,inputvr[i])
    profile1t3[i] = LyaRP(t3,inputvr[i])
    profile2[i] = LyaRP2(t,inputvr[i])
    profile2t2[i] = LyaRP2(t2,inputvr[i])
    profile2t3[i] = LyaRP2(t3,inputvr[i])
    profile3[i] = LyaRP3(t,inputvr[i])
    profile3t2[i] = LyaRP3(t2,inputvr[i])
    profile3t3[i] = LyaRP3(t3,inputvr[i])

fsize = 18
fig, ax = plt.subplots()
fig.set_figwidth(9)
fig.set_figheight(6)
ax.plot(inputvr/1000, profile2)
ax.plot(inputvr/1000, profile2t2)
ax.plot(inputvr/1000, profile2t3)
plt.xticks(fontsize=fsize)
plt.yticks(fontsize=fsize)
plt.ylim(bottom=0)
plt.grid()
ax.set_xlabel("Radial Velocity Component $v_r$ (km/s)", fontsize=fsize)
ax.set_ylabel("Value of $\mu (t)$", fontsize=fsize)
#plt.title("Photoionization Rate over Time", fontsize=fsize)
plt.show()