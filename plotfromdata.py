import numpy as np
import matplotlib.pyplot as plt
import matplotlib

vx = np.array([])
vy = np.array([])
f = np.array([])

filenum = 1

#file = open("C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/datafiles/p5s2adj_pi4_6p3e9_center_noatttest.txt", "r")
#file = np.loadtxt("C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Paper Data/cosexprp_5pi6_6p2e9_center_cosexppi.txt", delimiter=',')
file = np.loadtxt("/Users/ldyke/Desktop/Dartmouth/HSResearch/Code/Kepler/Python Orbit Code/datafiles/cosexprp_pi32_1p5e8_indirect_cosexppi_loctest.txt", delimiter=',')


#file2 = np.loadtxt("C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/datafiles/cosexprp_5pi6_6p36675e9_indirect_cosexppi.txt", delimiter=',')
#file2 = np.loadtxt("/Users/ldyke/Desktop/Dartmouth/HSResearch/Code/Kepler/Python Orbit Code/datafiles/cosexprp_5pi6_6p36675e9_indirect_cosexppi.txt", delimiter=',')

#file3 = np.loadtxt("C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/datafiles/cosexprp_5pi6_6p36675e9_direct_cosexppi.txt", delimiter=',')


for i in range(np.shape(file)[0]):
    vx = np.append(vx, file[i,0])
    vy = np.append(vy, file[i,1])
    f = np.append(f, file[i,2])

if filenum == 2:
    for i in range(np.shape(file2)[0]):
        vx = np.append(vx, file2[i,0])
        vy = np.append(vy, file2[i,1])
        f = np.append(f, file2[i,2])

if filenum == 3:
    for i in range(np.shape(file2)[0]):
        vx = np.append(vx, file2[i,0])
        vy = np.append(vy, file2[i,1])
        f = np.append(f, file2[i,2])
    for i in range(np.shape(file3)[0]):
        vx = np.append(vx, file3[i,0])
        vy = np.append(vy, file3[i,1])
        f = np.append(f, file3[i,2])

fig = plt.figure()
fig.set_figwidth(10)
fig.set_figheight(6)

#plt.scatter(vx[:], vy[:], c=f[:], marker='o', cmap='plasma')
#plt.scatter(vx[:], vy[:], c=f[:], marker='o', cmap='plasma', vmin=0, vmax=.6)
plt.scatter(vx[:], vy[:], c=f[:], marker='o', cmap='plasma', norm=matplotlib.colors.LogNorm(vmin=10**(-7), vmax=1))
cb = plt.colorbar()
cb.set_label('PDF(r,v,t)')
plt.xlabel("vx at Target in km/s")
plt.ylabel("vy at Target in km/s")
plt.suptitle('Phase space population at target (t $\\approx$ 4.753 years) drawn from Maxwellian at 100 au centered on vx = -26 km/s')
plt.title('Target at (.9952 au, .0980 au), Time Resolution Close to Target = 1000 s')
plt.show()

"""cb.set_label('PDF(r,v,t)', fontsize=12)
plt.xlabel("vx at Target in km/s", fontsize=14)
plt.ylabel("vy at Target in km/s", fontsize=14)
plt.suptitle('Phase space population at target (t = 6.2e9 s) \n Drawn from Maxwellian at 100 au centered on vx = -26 km/s', fontsize=14)
plt.title('Target at (-.866 au, .5 au), Time Resolution Close to Target = 1000 s', fontsize=14)
plt.show()"""



"""fig = plt.figure()
fig.set_figwidth(9)
fig.set_figheight(6)
levels = [.001, .01, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .98, 1.0]
plt.tricontour(vx[:], vy[:], f[:], levels)
cb = plt.colorbar()
cb.set_label('f(r,v,t)')
plt.xlabel("vx at Target in km/s")
plt.ylabel("vy at Target in km/s")
plt.suptitle('Phase space population at target (t = 6.246e9 s) drawn from Maxwellian at 100 au centered on vx = -26 km/s')
plt.title('Target at (-.707 au, .707 au)')
plt.show()"""