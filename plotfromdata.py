import numpy as np
import matplotlib.pyplot as plt
import matplotlib

vx = np.array([])
vy = np.array([])
f = np.array([])

filenum = 1

#file = open("C:\Users\lucas\Downloads\cosexprp_pi32_1p5e8_indirect_cosexppi.txt", "r")
file = np.loadtxt("C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/datafiles/cosexprp_5pi6_4e6_center_constantpi_tclose1000_r=1au.txt", delimiter=',')
#file = np.loadtxt("C:/Users/lucas/Downloads/Data Files-20230406T214257Z-001/Data Files/lyaminrp_5pi6_0y_direct_cosexppi_tclose400_1.txt", delimiter=',')
#file = np.loadtxt("/Users/ldyke/Downloads/drive-download-20221019T183112Z-001/cosexprp_pi32_1p5e8_indirect_cosexppi.txt", delimiter=',')


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

"""plt.scatter(vx[:], vy[:], c=f[:], marker='o', cmap='plasma')
#plt.scatter(vx[:], vy[:], c=f[:], marker='o', cmap='plasma', vmin=0, vmax=.6)
#plt.scatter(vx[:], vy[:], c=f[:], marker='o', cmap='plasma', norm=matplotlib.colors.LogNorm(vmin=10**(-100), vmax=1))
cb = plt.colorbar()
cb.set_label('PDF(r,v,t)')
plt.xlim([-25,25])
plt.ylim([-25,25])
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("$v_x$ at Target in km/s", fontsize=16)
plt.ylabel("$v_y$ at Target in km/s", fontsize=16)
plt.suptitle('Phase space population at target (t = 0 years) drawn from Maxwellian at 100 au centered on vx = -26 km/s')
plt.title('Target at (-.866 au, .5 au), Time Resolution Close to Target = 1000 s')
plt.show()"""

fsize = 18
#plt.scatter(vx[:], vy[:], c=f[:], marker='o', cmap='hsv', vmin=0, vmax=0.06218571051524244)
#plt.scatter(vx[:], vy[:], c=f[:], marker='o', cmap='hsv')
plt.scatter(vx[:], vy[:], c=f[:], marker='o', cmap='hsv', norm=matplotlib.colors.LogNorm(vmin=10**(-6), vmax=10**(-2)))
plt.rcParams.update({'font.size': fsize})
cb = plt.colorbar()
#cb.set_label('Time at which orbit passes through 100 au (s)')
#cb.set_label('Travel Time from 100 au to Point of Interest (s)')
cb.set_label('Normalized Phase Space Density')
#plt.xlim([-25, 25])
#plt.ylim([-25, 25])
plt.xticks(fontsize=fsize)
plt.yticks(fontsize=fsize)
plt.xlabel("$v_x$ at Target in km/s", fontsize=fsize)
plt.ylabel("$v_y$ at Target in km/s", fontsize=fsize)
#plt.suptitle('Phase Space population at x = 100 au reaching initial position at t = 5700000000 s')
#plt.suptitle('VDF at target, at t $\\approx$ ' + str(round(finalt/(oneyear), 3)) + ' years, drawn from Maxwellian at 100 au centered on $v_x$ = -26 km/s')
#plt.title('Target (-.97au, .2au): vx range -51500 m/s to -30500 m/s, vy range -30000 m/s to 30000 m/s')
#plt.title('Target at (' + str(round(ibexpos[0]/au, 3)) + ' au, ' + str(round(ibexpos[1]/au, 3)) + ' au), Time Resolution Close to Target = ' + str(tstepclose) + ' s')
#plt.title('Initial test distribution centered on vx = -41.5 km/s, vy = -1.4 km/s')
plt.show()



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