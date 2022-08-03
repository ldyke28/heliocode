import numpy as np
import matplotlib.pyplot as plt
import matplotlib

vx = np.array([])
vy = np.array([])
f = np.array([])

filenum = 1

#file = open("C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/datafiles/p5s2_meddownwind_sta_flipped.txt", "r")
file = np.loadtxt("/Users/ldyke/Desktop/Dartmouth/HSResearch/Code/Kepler/Python Orbit Code/datafiles/p5s2adj_pi4_6p246e9_center_newattenuation.txt", delimiter=',')
#file = np.loadtxt("/Users/ldyke/Desktop/Dartmouth/HSResearch/Code/Kepler/Python Orbit Code/datafiles/p5s2adj_meddownwind_attractive_center_expanded.txt", delimiter=',')


#file2 = np.loadtxt("C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/datafiles/p5s2adj_meddownwind_attractive_indirect.txt", delimiter=',')
#file2 = np.loadtxt("/Users/ldyke/Desktop/Dartmouth/HSResearch/Code/Kepler/Python Orbit Code/datafiles/p5s2_5pi6_sta_ind_t3.txt", delimiter=',')

#file3 = np.loadtxt("C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/datafiles/p5s2adj_meddownwind_attractive_test3.txt", delimiter=',')


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
fig.set_figwidth(9)
fig.set_figheight(6)

#plt.scatter(vx[:], vy[:], c=f[:], marker='o', cmap='plasma')
plt.scatter(vx[:], vy[:], c=f[:], marker='o', cmap='plasma', norm=matplotlib.colors.LogNorm(vmin=.001))
cb = plt.colorbar()
cb.set_label('f(r,v,t)')
plt.xlabel("vx at Target in km/s")
plt.ylabel("vy at Target in km/s")
plt.suptitle('Phase space population at target (t = 6.246e9 s) drawn from Maxwellian at 100 au centered on vx = -26 km/s')
plt.title('Target at (.707 au, .707 au)')
plt.show()


fig = plt.figure()
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
plt.show()