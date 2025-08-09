import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib
import scipy
import pylab as pl
import math
import pandas as pd

# Paper 2 maxima
#vxarray = np.array([-24.48, -25.98, -28.82, -31.88, -32.82, -33.06, -33.16, -33.00, -31.74, -28.5, -28.58, -32.34])
#vyarray = np.array([-4.1, 1.52, 3.7, 6.3, 6.84, 6.88, 6.86, 6.54, 4.78, 0.64, -4.32, 16.26])
#f = np.array([8.04*10**(-8), 1.30*10**(-7), 2.65*10**(-7), 3.85*10**(-7), 4.27*10**(-7), 4.38*10**(-7), 4.42*10**(-7), 4.34*10**(-7), 3.78*10**(-7), 2.54*10**(-7), 1.41*10**(-7), 3.30*10**(-8)])

# Paper 1 maxima
#vxarray = np.array([-28.76, -29.54, -33, -35.38, -36.2, -36.45, -36.52, -36.38, -35.26, -32.58, -31.74, -25.78])
#vyarray = np.array([1.22, 2.44, 6.92, 9.2, 9.68, 9.7, 9.7, 9.38, 7.7, 4.08, 2.94, -5.06])
#f = np.array([4.87*10**(-7), 5.88*10**(-7), 1.11*10**(-6), 1.56*10**(-6), 1.73*10**(-6), 1.79*10**(-6), 1.81*10**(-6), 1.79*10**(-6), 1.58*10**(-6), 1.14*10**(-6), 1.01*10**(-6), 2.02*10**(-7)])

# Paper 2 bulk velocities (approximation)
#vxarray = np.array([-24.093, -24.580, -26.877, -28.507, -28.976, -29.059, -29.076, -28.926, -27.940, -25.680, -25.023, -22.037])
#vyarray = np.array([2.594, 3.250, 6.279, 8.159, 8.612, 8.673, 8.676, 8.413, 6.989, 3.969, 3.117, 0.311])
#f = np.array([6.93*10**(-8), 8.42*10**(-8), 1.77*10**(-7), 2.98*10**(-7), 3.25*10**(-7), 3.35*10**(-7), 3.38*10**(-7), 3.31*10**(-7), 2.86*10**(-7), 1.95*10**(-7), 1.64*10**(-7), 2.09*10**(-8)])

# 3D Time Sequence (downwind) approximate bulk velocities
vxarray = np.array([-21.813, -22.019, -21.869, -22.658, -22.866, -23.036, -23.109, -23.083, -22.707, -21.843])
vyarray = np.array([15.397, 16.051, 16.569, 17.889, 18.667, 19.458, 19.845, 19.176, 18.271, 17.019])
vzarray = np.array([-5.148, -4.786, -5.605, -4.819, -4.826, -4.778, -4.842, -4.899, -4.833, -4.793])

# 3D Time Sequence (downwind, fifth order filter) approximate bulk velocities
vxarray = np.array([-21.496, -21.939, -22.493, -22.757, -22.960, -23.076, -23.206, -23.109, -22.745, -21.798])
vyarray = np.array([15.354, 16.165, 17.262, 18.303, 19.029, 19.579, 19.680, 19.245, 17.986, 16.727])
vzarray = np.array([-5.172, -4.726, -4.777, -4.848, -4.834, -4.818, -4.799, -4.833, -4.883, -4.770])

#labels = np.array(["-4 yr", "-3.824 yr", "-3 yr", "-2 yr", "-1 yr", "0 yr", "1 yr", "2 yr", "3 yr", "3.824 yr", "4 yr", "5.5 yr"])
labels = np.array(["-4 yr", "-3 yr", "-2 yr", "-1 yr", "0 yr", "1 yr", "2 yr", "3 yr", "4 yr", "5.5 yr"])

def get_text_positions(x_data, y_data, txt_width, txt_height):
    a = zip(y_data, x_data)
    text_positions = y_data.copy()
    for index, (y, x) in enumerate(a):
        local_text_positions = [i for i in a if i[0] > (y - txt_height) 
                            and (abs(i[1] - x) < txt_width * 2) and i != (y,x)]
        if local_text_positions:
            sorted_ltp = sorted(local_text_positions)
            if abs(sorted_ltp[0][0] - y) < txt_height: #True == collision
                differ = np.diff(sorted_ltp, axis=0)
                a[index] = (sorted_ltp[-1][0] + txt_height, a[index][1])
                text_positions[index] = sorted_ltp[-1][0] + txt_height
                for k, (j, m) in enumerate(differ):
                    #j is the vertical distance between words
                    if j > txt_height * 2: #if True then room to fit a word in
                        a[index] = (sorted_ltp[k][0] + txt_height, a[index][1])
                        text_positions[index] = sorted_ltp[k][0] + txt_height
                        break
    return text_positions

#def text_plotter(x_data, y_data, text_positions, axis,txt_width,txt_height):
#    for x,y,t in zip(x_data, y_data, text_positions):
#        axis.text(x - txt_width, 1.01*t, '%d'%int(y),rotation=0, color='blue')
#        if y != t:
#            axis.arrow(x, t,0,y-t, color='red',alpha=0.3, width=txt_width*0.1, 
#                       head_width=txt_width, head_length=txt_height*0.5, 
#                       zorder=0,length_includes_head=True)
            
def text_plotter(x_data, y_data, labels, text_positions, axis,txt_width,txt_height):
    for x,y,t,l in zip(x_data, y_data, text_positions, labels):
        axis.text(x - txt_width, 1.01*t, l,rotation=0, color='black', weight='bold')
        if y != t:
            axis.arrow(x, t,0,y-t, color='red',alpha=0.3, width=txt_width*0.1, 
                       head_width=txt_width, head_length=txt_height*0.5, 
                       zorder=0,length_includes_head=True)

"""fsize = 16
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
plt.plot(vxarray, vyarray, zorder=1)
for i in range(vxarray.size-1):
    veclength = np.sqrt((vxarray[i+1] - vxarray[i])**2 + (vyarray[i+1] - vyarray[i])**2)
    #plt.arrow((vxarray[i+1] + vxarray[i])/2, (vyarray[i+1] + vyarray[i])/2, (vxarray[i+1] - vxarray[i])/veclength*.000001, (vyarray[i+1] - vyarray[i])/veclength*.000001, shape='full', lw=0, length_includes_head=False, head_width=.2)
#plt.scatter(vxarray[:], vyarray[:], c=f[:], marker='o', cmap='rainbow', norm=matplotlib.colors.LogNorm(), zorder=2)
plt.scatter(vxarray[:], vyarray[:], marker='o', zorder=2)"""

fsize = 16
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(projection='3d')
ax.plot(vxarray, vyarray, vzarray, zorder=1)
#for i in range(vxarray.size-1):
#    veclength = np.sqrt((vxarray[i+1] - vxarray[i])**2 + (vyarray[i+1] - vyarray[i])**2)
    #plt.arrow((vxarray[i+1] + vxarray[i])/2, (vyarray[i+1] + vyarray[i])/2, (vxarray[i+1] - vxarray[i])/veclength*.000001, (vyarray[i+1] - vyarray[i])/veclength*.000001, shape='full', lw=0, length_includes_head=False, head_width=.2)
#plt.scatter(vxarray[:], vyarray[:], c=f[:], marker='o', cmap='rainbow', norm=matplotlib.colors.LogNorm(), zorder=2)
ax.scatter(vxarray[:], vyarray[:], vzarray[:], marker='o', zorder=2, color='black', s=10)
for i in range(vxarray.size):
    ax.text(vxarray[i],vyarray[i],vzarray[i], labels[i], size=10, zorder=1, color='k')

ax.set_xlabel("$v_x$", fontsize=fsize)
ax.set_ylabel("$v_y$", fontsize=fsize)
ax.set_zlabel("$v_z$", fontsize=fsize)
ax.set_title("Approximate Bulk Velocity Throughout the Solar Cycle (km/s)")
plt.show()

"""#set the bbox for the text. Increase txt_width for wider text.
txt_height = 0.04*(plt.ylim()[1] - plt.ylim()[0])
txt_width = 0.02*(plt.xlim()[1] - plt.xlim()[0])
#Get the corrected text positions, then write the text.
text_positions = get_text_positions(vxarray, vyarray, txt_width, txt_height)
text_plotter(vxarray, vyarray, labels, text_positions, ax, txt_width, txt_height)

plt.ylim(0,max(text_positions)+2*txt_height)
plt.xlim(-0.1,1.1)"""

ann = []
for i in range(labels.size):
    ann.append(ax.annotate(labels[i], (vxarray[i], vyarray[i]+0.1), weight='bold', fontsize=12))

mask = np.zeros(fig.canvas.get_width_height(), bool)

fig.canvas.draw()

for a in ann:
    bbox = a.get_window_extent()
    x0 = int(bbox.x0)
    x1 = int(math.ceil(bbox.x1))
    y0 = int(bbox.y0)
    y1 = int(math.ceil(bbox.y1))

    s = np.s_[x0:x1+1, y0:y1+1]
    if np.any(mask[s]):
        a.set_visible(False)
    else:
        mask[s] = True


plt.rcParams.update({'font.size': fsize})
cb = plt.colorbar()
cb.set_label('Phase Space Density at Corresponding Point')
plt.grid(zorder=0)
ax.set_axisbelow(True)
#plt.xlim([-30, -21])
#plt.ylim([0, 10])
plt.xticks(fontsize=fsize)
plt.yticks(fontsize=fsize)
plt.xlabel("$v_x$ at Target in km/s", fontsize=fsize)
plt.ylabel("$v_y$ at Target in km/s", fontsize=fsize)
plt.show()