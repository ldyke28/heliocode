import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import scipy

filename = 'C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Collaborations/FedericoVDF/VDF3D_HE013Ksw_PRB_Eclip256_R115_001_H_RegAll.h5'

with h5py.File(filename, "r") as f:
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
    dsvdf = f['VDF3D'] # returns h5py dataset object
    arrvdf = f['VDF3D'][()] # returns np.array of values
    #print(arrvdf[128,128,128])
    xloc = f['vx_grid'][()]
    yloc = f['vy_grid'][()]
    zloc = f['vz_grid'][()]

    totalsize = xloc.size*yloc.size*zloc.size
    newx = np.zeros(totalsize)
    newy = np.zeros(totalsize)
    newz = np.zeros(totalsize)
    newvdf = np.zeros(totalsize)

    #print(xloc)
    #print(yloc)
    #print(zloc)

    """xsize = xloc.size
    ysize = yloc.size
    zsize = zloc.size
    for i in tqdm(range(xsize*ysize*zsize)):
        #print(np.floor((i+1)/(xsize**2)))
        newz[i] = zloc[int(np.floor((i)/(xsize**2)))]
        #newz[i] = 0

    for i in tqdm(range(zsize)):
        for j in range(ysize*xsize):
            newy[j+i*ysize*xsize] = yloc[int(np.floor((j)/zsize))]

    for i in tqdm(range(zsize)):
        for j in range(ysize):
            for k in range(xsize):
                newx[k + j*zsize + i*ysize*zsize] = xloc[k]
                newvdf[k + j*zsize + i*ysize*zsize] = arrvdf[k,j,i]"""


    """lowerbound = int(np.floor(xsize*ysize*(zsize/2-1)))
    upperbound = int(np.floor(xsize*ysize*(zsize/2+1)))
    newx = newx[lowerbound:upperbound]
    newy = newy[lowerbound:upperbound]
    newz = newz[lowerbound:upperbound]
    newvdf = newvdf[lowerbound:upperbound]"""


    """fig3d = plt.figure()
    #fig3d.set_figwidth(10)
    #fig3d.set_figheight(7)
    ax3d = plt.axes(projection='3d')
    scatterplot = ax3d.scatter3D(newx[:], newy[:], newz[:], c=newvdf[:], cmap='rainbow', s=.1, norm=matplotlib.colors.LogNorm(vmin=10**(-7)))
    cb = fig3d.colorbar(scatterplot)
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.set_xlim3d(-100,100)
    ax3d.set_ylim3d(-100,100)
    ax3d.set_zlim3d(-2,2)
    ax3d.view_init(90,270)
    plt.show()"""



    """def maxwelliantest(data, a, b, c, d):
        #x, y, z = data
        x = data[0,:]
        y = data[1,:]
        z = data[2,:]
        return d*1/(np.pi)**(3/2)*1/(a*b*c)*np.exp(-((x-26)**2/a**2 + y**2/b**2 + z**2/c**2))
    
    #maxwell = scipy.stats.maxwell
    #params = maxwell.fit
    
    #popt, pcov = scipy.optimize.curve_fit(maxwelliantest, (newx, newy, newz), newvdf, bounds=(0, [200., 200., 200.]))
    popt, pcov = scipy.optimize.curve_fit(maxwelliantest, (newx, newy, newz), newvdf)
    print(popt)"""

    zgrid, ygrid, xgrid = np.meshgrid(zloc, yloc, xloc, indexing='ij') # order will be z, y, x for this

    interp = scipy.interpolate.RegularGridInterpolator((zloc, yloc, xloc), arrvdf)

    print(interp([-25,-32,0]))
