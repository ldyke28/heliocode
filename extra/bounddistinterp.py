import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import odeint
import scipy
from tqdm import tqdm
import h5py


datafilename1 = 'C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Collaborations/FedericoVDF/VDF3D_HE013Ksw_PRB_Eclip256_R070_001_H_RegAll.h5'
datafilename2 = 'C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Collaborations/FedericoVDF/VDF3D_HE013Ksw_PRB_Eclip256_R070_002_H_RegAll.h5'
datafilename3 = 'C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Collaborations/FedericoVDF/VDF3D_HE013Ksw_PRB_Eclip256_R070_003_H_RegAll.h5'
datafilename4 = 'C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Collaborations/FedericoVDF/VDF3D_HE013Ksw_PRB_Eclip256_R070_004_H_RegAll.h5'
datafilename5 = 'C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Collaborations/FedericoVDF/VDF3D_HE013Ksw_PRB_Eclip256_R070_005_H_RegAll.h5'
datafilename6 = 'C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Collaborations/FedericoVDF/VDF3D_HE013Ksw_PRB_Eclip256_R070_006_H_RegAll.h5'
datafilename7 = 'C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Collaborations/FedericoVDF/VDF3D_HE013Ksw_PRB_Eclip256_R070_007_H_RegAll.h5'
datafilename8 = 'C:/Users/lucas/OneDrive/Documents/Dartmouth/HSResearch/Collaborations/FedericoVDF/VDF3D_HE013Ksw_PRB_Eclip256_R070_008_H_RegAll.h5'

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

endlongangle = 360

endvelcoords = np.array([-26, 1, 2])

"""if endlongangle > 22.5 and endlongangle <=67.5:
    initpsd = interp2(endvelcoords)
elif endlongangle > 67.5 and endlongangle <=112.5:
    initpsd = interp3(endvelcoords)
elif endlongangle > 112.5 and endlongangle <=157.5:
    initpsd = interp4(endvelcoords)
elif endlongangle > 157.5 and endlongangle <=202.5:
    initpsd = interp5(endvelcoords)
elif endlongangle > 202.5 and endlongangle <=247.5:
    initpsd = interp6(endvelcoords)
elif endlongangle > 247.5 and endlongangle <=292.5:
    initpsd = interp7(endvelcoords)
elif endlongangle > 292.5 and endlongangle <=337.5:
    initpsd = interp8(endvelcoords)
elif endlongangle > 337.5 or endlongangle <= 22.5:
    initpsd = interp1(endvelcoords)"""

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

print(initpsd)