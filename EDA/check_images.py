import numpy as np
from os import listdir
import matplotlib.pyplot as plt
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly
import cv2

# check if all bands are aligned
cap = np.load('trees_images.npy')[0]
for i in range(20):
    a = cap[:,:,i]
    print(a.shape)
    #np.reshape(a, (3349, 3391))
    plt.imshow(a)
    plt.colorbar()
    plt.show()
    del a

a = np.load('trees_masks.npy')[0]
print(a.shape)
plt.imshow(a[:,:,0])
plt.show()

'''ds = gdal.Open('6040_2_2_M.tif', GA_ReadOnly)
ds_array = ds.ReadAsArray()

ds_array = np.transpose(ds_array, (1,2,0))

plt.imshow(ds_array[:,:,7])
plt.show()'''