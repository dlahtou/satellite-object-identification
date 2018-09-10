from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
from osgeo.gdalconst import GA_ReadOnly
import rasterio
import cv2
from os import mkdir
from os.path import join, isdir
import pickle as pkl


# opens a RGB 3000px+ image, and saves it as 169 corresponding 256x256 images
ds = gdal.Open('data/three_band/6040_2_2.tif', GA_ReadOnly)
ds_array = ds.ReadAsArray()

print("TYPE:")
print(type(ds_array))

print("Shape:")
print(ds_array.shape)

print("X width:")
print(ds.RasterXSize)
print("Y height:")
print(ds.RasterYSize)

'''ds = gdal.Translate('6040_2_2_0.tif', ds, projWin = [0, 100, 100, 200])

fig = plt.figure()
for i in range(3):
    fig.add_subplot(1,3,i+1)
    plt.imshow(ds_array[i,:,:])

    print(i)
    print(np.min(ds_array[i,:,:]))
    print(np.max(ds_array[i,:,:]))
plt.show()'''

parent_dir = 'data/three_band/clipped'
if not isdir(parent_dir):
    mkdir(parent_dir)

out_folder = 'data/three_band/clipped/6040_2_2'
if not isdir(out_folder):
    mkdir(out_folder)

graph = False
counter = 0
# 256 pixel blocks on x axis
for i in range(13):
    # 256 pixel blocks on y axis
    for j in range(13):
        print(counter)
        with open(join(out_folder, 'image'+str(counter)), 'wb') as open_file:
            pkl.dump(ds_array[:, j*256:(j+1)*256, i*256:(i+1)*256], open_file)
        
        if graph == True:
            fig = plt.figure()
            for k in range(3):
                fig.add_subplot(1,3,k+1)
                plt.imshow(ds_array[k,j*256:(j+1)*256,i*256:(i+1)*256])
            plt.show()
        counter +=1

'''with rasterio.open('/home/dlahtou/6040_2_2.tif') as source:
    source_data = source.read()
    profile = source.profile

print(profile['count'])'''

'''with open('/home/dlahtou/6040_2_2/image1', 'rb') as open_file:
    ds_array = pkl.load(open_file)

print(ds_array.shape)

fig = plt.figure()
for i in range(3):
    fig.add_subplot(1,3,i+1)
    plt.imshow(ds_array[i,:,:])

    print(i)
    print(np.min(ds_array[i,:,:]))
    print(np.max(ds_array[i,:,:]))'''