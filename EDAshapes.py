import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon
from osgeo.gdalconst import GA_ReadOnly
from shapely.wkt import loads
from osgeo import ogr
import pickle as pkl
from os.path import join, isdir
from os import mkdir
import cv2
import tensorflow as tf
from keras.layers import BatchNormalization, Conv2D, Dense, Conv2DTranspose, Input, MaxPooling2D, Concatenate
from keras.models import Model
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping
from tensorflow.metrics import mean_iou
import shapely
from osgeo import gdal


with open('data/grid_sizes.csv', 'r') as open_file:
    grids = pd.read_csv(open_file)

grids.set_index('Unnamed: 0', inplace=True)

target_image = '6040_2_2'

xmax = grids.loc[target_image, 'Xmax']
ymin = grids.loc[target_image, 'Ymin']

xmax_chunk = xmax/10
ymin_chunk = ymin/10

# get wkt shapes for trees in the first image
# wkt = loads(shapes.iloc[4,2])
'''
fig, ax = plt.subplots(figsize=(9, 9))

count = 0

for polygon in wkt:
    if count == 0:
        print(np.array(polygon.exterior))
    mpl_poly = Polygon(np.array(polygon.exterior))
    ax.add_patch(mpl_poly)
    count += 1

segment_square = Polygon(np.array([[0, 0], [xmax_chunk, 0], [xmax_chunk,ymin_chunk], [0, ymin_chunk]]), color='red', alpha=0.4)
ax.add_patch(segment_square)

ax.relim()
ax.autoscale_view()

plt.show()

wkt2 = f"POLYGON ((0 0, {xmax_chunk} 0, {xmax_chunk} {ymin_chunk}, 0 {ymin_chunk}, 0 0))"


poly1 = ogr.CreateGeometryFromWkt(shapes.iloc[4,2])
poly2 = ogr.CreateGeometryFromWkt(wkt2)

intersection = poly1.Intersection(poly2)

int_wkt =  intersection.ExportToWkt()
clipped_poly = loads(int_wkt)

fig, ax = plt.subplots(figsize=(9, 9))

for polygon in clipped_poly:
    mpl_poly = Polygon(np.array(polygon.exterior))
    ax.add_patch(mpl_poly)

ax.relim()
ax.autoscale_view()

plt.show()'''

def get_pixel_coords(coords, size, x_range, y_range):
    '''
    Returns coordinates of vertices as pixel locations, given target image size
    '''

    # shift coords top left to 0,0
    coords[:, 0] -= x_range[0]
    coords[:, 1] -= y_range[0]

    # scale coords to new size
    coords[:, 0] *= size[0] / (x_range[1]-x_range[0])
    coords[:, 1] *= size[1] / (y_range[1]-y_range[0])


    # pixel locations must be int
    coords = np.round(coords).astype(int)

    return coords


def make_vertices_lists(polygons, x_range, y_range, size=[256, 256]):
    if not polygons:
        return [], []
    
    format_coords = lambda x: np.array(list(x)).astype(np.float)

    print(type(polygons))
    
    if isinstance(polygons, shapely.geometry.polygon.Polygon):
        return ([get_pixel_coords(format_coords(polygons.exterior.coords),
                    size, x_range, y_range)],
               [])

    perimeters = [get_pixel_coords(format_coords(polygon.exterior.coords),
                    size, x_range, y_range)
                    for polygon in polygons]
    
    interiors = [get_pixel_coords(format_coords(polygon.interior.coords),
                    size, x_range, y_range)
                    for exterior in polygons
                    for polygon in exterior.interiors]
    
    return perimeters, interiors

def make_mask(size, perimeters, interiors):
    '''
    Returns a binary mask image given perimeters of shapes and interiors (holes)
    '''

    # initialize empty mask
    mask = np.zeros(tuple(size))

    if not perimeters:
        return mask
    
    # fill shapes with 1 (true) and then refill holes with 0 (false)
    cv2.fillPoly(mask, perimeters, 1)
    cv2.fillPoly(mask, interiors, 0)

    return mask

def make_warp(img1, img2):
    '''
    Returns a warp matrix that aligns image 1 to image 2

    image 1 can then be warped onto image 2

    Assume: Image 2 is the larger image
    '''

    shape1 = img1.shape
    shape2 = img2.shape

    x1 = shape1[0]
    y1 = shape1[1]

    x2 = shape2[0]
    y2 = shape2[1]

    if shape1 != shape2:
        img1 = cv2.resize(img1, (shape2[1], shape2[0]),
                          interpolation=cv2.INTER_CUBIC)

    img1 = img1[int(x1*0.2):int(x1*0.8), int(y1*0.2):int(y1*0.8),:]
    img1 = img1[int(x2*0.2):int(x2*0.8), int(y2*0.2):int(y2*0.8),:]
    
    img1 = rescale_image_values(img1)*255
    img2 = rescale_image_values(img2)*255

    img1 = np.sum(img1, axis=2)
    img2 = np.sum(img2, axis=2)

    matrix = np.eye(2,3)

    _, matrix = cv2.findTransformECC(img2, img1, matrix,
                                     motionType=cv2.MOTION_EUCLIDEAN)
    
    return matrix


def break_shapes(wkt, size=[3391, 3349], x_width=xmax, y_height=ymin, filename="6040_2_2", graph=False):
    '''saves 169 256x256 binary masks for a corresponding image'''
    parent_folder = 'data/shapes'
    if not isdir(parent_folder):
        mkdir(parent_folder)

    out_folder = f'data/shapes/{filename}'
    if not isdir(out_folder):
        mkdir(out_folder)
    
    # find appropriate chunk size within x_width and y_height
    x_chunk = x_width*256/size[0]
    y_chunk = y_height*256/size[1]

    counter = 0
    for i in range(13):
        for j in range(13):
            print(counter)
            xmin = i*x_chunk
            xmax = (i+1)*x_chunk
            ymin = j*y_chunk
            ymax = (j+1)*y_chunk

            wkt2 = f"POLYGON (({i*x_chunk} {j*y_chunk}, {(i+1)*x_chunk} {j*y_chunk}, {(i+1)*x_chunk} {(j+1)*y_chunk}, {i*x_chunk} {(j+1)*y_chunk}, {i*x_chunk} {j*y_chunk}))"
            poly1 = ogr.CreateGeometryFromWkt(wkt)
            poly2 = ogr.CreateGeometryFromWkt(wkt2)

            intersection = poly1.Intersection(poly2)

            int_wkt =  intersection.ExportToWkt()

            # save wkt directly. not desired
            '''
            with open(join(out_folder, 'wkt'+str(counter)+'.pkl'), 'wb') as open_file:
                pkl.dump(int_wkt, open_file)
            '''

            # TODO: save binary mask image
            perimeters, interiors = make_vertices_lists(loads(int_wkt), x_range=[xmin, xmax], y_range=[ymin, ymax])
            mask = make_mask([256, 256], perimeters, interiors)

            with open(join(out_folder, f'mask{counter}.pkl'), 'wb') as open_file:
                pkl.dump(mask, open_file)
            

            # graph each shapefile for examining correctness
            if graph == True:
                plt.imshow(mask)
                plt.show()

                clipped_poly = loads(int_wkt)
                fig, ax = plt.subplots(figsize=(9, 9))

                for polygon in clipped_poly:
                    mpl_poly = Polygon(np.array(polygon.exterior))
                    ax.add_patch(mpl_poly)

                ax.relim()
                ax.autoscale_view()

                plt.show()
            
            counter +=1

def jaccard(img1, img2):
    intersection = np.sum(img1 * img2)
    union = np.sum(img1+img2) - intersection

    if not union:
        return 1

    return 1 - (intersection / union)    

def train_model(x):
    save_images = {}

    # GO DEEP
    layer_1 = slim.conv2d(x, 64, [3, 3])
    layer_2 = slim.conv2d(layer_1, 64, [3, 3])

    layer_3 = slim.max_pool2d(layer_2)

    # GO DEEPER
    layer_4 = slim.conv2d(layer_3, 128, [3, 3])
    layer_5 = slim.conv2d(layer_4, 128, [3, 3])

    # BRING IT BACK
    layer_6 = slim.conv2d_transpose(layer_5, 64, [2, 2], 2)
    combined_layers = tf.concat(layer_2, layer_6)
    layer_7 = slim.conv2d(combined_layers, 64, [3, 3])
    layer_8 = slim.conv2d(layer_7, 64, [3, 3])

    layer_9 = slim.conv2d(layer_8)
    pred = tf.sigmoid(layer_9)

    return pred

def train_keras_model(x, y):
    with tf.device('/gpu:0'):
        inputs = Input((256,256,3))

        layer_1 = Conv2D(64, [3, 3], activation='elu', padding='same')(inputs)
        layer_2 = Conv2D(64, [3, 3], activation='elu', padding='same')(layer_1)

        layer_3 = MaxPooling2D(pool_size=(2,2))(layer_2)

        layer_4 = Conv2D(128, [3, 3], activation='elu', padding='same')(layer_3)
        layer_5 = Conv2D(128, [3, 3], activation='elu', padding='same')(layer_4)

        layer_6 = MaxPooling2D(pool_size=(2,2))(layer_5)

        layer_7 = Conv2D(256, [3, 3], activation='elu', padding='same')(layer_6)
        layer_8 = Conv2D(256, [3, 3], activation='elu', padding='same')(layer_7)

        layer_9 = MaxPooling2D(pool_size=(2,2))(layer_8)

        layer_10 = Conv2D(512, [3, 3], activation='elu', padding='same')(layer_9)
        layer_11 = Conv2D(512, [3, 3], activation='elu', padding='same')(layer_10)

        final_depth = MaxPooling2D(pool_size=(2,2))(layer_11)
        final_conv = Conv2D(1024, [3,3], activation='elu', padding='same')(final_depth)
        final_conv2 = Conv2D(1024, [3,3], activation='elu', padding='same')(final_conv)

        final_upconv = Conv2DTranspose(512, (2,2), strides=(2,2), padding='same')(final_conv2)
        final_con = concatenate([layer_10, final_upconv])
        final_re = Conv2D(512, [3, 3], activation='elu', padding='same')(final_con)
        final_re2 = Conv2D(512, [3, 3], activation='elu', padding='same')(final_re)

        layer_12 = Conv2DTranspose(256, (2,2), strides=(2,2), padding='same')(final_re2)
        layer_13 = concatenate([layer_7, layer_12])
        layer_14 = Conv2D(256, [3, 3], activation='elu', padding='same')(layer_13)
        layer_15 = Conv2D(256, [3, 3], activation='elu', padding='same')(layer_14)
        
        layer_16 = Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(layer_15)
        layer_17 = concatenate([layer_4, layer_16])
        layer_18 = Conv2D(128, [3, 3], activation='elu', padding='same')(layer_17)
        layer_19 = Conv2D(128, [3, 3], activation='elu', padding='same')(layer_18)

        layer_20 = Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(layer_19)
        layer_21 = concatenate([layer_1, layer_20])
        layer_22 = Conv2D(64, [3, 3], activation='elu', padding='same')(layer_21)
        layer_23 = Conv2D(64, [3, 3], activation='elu', padding='same')(layer_22)

        outputs = Conv2D(1, (1,1), activation='sigmoid')(layer_23)

        model = Model(inputs=[inputs], outputs=[outputs])

        stale = EarlyStopping(monitor='loss', patience=3)

        model.compile(optimizer='adam', loss='binary_crossentropy')
        model.fit(x=x, y=y, epochs=10, callbacks=[stale], batch_size=13)

    return model

def rescale_image_values(img):
    shape1 = img.shape
    img = np.reshape(img,
                    [shape1[0] * shape1[1], shape1[2]]
                    ).astype(np.float32)

    min_ = np.percentile(img, 1, axis=0)
    max_ = np.percentile(img, 99, axis=0) - min_

    img = (img - min_) / max_

    img.clip(0., 1.)
    img = np.reshape(img, shape1)

    return img

def get_image_IDs():
    with open('data/train_wkt_v4.csv', 'r') as open_file:
        data = pd.read_csv(open_file)

    IDs = data['ImageId'].unique()

    return IDs

def save_multiband_image(image_id):
    paths = dict()
    paths['RGB'] = f'data/three_band/{image_id}.tif'
    paths['A'] = f'data/sixteen_band/{image_id}_A.tif'
    paths['M'] = f'data/sixteen_band/{image_id}_M.tif'
    paths['P'] = f'data/sixteen_band/{image_id}_P.tif'

    images = dict()
    for key, path in paths.items():
        ds = gdal.Open(path, GA_ReadOnly)
        image = ds.ReadAsArray()
        images[key] = image
    
    target_shape = images['RGB'].shape

    # calculate warp matrix
    warps = dict()
    for key, image in images.items():
        if key == 'RGB':
            continue
        warps[key] = make_warp(image, images['RGB'])
    
    # perform transform
    warped_images = dict()
    for key, warp in warps.items():
        warped_images[key] = cv2.warpAffine(images[key], warp,
                                            (target_shape[1], target_shape[0]),
                                            flags= cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP,
                                            borderMode = cv2.BORDER_REPLICATE)
    
    # concatenate images
    out_image = rescale_image_values(images['RGB'])
    for image in warped_images.values():
        res_image = rescale_image_values(image)
        out_image = np.concatenate((out_image, res_image), axis=2)
    
    parent_folder = 'data/combined_images'
    if not isdir(parent_folder):
        mkdir(parent_folder)

    out_path = f'data/combined_images/{image_id}.npz'
    np.savez(out_path, out_image)

    return


if __name__ == '__main__':
    image_IDs = get_image_IDs()
    for ID in image_IDs:
        save_multiband_image(ID)


    '''with open('data/train_wkt_v4.csv', 'r') as open_file:
        shapes = pd.read_csv(open_file)
    break_shapes(shapes.iloc[4,2], graph=False)'''

    '''x = []
    y = []

    for i in range(169):
        with open(f'data/shapes/6040_2_2/mask{i}.pkl', 'rb') as open_file:
            k = pkl.load(open_file)
            assert k.shape == (256, 256)
            k = np.expand_dims(k, axis=2)
            y.append(k)
        with open(f'data/three_band/clipped/6040_2_2/image{i}', 'rb') as open_file:
            k = pkl.load(open_file)
            assert k.shape == (3, 256, 256)
            k = np.moveaxis(k, 0, 2)
            x.append(k)

    #model = train_keras_model(np.asarray(x), np.asarray(y))

    #v = model.predict(np.asarray(x[:2]))
    print(np.sum(x[0], axis=2))

    sub_image = 18

    im = rescale_image_values(x[sub_image])

    print(im)

    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(im)
    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(im)
    ax2.imshow(y[sub_image][:,:,0], alpha = 0.2, cmap='Greys')
    plt.show()'''