import pickle as pkl
from os import listdir, mkdir
from os.path import isdir, isfile, join

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import (BatchNormalization, Concatenate, Conv2D,
                          Conv2DTranspose, Dense, Input, MaxPooling2D)
from keras.layers.core import Dropout
from keras.layers.merge import concatenate
from keras.models import Model, load_model
from matplotlib.patches import Polygon
from osgeo import gdal, ogr
from osgeo.gdalconst import GA_ReadOnly
from shapely.wkt import loads
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

with open('data/grid_sizes.csv', 'r') as open_file:
    grids = pd.read_csv(open_file)

with open('data/train_wkt_v4.csv', 'r') as open_file:
    shapes = pd.read_csv(open_file)

grids.set_index('Unnamed: 0', inplace=True)

target_image = '6040_2_2'

xmax = grids.loc[target_image, 'Xmax']
ymin = grids.loc[target_image, 'Ymin']

'''xmax_chunk = xmax/10
ymin_chunk = ymin/10

# get wkt shapes for trees in the first image
wkt = loads(shapes.iloc[4,2])

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
    coords[:, 0] *= size[1] / (x_range[1]-x_range[0])
    coords[:, 1] *= size[0] / (y_range[1]-y_range[0])

    # pixel locations must be int
    coords = np.round(coords).astype(int)

    return coords

def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

def combined_dice_ce(y_true, y_pred):
    smooth = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice_loss = - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth) 
    return dice_loss + binary_crossentropy(y_pred, y_true)

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
                    for polygon in exterior.interiors
                    if not isinstance(polygon, shapely.geometry.polygon.LinearRing)]
    
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

    print(f'original shape {shape1}')

    x1 = shape1[0]
    y1 = shape1[1]

    x2 = shape2[0]
    y2 = shape2[1]

    if shape1 != shape2:
        img1 = cv2.resize(img1, (shape2[1], shape2[0]),
                          interpolation=cv2.INTER_CUBIC)

    img1 = img1[int(x1*0.2):int(x1*0.8), int(y1*0.2):int(y1*0.8),:]
    img2 = img2[int(x2*0.2):int(x2*0.8), int(y2*0.2):int(y2*0.8),:]

    print(f'img1 warp shape: {img1.shape}')
    print(f'output warp shape: {img2.shape}')
    
    img1 = rescale_image_values(img1)*255
    img2 = rescale_image_values(img2)*255

    img1 = np.sum(img1, axis=2)
    img2 = np.sum(img2, axis=2)

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    matrix = np.eye(2,3).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-7)
    try:
        _, matrix = cv2.findTransformECC(img2, img1, matrix,
                                         motionType=cv2.MOTION_EUCLIDEAN, criteria=criteria)
    except:
        pass
    
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


def train_keras_model(x, y):
    with tf.device('/gpu:0'):
        inputs = Input((256,256,20))

        layer_1 = Conv2D(64, [3, 3], activation='relu', padding='same')(inputs)
        layer_2 = Conv2D(64, [3, 3], activation='relu', padding='same')(layer_1)

        layer_3 = MaxPooling2D(pool_size=(2,2))(layer_2)
        layer_4 = Conv2D(128, [3, 3], activation='relu', padding='same')(layer_3)
        layer_5 = Conv2D(128, [3, 3], activation='relu', padding='same')(layer_4)

        layer_6 = MaxPooling2D(pool_size=(2,2))(layer_5)
        layer_7 = Conv2D(256, [3, 3], activation='relu', padding='same')(layer_6)
        layer_8 = Conv2D(256, [3, 3], activation='relu', padding='same')(layer_7)

        layer_9 = MaxPooling2D(pool_size=(2,2))(layer_8)
        layer_10 = Conv2D(512, [3, 3], activation='relu', padding='same')(layer_9)
        layer_11 = Conv2D(512, [3, 3], activation='relu', padding='same')(layer_10)

        final_depth = MaxPooling2D(pool_size=(2,2))(layer_11)
        final_conv = Conv2D(1024, [3,3], activation='relu', padding='same')(final_depth)
        final_conv2 = Conv2D(1024, [3,3], activation='relu', padding='same')(final_conv)

        final_upconv = Conv2DTranspose(512, (2,2), strides=(2,2), padding='same')(final_conv2)
        final_con = concatenate([layer_10, final_upconv])
        final_re = Conv2D(512, [3, 3], activation='relu', padding='same')(final_con)
        final_re2 = Conv2D(512, [3, 3], activation='relu', padding='same')(final_re)

        layer_12 = Conv2DTranspose(256, (2,2), strides=(2,2), padding='same')(final_re2)
        layer_13 = concatenate([layer_7, layer_12])
        layer_14 = Conv2D(256, [3, 3], activation='relu', padding='same')(layer_13)
        layer_15 = Conv2D(256, [3, 3], activation='relu', padding='same')(layer_14)
        
        layer_16 = Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(layer_15)
        layer_17 = concatenate([layer_4, layer_16])
        layer_18 = Conv2D(128, [3, 3], activation='relu', padding='same')(layer_17)
        layer_19 = Conv2D(128, [3, 3], activation='relu', padding='same')(layer_18)

        layer_20 = Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(layer_19)
        layer_21 = concatenate([layer_1, layer_20])
        layer_22 = Conv2D(64, [3, 3], activation='relu', padding='same')(layer_21)
        layer_23 = Conv2D(64, [3, 3], activation='relu', padding='same')(layer_22)

        outputs = Conv2D(1, (1,1), activation='sigmoid')(layer_23)

        model = Model(inputs=[inputs], outputs=[outputs])

        stale = EarlyStopping(patience=3, verbose=1)

        checkpoint_model = ModelCheckpoint('trees_model.h5', verbose=1, save_best_only=True)

        

        #adam = Adam(lr=0.01)

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])

        model.fit(x=x, y=y, epochs=8, callbacks=[stale, checkpoint_model], batch_size=4, validation_split=0.1)

        model.save('trees_model.h5')

    return model

def rescale_image_values(img):
    shape1 = img.shape
    if len(shape1) == 3:
        img = np.reshape(img,
                        [shape1[0] * shape1[1], shape1[2]]
                        ).astype(np.float32)
    elif len(shape1) == 2:
        img = np.reshape(img, [shape1[0] * shape1[1]]).astype(np.float32)

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

    for path in paths.values():
        print(path)
        assert isfile(path)

    images = dict()
    for key, path in paths.items():
        ds = gdal.Open(path, GA_ReadOnly)
        image = ds.ReadAsArray()
        if key == 'P':
            images[key] = np.expand_dims(image, 2)
        else:
            images[key] = np.transpose(image, (1, 2, 0))
        
        print(f'{key} shape: {images[key].shape}')

    target_shape = images['RGB'].shape


    '''# calculate warp matrix
    warps = dict()
    for key, image in images.items():
        if key == 'RGB' or key == 'P':
            continue
        print(f'warping {key}')
        warps[key] = make_warp(image, images['RGB'])
    
    # perform transform
    warped_images = dict()
    for key, warp in warps.items():
        warped_images[key] = cv2.warpAffine(images[key], warp,
                                            (target_shape[1], target_shape[0]),
                                            flags= cv2.INTER_LINEAR,
                                            borderMode = cv2.BORDER_REPLICATE)'''
    
    # concatenate images
    out_image = rescale_image_values(images['RGB'])
    print(target_shape)
    for key, image in images.items():
        #res_image = rescale_image_values(image)
        if key == 'RGB':
            continue
        resized_image = cv2.resize(image, (target_shape[1], target_shape[0]))
        print(resized_image.shape)
        if len(resized_image.shape) == 2:
            resized_image = np.expand_dims(resized_image, axis=2)
        out_image = np.concatenate((out_image, resized_image), axis=2)
    
    print(out_image.shape)
    
    parent_folder = 'data/combined_images'
    if not isdir(parent_folder):
        mkdir(parent_folder)

    out_path = f'data/combined_images/{image_id}.npz'
    np.savez(out_path, out_image)

    return

def make_masks(target_class='Trees'):
    classes = {'Trees': 5,
                'Buildings': 1}


    image_IDs = get_image_IDs()

    with open('data/train_wkt_v4.csv', 'r') as open_file:
        shapes = pd.read_csv(open_file)
    
    with open('data/grid_sizes.csv', 'r') as open_file:
        grids = pd.read_csv(open_file)
    grids.set_index('Unnamed: 0', inplace=True)

    parent_folder = f'data/masks/{target_class}'
    if not isdir(parent_folder):
        mkdir(parent_folder)

    for image_id in image_IDs:
        image_shapes = shapes.loc[(shapes['ImageId'] == image_id) & (shapes['ClassType'] == classes[target_class])]['MultipolygonWKT'].values[0]

        xmax = grids.loc[image_id, 'Xmax']
        ymin = grids.loc[image_id, 'Ymin']

        perimeters, interiors = make_vertices_lists(loads(image_shapes), [0, xmax], [ymin, 0], size=[3349, 3391])
        mask = make_mask([3349,3391], perimeters, interiors)

        # there's a bug somewhere, y axis is inverted
        mask = np.flip(mask, 0)

        print(f'saving mask {image_id} into {parent_folder}')
        print(f'mask shape: {mask.shape}')
        print(f'nonzero_values: {np.count_nonzero(mask)}')

        np.save(parent_folder + f'{image_id}_mask.npy', mask)
    
    return

def make_clipped_images(mask_type='Buildings', save=True, number=600, sq_dims=256, package=None):
    image_IDs = get_image_IDs()

    clipped_shapes_folder = f'data/clipped_masks/{mask_type}'
    if not isdir(clipped_shapes_folder):
        mkdir(clipped_shapes_folder)
    
    clipped_images_folder = f'data/clipped_images/{mask_type}'
    if not isdir(clipped_images_folder):
        mkdir(clipped_images_folder)

    return_masks = []
    return_images = []

    counter = 0
    for image_id in image_IDs:
        print(counter)

        if counter > number:
            break

        npz = np.load(f'data/combined_images/{image_id}.npz')
        image_array = npz['arr_0']
        print(f'image {image_id}: {image_array.shape}')
        
        try:
            image_mask = np.load(f'data/masks/{mask_type}/{image_id}_mask.npy')
        except:
            continue
        print(f'mask {image_id}: {image_mask.shape}')
        print(np.count_nonzero(image_mask))

        for i in range(13):
            for j in range(13):
                clipped_image = image_array[i*sq_dims:(i+1)*sq_dims, j*sq_dims:(j+1)*sq_dims, :]
                clipped_mask = image_mask[i*sq_dims:(i+1)*sq_dims, j*sq_dims:(j+1)*sq_dims]
                if np.count_nonzero(clipped_mask) == 0:
                    continue

                clipped_mask = np.expand_dims(clipped_mask, axis=2)

                if save:
                # save clipped image
                    np.save(f'{clipped_images_folder}/{image_id}_clip_{counter}.npy', clipped_image)

                    # save cwarped_imageslipped mask
                    np.save(f'{clipped_shapes_folder}/{image_id}_mask_{counter}.npy', clipped_mask)
                else:
                    return_images.append(clipped_image)
                    return_masks.append(clipped_mask)
                
                counter += 1
    
    return return_images[:number], return_masks[:number]

# stupid example where i make the "satellite images" as binary masks of green regions
def make_clipped_images_green_masks(mask_type="Trees", number=40):
    x, y = make_clipped_images(mask_type=mask_type, number=number, save=False)

    y = np.asarray(y)

    x = np.concatenate((np.zeros(y.shape), y, np.zeros(y.shape)), axis=3)

    return x, y

        
def run_big_model(mask_type='Buildings'):
    x = []
    y = []

    x_folder = 'data/clipped_images/Buildings'
    y_folder = 'data/clipped_masks/Buildings'

    counter = 0
    for file_ in listdir(x_folder):
        if counter > 600:
            break
        x.append(np.load(join(x_folder, file_)))
        counter += 1
    
    counter = 0
    for file_ in listdir(y_folder):
        if counter > 600:
            break
        y.append(np.load(join(y_folder, file_)))
        counter += 1

    model = train_keras_model(np.asarray(x), np.asarray(y))

if __name__ == '__main__':
    

    '''image_IDs = get_image_IDs()

    for ID in image_IDs:
        save_multiband_image(ID)'''

    '''a = np.load('data/combined_images/6040_2_2.npz')
    for i in range(3,19):
        np.save(f'band{i}.npy', a['arr_0'][:, :, i])'''

    #make_masks('Buildings')

    number = 40
    predict = True

    x, y = make_clipped_images_green_masks(number=40)

    x = np.asarray(x).astype(np.float32)
    y = np.asarray(y).astype(bool)

    if predict:
        model = train_keras_model(np.asarray(x), np.asarray(y))

        predicts = model.predict(x[:number])
        np.save('trees_predicts.npy', predicts)

        print(predicts)
    
    # at one point, I ran this to make a 19-channel image for each file
    #for ID in image_IDs:
    #    save_multiband_image(ID)

    #make_clipped_images()
    #break_shapes(shapes.iloc[4,2], graph=False)

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
