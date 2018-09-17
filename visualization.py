import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly
from keras.models import load_model

def show_side_by_side():
    a = np.load('/home/dlahtou/6040_2_2_mask_157.npy')
    b = np.load('/home/dlahtou/6040_2_2_clip_157.npy')

    print(a.shape)

    fig = plt.figure()
    fig.add_subplot(1,3,3)
    plt.imshow(np.squeeze(a, 2), cmap='Reds', alpha=0.8)
    #plt.axis('off')
    plt.axis('off')
    plt.title('Output Labels')
    fig.add_subplot(1,3,1)
    plt.imshow(b[:,:,:3])
    plt.tick_params(
        axis='y',
        which='both', 
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.xticks([0,256], ['0', '80m'])
    plt.title('Raw Input')
    fig.add_subplot(1,3,2)
    plt.imshow(b[:,:,:3])
    plt.imshow(np.squeeze(a, 2), cmap='Reds', alpha=0.5)
    plt.axis('off')
    plt.title('Identify Target')
    fig.savefig('TEST.png', transparent=True)
    plt.show()

#show_side_by_side()
def show_side_by_side2(pred='6040_2_2_pred_157.npy', mask='/home/dlahtou/6040_2_2_mask_157.npy', clip='/home/dlahtou/6040_2_2_clip_157.npy', save_path=None):
    if isinstance(pred, str):
        a = np.load(pred)
        b = np.load(mask)
        c = np.load(clip)
        a = a > 0.121
    else:
        a = pred
        b = mask
        c = clip

    fig = plt.figure()
    fig.add_subplot(1,3,3)
    plt.imshow(np.squeeze(a, 2), cmap='Reds', alpha=0.8)
    #plt.axis('off')
    plt.axis('off')
    plt.title('Predicted Labels')
    fig.add_subplot(1,3,1)
    plt.imshow(c[:,:,:3], cmap='Greens')
    plt.tick_params(
        axis='y',
        which='both', 
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.xticks([0,256], ['0', '80m'])
    plt.title('Raw Input')
    fig.add_subplot(1,3,2)
    plt.imshow(c[:,:,:3], cmap='Greens')
    plt.imshow(np.squeeze(b, 2), cmap='Reds', alpha=0.5)
    plt.axis('off')
    plt.title('Ground Truth Mask + Raw Input')
    if save_path:
        fig.savefig(save_path, transparent=True)
    else:
        plt.show()

#show_side_by_side2()

'''preds = np.load('/home/dlahtou/trees_predicts.npy')
masks = np.load('/home/dlahtou/trees_predicts.npy')
clips = np.load('trees_images.npy')

for i in range(20):
    show_side_by_side2(preds[i], masks[i], clips[i])'''


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

def rescale2(img):
    shape1 = img.shape
    
    totals = np.sum(img, axis=2)
    print(totals.shape)

    img = img / totals[:,:, np.newaxis]

    return img
'''
ds = gdal.Open('data/three_band/6040_2_2.tif', GA_ReadOnly)
ds_array = ds.ReadAsArray()

ds_array = np.transpose(ds_array, (1,2,0))
ds_array = rescale_image_values(ds_array)

mask = np.load('data/masks/Trees/6040_2_2_mask.npy')

fig = plt.figure(figsize=[15,15])
plt.imshow(ds_array)
plt.imshow(mask, cmap='Greens', alpha=0.5)

plt.axis('off')
fig.savefig('full_sat_image_with_mask.png', transparent=True)
'''