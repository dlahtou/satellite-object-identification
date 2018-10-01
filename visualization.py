import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly
from keras.models import load_model

threshold = 0.5

def show_side_by_side():
    a = np.load('/home/dlahtou/6040_2_2_mask_157.npy')
    b = np.load('/home/dlahtou/6040_2_2_clip_157.npy')

    print(a.shape)

    fig = plt.figure()
    fig.add_subplot(1,3,3)
    plt.imshow(np.squeeze(a, 2), cmap='Greens', alpha=0.8)
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
    plt.imshow(np.squeeze(a, 2), cmap='Greens', alpha=0.5)
    plt.axis('off')
    plt.title('Identify Target')
    fig.savefig('TEST.png', transparent=True)
    plt.show()

#show_side_by_side()
def show_side_by_side2(pred='6040_2_2_pred_157.npy', mask='/home/dlahtou/6040_2_2_mask_157.npy', clip='/home/dlahtou/6040_2_2_clip_157.npy', save_path=None, image_num='NO_NUM'):
    if isinstance(pred, str):
        a = np.load(pred)
        b = np.load(mask)
        c = np.load(clip)
        #a = b
        #a = a > 0.121
    else:
        a = pred
        b = mask
        c = clip
        a = a #> threshold

    fig = plt.figure(figsize=[15,10])
    fig.add_subplot(1,3,3)
    plt.imshow(np.squeeze(a, 2), cmap='Greens', alpha=0.8)
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
    plt.title(f'Raw Input {image_num}')
    fig.add_subplot(1,3,2)
    plt.imshow(c[:,:,:3], cmap='Greens')
    plt.imshow(np.squeeze(b, 2), cmap='Greens', alpha=0.5)
    plt.axis('off')
    plt.title('Ground Truth Mask + Raw Input')
    if save_path:
        fig.savefig(save_path, transparent=True)
    else:
        plt.show()

'''show_side_by_side2(save_path='trees_sample_output.png', image_num='Image')

sss=False
showall_masks=False

preds = np.load('/home/dlahtou/Buildings_predicts2.npy')
masks = np.load('/home/dlahtou/Buildings_masks2.npy')
clips = np.load('/home/dlahtou/Buildings_images2.npy')'''

# show_side_by_side2(pred=preds[7], mask=masks[7], clip=clips[7], save_path='buildings_sample_output.png', image_num='Image')

'''if sss:
    for i in range(20):
        show_side_by_side2(preds[i], masks[i], clips[i], image_num=i)'''

def overlay_masks(pred, mask, image_num="NO IMAGE NUM PROVIDED"):
    image = np.zeros((pred.shape[0], pred.shape[1], 3))
    image[:,:,0] = np.squeeze(pred > threshold, 2)
    image[:,:,1] = np.squeeze(mask, 2) 

    fig = plt.figure()
    plt.imshow(image)
    plt.title(f'image {image_num}')
    plt.axis('off')
    plt.show()

def jaccard(img1, img2):
    intersection = np.sum(np.multiply(img1, img2))
    union = np.sum(np.add(img1, img2)) - intersection

    if not union:
        return 1

    return (intersection / union)   

def overlay_masks2(pred, mask, clip, save_path='overlay_sample_output.png'):
    pred = np.squeeze(pred > threshold, 2).astype(int)
    mask = np.squeeze(mask, 2).astype(int)

    print(np.unique(mask))
    print(np.unique(pred))

    print(f'jaccard index: {jaccard(pred, mask)}')

    fig = plt.figure(figsize=[15,10])
    fig.add_subplot(1,4,1)
    plt.imshow(clip)
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
    plt.title(f'Raw Input Image')

    fig.add_subplot(1,4,2)
    plt.imshow(mask, cmap='Greens')
    plt.axis('off')
    plt.title('Ground Truth Mask')

    fig.add_subplot(1,4,3)
    plt.imshow(pred, cmap='Reds')
    plt.axis('off')
    plt.title('Predicted Labels')

    # overlaid RG(B) masks
    fig.add_subplot(1, 4, 4)
    image = np.zeros((pred.shape[0], pred.shape[1], 3))
    image[:,:,0] = pred*0.8
    image[:,:,1] = mask*0.7
    plt.imshow(image)
    plt.axis('off')
    plt.title('Overlaid Mask + Predictions')

    fig.savefig(save_path, transparent=True)

    plt.show()

if showall_masks:
    for i in range(20):
        overlay_masks(preds[i], masks[i], image_num=i)

# overlay_masks2(preds[7], masks[7], clips[7])

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