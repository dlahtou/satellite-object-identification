import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly
from keras.models import load_model
import tensorflow as tf
from keras import backend as K

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

def predict_mask():
    a = np.load('data/clipped_masks/6040_2_2_mask_157.npy')
    b = np.load('data/clipped_images/6040_2_2_clip_157.npy')

    print(a.shape)
    model = load_model('my_model.h5', custom_objects={'mean_iou': mean_iou})
    
    k = model.predict(np.asarray([a]))
    rak = k[0]

    np.save('6040_2_2_pred_157.npy', rak)

predict_mask()