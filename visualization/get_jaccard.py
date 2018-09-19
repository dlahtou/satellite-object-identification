import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.models import load_model


def get_jaccard_vals(img1, img2):
    intersection = np.sum(np.multiply(img1, img2))
    union = np.sum(np.add(img1, img2)) - intersection

    return intersection, union

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

def save_jaccards():
    print('loading model')
    model = load_model('unnamed_model.h5', custom_objects={'mean_iou': mean_iou})

    masks = np.load('Buildings_masks.npy')
    images = np.load('Buildings_images.npy')

    print('making predictions')
    predicts = model.predict(images)
    np.save('Buildings_predicts.npy', predicts)

    predicts = predicts > 0.5

    predicts = predicts.astype(int)
    masks = masks.astype(int)

    print('calculating jaccards')
    jaccard_ints = []
    jaccard_unions = []
    for images in zip(predicts, masks):
        intersection, union = get_jaccard_vals(images[0], images[1])

        jaccard_ints.append(intersection)
        jaccard_unions.append(union)

    print(f'jaccard index: {sum(jaccard_ints) / sum(jaccard_unions)}')

    np.save('jaccards.npy', np.asarray([jaccard_ints, jaccard_unions]))

if __name__ == '__main__':
    save_jaccards()
