import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import numpy as np

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

def train_keras_model(x, y, n_channels=20, save_filepath='unnamed_model.h5'):
    with tf.device('/gpu:0'):
        inputs = Input((256,256,n_channels))

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

        checkpoint_model = ModelCheckpoint(save_filepath, verbose=1, save_best_only=True)        

        #adam = Adam(lr=0.01)

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])

        model.fit(x=x, y=y, epochs=8, callbacks=[stale, checkpoint_model], batch_size=4, validation_split=0.1)

        model.save(save_filepath)

    return model
