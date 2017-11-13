# coding=utf-8

"""
Created by jayveehe on 2017/8/27.
http://github.com/jayveehe
"""
from keras import Input, optimizers
from keras.engine import Model
from keras.layers import Embedding, Activation, Dense, Reshape, Dropout
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu':0})))


def build_model(feature_dim=4560, output_dim=1):
    input_layer = Input(shape=(feature_dim,))
    # embedded_512 = Embedding(input_dim=feature_dim, output_dim=512)(input_layer)
    # reshape_vec = Reshape(target_shape=(1, 512))(embedded_512)
    fc_2048 = Dense(2048, activation='tanh')(input_layer)
    fc_1024 = Dense(1024, activation='tanh')(fc_2048)
    dropout_1 = Dropout(0.5)(fc_1024)
    fc_512 = Dense(512, activation='tanh')(dropout_1)
    fc_512 = Dense(512, activation='tanh')(fc_512)
    fc_512 = Dense(512, activation='tanh')(fc_512)
    fc_512 = Dense(512, activation='tanh')(fc_512)
    fc_256 = Dense(256, activation='relu')(fc_512)
    dropout_2 = Dropout(0.5)(fc_256)
    fc_64 = Dense(64, activation='relu')(dropout_2)
    output_layer = Dense(output_dim, activation='sigmoid')(fc_64)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=optimizers.RMSprop(0.001, rho=0.9, epsilon=1e-06),
                  loss='binary_crossentropy',
                  metrics=['mse'])
    return model
