# coding=utf-8

"""
Created by jayveehe on 2017/8/27.
http://github.com/jayveehe
"""
from keras import Input
from keras.engine import Model
from keras.layers import Embedding, Activation, Dense, Reshape


def build_model(feature_dim=4560, output_dim=1):
    input_layer = Input(shape=(feature_dim,))
    # embedded_512 = Embedding(input_dim=feature_dim, output_dim=512)(input_layer)
    # reshape_vec = Reshape(target_shape=(1, 512))(embedded_512)
    fc_256 = Dense(256, activation='tanh')(input_layer)
    fc_64 = Dense(64, activation='tanh')(fc_256)
    output_layer = Dense(output_dim, activation='sigmoid')(fc_64)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['mse'])
    return model
