# coding=utf-8

"""
Created by jayveehe on 2017/9/1.
http://code.dianpingoa.com/hejiawei03
"""
import gzip
import json
import os
import sys

import keras
import numpy as np
from keras.callbacks import ModelCheckpoint

from utils.data_utils import gzip_sample_generator

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print 'Related File:%s\t----------project_path=%s' % (__file__, PROJECT_PATH)
sys.path.append(PROJECT_PATH)

from utils.keras_utils import build_model
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu':0})))


def _gzip_sample_generator(gzip_file_path_list, batch_size=100, total_limit=1000, per_file_limit=500):
    while 1:
        count = 0
        total_count = 0
        batch_feature_list = []
        batch_label_list = []
        for i in xrange(len(gzip_file_path_list)):
            if total_count > total_limit:
                print 'total_count = %s, break' % total_count
                break
            gzip_file_path = gzip_file_path_list[i]
            # label_path = label_path_list[i]
            # print 'current file: %s'%gzip_file_path
            single_file_sample_limit = per_file_limit
            with gzip.open(gzip_file_path, 'rb') as gzip_in:
                try:
                    # print feature_path
                    for gzip_line in gzip_in:
                        input_vec = [float(a) for a in gzip_line.split(',')]
                        stock_id = long(input_vec[0])
                        stock_score = input_vec[1]
                        feature_vec = input_vec[2:]
                        label_vec = stock_score
                        batch_feature_list.append(feature_vec)
                        batch_label_list.append([label_vec])
                        # yield feature_vec, label_vec
                        count += 1
                        total_count += 1
                        # print total_count
                        # single_file_sample_limit -= 1
                        if len(batch_feature_list) == batch_size:
                            count = 0
                            batch_feature_array = np.array(batch_feature_list)
                            batch_feature_array = batch_feature_array.reshape(100, 4561)
                            batch_label_array = np.array(batch_label_list)
                            # batch_label_array = batch_label_array.reshape((100, 1, 1))
                            yield (batch_feature_array, batch_label_array)
                            batch_feature_list = []
                            batch_label_list = []
                            # if single_file_sample_limit < 0:
                            # batch_feature_list = []
                            # batch_label_list = []
                            # print 'reach single file limit'
                            # break
                except IOError, e:
                    print e
                    continue


def train_base_projection(model_output_path,
                          valid_limit=500,
                          former_model_path=None, epochs=100, batch_size=100, nb_worker=4,
                          mini_batch_size=3000, limit=2000):
    train_file_numbers = range(1, 540) + range(750, 800) + range(870, 920) + range(970, 1020) + range(1100, 1200) + range(1220, 1270)
    valid_file_numbers = range(400, 440) + range(700, 750) + range(845, 870) + range(945, 970) + range(1045, 1100)
    test_file_numbers = range(540, 640) + range(800, 845) + range(920, 945) + range(1020, 1045) + range(1200, 1214)
    DATA_ROOT = '/media/user/Data0/hjw/datas/Quant_Datas_v4.0/gzip_datas'
    train_filepath_list = [os.path.join(DATA_ROOT, '%s_trans.gz' % fn) for fn in train_file_numbers]
    valid_filepath_list = [os.path.join(DATA_ROOT, '%s_trans.gz' % fn) for fn in valid_file_numbers]
    basic_model = build_model(feature_dim=4561, output_dim=1)
    if former_model_path:
        basic_model =  keras.models.load_model(former_model_path)
        print 'loading model from %s'%former_model_path
        # basic_model =  basic_model.load_weights(former_model_path)
    # train_filepath_list = ['/Users/jayveehe/git_project/FundDataAnalysis/pipelines/datas/tmp_data/993_trans_norm.gz']
    # valid_filepath_list = ['/Users/jayveehe/git_project/FundDataAnalysis/pipelines/datas/tmp_data/993_trans_norm.gz']
    train_generator = _gzip_sample_generator(train_filepath_list, batch_size=10, total_limit=1000000,
                                             per_file_limit=10000)
    valid_generator = _gzip_sample_generator(valid_filepath_list, batch_size=10, total_limit=100000,
                                             per_file_limit=3000)
    checkpointer = ModelCheckpoint(filepath=model_output_path, verbose=1, save_best_only=True)
    early_stopper = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=0, mode='auto')
    basic_model.fit_generator(generator=train_generator,
                              nb_epoch=epochs,
                              steps_per_epoch=limit / batch_size,
                              # samples_per_epoch=limit,
                              validation_data=valid_generator,
                              validation_steps=valid_limit / batch_size,
                              # nb_val_samples=valid_limit,
                              nb_worker=nb_worker, pickle_safe=True, callbacks=[checkpointer, early_stopper])


if __name__ == '__main__':
    train_base_projection('keras_model_no_norm.mod', former_model_path=None,  valid_limit=100000, limit=1000000, batch_size = 100, nb_worker = 20)
