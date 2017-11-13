# coding=utf-8

"""
Created by jayveehe on 2017/9/23.
http://code.dianpingoa.com/hejiawei03
"""
import gzip

import numpy as np


def gzip_sample_generator(gzip_file_path_list, batch_size=100, total_limit=1000, per_file_limit=500):
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
                        stock_id = int(input_vec[0])
                        stock_score = input_vec[1]
                        feature_vec = input_vec[2:]
                        label_vec = stock_score
                        batch_feature_list.append(feature_vec)
                        batch_label_list.append(label_vec)
                        # yield feature_vec, label_vec
                        count += 1
                        total_count += 1
                        # print total_count
                        # single_file_sample_limit -= 1
                        if len(batch_feature_list) == batch_size:
                            count = 0
                            batch_feature_array = np.array(batch_feature_list)
                            batch_feature_array = batch_feature_array.reshape(batch_size, 4561)
                            batch_label_array = np.array(batch_label_list)
                            # batch_label_array = batch_label_list
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
