# coding=utf-8

"""
Created by jayveehe on 2017/9/2.
http://code.dianpingoa.com/hejiawei03
"""
import gzip

import keras
import numpy as np
import os


def test_predictions(gzip_file_path_list, model_path, result_output='keras_result.txt'):
    pred_model = keras.models.load_model(model_path)
    mean_rank_list = []
    rank_rate_list = []
    if not result_output:
        res_output = open(result_output, 'w')
        res_output.write('gzip_file_path, mean_rank, rank_rate\n')
    for i in xrange(len(gzip_file_path_list)):
        gzip_file_path = gzip_file_path_list[i]
        # label_path = label_path_list[i]
        # print 'current file: %s'%gzip_file_path
        with gzip.open(gzip_file_path, 'rb') as gzip_in:
            true_score_list = []
            pred_score_list = []
            index = 0
            for gzip_line in gzip_in:
                input_vec = [float(a) for a in gzip_line.split(',')]
                stock_id = long(input_vec[0])
                stock_score = input_vec[1]
                feature_vec = input_vec[2:]
                label_vec = stock_score
                pred_label = pred_model.predict(feature_vec, batch_size=1)
                pred_score_list.append((index, pred_label, stock_id))
                true_score_list.append((index, stock_score, stock_id))
                index += 1
            mean_rank, rank_rate = test_rank(true_score_list, pred_score_list, topk=50)
            mean_rank_list.append(mean_rank)
            rank_rate_list.append(rank_rate)
            print '%s \tmean rank: %s\trank rate: %s' % (gzip_file_path, mean_rank, rank_rate)
            if result_output:
                res_output.write('%s,%s,%s\n' % (gzip_file_path, mean_rank, rank_rate))
    print 'Total file: %s\nmean: %s\nstd: %s' % (
        len(gzip_file_path_list), np.mean(rank_rate_list), np.std(rank_rate_list))


def test_rank(true_score_list, pred_score_list, topk=50):
    true_score_list.sort(key=lambda x: x[1], reverse=True)
    pred_score_list.sort(key=lambda x: x[1], reverse=True)
    stock_rank_dict = {}
    for ts in true_score_list:
        stock_rank_dict[ts[2]] = ts
    # calc rank error
    rank_list = []
    for ps in pred_score_list[:topk]:
        rank_list.append(stock_rank_dict[ps[2]][0])  # 根据pred score排名前topk，取其真实排名
    mean_rank = np.mean(rank_list)
    rank_rate = mean_rank / (len(true_score_list) + 0.0)
    return mean_rank, rank_rate


if __name__ == '__main__':
    test_file_numbers = range(540, 640) + range(800, 845) + range(920, 945) + range(1020, 1045) + range(1200, 1214)
    DATA_ROOT = '/media/user/Data0/hjw/datas/Quant_Datas_v3.0/gzip_datas'
    test_filepath_list = [os.path.join(DATA_ROOT, '%s_trans_norm.gz' % fn) for fn in test_file_numbers]
    test_predictions(test_filepath_list, model_path='keras_model.mod')
