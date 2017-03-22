# coding=utf-8

"""
该脚本用于训练模型

Created by jayvee on 17/3/4.
https://github.com/JayveeHe
"""
import cPickle
import multiprocessing
import os

import sys

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print 'Related File:%s\t----------project_path=%s' % (__file__, PROJECT_PATH)
sys.path.append(PROJECT_PATH)

from pipelines.analysis_data import test_datas_wrapper
from pipelines.data_preprocess import load_csv_data
from utils.logger_utils import data_process_logger
from utils.model_utils import train_with_lightgbm
import cPickle
import numpy as np


def load_pickle_datas(tmp_pickle_path):
    with open(tmp_pickle_path, 'rb') as fin:
        pickle_data = cPickle.load(fin)
        return pickle_data


def train_lightGBM_new_data(train_file_number_list):
    """
    利用新数据(2899维)训练lightGBM模型

    Returns:

    """

    # train_file_number_list = range(1, 300)
    model_tag = 'Quant_Data_300_norm'
    # load with multi-processor
    process_count = 12
    proc_pool = multiprocessing.Pool(process_count)
    multi_results = []
    for i in train_file_number_list:
        # data_process_logger.info('loading %s file' % i)
        # csv_path = '%s/datas/Quant-Datas/pickle_datas/%s.csv' % (PROJECT_PATH, i)
        pickle_path = '%s/datas/Quant-Datas/pickle_datas/%s_trans_norm.pickle' % (PROJECT_PATH, i)
        data_res = proc_pool.apply_async(load_pickle_datas, args=(pickle_path,))
        multi_results.append(data_res)
        # datas = load_csv_data(csv_path, normalize=True, is_combine=True)
        # train_datas += datas
    proc_pool.close()
    proc_pool.join()
    # fetch datas from pool
    tmp_data = multi_results[0].get()
    train_datas = tmp_data
    data_process_logger.info('combining datas...')
    for i in xrange(1, len(multi_results)):
        datas = multi_results[i].get()
        train_datas = np.row_stack((train_datas, datas))
    output_lightgbm_path = '%s/models/lightgbm_%s.model' % (PROJECT_PATH, model_tag)
    # lightgbm_params = {'learning_rates': lambda iter_num: 0.05 * (0.99 ** iter_num)}
    train_with_lightgbm(train_datas, output_lightgbm_path, num_boost_round=30000, early_stopping_rounds=150,
                        learning_rates=lambda iter_num: max(0.8 * (0.99 ** iter_num), 0.0005))


def test_old_datas():
    model_tag = 'dart_iter30000_norm_combined_400000full'
    train_datas = []
    # load with multi-processor
    process_count = 12
    proc_pool = multiprocessing.Pool(process_count)
    multi_results = []
    for i in range(200, 401):
        # data_process_logger.info('loading %s file' % i)
        csv_path = '%s/datas/%s.csv' % (PROJECT_PATH, i)
        data_res = proc_pool.apply_async(load_csv_data, args=(csv_path, True, True))
        multi_results.append(data_res)
        # datas = load_csv_data(csv_path, normalize=True, is_combine=True)
        # train_datas += datas
    proc_pool.close()
    proc_pool.join()
    # fetch datas from pool
    data_process_logger.info('combining csv datas...')
    for i in xrange(len(multi_results)):
        datas = multi_results[i].get()
        train_datas += datas
        # dump normalized train datas
    data_process_logger.info('dumping norm datas...')
    cPickle.dump(train_datas, open('%s/datas/norm_datas/200_norm_combined_datas_full.dat' % PROJECT_PATH, 'wb'),
                 protocol=2)
    # load train normalized train datas
    # data_process_logger.info('loading datas...')
    # train_datas = cPickle.load(open('%s/datas/norm_datas/200_norm_datas_full.dat' % PROJECT_PATH, 'rb'))
    # random sample the train datas
    # SAMPLE_SIZE = 20000
    # data_process_logger.info('random sampling %s obs...' % SAMPLE_SIZE)
    # random.shuffle(train_datas)
    # train_datas = train_datas[:SAMPLE_SIZE]
    # ------- start training ---------
    # output_gbrt_path = '%s/models/gbrt_%s.model' % (PROJECT_PATH, model_tag)
    # train_gbrt_model(train_datas, output_gbrt_path)

    # output_svr_path = '%s/models/svr_%s.model' % (PROJECT_PATH, model_tag)
    # train_svr_model(train_datas, output_svr_path)

    output_lightgbm_path = '%s/models/lightgbm_%s.model' % (PROJECT_PATH, model_tag)
    # lightgbm_params = {'learning_rates': lambda iter_num: 0.05 * (0.99 ** iter_num)}
    train_with_lightgbm(train_datas, output_lightgbm_path, num_boost_round=30000, early_stopping_rounds=150,
                        learning_rates=lambda iter_num: max(0.8 * (0.99 ** iter_num), 0.0005))
    # --------- Testing -------
    data_process_logger.info('--------LightGBM:----------')
    data_process_logger.info('using model: %s/models/lightgbm_%s.model' % (PROJECT_PATH, model_tag))
    lightgbm_mod = cPickle.load(open('%s/models/lightgbm_%s.model' % (PROJECT_PATH, model_tag), 'rb'))
    # data_process_logger.info('test trianing file')
    # test_datas_wrapper(range(1,100),lightgbm_mod)
    data_process_logger.info('test test file')
    # print  list(lightgbm_mod.feature_importances_)
    test_datas_wrapper([100, 150, 200, 310], lightgbm_mod, is_combined=True, normalize=True)
    test_datas_wrapper(range(500, 551), lightgbm_mod, is_combined=True, normalize=True)


if __name__ == '__main__':
    pass
    train_lightGBM_new_data(range(1, 301))
