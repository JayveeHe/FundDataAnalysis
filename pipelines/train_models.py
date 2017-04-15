# coding=utf-8

"""
该脚本用于训练模型

Created by jayvee on 17/3/4.
https://github.com/JayveeHe
"""
from __future__ import division

import cPickle
import multiprocessing
import os
import random

import sys

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print 'Related File:%s\t----------project_path=%s' % (__file__, PROJECT_PATH)
sys.path.append(PROJECT_PATH)

from utils.logger_utils import data_process_logger
from utils.model_utils import train_with_lightgbm, cv_with_lightgbm
import cPickle
import numpy as np

DATA_ROOT = '/media/user/Data0/hjw'


# DATA_ROOT = '/Users/jayvee/CS/Python/FundDataAnalysis'


def load_pickle_datas(tmp_pickle_path):
    with open(tmp_pickle_path, 'rb') as fin:
        data_process_logger.info('processing %s' % tmp_pickle_path)
        pickle_data = cPickle.load(fin)
        return pickle_data


def train_lightGBM_new_data(train_file_number_list, train_params, former_model=None, output_lightgbm_path=None,
                            save_rounds=1000,
                            num_total_iter=100, process_count=12, cv_fold=None):
    """
    利用新数据(2899维)训练lightGBM模型

    Args:
        cv_fold: 是否进行CV,默认None,进行则填fold数
        train_params: 训练参数
        process_count: 并行进程数
        num_total_iter: 总迭代次数
        save_rounds: 存储的迭代间隔
        output_lightgbm_path: 模型的保存路径
        train_file_number_list: 训练文件序号列表
        former_model: 如果需要从已有模型继续训练,则导入

    Returns:

    """

    # train_file_number_list = range(1, 300)
    # load with multi-processor
    proc_pool = multiprocessing.Pool(process_count)
    multi_results = []
    for i in train_file_number_list:
        # data_process_logger.info('loading %s file' % i)
        # csv_path = '%s/datas/Quant-Datas/pickle_datas/%s.csv' % (PROJECT_PATH, i)
        data_root_path = '%s/datas/Quant-Datas-2.0' % (DATA_ROOT)
        pickle_path = '%s/pickle_datas/%s_trans_norm.pickle' % (data_root_path, i)
        data_process_logger.info('add file: %s' % pickle_path)
        data_res = proc_pool.apply_async(load_pickle_datas, args=(pickle_path,))
        multi_results.append(data_res)
        # datas = load_csv_data(csv_path, normalize=True, is_combine=True)
        # train_datas += datas
    proc_pool.close()
    proc_pool.join()
    # fetch datas from pool
    stock_ids, stock_scores, vec_values = multi_results[0].get()
    # train_datas = tmp_data
    label_list = stock_scores
    vec_list = vec_values
    data_process_logger.info('combining datas...')
    for i in xrange(0, len(multi_results)):
        data_process_logger.info('combining No.%s data' % i)
        try:
            stock_ids, stock_scores, vec_values = multi_results[i].get()
            # train_datas = np.row_stack((train_datas, datas)) # np.2darray
            # train_datas = np.vstack((train_datas, datas))
            # train_datas.extend(datas)
            # label_list.extend(stock_scores)
            for index in range(len(vec_values)):
                vec = vec_values[index]
                label = stock_scores[index]
                if len(vec) == len(vec_list[-1]):
                    vec_list.append(vec)
                    label_list.append(label)
                else:
                    print 'not equaling n_feature: %s' % len(vec)
        except Exception, e:
            data_process_logger.error('No.%s data failed, details=%s' % (i, str(e.message)))
            continue
    # 组装train_datas
    train_datas = (label_list, vec_list)
    if not output_lightgbm_path:
        model_tag = 'Quant_Data_%s_norm' % (len(train_file_number_list))
        output_lightgbm_path = '%s/models/lightgbm_%s.model' % (PROJECT_PATH, model_tag)
    # lightgbm_params = {'learning_rates': lambda iter_num: 0.05 * (0.99 ** iter_num)}
    # num_total_iter = 55
    train_params['num_threads'] = process_count
    if not cv_fold:
        train_with_lightgbm(train_datas, former_model=former_model, save_rounds=save_rounds,
                            output_path=output_lightgbm_path, params=train_params,
                            num_boost_round=num_total_iter,
                            early_stopping_rounds=301,
                            learning_rates=lambda iter_num: max(1 * (0.98 ** iter_num / (num_total_iter * 0.05)),
                                                                0.008),
                            thread_num=process_count)
    else:
        cv_with_lightgbm(train_datas, former_model=former_model, save_rounds=save_rounds,
                         output_path=output_lightgbm_path, params=train_params,
                         num_boost_round=num_total_iter,
                         early_stopping_rounds=301,
                         learning_rates=lambda iter_num: max(1 * (0.98 ** iter_num / (num_total_iter * 0.05)), 0.008),
                         thread_num=process_count, cv_fold=cv_fold)


def trainer_select(model_pattern):
    model_pattern = model_pattern.lower()
    if model_pattern not in ['wobble', 'full', 'full_15leaves']:
        data_process_logger.error('Pattern not match!')
        return -1
    # ------ Wobble ------
    if model_pattern == 'wobble':
        # Wobble
        model_tag = 'Wobble_gbdt_7leaves_iter50000'
        lightgbm_mod = cPickle.load(open('%s/models/lightgbm_%s.model' % (PROJECT_PATH, model_tag), 'rb'))
        old_datas_numbers = range(500, 940)
        random.shuffle(old_datas_numbers)
        params = {
            'objective': 'regression_l2',
            'num_leaves': 7,
            'boosting': 'gbdt',
            'feature_fraction': 0.8,
            'bagging_fraction': 0.7,
            'bagging_freq': 50,
            'verbose': 0,
            'is_unbalance': False,
            'metric': 'l1,l2,huber',
            # 'num_threads': process_count,
            'min_data_in_leaf': 80,
            'lambda_l2': 1.5,
            'save_binary': True,
            'two_round': False,
            'max_bin': 255
        }
        model_tag = 'Wobble_gbdt_7leaves_iter50000'
        train_lightGBM_new_data(
            range(300, 401) + range(840, 941) + range(1245, 1301) + range(1400, 1511),
            params,
            former_model=lightgbm_mod,
            output_lightgbm_path='%s/models/lightgbm_%s.model' % (PROJECT_PATH, model_tag),
            save_rounds=500, num_total_iter=50000, process_count=30)
    # ------ Full ------
    if model_pattern == 'full':
        # Full
        model_tag = 'Full_gbdt_7leaves_iter50000'
        lightgbm_mod = cPickle.load(open('%s/models/lightgbm_%s.model' % (PROJECT_PATH, model_tag), 'rb'))
        params = {
            'objective': 'regression_l2',
            'num_leaves': 7,
            'boosting': 'gbdt',
            'feature_fraction': 0.8,
            'bagging_fraction': 0.7,
            'bagging_freq': 50,
            'verbose': 0,
            'is_unbalance': False,
            'metric': 'l1,l2,huber',
            # 'num_threads': process_count,
            'min_data_in_leaf': 80,
            'lambda_l2': 1.5,
            'save_binary': True,
            'two_round': False,
            'max_bin': 255
        }
        model_tag = 'Full_gbdt_7leaves_iter50000'
        train_lightGBM_new_data(
            range(1045, 1145) + range(1195, 1245) + range(1300, 1450),
            params,
            former_model=lightgbm_mod,
            output_lightgbm_path='%s/models/lightgbm_%s.model' % (PROJECT_PATH, model_tag),
            save_rounds=500, num_total_iter=50000, process_count=30)
    # ------ Full ------
    if model_pattern == 'full_15leaves':
        # Full
        model_tag = 'Full_gbdt_15leaves'
        lightgbm_mod = cPickle.load(open('%s/models/lightgbm_%s.model' % (PROJECT_PATH, model_tag), 'rb'))
        # lightgbm_mod = None
        params = {
            'objective': 'regression_l2',
            'num_leaves': 15,
            'boosting': 'gbdt',
            'feature_fraction': 0.8,
            'bagging_fraction': 0.7,
            'bagging_freq': 50,
            'verbose': 0,
            'is_unbalance': False,
            'metric': 'l1,l2,huber',
            # 'num_threads': process_count,
            'min_data_in_leaf': 80,
            'lambda_l2': 1.5,
            'save_binary': True,
            'two_round': False,
            'max_bin': 255
        }
        model_tag = 'Full_gbdt_15leaves'
        train_lightGBM_new_data(
            range(300, 400) + range(840, 941) + range(1042, 1145) + range(1200, 1301) + range(1400, 1511),
            params,
            former_model=lightgbm_mod,
            output_lightgbm_path='%s/models/lightgbm_%s.model' % (PROJECT_PATH, model_tag),
            save_rounds=500, num_total_iter=50000, process_count=32)
    # ------ Full CV ------
    if model_pattern == 'full_15leaves_cv':
        # Full
        model_tag = 'Full_gbdt_15leaves_cv'
        # lightgbm_mod = cPickle.load(open('%s/models/lightgbm_%s.model' % (PROJECT_PATH, model_tag), 'rb'))
        lightgbm_mod = None
        params = {
            'objective': 'regression_l2',
            'num_leaves': 15,
            'boosting': 'gbdt',
            'feature_fraction': 0.8,
            'bagging_fraction': 0.7,
            'bagging_freq': 50,
            'verbose': 0,
            'is_unbalance': False,
            'metric': 'l1,l2,huber',
            # 'num_threads': process_count,
            'min_data_in_leaf': 80,
            'lambda_l2': 1.5,
            'save_binary': True,
            'two_round': False,
            'max_bin': 255
        }
        model_tag = 'Full_gbdt_15leaves_cv'
        train_lightGBM_new_data(
            range(300, 400) + range(840, 941) + range(1042, 1145) + range(1200, 1301) + range(1400, 1511),
            params,
            former_model=lightgbm_mod,
            output_lightgbm_path='%s/models/lightgbm_%s.model' % (PROJECT_PATH, model_tag),
            save_rounds=500, num_total_iter=50000, process_count=32)


if __name__ == '__main__':
    pass
    # lightgbm_mod = None
    # 继续训练
    # model_tag = 'Full_gbdt_7leaves_iter50000'
    # # data_process_logger.info('continue training with model: %s/models/lightgbm_%s.model' % (PROJECT_PATH, model_tag))
    # lightgbm_mod = cPickle.load(open('%s/models/lightgbm_%s.model' % (PROJECT_PATH, model_tag), 'rb'))

    # training
    trainer_select('full_15leaves_cv')

    # train_lightGBM_new_data(
    #     range(1, 5),
    #     former_model=lightgbm_mod,
    #     output_lightgbm_path='%s/models/lightgbm_%s.model' % (PROJECT_PATH, model_tag),
    #     save_rounds=500, num_total_iter=30000, process_count=1)
    # train_lightGBM_new_data(range(840, 841), former_model=lightgbm_mod,
    #                         output_lightgbm_path='%s/models/lightgbm_%s.model' % (PROJECT_PATH, model_tag),
    #                         save_rounds=500, num_total_iter=50000)
