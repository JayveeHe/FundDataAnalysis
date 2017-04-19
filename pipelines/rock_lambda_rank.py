# coding=utf-8

"""
Created by jayvee on 17/4/18.
https://github.com/JayveeHe
"""
from __future__ import division
import multiprocessing
import os
import random

import sys

from lightgbm import Booster
import numpy as np

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print 'Related File:%s\t----------project_path=%s' % (__file__, PROJECT_PATH)
sys.path.append(PROJECT_PATH)

try:
    import cPickle as pickle
except:
    import pickle

from pipelines.analysis_data import result_validation
from pipelines.train_models import DATA_ROOT
from utils.logger_utils import data_process_logger, data_analysis_logger
from utils.lambda_rank_utils import process_single_pickle_data, train_lambda_rank


# for mac-air test
# DATA_ROOT = PROJECT_PATH


def pipeline_train_lambda_rank(train_file_number_list, train_params, former_model=None, output_lightgbm_path=None,
                               save_rounds=1000,
                               num_total_iter=100, process_count=12, cv_fold=None):
    """
    利用新数据(2899维)训练lambda rank模型

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
        data_res = proc_pool.apply_async(process_single_pickle_data, args=(pickle_path, i))
        multi_results.append(data_res)
        # datas = load_csv_data(csv_path, normalize=True, is_combine=True)
        # train_datas += datas
    proc_pool.close()
    proc_pool.join()
    # fetch datas from pool
    stock_ids, stock_scores, vec_values, stock_rank_labels, query_count = multi_results[0].get()
    # train_datas = tmp_data
    # label_list = stock_rank_labels
    # vec_list = vec_values
    # query_datas = [query_count]
    label_list = []
    vec_list = []
    query_datas = []
    data_process_logger.info('combining datas...')
    for i in xrange(0, len(multi_results)):
        data_process_logger.info('combining No.%s data' % i)
        try:
            stock_ids, stock_scores, vec_values, stock_rank_labels, query_count = multi_results[i].get()
            # train_datas = np.row_stack((train_datas, datas)) # np.2darray
            # train_datas = np.vstack((train_datas, datas))
            # train_datas.extend(datas)
            # label_list.extend(stock_scores)
            tmp_vec_list = []
            tmp_label_list = []
            for index in range(len(vec_values)):
                vec = vec_values[index]
                label = stock_rank_labels[index]
                # if len(vec) == len(vec_list[-1]):
                if len(vec) == 2897:
                    tmp_vec_list.append(vec)
                    tmp_label_list.append(label)
                else:
                    raise IndexError('not equaling n_feature: %s' % len(vec))
                    # query_count -= 1
            if query_count == len(tmp_vec_list):
                vec_list.extend(tmp_vec_list)
                label_list.extend(tmp_label_list)
                query_datas.append(query_count)
            else:
                raise IndexError('query count %s not equal to len(vec_list): %s' % (query_count, len(vec_list)))
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
    # train_params['group']=query_datas
    train_lambda_rank(train_datas, group_datas=query_datas, former_model=former_model, save_rounds=save_rounds,
                      output_path=output_lightgbm_path, params=train_params,
                      num_boost_round=num_total_iter,
                      early_stopping_rounds=301,
                      learning_rates=lambda iter_num: max(1 * (0.98 ** iter_num / (num_total_iter * 0.05)),
                                                          0.008),
                      thread_num=process_count)


def pipeline_test_lambdarank_wrapper(input_file_numbers, model, normalize=True, predict_iteration=None):
    """
    进行结果测试
    Args:
        input_file_numbers:
        model:
        normalize:
        predict_iteration:

    Returns:

    """
    mean_rank_rates = []
    file_number_list = []
    if predict_iteration:
        model.save_model('tmp_lambdarank_model.txt', num_iteration=predict_iteration)
        model = Booster(model_file='tmp_lambdarank_model.txt')
    for i in input_file_numbers:
        data_root_path = '%s/datas/Quant-Datas-2.0' % (DATA_ROOT)
        if normalize:
            fin_path = '%s/pickle_datas/%s_trans_norm.pickle' % (data_root_path, i)
        else:
            fin_path = '%s/pickle_datas/%s_trans.pickle' % (data_root_path, i)
        try:
            mean_rank_rate = test_single_lambdarank_file(fin_path, model)
            mean_rank_rates.append(mean_rank_rate)
            file_number_list.append(i)
        except Exception, e:
            data_process_logger.info('test file failed: file path=%s, details=%s' % (fin_path, e))
    mean_rank_rate = np.mean(mean_rank_rates)
    std_rank_rate = np.std(mean_rank_rates)
    var_rank = np.var(mean_rank_rates)
    data_process_logger.info(
        'Tested %s files, all input files mean rank rate is %s, all input files std is %s, var is %s' % (
            len(input_file_numbers), mean_rank_rate, std_rank_rate, var_rank))
    return file_number_list, mean_rank_rates


def test_single_lambdarank_file(fin_path, model_file):
    try:
        # global g_model
        data_analysis_logger.info('testing %s' % fin_path)
        stock_ids, stock_scores, vec_values, stock_rank_labels, query_count = process_single_pickle_data(fin_path)
        ylist = model_file.predict(vec_values, [query_count])
        origin_score_list = stock_scores
        combined_score_list = np.column_stack((ylist, origin_score_list))
        # input_datas = input_datas.tolist()
        # origin_ranked_list = sorted(input_datas, cmp=lambda x, y: 1 if x[1] - y[1] > 0 else -1)
        combined_score_list = combined_score_list.tolist()
        origin_ranked_list = sorted(combined_score_list, cmp=lambda x, y: 1 if x[1] - y[1] > 0 else -1)  # 根据原始值值进行升序排序
        # 得到原始值的排名序号
        index_ylist = [(i, origin_ranked_list[i][0], origin_ranked_list[i][1]) for i in range(len(origin_ranked_list))]
        predict_ranked_index_ylist = sorted(index_ylist,
                                            cmp=lambda x, y: 1 if x[1] - y[1] < 0 else -1)  # 根据预测值进行降序排序,保留原始值排名序号
        mean_rank_rate = result_validation(predict_ranked_index_ylist, N=50)

        if mean_rank_rate >= 0.4:
            data_analysis_logger.info('the file path is %s, obs = %s' % (fin_path, len(stock_scores)))
            # mean_rank_rates.append(mean_rank_rate)
            # file_number_list.append(i)
        return mean_rank_rate
    except Exception, e:
        data_process_logger.info('test file failed: file path=%s, details=%s' % (fin_path, e))


def test_train():
    params = {
        'objective': 'lambdarank',
        'num_leaves': 15,
        'boosting': 'gbdt',
        'feature_fraction': 0.8,
        'bagging_fraction': 0.7,
        'bagging_freq': 50,
        'verbose': 0,
        'is_unbalance': False,
        'metric': 'ndcg',
        # 'num_threads': 1,
        'min_data_in_leaf': 80,
        'lambda_l2': 1.5,
        'save_binary': True,
        'two_round': False,
        'max_bin': 127,
        'eval_at': [30, 50, 100]
    }
    init_model_tag = 'lambdarank_15leaves_shuffle60'
    lightgbm_mod = pickle.load(open('%s/models/lightgbm_%s.model' % (PROJECT_PATH, init_model_tag), 'rb'))
    model_tag = 'lambdarank_15leaves_shuffle500'
    lightgbm_mod = None
    train_file_numbers = range(300, 400) + range(840, 941) + range(1042, 1145) + range(1200, 1301) + range(1400, 1511)
    # random.shuffle(train_file_numbers)
    pipeline_train_lambda_rank(
        # [1, 2, 3, 4, 5],
        # train_file_numbers[:500],
        train_file_numbers,
        # range(1000,1100),
        train_params=params,
        former_model=lightgbm_mod,
        output_lightgbm_path='%s/models/lightgbm_%s.model' % (PROJECT_PATH, model_tag),
        save_rounds=500,
        num_total_iter=50000,
        process_count=32)


def test_predict():
    model_tag = 'lambdarank_15leaves_shuffle30'
    lightgbm_mod = pickle.load(open('%s/models/lightgbm_%s.model' % (PROJECT_PATH, model_tag), 'rb'))
    # data_root_path = '%s/datas/Quant-Datas-2.0' % (DATA_ROOT)
    # fin_path = '%s/pickle_datas/%s_trans_norm.pickle' % (data_root_path, 1)
    # test_single_lambdarank_file(fin_path, lightgbm_mod)
    pipeline_test_lambdarank_wrapper(
        # range(1, 10),
        range(1, 300) + range(401, 840) + range(941, 1042) + range(1145, 1200) + range(1301, 1400) + range(1511, 1521),
        model=lightgbm_mod)


if __name__ == '__main__':
    test_train()
    # test_predict()
