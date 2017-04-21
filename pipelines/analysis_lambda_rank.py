# coding=utf-8

"""
Created by jayvee on 17/4/20.
https://github.com/JayveeHe
"""
from __future__ import division
import os

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
from utils.lambda_rank_utils import process_single_pickle_data


# for mac-air test
DATA_ROOT = PROJECT_PATH


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
            if mean_rank_rate:
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
        ylist = model_file.predict(vec_values)
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


def test_predict():
    """
    测试的主入口
    Returns:

    """
    model_tag = 'lambdarank_15leaves_full'
    lightgbm_mod = pickle.load(open('%s/models/lightgbm_%s.model' % (PROJECT_PATH, model_tag), 'rb'))
    # data_root_path = '%s/datas/Quant-Datas-2.0' % (DATA_ROOT)
    # fin_path = '%s/pickle_datas/%s_trans_norm.pickle' % (data_root_path, 1)
    # test_single_lambdarank_file(fin_path, lightgbm_mod)
    f_numbers, f_rank_rates = pipeline_test_lambdarank_wrapper(
        range(1, 3) + [11],
        # range(1, 300) + range(401, 840) + range(941, 1042) + range(1145, 1200) + range(1301, 1400) + range(1511, 1521),
        model=lightgbm_mod)
    result_tag = 'iter1w5'
    with open('%s/pipelines/test_%s_%s_result_%s.csv' % (PROJECT_PATH, model_tag, result_tag, len(f_numbers)),
              'wb') as fout:
        for i in range(len(f_numbers)):
            fout.write('%s,%s\n' % (f_numbers[i], f_rank_rates[i]))
        data_process_logger.info(
            'result csv: %s/pipelines/test_%s_%s_result_%s.csv' % (PROJECT_PATH, model_tag, result_tag, len(f_numbers)))


if __name__ == '__main__':
    # model_tag = 'lambdarank_15leaves_full'
    # lightgbm_mod = pickle.load(open('%s/models/lightgbm_%s.model' % (PROJECT_PATH, model_tag), 'rb'))
    # fin_path = '/Users/jayvee/CS/Python/FundDataAnalysis/datas/Quant-Datas-2.0/pickle_datas/%s_trans_norm.pickle' % (1)
    # test_single_lambdarank_file(fin_path, lightgbm_mod)
    test_predict()
