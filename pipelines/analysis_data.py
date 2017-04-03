# coding=utf-8

"""
该脚本用于分析数据

Created by jayvee on 17/2/23.
https://github.com/JayveeHe
"""
import os
import sys

import cPickle

from sklearn.cross_validation import cross_val_score
import numpy as np

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print 'Related File:%s\t----------project_path=%s' % (__file__, PROJECT_PATH)
sys.path.append(PROJECT_PATH)

from lightgbm import Booster
from pipelines.data_preprocess import load_csv_data
from utils.logger_utils import data_process_logger
from utils.logger_utils import data_analysis_logger


def cross_valid(input_x_datas, input_y_datas, cv_model):
    cv = cross_val_score(cv_model, X=input_x_datas, y=input_y_datas, cv=5, scoring='mean_squared_error')
    print cv


def test_datas_wrapper(input_files, model, normalize=True, is_combined=False):
    """
    input:(file_names,model)
    output: mean rank rate
    """
    mean_rank_rates = []
    for i in input_files:
        input_datas = load_csv_data('%s/datas/%s.csv' % (PROJECT_PATH, i), is_combine=is_combined, normalize=normalize)
        data_process_logger.info('testing file: %s.csv' % i)
        mean_rank_rate = test_datas(input_datas, model)
        if mean_rank_rate >= 0.4:
            data_analysis_logger.info('the file number is %s, obs = %s' % (i, len(input_datas)))
        mean_rank_rates.append(mean_rank_rate)
    mean_rank_rate = np.mean(mean_rank_rates)
    std_rank_rate = np.std(mean_rank_rates)
    var_rank = np.var(mean_rank_rates)
    data_process_logger.info(
        'all input files mean rank rate is %s, all input files std is %s, var is %s' % (
            mean_rank_rate, std_rank_rate, var_rank))


def test_quant_data_wrapper(input_file_numbers, model, normalize=True, predict_iteration=None):
    """
    input:(file_names,model)
    output: mean rank rate
    """
    mean_rank_rates = []
    file_number_list = []
    if predict_iteration:
        model.save_model('tmp_model.txt', num_iteration=predict_iteration)
        model = Booster(model_file='tmp_model.txt')
    for i in input_file_numbers:
        data_root_path = '%s/datas/Quant-Datas-2.0' % (PROJECT_PATH)
        if normalize:
            fin_path = '%s/pickle_datas/%s_trans_norm.pickle' % (data_root_path, i)
        else:
            fin_path = '%s/pickle_datas/%s_trans.pickle' % (data_root_path, i)
        try:
            with open(fin_path, 'rb') as fin_data_file:
                input_datas = cPickle.load(fin_data_file)
                data_process_logger.info('testing file: %s' % fin_path)
                mean_rank_rate = test_datas(input_datas, model)
                if mean_rank_rate >= 0.4:
                    data_analysis_logger.info('the file number is %s, obs = %s' % (i, len(input_datas)))
                mean_rank_rates.append(mean_rank_rate)
                file_number_list.append(i)
        except:
            data_process_logger.info('test file failed: file path=%s' % (fin_path))
    mean_rank_rate = np.mean(mean_rank_rates)
    std_rank_rate = np.std(mean_rank_rates)
    var_rank = np.var(mean_rank_rates)
    data_process_logger.info(
        'Tested %s files, all input files mean rank rate is %s, all input files std is %s, var is %s' % (
            len(input_file_numbers), mean_rank_rate, std_rank_rate, var_rank))
    return file_number_list, mean_rank_rates


def test_datas(input_datas, model):
    # input_datas = list(input_datas)
    input_ranked_list = sorted(input_datas, cmp=lambda x, y: 1 if x[1] - y[1] > 0 else -1)
    xlist = [a[2:] for a in input_ranked_list]
    origin_score_list = [a[1] for a in input_ranked_list]
    # pca_mod = cPickle.load(open('%s/models/pca_norm_5.model' % project_path, 'rb'))
    # xlist = list(pca_mod.transform(xlist))
    ylist = model.predict(xlist)
    index_ylist = [(i, ylist[i], origin_score_list[i]) for i in range(len(ylist))]
    ranked_index_ylist = sorted(index_ylist, cmp=lambda x, y: 1 if x[1] - y[1] > 0 else -1)
    # for i in range(len(ranked_index_ylist)):
    # data_process_logger.info('pre: %s\t origin: %s\t delta: %s\tpredict_score: %s\torigin_score: %s' % (
    #    i, ranked_index_ylist[i][0], i - ranked_index_ylist[i][0], ranked_index_ylist[i][1],
    #    ranked_index_ylist[i][2]))
    mean_rank_rate = result_validation(ranked_index_ylist)
    return mean_rank_rate


def result_validation(ranked_index_ylist, N=50, threshold=0.35):
    buyer_list = ranked_index_ylist[:N]
    total_error = 0
    origin_rank_list = []
    for i in range(len(buyer_list)):
        origin_rank_list.append(buyer_list[i][0] + 1)
        total_error += abs((buyer_list[i][0] - i))
    mean_rank = np.mean(origin_rank_list)
    data_process_logger.info('mean_rank = %s' % mean_rank)
    data_process_logger.info('mean error = %s' % ((0.0 + total_error) / N))
    mean_rank_rate = mean_rank / len(ranked_index_ylist)
    data_process_logger.info('mean_rank_rate = %s' % mean_rank_rate)
    return mean_rank_rate


if __name__ == '__main__':
    # --------- Testing -------
    model_tag = 'New_Quant_Data_rebalanced_norm_gbdt_7leaves_iter30000'
    data_process_logger.info('--------LightGBM:----------')
    data_process_logger.info('using model: %s/models/lightgbm_%s.model' % (PROJECT_PATH, model_tag))
    # lightgbm_mod = cPickle.load(open('%s/models/lightgbm_%s.model' % (PROJECT_PATH, model_tag), 'rb'))
    # lightgbm_mod = Booster(
    #    model_file='%s/models/lightgbm_%s_continued.model' % (PROJECT_PATH, model_tag))
    lightgbm_mod = cPickle.load(open('%s/models/lightgbm_%s.model' % (PROJECT_PATH, model_tag), 'rb'))
    # data_process_logger.info('test trianing file')
    # test_datas_wrapper(range(1,100),lightgbm_mod)
    data_process_logger.info('test test file')
    f_numbers, f_rank_rates = test_quant_data_wrapper(
        range(740, 840) + range(940, 1040) + range(1145, 1195) + range(1245, 1295) + range(1345, 1445), lightgbm_mod,
        normalize=True)
    # save test result to csv
    with open('%s/pipelines/test_result_%s.csv' % (PROJECT_PATH, len(f_numbers)), 'wb') as fout:
        for i in range(len(f_numbers)):
            fout.write('%s,%s\n' % (f_numbers[i], f_rank_rates[i]))
