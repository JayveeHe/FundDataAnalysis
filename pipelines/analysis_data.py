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


def test_datas(input_datas, model):
    input_ranked_list = sorted(input_datas, cmp=lambda x, y: 1 if x[2] - y[2] < 0 else -1)
    xlist = [a[3] for a in input_ranked_list]
    origin_score_list = [a[2] for a in input_ranked_list]
    # pca_mod = cPickle.load(open('%s/models/pca_norm_5.model' % project_path, 'rb'))
    # xlist = list(pca_mod.transform(xlist))
    ylist = model.predict(xlist)
    index_ylist = [(i, ylist[i], origin_score_list[i]) for i in range(len(ylist))]
    ranked_index_ylist = sorted(index_ylist, cmp=lambda x, y: 1 if x[1] - y[1] < 0 else -1)
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
        origin_rank_list.append(buyer_list[i][0])
        total_error += abs((buyer_list[i][0] - i))
    mean_rank = np.mean(origin_rank_list)
    data_process_logger.info('mean_rank = %s' % mean_rank)
    mean_rank_rate = mean_rank / len(ranked_index_ylist)
    data_process_logger.info('mean_rank_rate = %s' % mean_rank_rate)
    return mean_rank_rate


if __name__ == '__main__':
    # start = time.time()
    ## print '====================== 20 normalize test set'
    # gbrt_mod = cPickle.load(open('./models/gbrt_model_%s.mod' % model_tag, 'rb'))
    # datas = load_csv_data('./datas/4.csv', normalize=True)
    # test_datas(datas, gbrt_mod)
    # print '====================== 20 normalize train set'
    # gbrt_mod = cPickle.load(open('./models/gbrt_model_%s.mod' % model_tag, 'rb'))
    # datas = load_csv_data('./datas/310.csv', normalize=True)
    # test_datas(datas, gbrt_mod)
    # out_data = normalize_data(train_datas)
    # for item in out_data:
    #	print 'item is : %s\t%s\t%s\n'%(item[0],item[2],item[3])
    # label_set = []
    # vec_set = []
    # for i in range(len(train_datas)):
    #    label_set.append(train_datas[i][2])
    #    vec_set.append(train_datas[i][3])
    # timators gbrt_mod = train_model(train_datas)
    # # model_tag = '50'
    # ------- grid ------
    # vec_set = [a[3] for a in train_datas]
    # label_set = [a[2] for a in train_datas]
    # train_regression_age_model(input_xlist=vec_set, input_ylist=label_set, model_label=model_tag)

    # end = time.time()
    # print "spend the time %s" % (end - start)
    ## xlist = [a[3] for a in datas[100:200]]
    ## ylist = [a[2] for a in datas[100:200]]
    ## cross_valid(xlist, ylist, gbrt_mod)

    # --------- Testing -------
    model_tag = 'iter10000_norm_combined_sample_20000'
    a = 550
    b = 560
    c = 150
    # gbrt_mod = cPickle.load(open('%s/models/gbrt_%s.model' % (project_path, model_tag), 'rb'))
    # data_process_logger.info('--------gbrt:----------')
    # data_process_logger.info('using model: %s/models/gbrt_%s.model' % (project_path, model_tag))
    # data_process_logger.info('testing file: /datas/%s.csv' % a)
    # datas = load_csv_data('./datas/%s.csv' % a)
    # data_process_logger.info('testing')
    # test_datas(datas, gbrt_mod)
    # data_process_logger.info('===============')
    # data_process_logger.info('testing file: /datas/%s.csv' % b)
    # datas = load_csv_data('./datas/%s.csv' % b)
    # test_datas(datas, gbrt_mod)
    # data_process_logger.info('--------SVR:----------')
    # data_process_logger.info('using model: %s/models/svr_%s.model' % (project_path, model_tag))
    # svr_mod = cPickle.load(open('%s/models/svr_%s.model' % (project_path, model_tag), 'rb'))
    # data_process_logger.info('testing file: /datas/%s.csv' % a)
    # datas = load_csv_data('./datas/%s.csv' % a)
    # data_process_logger.info('testing')
    # test_datas(datas, svr_mod)
    # data_process_logger.info('===============')
    # data_process_logger.info('testing file: /datas/%s.csv' % b)
    # datas = load_csv_data('./datas/%s.csv' % b)
    # test_datas(datas, svr_mod)
    data_process_logger.info('--------LightGBM:----------')
    data_process_logger.info('using model: %s/models/lightgbm_%s.model' % (PROJECT_PATH, model_tag))
    lightgbm_mod = cPickle.load(open('%s/models/lightgbm_%s.model' % (PROJECT_PATH, model_tag), 'rb'))
    # data_process_logger.info('test trianing file')
    # test_datas_wrapper(range(1,100),lightgbm_mod)
    data_process_logger.info('test test file')
    test_datas_wrapper(range(1, 11), lightgbm_mod, is_combined=True, normalize=True)
    # data_process_logger.info('testing file: /datas/%s.csv' % 570)
    # 3datas = load_csv_data('./datas/%s.csv' % 570)
    # 3data_process_logger.info('testing')
    # 3test_datas(datas, lightgbm_mod)
    # 3data_process_logger.info('===============')
    # 3data_process_logger.info('testing file: /datas/%s.csv' % 580)
    # 3datas = load_csv_data('./datas/%s.csv' % 580)
    # 3test_datas(datas, lightgbm_mod)
    # 3data_process_logger.info('===============')
    # 3data_process_logger.info('testing file: /datas/%s.csv' % 590)
    # 3datas = load_csv_data('./datas/%s.csv' % 590)
    # 3test_datas(datas, lightgbm_mod)
    # 3test_datas_wrapper(range(20,30), lightgbm_mod)
    # 3test_datas_wrapper(range(560,600), lightgbm_mod)
    # 3test_datas_wrapper(range(550,560),lightgbm_mod)
