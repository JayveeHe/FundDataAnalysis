# coding=utf-8

"""
该脚本用于训练模型

Created by jayvee on 17/3/4.
https://github.com/JayveeHe
"""
import cPickle
import os
import random

import sys

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print 'Related File:%s\t----------project_path=%s' % (__file__, PROJECT_PATH)
sys.path.append(PROJECT_PATH)

from pipelines.data_preprocess import load_csv_data
from utils.logger_utils import data_process_logger
from utils.model_utils import train_with_lightgbm

if __name__ == '__main__':
    model_tag = 'iter10000_norm_combined_sample_20000'
    train_datas = []

    # for i in range(1, 11):
    #     data_process_logger.info('loading %s file' % i)
    #     datas = load_csv_data('%s/datas/%s.csv' % (PROJECT_PATH, i), normalize=True, is_combine=True)
    #     train_datas += datas
    # # dump normalized train datas
    # data_process_logger.info('dumping norm datas...')
    # cPickle.dump(train_datas, open('%s/datas/norm_datas/10_norm_combined_datas.dat' % PROJECT_PATH, 'wb'))
    # load train normalized train datas
    data_process_logger.info('loading datas...')
    train_datas = cPickle.load(open('%s/datas/norm_datas/10_norm_combined_datas.dat' % PROJECT_PATH, 'rb'))
    # random sample the train datas
    SAMPLE_SIZE = 20000
    data_process_logger.info('random sampling %s obs...' % SAMPLE_SIZE)
    random.shuffle(train_datas)
    train_datas = train_datas[:SAMPLE_SIZE]
    # ------- start training ---------
    # output_gbrt_path = '%s/models/gbrt_%s.model' % (PROJECT_PATH, model_tag)
    # train_gbrt_model(train_datas, output_gbrt_path)

    # output_svr_path = '%s/models/svr_%s.model' % (PROJECT_PATH, model_tag)
    # train_svr_model(train_datas, output_svr_path)

    output_lightgbm_path = '%s/models/lightgbm_%s.model' % (PROJECT_PATH, model_tag)
    # lightgbm_params = {'learning_rates': lambda iter_num: 0.05 * (0.99 ** iter_num)}
    train_with_lightgbm(train_datas, output_lightgbm_path, num_boost_round=40000,learning_rates=lambda iter_num: 0.5 * (0.99 ** iter_num))
