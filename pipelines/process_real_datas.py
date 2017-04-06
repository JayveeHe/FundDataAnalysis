# coding=utf-8

"""
Created by jayvee on 17/4/4.
https://github.com/JayveeHe
"""
import csv
import multiprocessing
import os

import sys

from lightgbm import Booster

try:
    import cPickle as pickle
except:
    import pickle

from sklearn import preprocessing
from sklearn.preprocessing import Imputer
import numpy as np

from utils.logger_utils import data_process_logger

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print 'Related File:%s\t----------project_path=%s' % (__file__, PROJECT_PATH)
sys.path.append(PROJECT_PATH)


def turn_csv_into_result(origin_csv_path, output_csv_path, predict_model, predict_iteration, is_norm=True,
                         is_norm_score=True):
    """
    把原始的feature csv转为排名后的结果csv
    Args:
        predict_model:
        origin_csv_path:
        output_csv_path:
        predict_iteration: 预测用的
        is_norm: 是否进行标准化
        is_norm_score: 是否对分数进行标准化

    Returns:

    """
    data_process_logger.info('handling %s' % origin_csv_path)
    with open(origin_csv_path, 'rb') as fin_csv, open(output_csv_path, 'wb') as fout_csv:
        reader = csv.reader(fin_csv)
        writer = csv.writer(fout_csv)
        # count = 0
        origin_datas = []
        data_process_logger.info('start reading %s' % origin_csv_path)
        # 首先进行缺失值的补充和标准化
        for line in reader:
            single_vec_value = [float(i) if i != 'NaN' else np.nan for i in line]
            origin_datas.append(single_vec_value)
            # data_process_logger.info('handled line %s' % count)
            # count += 1
        # inferring missing data
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        imp.fit(origin_datas)
        transformed_datas = imp.transform(origin_datas)
        if is_norm:
            # standardising datas
            stock_ids = transformed_datas[:, 0]
            stock_scores = transformed_datas[:, 1]
            vec_values = transformed_datas[:, 2:]
            scaled_vec_values = preprocessing.scale(vec_values)
            if is_norm_score:
                stock_scores = preprocessing.scale(stock_scores)
            transformed_datas = np.column_stack((stock_ids, stock_scores, scaled_vec_values))
        # 进行预测
        xlist = [a[2:] for a in transformed_datas]  # vec values
        origin_score_list = [a[1] for a in transformed_datas]
        stock_ids = [a[0] for a in transformed_datas]
        score_list = predict_model.predict(xlist, num_iteration=predict_iteration)
        # 对预测结果进行排序并输出csv
        result = np.column_stack((stock_ids, score_list, origin_score_list))
        sorted_result = sorted(result, cmp=lambda x, y: 1 if x[1] - y[1] > 0 else -1)
        for row in sorted_result:
            writer.writerow(row)
        # writting transformed datas
        data_process_logger.info('complete writting %s' % output_csv_path)
        return sorted_result


def processing_real_data(model_path, file_numbers=[], workspace_root='./', predict_iter=None):
    """
    进行并行化处理实测数据
    Args:
        predict_iter:
        model_path: 预测使用的模型路径 (lightGBM模型)
        process_count: 并行核心数
        workspace_root: csv文件所在的目录
        file_numbers: 处理的文件编号列表
    Returns:

    """
    output_path = os.path.join(workspace_root, 'results')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    predict_mod = pickle.load(open(model_path, 'rb'))
    # if predict_iter:
    #     predict_mod.save_model('tmp_model.txt', num_iteration=predict_iter)
    #     predict_mod = Booster(model_file='tmp_model.txt')
    for file_n in file_numbers:
        fin_path = (workspace_root + '/%s.csv') % file_n
        fout_path = (output_path + '/%s_result.csv') % file_n
        turn_csv_into_result(fin_path, fout_path, predict_mod, predict_iter, True,
                             True)
    data_process_logger.info('Done with %s files' % len(file_numbers))


if __name__ == '__main__':
    wsr = '%s/datas/Quant-Datas' % PROJECT_PATH
    fn = [1, 2, 3, 4]
    mpath = '%s/models/best_models/lightgbm_New_Quant_Data_rebalanced_norm_gbdt_7leaves_iter30000_best.model' % PROJECT_PATH
    processing_real_data(mpath, fn, wsr, predict_iter=27000)
