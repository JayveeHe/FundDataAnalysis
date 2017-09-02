# coding=utf-8

"""
Created by jayvee on 17/4/4.
https://github.com/JayveeHe
"""
import csv
import multiprocessing
import os
import re

import sys

from lightgbm import Booster

try:
    import cPickle as pickle
except:
    import pickle

from sklearn import preprocessing
from sklearn.preprocessing import Imputer
import numpy as np

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print 'Related File:%s\t----------project_path=%s' % (__file__, PROJECT_PATH)
sys.path.append(PROJECT_PATH)

from pipelines.train_models import DATA_ROOT
from utils.logger_utils import data_process_logger


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
        count = 1
        n_feature = 4563
        for line in reader:
            if len(line) == n_feature:
                single_vec_value = [float(i) if i != 'NaN' else np.nan for i in line]
                 # process the 453th col, remove future feature.
                single_vec_value = single_vec_value[:453]+single_vec_value[454:]
                origin_datas.append(single_vec_value)
                # data_process_logger.info('handled line %s' % count)

            else:
                data_process_logger.info(
                    'casting line: %s in file %s, it has %s features while the first line has %s' % (
                        count, origin_csv_path, len(line), n_feature))
            count += 1
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
        line_numbers = range(1, len(xlist) + 1)
        # 对预测结果进行排序并输出csv
        result = np.column_stack((line_numbers, stock_ids, score_list, origin_score_list))
        sorted_result = sorted(result, cmp=lambda x, y: 1 if x[2] - y[2] > 0 else -1)
        writer.writerow(['origin_line', 'stock_id', 'predict_score', 'origin_score'])
        for row in sorted_result:
            writer.writerow([int(row[0]), str(row[1]), row[2], row[3]])
        # writting transformed datas
        data_process_logger.info('complete writting %s' % output_csv_path)
        return sorted_result


def batch_process_real_data(model_path, file_numbers=[], workspace_root='./', model_tag='real', predict_iter=None):
    """
    进行批量处理实测数据,将结果存在相应文件下的<model_tag>文件夹下
    Args:
        model_tag: 模型tag
        predict_iter:使用的模型迭代次数
        model_path: 预测使用的模型路径 (lightGBM模型)
        workspace_root: csv文件所在的目录
        file_numbers: 处理的文件编号列表
    Returns:
        None
    """
    output_path = os.path.join(workspace_root, '%s_results' % model_tag)
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
                             False)
    data_process_logger.info('Done with %s files' % len(file_numbers))


def predict_with_oldbest(fin_csv_path, fout_csv_path=None, tag='OldBest'):
    """
    使用Old_Best模型进行单个文件的训练
    Args:
        tag:
        fin_csv_path:
        fout_csv_path:

    Returns:

    """
    if not fout_csv_path:
        csv_dir_path = os.path.dirname(fin_csv_path)
        csv_filename = re.findall('%s/(.*)\.csv' % csv_dir_path, fin_csv_path)
        csv_output_dir = os.path.join(csv_dir_path, 'Old_Best_results')
        if not os.path.exists(csv_output_dir):
            os.mkdir(csv_output_dir)
        fout_csv_path = os.path.join(csv_output_dir, '%s_%s_result.csv' % (csv_filename[0], tag))
    old_best_mod = pickle.load(open(
        '%s/models/best_models/lightgbm_New_Quant_Data_rebalanced_norm_gbdt_7leaves_iter30000_best.model' % PROJECT_PATH))
    turn_csv_into_result(fin_csv_path, fout_csv_path, old_best_mod, predict_iteration=27000)


def predict_with_full(fin_csv_path, fout_csv_path=None, tag='Full'):
    """
    使用 Full 模型进行单个文件的训练
    Args:
        fin_csv_path:
        fout_csv_path:

    Returns:

    """
    if not fout_csv_path:
        csv_dir_path = os.path.dirname(fin_csv_path)
        csv_filename = re.findall('%s/(.*)\.csv' % csv_dir_path, fin_csv_path)
        csv_output_dir = os.path.join(csv_dir_path, 'Full_results')
        if not os.path.exists(csv_output_dir):
            os.mkdir(csv_output_dir)
        fout_csv_path = os.path.join(csv_output_dir, '%s_%s_result.csv' % (csv_filename[0], tag))
    full_mod = pickle.load(open(
        '%s/models/best_models/lightgbm_Full_gbdt_15leaves.model' % PROJECT_PATH))
    res = turn_csv_into_result(fin_csv_path, fout_csv_path, full_mod, predict_iteration=50000)
    if res:
        return fout_csv_path
    else:
        return None


if __name__ == '__main__':
    wsr = '/media/user/Data0/DataTest3.0'
    # wsr = '/Users/jayvee/CS/Python/FundDataAnalysis/datas/Quant-Datas-2.0'
    fn = range(1, 91)
    model_path = '%s/models/best_models/lightgbm_Full_gbdt_7leaves_3.0.model' % PROJECT_PATH
    # model_path = '%s/models/best_models/lightgbm_Full_gbdt_15leaves.model' % PROJECT_PATH
    model_tag = 'New_7leaves'
    batch_process_real_data(model_path, fn, wsr, model_tag=model_tag, predict_iter=5275)
    # model_path = '%s/models/lightgbm_New_Quant_Data_rebalanced_norm_gbdt_7leaves_iter30000_best.model' % PROJECT_PATH
    # model_tag = 'Old_Best'
    # batch_process_real_data(model_path, fn, wsr, model_tag=model_tag, predict_iter=27000)

    # process daily csv
    # daily_csv_path = '/Users/jayvee/CS/Python/FundDataAnalysis/datas/daily/newdata_2739.csv'
    # predict_with_oldbest(daily_csv_path)
    # predict_with_full(daily_csv_path)
