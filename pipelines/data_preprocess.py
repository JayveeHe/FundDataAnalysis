# coding=utf-8

"""
Created by jayvee on 17/3/3.
https://github.com/JayveeHe
"""
import csv
import os
import sys

import cPickle
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import Imputer

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print 'Related File:%s\t----------project_path=%s' % (__file__, PROJECT_PATH)
sys.path.append(PROJECT_PATH)

import multiprocessing

from utils.logger_utils import data_process_logger


def load_csv_data(csv_path, normalize=True, is_combine=False):
    """

    Args:
        csv_path:
        normalize: 是否进行标准化
        is_combine: 是否进行norm特征和的拼接

    Returns:

    """
    from sklearn import preprocessing
    with open(csv_path, 'rb') as fin:
        data_process_logger.info('loading file: %s' % csv_path)
        datas = []
        temp_list = []
        score_list = []
        date_list = []
        id_list = []
        vec_list = []
        for line in fin:
            line = line.strip()
            tmp = line.split(',')
            stock_id = tmp[0]
            trade_date = tmp[1]
            score = eval(tmp[2])
            score_list.append(score)
            vec_value = [eval(a) for a in tmp[3:]]
            vec_list.append(vec_value)
            date_list.append(trade_date)
            id_list.append(stock_id)
            temp_list.append((stock_id, trade_date, score, vec_value))
        # all not normalize
        if not normalize:
            avg = np.mean(score_list)
            std = np.std(score_list)
            for item in temp_list:
                normalize_score = (item[2] - avg) / std
                datas.append((item[0], item[1], normalize_score, item[3]))
            return datas
        else:
            score_scale = preprocessing.scale(score_list)
            score_scale_list = list(score_scale)
            vec_scale = preprocessing.scale(vec_list)
            vec_scale_list = vec_scale
            for i in range(len(id_list)):
                if is_combine:
                    datas.append((id_list[i], date_list[i], score_scale_list[i], list(vec_scale_list[i]) + vec_list[i]))
                else:
                    datas.append((id_list[i], date_list[i], score_scale_list[i], list(vec_scale_list[i])))
            # avg = np.mean(score_list)
            #            std = np.std(score_list)
            #            for item in temp_list:
            #                normalize_score = (item[2] - avg) / std
            #                datas.append((item[0], item[1], normalize_score, item[3]))
            return datas


def normalize_data(input_data):
    """
    author:zxj
    func:normalize
    input:origin input data
    return:tuple of (normalize_score,fea_vec,id,date)
    """
    output_data = []
    from itertools import groupby
    import numpy as np
    score_list = [(input_data[i][1], (input_data[i][2], input_data[i][3], input_data[i][0])) \
                  for i in range(len(input_data))]
    score_group_list = groupby(score_list, lambda p: p[0])
    # for key,group in score_group_list:
    #	print list(group)[0][1]
    for key, group in score_group_list:
        temp_list = list(group)
        score_list = [a[1][0] for a in temp_list]
        score_list = np.array(score_list).astype(np.float)
        print "the score list is %s" % (''.join(str(v) for v in score_list))
        vec_list = [a[1][1] for a in temp_list]
        id_list = [a[1][2] for a in temp_list]
        avg = np.mean(score_list)
        std = np.std(score_list)
        for i in range(len(score_list)):
            # normalize
            normalize_score = (score_list[i] - avg) / std
            output_data.append((normalize_score, vec_list[i], id_list[i], key))
    return output_data


def infer_missing_datas(fin_csv_path, fout_csv_path, fout_pickle_path, is_norm=False, is_norm_score=True):
    """
    处理NaN数据,并将处理后的数据分别存储为csv与pickle文件
    Args:
        is_norm: 是否进行标准化
        is_norm_score: 是否对score进行标准化
        fin_csv_path:
        fout_csv_path:
        fout_pickle_path:

    Returns:

    """
    with open(fin_csv_path, 'rb') as fin_csv, \
            open(fout_csv_path, 'wb') as fout_csv, \
            open(fout_pickle_path, 'wb') as fout_pickle:
        origin_datas = []
        reader = csv.reader(fin_csv)
        writer = csv.writer(fout_csv)
        # count = 0
        data_process_logger.info('start reading %s' % fin_csv_path)
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
        # writting transformed datas
        data_process_logger.info('start writting %s' % fout_csv_path)
        for row in transformed_datas:
            writer.writerow(row)
            # data_process_logger.info('line %s written' % count)
            # count += 1
            # result = ','.join(row)
            # fout_csv.write(result + '\n')
        data_process_logger.info('start dumping %s' % fout_pickle_path)
        cPickle.dump(transformed_datas, fout_pickle, protocol=2)
        data_process_logger.info('%s done' % fin_csv_path)
        return transformed_datas


def parallel_inferring(file_number_list, process_count=12):
    """
    并行化进行数据清理
    Returns:

    """
    data_process_logger.info('Start parallel inferring, process count = %s' % process_count)
    proc_pool = multiprocessing.Pool(process_count)
    # multi_results = []
    for i in file_number_list:
        # data_process_logger.info('loading %s file' % i)
        # csv_path = '%s/datas/%s.csv' % (PROJECT_PATH, i)
        fin_csv_path = '%s/datas/Quant-Datas/%s.csv' % (PROJECT_PATH, i)
        fout_csv_path = '%s/datas/Quant-Datas/transformed_datas/%s_trans_norm.csv' % (PROJECT_PATH, i)
        fout_pickle_path = '%s/datas/Quant-Datas/pickle_datas/%s_trans_norm.pickle' % (PROJECT_PATH, i)
        data_res = proc_pool.apply_async(infer_missing_datas,
                                         args=(fin_csv_path, fout_csv_path, fout_pickle_path, True, True))
        # multi_results.append(data_res)
        # datas = load_csv_data(csv_path, normalize=True, is_combine=True)
        # train_datas += datas
    proc_pool.close()
    proc_pool.join()
    data_process_logger.info('Done with %s files' % len(file_number_list))


if __name__ == '__main__':
    # print len(load_csv_data('%s/datas/%s.csv' % (PROJECT_PATH, 1), is_combine=True))
    # infer_missing_datas(fin_csv_path='%s/datas/Quant-Datas/%s.csv' % (PROJECT_PATH, 1),
    #                     fout_csv_path='%s/datas/Quant-Datas/transformed_datas/%s_trans.csv' % (PROJECT_PATH, 1),
    #                     fout_pickle_path='%s/datas/Quant-Datas/pickle_datas/%s_trans.pickle' % (PROJECT_PATH, 1))
    # pickle_data = cPickle.load()
    # print len(pickle_data)
    parallel_inferring(file_number_list=range(1, 769), process_count=12)
