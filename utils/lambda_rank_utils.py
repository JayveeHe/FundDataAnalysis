# coding=utf-8

"""
Created by jayvee on 17/4/18.
https://github.com/JayveeHe
"""
from __future__ import division
import os

import sys
import numpy as np

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print 'Related File:%s\t----------project_path=%s' % (__file__, PROJECT_PATH)
sys.path.append(PROJECT_PATH)

from utils.logger_utils import data_process_logger

try:
    import cPickle as pickle
except:
    import pickle


# @profile
def process_single_pickle_data(pickle_file_path, query_label=1):
    """
    处理单个pickle后的data文件,进行排序、query data的生成等
    Returns: tuple of ranked-features and query data

    """

    def rank_value(rank, max_rank):
        """
        获取排名权重
        Returns:

        """
        rank_steps = [0.01, 0.05, 0.1, 0.15, 0.3, 0.4, 0.5, 0.7, 0.9, 1]
        rank_weights = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        basic_rank = rank / max_rank
        for rank_index in range(len(rank_steps)):
            if basic_rank <= rank_steps[rank_index]:
                return rank_weights[rank_index]
        return 0

    with open(pickle_file_path, 'rb') as fin:
        # stock_ids, stock_scores, vec_values = pickle.load(fin)
        pickle_obj = pickle.load(fin)
        combined_obj = [(pickle_obj[0][i], pickle_obj[1][i], pickle_obj[2][i]) for i in range(len(pickle_obj[0]))]
        ranked_datas = sorted(combined_obj, cmp=lambda x, y: 1 if x[1] - y[1] > 0 else -1)
        stock_ids = [a[0] for a in ranked_datas]
        stock_scores = [a[1] for a in ranked_datas]
        stock_rank_labels = [rank_value(a, len(ranked_datas)) for a in range(len(ranked_datas))]  # label for rank
        vec_values = [a[2] for a in ranked_datas]
        return stock_ids, stock_scores, vec_values, stock_rank_labels, len(ranked_datas)


def train_lambda_rank(input_datas, group_datas, former_model=None, save_rounds=-1,
                      output_path='./models/lightgbm_model.mod',
                      num_boost_round=60000, early_stopping_rounds=30,
                      learning_rates=lambda iter_num: 0.05 * (0.99 ** iter_num) if iter_num < 1000 else 0.001,
                      params=None, thread_num=12):
    """
    使用LightGBM进行训练 lambdarank模型
    Args:
        group_datas:
        save_rounds: 保存的迭代间隔
        input_datas: load_csv_data函数的返回值
        former_model: 如果需要从已有模型继续训练,则导入
        thread_num: 并行训练数
        learning_rates: 学习率函数
        early_stopping_rounds: early stop次数
        num_boost_round: 迭代次数
        params: dict形式的参数
        output_path: 模型输出位置

    Returns:
        训练后的模型实例
    """
    import lightgbm as lgb
    # param = {'num_leaves': 31, 'num_trees': 100, 'objective': 'binary'}
    # num_round = 10
    data_process_logger.info('start training lightgbm')
    # train
    if not params:
        params = {
            'objective': 'lambdarank',
            'num_leaves': 128,
            'boosting': 'gbdt',
            'feature_fraction': 0.9,
            'bagging_fraction': 0.7,
            'bagging_freq': 100,
            'verbose': 0,
            'is_unbalance': False,
            'metric': 'l1,l2,huber',
            'num_threads': thread_num
            # 'group': group_datas
        }
    # Quant-data process
    # label_set = input_datas[:, 1]
    # vec_set = input_datas[:, 2:]
    data_process_logger.info('spliting feature datas')
    # label_set = [a[1] for a in input_datas]
    # vec_set = [a[2:] for a in input_datas]
    label_set = input_datas[0]
    vec_set = input_datas[1]
    data_process_logger.info('turning list into np2d-array')
    label_set = np.array(label_set)
    vec_set = np.array(vec_set)
    print 'dataset shape: ', vec_set.shape
    data_process_logger.info('training lightgbm')
    data_process_logger.info('params: \n%s' % params)
    data_process_logger.info('building dataset')
    train_set = lgb.Dataset(vec_set, label_set, group=group_datas, free_raw_data=False)
    data_process_logger.info('complete building dataset')
    # 处理存储间隔
    # gbm = former_model
    tmp_model = former_model
    tmp_num = num_boost_round
    while tmp_num > save_rounds > 0:
        gbm = lgb.train(params, train_set, num_boost_round=save_rounds,
                        early_stopping_rounds=early_stopping_rounds,
                        learning_rates=learning_rates,
                        valid_sets=[train_set],
                        init_model=tmp_model)
        # m_json = gbm.dump_model()
        data_process_logger.info('saving lightgbm during training')
        tmp_num -= save_rounds
        # save
        with open(output_path, 'wb') as fout:
            pickle.dump(gbm, fout, protocol=2)
        print 'saved model: %s' % output_path
        # tmp_model = cPickle.load(open(output_path, 'rb'))
        tmp_model = gbm
    gbm = lgb.train(params, train_set, num_boost_round=tmp_num,
                    early_stopping_rounds=early_stopping_rounds,
                    learning_rates=learning_rates,
                    valid_sets=[train_set],
                    init_model=tmp_model)
    data_process_logger.info('Final saving lightgbm')
    with open(output_path, 'wb') as fout:
        pickle.dump(gbm, fout)
    data_process_logger.info('saved model: %s' % output_path)
    return gbm


if __name__ == '__main__':
    process_single_pickle_data(
        '/Users/jayvee/CS/Python/FundDataAnalysis/datas/Quant-Datas-2.0/pickle_datas/1_trans_norm.pickle')
