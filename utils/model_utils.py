# coding=utf-8

"""
Created by jayvee on 17/3/4.
https://github.com/JayveeHe
"""
from __future__ import division
import os
import sys

import cPickle
import numpy as np
from lightgbm import Booster
from sklearn import grid_search
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.externals import joblib

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print 'Related File:%s\t----------project_path=%s' % (__file__, PROJECT_PATH)
sys.path.append(PROJECT_PATH)

from utils.logger_utils import data_process_logger


def train_with_lightgbm(input_datas, former_model=None, save_rounds=-1, output_path='./models/lightgbm_model.mod',
                        num_boost_round=60000, early_stopping_rounds=30,
                        learning_rates=lambda iter_num: 0.05 * (0.99 ** iter_num) if iter_num < 1000 else 0.001,
                        params=None, thread_num=12):
    """
    使用LightGBM进行训练
    Args:
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
            'objective': 'regression_l2',
            'num_leaves': 128,
            'boosting': 'gbdt',
            'feature_fraction': 0.9,
            'bagging_fraction': 0.7,
            'bagging_freq': 100,
            'verbose': 0,
            'is_unbalance': False,
            'metric': 'l1,l2,huber',
            'num_threads': thread_num
        }
    # gbm = lgb.LGBMRegressor(objective='regression_l2',
    #                         num_leaves=31,
    #                         learning_rate=0.001,
    #                         n_estimators=50, nthread=2, silent=False)
    # params_grid = {'max_bin': [128, 255, 400], 'num_leaves': [21, 31],
    #                'learning_rate': [0.01, 0.1, 0.005], 'n_estimators': [11, 15, 21]}
    # gbm = GridSearchCV(gbm, params_grid, n_jobs=2)
    # label_set = []
    # vec_set = []
    # for i in range(len(input_datas)):
    #     label_set.append(input_datas[i][2])
    #     vec_set.append(input_datas[i][3])

    # Quant-data process
    # label_set = input_datas[:, 1]
    # vec_set = input_datas[:, 2:]
    data_process_logger.info('spliting feature datas')
    label_set = [a[1] for a in input_datas]
    vec_set = [a[2:] for a in input_datas]
    data_process_logger.info('turning list into np2d-array')
    label_set = np.array(label_set)
    vec_set = np.array(vec_set)
    print 'dataset shape: ', vec_set.shape
    data_process_logger.info('training lightgbm')
    data_process_logger.info('params: \n%s' % params)
    data_process_logger.info('building dataset')
    train_set = lgb.Dataset(vec_set, label_set, free_raw_data=False)
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
        # gbm.save_model(output_path)

        # print('Save model')
        # joblib.dump(gbm, output_path)
        # data_process_logger.info('saved model: %s' % output_path)
        # print('Load model')
        # gbm2 = joblib.load('gbm.pkl')

        with open(output_path, 'wb') as fout:
            cPickle.dump(gbm, fout, protocol=2)
        print 'saved model: %s' % output_path
        # tmp_model = cPickle.load(open(output_path, 'rb'))
        tmp_model = gbm
        # tmp_model = Booster(params=params, model_file=output_path)
        # with open(output_path, 'wb') as fout:
        #     cPickle.dump(tmp_model, fout)
    gbm = lgb.train(params, train_set, num_boost_round=tmp_num,
                    early_stopping_rounds=early_stopping_rounds,
                    learning_rates=learning_rates,
                    valid_sets=[train_set],
                    init_model=tmp_model)
    # gbm.fit()
    # data_process_logger.info('Best parameters found by grid search are: %s' % gbm.best_params_)
    data_process_logger.info('Final saving lightgbm')
    # joblib.dump(gbm, output_path)
    # gbm.save_model(output_path)
    # tmp_model = Booster(params=params, model_file=output_path)
    # with open(output_path, 'wb') as fout:
    #     cPickle.dump(tmp_model, fout)
    with open(output_path, 'wb') as fout:
        cPickle.dump(gbm, fout)
    data_process_logger.info('saved model: %s' % output_path)
    return gbm


def cv_with_lightgbm(input_datas, output_path='./models/lightgbm_model.mod',
                     num_boost_round=60000, early_stopping_rounds=30,
                     learning_rates=lambda iter_num: 0.05 * (0.99 ** iter_num), params=None):
    """
    使用LightGBM进行训练
    Args:
        early_stopping_rounds: early stop次数
        num_boost_round: 迭代次数
        params: dict形式的参数
        output_path:
        input_datas: load_csv_data函数的返回值

    Returns:

    """
    import lightgbm as lgb
    # param = {'num_leaves': 31, 'num_trees': 100, 'objective': 'binary'}
    # num_round = 10
    data_process_logger.info('start training lightgbm')
    # train
    if not params:
        params = {
            'boosting_type': 'gbdt',
            'objective': 'regression_l2',
            'num_leaves': 15,
            'boosting': 'dart',
            'feature_fraction': 0.9,
            'bagging_fraction': 0.7,
            'bagging_freq': 20,
            'verbose': 0,
            'metric': 'l1,l2,huber',
            'num_threads': 12
        }
    # gbm = lgb.LGBMRegressor(objective='regression_l2',
    #                         num_leaves=31,
    #                         learning_rate=0.001,
    #                         n_estimators=50, nthread=2, silent=False)
    # params_grid = {'max_bin': [128, 255, 400], 'num_leaves': [21, 31],
    #                'learning_rate': [0.01, 0.1, 0.005], 'n_estimators': [11, 15, 21]}
    # gbm = GridSearchCV(gbm, params_grid, n_jobs=2)
    label_set = []
    vec_set = []
    for i in range(len(input_datas)):
        label_set.append(input_datas[i][2])
        vec_set.append(input_datas[i][3])
    data_process_logger.info('training lightgbm')
    train_set = lgb.Dataset(vec_set, label_set)
    gbm = lgb.cv(params, train_set, num_boost_round=num_boost_round,
                 early_stopping_rounds=early_stopping_rounds,
                 learning_rates=learning_rates,
                 valid_sets=[train_set])
    # gbm.fit()
    # data_process_logger.info('Best parameters found by grid search are: %s' % gbm.best_params_)
    data_process_logger.info('saving lightgbm')
    with open(output_path, 'wb') as fout:
        cPickle.dump(gbm, fout)
    return gbm


def train_gbrt_model(input_datas, output_path='./models/gbrt_model.mod', n_estimators=500, loss='lad', subsample=0.7,
                     max_depth=4, max_leaf_nodes=10):
    """
    训练sklearn的gbrt模型
    Args:
        input_datas:
        output_path:
        n_estimators:
        loss:
        subsample:
        max_depth:
        max_leaf_nodes:

    Returns:

    """
    label_set = []
    vec_set = []
    for i in range(len(input_datas)):
        label_set.append(input_datas[i][2])
        vec_set.append(input_datas[i][3])
    gbrt_model = GradientBoostingRegressor(n_estimators=n_estimators, loss=loss, subsample=subsample,
                                           max_depth=max_depth, max_leaf_nodes=max_leaf_nodes)
    print 'training'
    gbrt_model.fit(vec_set[:], label_set[:])
    print 'saving'
    with open(output_path, 'wb') as fout:
        cPickle.dump(gbrt_model, fout)
    return gbrt_model


def train_svr_model(input_datas, output_path='./models/svr_model.mod', C=10, cache_size=200, coef0=0.0, degree=3,
                    epsilon=0.1, gamma=0.0001,
                    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False):
    """
    训练sklearn的svr模型
    Args:
        input_datas:
        output_path:
        C:
        cache_size:
        coef0:
        degree:
        epsilon:
        gamma:
        kernel:
        max_iter:
        shrinking:
        tol:
        verbose:

    Returns:

    """
    from sklearn.svm import SVR
    label_set = []
    vec_set = []
    for i in range(len(input_datas)):
        label_set.append(input_datas[i][2])
        vec_set.append(input_datas[i][3])
    svr_model = SVR(C=C, cache_size=cache_size, coef0=coef0, degree=degree, epsilon=epsilon, gamma=gamma,
                    kernel=kernel, max_iter=max_iter, shrinking=shrinking, tol=tol, verbose=verbose)
    print 'training'
    svr_model.fit(vec_set[:], label_set[:])
    print 'saving'
    with open(output_path, 'wb') as fout:
        cPickle.dump(svr_model, fout)
    return svr_model


def train_regression_age_model(input_xlist, input_ylist, model_label):
    """
    train age regression model, with grid search
    Args:
        input_xlist:
        input_ylist:
        model_label:

    Returns:

    """
    from sklearn import svm
    from sklearn.ensemble import GradientBoostingRegressor
    data_process_logger.info('loading model')
    input_xlist = np.float64(input_xlist)
    # SVR
    data_process_logger.info('training svr')
    clf = svm.SVR()
    parameters = {'C': [1e3, 5e3, 1e2, 1e1, 1e-1], 'kernel': ['rbf', 'sigmoid'],
                  'gamma': [0.0001, 0.001, 0.01, 0.1, 0.05]}
    svr_mod = grid_search.GridSearchCV(clf, parameters, n_jobs=12, scoring='mean_absolute_error')
    svr_mod.fit(input_xlist, input_ylist)
    print svr_mod.best_estimator_
    fout = open('%s/models/svr_%s.model' % (PROJECT_PATH, model_label), 'wb')
    cPickle.dump(svr_mod, fout)
    for item in svr_mod.grid_scores_:
        print item
    # GBRT
    data_process_logger.info('training gbrt')
    gbrt_mod = GradientBoostingRegressor()
    gbrt_parameters = {'n_estimators': [300, 350], 'max_depth': [2, 3, 4],
                       'max_leaf_nodes': [10, 20], 'loss': ['huber', 'lad'], 'subsample': [0.2, 0.5, 0.7]}
    gbrt_mod = grid_search.GridSearchCV(gbrt_mod, gbrt_parameters, n_jobs=12, scoring='mean_absolute_error')
    gbrt_mod.fit(input_xlist, input_ylist)
    gbrt_out = open('%s/models/gbrt_%s.model' % (PROJECT_PATH, model_label), 'wb')
    cPickle.dump(gbrt_mod, gbrt_out)
    print gbrt_mod.best_estimator_
    for item in gbrt_mod.grid_scores_:
        print item
        # clf.fit(reduced_xlist, input_ylist)


if __name__ == '__main__':
    model_tag = 'iter40000_norm_sample_700000'
    # train_datas = []

    # for i in range(1, 501):
    #    data_process_logger.info('loading %s file' % i)
    #    datas = load_csv_data('./datas/%s.csv' % i, normalize=True)
    #    train_datas += datas
    # dump normalized train datas
    # data_process_logger.info('dumping norm datas...')
    # cPickle.dump(train_datas, open('%s/datas/norm_datas/20_norm_datas.dat' % project_path, 'wb'))
    # load train normalized train datas
    # data_process_logger.info('loading datas...')
    # train_datas = cPickle.load(open('%s/datas/norm_datas/100_norm_datas.dat' % project_path, 'rb'))
    # random sample the train datas
    # SAMPLE_SIZE = 700000
    # data_process_logger.info('random sampling %s obs...' % SAMPLE_SIZE)
    # random.shuffle(train_datas)
    # train_datas = train_datas[:SAMPLE_SIZE]
    # start training
    # output_gbrt_path = './models/gbrt_%s.model' % model_tag
    # output_svr_path = './models/svr_%s.model' % model_tag
    # output_lightgbm_path = './models/lightgbm_%s.model' % model_tag
    # train_svr_model(train_datas, output_svr_path)
    # train_gbrt_model(train_datas, output_gbrt_path)

    # train_with_lightgbm(train_datas, output_lightgbm_path)
