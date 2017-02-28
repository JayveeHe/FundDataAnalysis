# coding=utf-8

"""
Created by jayvee on 17/2/23.
https://github.com/JayveeHe
"""
import os
import random
import sys
import pickle
import time

import cPickle

from sklearn import grid_search
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

project_path = os.path.dirname(os.path.abspath(__file__))
print 'Related File:%s\t----------project_path=%s' % (__file__, project_path)
sys.path.append(project_path)

from utils.logger_utils import data_process_logger


def load_csv_data(csv_path, normalize=True):
    from sklearn import preprocessing
    with open(csv_path, 'rb') as fin:
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
            vec_scale_list = list(vec_scale)
            for i in range(len(id_list)):
                datas.append((id_list[i], date_list[i], score_scale_list[i], list(vec_scale_list[i])))
            # avg = np.mean(score_list)
            #            std = np.std(score_list)
            #            for item in temp_list:
            #                normalize_score = (item[2] - avg) / std
            #                datas.append((item[0], item[1], normalize_score, item[3]))
            return datas


def train_gbrt_model(input_datas, output_path='./models/gbrt_model.mod', n_estimators=300, loss='lad', subsample=0.7,
                     max_depth=4, max_leaf_nodes=10):
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
        pickle.dump(gbrt_model, fout)
    return gbrt_model


def train_svr_model(input_datas, output_path='./models/svr_model.mod', C=10, cache_size=200, coef0=0.0, degree=3,
                    epsilon=0.1, gamma=0.0001,
                    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False):
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
        pickle.dump(svr_model, fout)
    return svr_model


def train_regression_age_model(input_xlist, input_ylist, model_label):
    """
    train age regression model
    :param input_xlist:
    :param input_ylist:
    :param model_label:
    :return:
    """
    from sklearn import svm
    from sklearn.ensemble import GradientBoostingRegressor
    data_process_logger.info('loading model')
    input_xlist = np.float64(input_xlist)
    # SVR
    data_process_logger.info('training svr')
    clf = svm.SVR()
    parameters = {'C': [1e3, 5e3, 1e4, 5e4, 1e5, 1e2, 1e1, 1e-1], 'kernel': ['rbf', 'sigmoid'],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 0.05]}
    svr_mod = grid_search.GridSearchCV(clf, parameters, n_jobs=10, scoring='mean_absolute_error')
    svr_mod.fit(input_xlist, input_ylist)
    print svr_mod.best_estimator_
    fout = open('%s/models/svr_%s.model' % (project_path, model_label), 'wb')
    cPickle.dump(svr_mod, fout)
    for item in svr_mod.grid_scores_:
        print item
    # GBRT
    data_process_logger.info('training gbrt')
    gbrt_mod = GradientBoostingRegressor()
    gbrt_parameters = {'n_estimators': [100, 200, 300, 350], 'max_depth': [2, 3, 4],
                       'max_leaf_nodes': [10, 20, 30], 'loss': ['huber', 'ls', 'lad'], 'subsample': [0.2, 0.5, 0.7]}
    gbrt_mod = grid_search.GridSearchCV(gbrt_mod, gbrt_parameters, n_jobs=10, scoring='mean_absolute_error')
    gbrt_mod.fit(input_xlist, input_ylist)
    gbrt_out = open('%s/models/gbrt_%s.model' % (project_path, model_label), 'wb')
    cPickle.dump(gbrt_mod, gbrt_out)
    print gbrt_mod.best_estimator_
    for item in gbrt_mod.grid_scores_:
        print item
        # clf.fit(reduced_xlist, input_ylist)


def cross_valid(input_x_datas, input_y_datas, cv_model):
    cv = cross_val_score(cv_model, X=input_x_datas, y=input_y_datas, cv=5, scoring='mean_squared_error')
    print cv


def test_datas(input_datas, model):
    error_num = 0
    input_ranked_list = sorted(input_datas, cmp=lambda x, y: 1 if x[2] - y[2] < 0 else -1)
    xlist = [a[3] for a in input_ranked_list]
    ylist = model.predict(xlist)
    index_ylist = [(i, ylist[i]) for i in range(len(ylist))]
    ranked_index_ylist = sorted(index_ylist, cmp=lambda x, y: 1 if x[1] - y[1] < 0 else -1)
    for i in range(len(ranked_index_ylist)):
        data_process_logger.info('pre: %s\t origin: %s\t delta: %s\tpredict_score: %s' % (
            i, ranked_index_ylist[i][0], i - ranked_index_ylist[i][0], ranked_index_ylist[i][1]))
        if abs((i - ranked_index_ylist[i][0])) > 700 and i < 35:
            error_num += 1
    data_process_logger.info("error num is %s" % (error_num))
    gap = result_validation(ranked_index_ylist)
    if gap == -1:
        data_process_logger.info("happiness,the result is OK!")
    else:
        data_process_logger.info("sad,the result is bad, the gap is %s" % (gap))


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
    std_rank = np.std(origin_rank_list)
    data_process_logger.info('std_rank = %s' % std_rank)
    if mean_rank <= threshold * len(ranked_index_ylist):
        # if total_error <= threshold:
        return -1
    else:
        return (total_error - threshold * len(ranked_index_ylist)) / float(threshold * len(ranked_index_ylist))


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


if __name__ == '__main__':
    start = time.time()
    model_tag = 'norm_5'
    train_datas = []

    for i in range(1, 11):
        print 'loading %s file' % i
        datas = load_csv_data('./datas/%s.csv' % i, normalize=True)
        train_datas += datas
    # random sample the train datas
    data_process_logger.info('random sampling...')
    SAMPLE_SIZE = 5000
    random.shuffle(train_datas)
    train_datas = train_datas[:SAMPLE_SIZE]
    # start training
    output_gbrt_path = './models/gbrt_%s.model' % model_tag
    output_svr_path = './models/svr_%s.model' % model_tag
    train_svr_model(train_datas, output_svr_path)
    train_gbrt_model(train_datas, output_gbrt_path)

    # print '====================== 20 normalize test set'
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
    # train_regression_age_model(input_xlist=vec_set, input_ylist=label_set, model_label=model_tag)
    # end = time.time()
    # print "spend the time %s" % (end - start)
    ## xlist = [a[3] for a in datas[100:200]]
    ## ylist = [a[2] for a in datas[100:200]]
    ## cross_valid(xlist, ylist, gbrt_mod)
    gbrt_mod = cPickle.load(open('%s/models/gbrt_%s.model' % (project_path, model_tag), 'rb'))
    data_process_logger.info('--------gbrt:----------')
    datas = load_csv_data('./datas/1.csv')
    data_process_logger.info('testing')
    test_datas(datas, gbrt_mod)
    data_process_logger.info('===============')
    datas = load_csv_data('./datas/150.csv')
    test_datas(datas, gbrt_mod)
    data_process_logger.info('--------SVR:----------')
    svr_mod = cPickle.load(open('%s/models/svr_%s.model' % (project_path, model_tag), 'rb'))
    datas = load_csv_data('./datas/1.csv')
    data_process_logger.info('testing')
    test_datas(datas, svr_mod)
    data_process_logger.info('===============')
    datas = load_csv_data('./datas/150.csv')
    test_datas(datas, svr_mod)
