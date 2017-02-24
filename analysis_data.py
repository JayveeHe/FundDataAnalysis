# coding=utf-8

"""
Created by jayvee on 17/2/23.
https://github.com/JayveeHe
"""
import os
import sys
import pickle

import cPickle

from sklearn import grid_search
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

project_path = os.path.dirname(os.path.abspath(__file__))
print 'Related File:%s\t----------project_path=%s' % (__file__, project_path)
sys.path.append(project_path)

from logger_utils import data_process_logger


def load_csv_data(csv_path, normalize=True):
    with open(csv_path, 'rb') as fin:
        datas = []
        temp_list = []
        score_list = []
        for line in fin:
            line = line.strip()
            tmp = line.split(',')
            stock_id = tmp[0]
            trade_date = tmp[1]
            score = eval(tmp[2])
            score_list.append(score)
            vec_value = [eval(a) for a in tmp[3:]]
            temp_list.append((stock_id, trade_date, score, vec_value))
        if not normalize:
            return temp_list
        else:
            avg = np.mean(score_list)
            std = np.std(score_list)
            for item in temp_list:
                normalize_score = (item[2] - avg) / std
                datas.append((item[0], item[1], normalize_score, item[3]))
            return datas


def train_model(input_datas, output_path='./models/gbrt_model.mod'):
    label_set = []
    vec_set = []
    for i in range(len(input_datas)):
        label_set.append(input_datas[i][2])
        vec_set.append(input_datas[i][3])
    gbrt_model = GradientBoostingRegressor(n_estimators=301, loss='lad')
    print 'training'
    gbrt_model.fit(vec_set[:], label_set[:])
    print 'saving'
    with open(output_path, 'wb') as fout:
        pickle.dump(gbrt_model, fout)
    return gbrt_model


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
                       'max_leaf_nodes': [10, 20, 30], 'loss': ['huber', 'ls', 'lad']}
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
    input_ranked_list = sorted(input_datas, cmp=lambda x, y: 1 if x[2] - y[2] > 0 else -1)
    xlist = [a[3] for a in input_ranked_list]
    ylist = model.predict(xlist)
    index_ylist = [(i, ylist[i]) for i in range(len(ylist))]
    ranked_index_ylist = sorted(index_ylist, cmp=lambda x, y: 1 if x[1] - y[1] > 0 else -1)
    for i in range(len(ranked_index_ylist)):
        print 'pre: %s\t origin: %s\t delta: %s' % (i, ranked_index_ylist[i][0], i - ranked_index_ylist[i][0])
        if abs((i - ranked_index_ylist[i][0])) > 700 and i < 35:
            error_num += 1
    print "error num is %s" % (error_num)
    gap = result_validation(ranked_index_ylist)
    if gap == -1:
        print "happiness,the result is OK!"
    else:
        print "sad,the result is bad, the gap is %s" % (gap)


def result_validation(ranked_index_ylist, N=50, threshold=0.35):
    buyer_list = ranked_index_ylist[:N]
    total_error = 0
    origin_rank_list = []
    for i in range(len(buyer_list)):
        origin_rank_list.append(buyer_list[i][0])
        total_error += abs((buyer_list[i][0] - i))
    mean_rank = np.mean(origin_rank_list)
    print 'mean_rank = %s' % mean_rank
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
    train_datas = []
    # for i in range(1, 21):
    #     print 'loading %s file' % i
    #     datas = load_csv_data('./datas/%s.csv' % i)
    #     train_datas += datas
    # train_datas = load_csv_data('./datas/%s.csv' % 310)
    model_tag = 'norm_10'
    # output_path = './models/gbrt_model_%s.mod' % model_tag
    # train_model(train_datas, output_path)
    print '====================== 20 normalize test set'
    gbrt_mod = cPickle.load(open('./models/gbrt_model_%s.mod' % model_tag, 'rb'))
    datas = load_csv_data('./datas/4.csv', normalize=True)
    test_datas(datas, gbrt_mod)
    print '====================== 20 normalize train set'
    gbrt_mod = cPickle.load(open('./models/gbrt_model_%s.mod' % model_tag, 'rb'))
    datas = load_csv_data('./datas/310.csv', normalize=True)
    test_datas(datas, gbrt_mod)
    # out_data = normalize_data(train_datas)
    # for item in out_data:
    #	print 'item is : %s\t%s\t%s\n'%(item[0],item[2],item[3])
    # label_set = []
    # vec_set = []
    # for i in range(len(train_datas)):
    #    label_set.append(train_datas[i][2])
    #    vec_set.append(train_datas[i][3])
    ## gbrt_mod = train_model(train_datas)
    # model_tag = '50'
    # train_regression_age_model(input_xlist=vec_set, input_ylist=label_set, model_label=model_tag)
    ## xlist = [a[3] for a in datas[100:200]]
    ## ylist = [a[2] for a in datas[100:200]]
    ## cross_valid(xlist, ylist, gbrt_mod)
    # gbrt_mod = cPickle.load(open('%s/models/gbrt_%s.model' % (project_path, model_tag), 'rb'))
    # svr_mod = cPickle.load(open('%s/models/svr_%s.model' % (project_path, model_tag), 'rb'))
    # print '--------GBRT:----------'
    # datas = load_csv_data('./datas/310.csv')
    # data_process_logger.info('testing')
    # test_datas(datas, gbrt_mod)
    # print '==============='
    # datas = load_csv_data('./datas/4.csv')
    # test_datas(datas, gbrt_mod)
    # print '--------SVR:----------'
    # datas = load_csv_data('./datas/310.csv')
    # data_process_logger.info('testing')
    # test_datas(datas, svr_mod)
    # print '==============='
    # datas = load_csv_data('./datas/4.csv')
    # test_datas(datas, svr_mod)
