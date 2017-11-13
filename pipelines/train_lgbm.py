# coding=utf-8

"""
Created by jayveehe on 2017/9/16.
http://code.dianpingoa.com/hejiawei03
"""
import os
import sys

import cPickle
import lightgbm as lgb
from lightgbm import Dataset

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print 'Related File:%s\t----------project_path=%s' % (__file__, PROJECT_PATH)
sys.path.append(PROJECT_PATH)

from utils.data_utils import gzip_sample_generator
from utils.logger_utils import data_process_logger


def train_lgb(model_output_path,
              valid_limit=500, thread_num=2, save_rounds=100, num_boost_round=2000,
              former_model_path=None, max_epochs=100, batch_size=100, nb_worker=4,
              mini_batch_size=3000, limit=2000, iteration_per_epoch=100):
    train_file_numbers = range(1, 540) + range(750, 800) + range(870, 920) + range(970, 1020) + range(1100, 1200)
    valid_file_numbers = range(400, 440) + range(700, 750) + range(845, 870) + range(945, 970) + range(1045, 1100)
    test_file_numbers = range(540, 640) + range(800, 845) + range(920, 945) + range(1020, 1045) + range(1200, 1214)
    DATA_ROOT = '/media/user/Data0/hjw/datas/Quant_Datas_v4.0/gzip_datas'
    train_filepath_list = [os.path.join(DATA_ROOT, '%s_trans.gz' % fn) for fn in train_file_numbers]
    valid_filepath_list = [os.path.join(DATA_ROOT, '%s_trans.gz' % fn) for fn in valid_file_numbers]
    # basic_model = build_model(feature_dim=4560, output_dim=1)
    # train_filepath_list = ['/Users/jayveehe/git_project/FundDataAnalysis/pipelines/datas/gzip_datas/993_trans.gz']
    # valid_filepath_list = ['/Users/jayveehe/git_project/FundDataAnalysis/pipelines/datas/gzip_datas/993_trans.gz']
    # train_generator = gzip_sample_generator(train_filepath_list, batch_size=1000, total_limit=1000000,
    #                                         per_file_limit=10000)
    valid_generator = gzip_sample_generator(valid_filepath_list, batch_size=50000, total_limit=100000,
                                            per_file_limit=10000)
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
    if former_model_path:
        former_model = cPickle.load(open(former_model_path, 'rb'))
    else:
        former_model = None
    tmp_model = former_model
    tmp_num = num_boost_round
    valid_x, valid_y = next(valid_generator)
    valid_set = Dataset(valid_x, valid_y, free_raw_data=False)
    gbm = None
    eval_res = {}
    for epoch in xrange(max_epochs):
        train_generator = gzip_sample_generator(train_filepath_list, batch_size=50000, total_limit=1000000,
                                                per_file_limit=10000)
        for iter_n in xrange(iteration_per_epoch):
            train_x, train_y = next(train_generator)
            tmp_dataset = Dataset(train_x, train_y, free_raw_data=False)
            # if not gbm:
            gbm = lgb.train(params, tmp_dataset, num_boost_round=save_rounds,
                            early_stopping_rounds=20, keep_training_booster=True,
                            learning_rates=lambda iter_num: max(1 * (0.98 ** iter_num / (iteration_per_epoch * 0.05)),
                                                                0.008),
                            valid_sets=[valid_set],
                            init_model=tmp_model, evals_result=eval_res)
            # else:
            #     gbm.update(train_set=tmp_dataset)
            tmp_model = gbm
        # print 'eval result: %s' % eval_res
        print 'saving model'
        gbm.save_model(model_output_path)

        # while tmp_num > save_rounds > 0:
        #     gbm = lgb.train(params, train_set, num_boost_round=save_rounds,
        #                     early_stopping_rounds=early_stopping_rounds,
        #                     learning_rates=lambda iter_num: max(1 * (0.98 ** iter_num / (num_total_iter * 0.05)),
        #                                                             0.008),
        #                     valid_sets=[eval_set],
        #                     init_model=tmp_model)
        #     # m_json = gbm.dump_model()
        #     data_process_logger.info('saving lightgbm during training')
        #     tmp_num -= save_rounds
        #     # save
        #     with open(output_path, 'wb') as fout:
        #         cPickle.dump(gbm, fout, protocol=2)
        #     print 'saved model: %s' % output_path
        #     # tmp_model = cPickle.load(open(output_path, 'rb'))
        #     tmp_model = gbm
        # if former_model_path:
        #     basic_model.load_weights(former_model_path)
        # train_filepath_list = ['/Users/jayveehe/git_project/FundDataAnalysis/pipelines/datas/tmp_data/993_trans_norm.gz']
        # valid_filepath_list = ['/Users/jayveehe/git_project/FundDataAnalysis/pipelines/datas/tmp_data/993_trans_norm.gz']

        # checkpointer = ModelCheckpoint(filepath=model_output_path, verbose=1, save_best_only=True)
        # early_stopper = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
        # basic_model.fit_generator(generator=train_generator,
        #                           nb_epoch=epochs,
        #                           steps_per_epoch=limit / batch_size,
        #                           # samples_per_epoch=limit,
        #                           validation_data=valid_generator,
        #                           validation_steps=valid_limit / batch_size,
        #                           # nb_val_samples=valid_limit,
        #                           nb_worker=nb_worker, pickle_safe=True, callbacks=[checkpointer, early_stopper])


if __name__ == '__main__':
    train_lgb('test_lgb.mod', thread_num=2, save_rounds=50, iteration_per_epoch=20)
