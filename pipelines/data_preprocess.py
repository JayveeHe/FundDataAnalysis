# coding=utf-8

"""
Created by jayvee on 17/3/3.
https://github.com/JayveeHe
"""
import os
import sys
import numpy as np

from utils.logger_utils import data_process_logger

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print 'Related File:%s\t----------project_path=%s' % (__file__, PROJECT_PATH)
sys.path.append(PROJECT_PATH)


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


if __name__ == '__main__':
    print len(load_csv_data('%s/datas/%s.csv' % (PROJECT_PATH, 1), is_combine=True))
