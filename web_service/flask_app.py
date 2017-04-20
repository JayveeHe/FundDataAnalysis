# coding=utf-8

"""
Created by jayvee on 17/4/15.
https://github.com/JayveeHe
"""
import json

import re
from flask import Flask, render_template, request, make_response
from flask import redirect
from flask import url_for
from flask_uploads import UploadSet, DATA, configure_uploads

import os
import sys

try:
    import cPickle as pickle
except:
    import pickle
abs_path = os.path.dirname(os.path.abspath(__file__))
abs_father_path = os.path.dirname(abs_path)
PROJECT_PATH = abs_father_path
print 'Used file: %s\nProject path=%s' % (__file__, PROJECT_PATH)
sys.path.append(PROJECT_PATH)
# add flask path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipelines.process_real_datas import turn_csv_into_result
from utils.lightgbm_operator import LightgbmOperator
from utils.logger_utils import data_process_logger

app = Flask(__name__)

data_process_logger.info('initing lightGBM operator')

# init
oldbest_mod = pickle.load(open(
    '%s/models/best_models/lightgbm_New_Quant_Data_rebalanced_norm_gbdt_7leaves_iter30000_best.model' % PROJECT_PATH))
oldbest_mod.save_model('flask_model.txt', num_iteration=27000)
oldbest_operator = LightgbmOperator('flask_model.txt', 'New_Quant_Data_rebalanced_norm_gbdt_7leaves_iter30000_best')

full_mod = pickle.load(open(
    '%s/models/best_models/lightgbm_Full_gbdt_15leaves.model' % PROJECT_PATH))
full_mod.save_model('flask_model.txt', num_iteration=50000)
full_operator = LightgbmOperator('flask_model.txt', 'Full_gbdt_15leaves')

data_process_logger.info('complete init lightGBM')

# 初始化upload
csv_sets = UploadSet('datacsv', DATA)
app.config['UPLOADED_DATACSV_DEST'] = './static/csvs'
app.config['UPLOADED_DATACSV_ALLOW'] = DATA
configure_uploads(app, csv_sets)


@app.route('/predict_old', methods=['GET', 'POST'])
def upload_predict_old():
    if request.method == 'POST' and 'csv' in request.files:
        filename = csv_sets.save(request.files['csv'])
        test_csv_path = '%s/web_service/static/csvs/%s' % (PROJECT_PATH, filename)
        csv_dir_path = os.path.dirname(test_csv_path)
        csv_filename = re.findall('%s/(.*)\.csv' % csv_dir_path, test_csv_path)
        csv_output_dir = os.path.join(csv_dir_path, 'Old_Best_results')
        if not os.path.exists(csv_output_dir):
            os.mkdir(csv_output_dir)
        fout_csv_path = os.path.join(csv_output_dir, '%s_%s_result.csv' % (csv_filename[0], 'oldbest'))
        # predict
        fout_csv_path = turn_csv_into_result(test_csv_path, fout_csv_path, oldbest_operator.model,
                                             predict_iteration=None)
        if fout_csv_path:
            return redirect('static/csvs/Old_Best_results/%s_%s_result.csv' % (csv_filename[0], 'oldbest'))
        else:
            return None
    return render_template('demo.html', full_model_tag=full_operator.model_tag,
                           oldbest_model_tag=oldbest_operator.model_tag)


@app.route('/predict_full', methods=['GET', 'POST'])
def upload_predict_full():
    if request.method == 'POST' and 'csv' in request.files:
        filename = csv_sets.save(request.files['csv'])
        test_csv_path = '%s/web_service/static/csvs/%s' % (PROJECT_PATH, filename)
        csv_dir_path = os.path.dirname(test_csv_path)
        csv_filename = re.findall('%s/(.*)\.csv' % csv_dir_path, test_csv_path)
        csv_output_dir = os.path.join(csv_dir_path, 'Full_results')
        if not os.path.exists(csv_output_dir):
            os.mkdir(csv_output_dir)
        fout_csv_path = os.path.join(csv_output_dir, '%s_%s_result.csv' % (csv_filename[0], 'full'))
        # predict
        fout_csv_path = turn_csv_into_result(test_csv_path, fout_csv_path, full_operator.model,
                                             predict_iteration=None)
        if fout_csv_path:
            return redirect('static/csvs/Full_results/%s_%s_result.csv' % (csv_filename[0], 'full'))
        else:
            return None
    return render_template('demo.html', full_model_tag=full_operator.model_tag,
                           oldbest_model_tag=oldbest_operator.model_tag)


@app.route('/', methods=['GET'])
def index_page():
    return render_template('demo.html', full_model_tag=full_operator.model_tag,
                           oldbest_model_tag=oldbest_operator.model_tag)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2335, debug=False)
    # print url_for('static/csvs/Old_Best_results', filename='newdata_2739_1_oldbest_result.csv')
