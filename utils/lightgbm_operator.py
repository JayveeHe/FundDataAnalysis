# coding=utf-8

"""
Created by jayvee on 17/4/15.
https://github.com/JayveeHe
"""
from lightgbm import Booster


class LightgbmOperator(object):
    def __init__(self, bst_path, model_tag):
        """
        初始化
        Args:
            bst_path: 通过model.save()保存的地址
        """
        self.model = Booster(model_file=bst_path)
        self.model_tag = model_tag

    def predict(self, input_datas):
        # if not isinstance(input_datas,list) and not isinstance(input_datas,np.array):
        return self.model.predict(input_datas)
