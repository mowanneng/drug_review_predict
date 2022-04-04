import pandas as pd
from data_process import DataProcess
from data_process import GetUsefulLevel
import os
from xgboost.sklearn import XGBClassifier
from xgboost.sklearn import Booster
import json
import pickle
def get_data(path):
    df = pd.read_csv(path)
    df.drop(axis=1, columns='recordId', inplace=True)
    df.drop(axis=1, columns='rating', inplace=True)
    df.dropna(axis=0, how='any', inplace=True)
    data_process = DataProcess()
    df['reviewComment'] = df['reviewComment'].map(data_process.text_replace)  # 替换评论数据中的乱码字符
    df['reviewComment'] = df['reviewComment'].map(data_process.sen_analy)  # 获取评论数据情感倾向
    df['date'] = df['date'].map(data_process.time_format)  # 获取年份数据
    df['sideEffects'] = data_process.side_effect_level(df['sideEffects'])  # 对sideEffects划分等级
    useful_level = GetUsefulLevel(df['usefulCount'])
    df['usefulCount'] = df['usefulCount'].map(useful_level.get_current_level)  # 获取count的分级，分为25级
    columns = ['drugName', 'condition']
    for column in columns:
        filename = column + '.json'
        base_path = r'D:\Python\drug_review_predict\save_file'
        save_path = os.path.join(base_path, filename)
        with open(save_path, 'r') as f:
            target_dict = json.load(f)
        df[column] = df[column].map(target_dict)
    return df
if __name__ == '__main__':
    test_path=r'D:\Python\drug_review_predict\dataset\testing.csv'
    result_path=r'D:\Python\drug_review_predict\result\result.csv'
    source_data=pd.read_csv(test_path)
    source_data.drop(axis=1, columns='rating', inplace=True)
    source_data.dropna(axis=0, how='any', inplace=True)
    test_data = get_data(test_path)
    xgb_model = pickle.load(open("../save_file/xgb.pkl", "rb"))
    test_pred = xgb_model.predict(test_data)
    pred = test_pred.tolist()
    source_data['rating']=pred
    source_data.to_csv(result_path)
    print('完成testing数据预测')


