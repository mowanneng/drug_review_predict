import pandas as pd
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, precision_score, recall_score
from data_process import DataProcess
from data_process import GetUsefulLevel
import os
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
import warnings
import pickle
import json
from collections import Counter

def importance_features_top(model_str, model, x_train):
    """打印模型的重要指标，排名top10指标"""
    print("print XGBoost importance features")
    feature_importances_ = model.feature_importances_
    feature_names = x_train.columns
    importance_col = pd.DataFrame([*zip(feature_names, feature_importances_)],
                                  columns=['a', 'b'])
    importance_col_desc = importance_col.sort_values(by='b', ascending=False)
    print(importance_col_desc)


def print_precison_recall_f1(y_true, y_pre):
    """打印精准率、召回率和F1值"""
    print(classification_report(y_true, y_pre))
    f1_mac = round(f1_score(y_true, y_pre, average='macro'), 5)
    p_mac = round(precision_score(y_true, y_pre, average='macro'), 5)
    r_mac = round(recall_score(y_true, y_pre, average='macro'), 5)

    print("Precision_mac: {}, Recall_mac: {}, F1_mac: {} ".format(p_mac, r_mac, f1_mac))

    f1_mic = round(f1_score(y_true, y_pre, average='micro'), 5)
    p_mic = round(precision_score(y_true, y_pre, average='micro'), 5)
    r_mic = round(recall_score(y_true, y_pre, average='micro'), 5)
    print("Precision_mic: {}, Recall_mic: {}, F1_mic: {} ".format(p_mic, r_mic, f1_mic))
    score = {'macro': {'f1_mac': f1_mac, 'p_mac': p_mac, 'r_mac': r_mac},
             'micro': {'f1_mic': f1_mic, 'p_mic': p_mic, 'r_mic': r_mic}}
    return score

def get_data(path,mode='train'):
    df = pd.read_csv(path)
    df.dropna(axis=0, how='any', inplace=True)
    df.drop(axis=1, columns='recordId', inplace=True)
    data_process = DataProcess()
    df['reviewComment'] = df['reviewComment'].map(data_process.text_replace)  # 替换评论数据中的乱码字符
    df['reviewComment'] = df['reviewComment'].map(data_process.sen_analy)  # 获取评论数据情感倾向
    df['date'] = df['date'].map(data_process.time_format)  # 获取年份数据
    df['sideEffects'] = data_process.side_effect_level(df['sideEffects'])  # 对sideEffects划分等级
    useful_level = GetUsefulLevel(df['usefulCount'])
    df['usefulCount']= df['usefulCount'].map(useful_level.get_current_level)  # 获取count的分级，分为10级

    columns = ['drugName', 'condition']
    for column in columns:
        if mode=='train':
            df[column] = data_process.target_encoder(df[column], df['rating'],column)  # 对drugName进行目标编码
        else:
            filename = column + '.json'
            base_path = r'D:\Python\drug_review_predict\save_file'
            save_path = os.path.join(base_path, filename)
            with open(save_path, 'r') as f:
                target_dict = json.load(f)
            df[column] = df[column].map(target_dict)
    return df

def oversamp_data(train_data):
    '''对标签为1~4的数据进行过采样'''
    x_new_data=pd.DataFrame(columns=train_data.columns[:-1])
    y_new_data = pd.DataFrame(columns=[train_data.columns[-1]])
    counter=Counter(y_new_data)
    for label in range(1, 5):
        smo = SMOTE(random_state=0)
        x_train = train_data[train_data['rating'].isin([label, 5])].iloc[:, :-1]
        y_train = pd.DataFrame(train_data[train_data['rating'].isin([label, 5])].iloc[:, -1], columns=['rating'])
        x_train_smo, y_train_smo = smo.fit_resample(x_train, y_train)
        x_new_data = x_new_data.append(x_train_smo[y_train_smo['rating'] == label], ignore_index=True)
        y_new_data = y_new_data.append(y_train_smo[y_train_smo['rating'] == label], ignore_index=True)

    x_new_data = x_new_data.append(train_data[train_data['rating'] == 5].iloc[:, :-1], ignore_index=True)  # 将标签为5的数据合并
    y_new_data = y_new_data.append(pd.DataFrame(train_data[train_data['rating'] == 5].iloc[:, -1], columns=['rating']),
                                   ignore_index=True)  # 将标签为5的数据合并
    return x_new_data,y_new_data


def xgb_train(x_train, y_train,find_params=False):
    if find_params:
        cv_params = {'n_estimators': [400, 500, 600], 'max_depth': [5, 6, 7, 8, 9, 10],
                     'min_child_weight': [1, 2, 3, 4], 'gamma': [0.1, 0.2, 0.3, 0.4],
                     'subsample': [0.6, 0.7, 0.8],
                     'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
                     'learning_rate': [0.05, 0.07, 0.1, 0.2]}
        other_params = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1,
                                     'seed': 0,
                                     'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0,
                                     'reg_lambda': 1}
        model = XGBClassifier(**other_params,num_class=5)
        grid = GridSearchCV(estimator=model, param_grid=cv_params, scoring='f1_micro', cv=3, n_jobs=4)
        grid.fit(x_train, y_train)
        best_estimator = grid.best_estimator_
        return(best_estimator)

    else:
        xgboost_clf = XGBClassifier(learning_rate=0.1, min_child_weight=6, max_depth=8,
                                    objective='multi:softmax', num_class=5)

        xgboost_clf.fit(x_train.values, y_train.values)
        return xgboost_clf


if __name__ == '__main__':
    # warnings.filterwarnings("ignore")
    path = r'D:\Python\drug_review_predict\dataset'
    train_path = os.path.join(path, 'training.csv')
    val_path = os.path.join(path, 'validation.csv')
    train_data = get_data(train_path,mode='train')
    x_train,y_train=oversamp_data(train_data)
    xgb_model = xgb_train(x_train, y_train,find_params=False)
    print('Training score:')
    train_pred = xgb_model.predict(x_train)
    pred = train_pred.tolist()
    y_true= y_train['rating'].values.tolist()
    train_score = print_precison_recall_f1(y_true, pred)

    val_data=get_data(val_path,mode='val')
    x_val=val_data.iloc[:,:-1]
    y_val=val_data.iloc[:,-1]
    print('Validation score:')
    val_pred = xgb_model.predict(x_val)
    val_pred = val_pred.tolist()
    y_val = y_val.tolist()
    val_score = print_precison_recall_f1(y_val, val_pred)
    pickle.dump(xgb_model, open(r"D:\Python\drug_review_predict\save_file\xgb.pkl", "wb"))










