import pandas as pd
import numpy as np
from train import trainer
from predict import predictor
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)
train_df = pd.read_csv(dir_path +'/data/train_df_test.csv')
test_df = pd.read_csv(dir_path +'/data/test_df_test.csv')
submission = pd.read_csv(dir_path +'/data/sample_submission.csv')

train_X = train_df.drop(['SalePrice','SalePrice_log','Id'], axis=1)
train_Y = train_df['SalePrice_log']
test_X = test_df.drop(['SalePrice','Id'], axis=1)


tr = trainer()
pred = predictor()
lgbm_params = {
    'objective':'regression',
    'random_seed':1234,
    'learning_rate':0.05,
    'n_estimators':1000,
    'num_leaves': 33,
    'max_bin': 125,
    'bagging_fraction': 0.7197362581993618,
    'bagging_freq': 4,
    'feature_fraction': 0.4684501358427995,
    'min_data_in_leaf': 14,
    'min_sum_hessian_in_leaf': 2,
}
xgb_params = {
    'learning_rate':0.05,
    'seed':1234,
    'max_depth':6,
    'colsample_bytree':0.330432640328732,
    'sublsample':0.7158427239902707
}

tr.train_lgb(train_X, train_Y, lgbm_params)
tr.train_xgb(train_X, train_Y, params=xgb_params)

pred.predict_lgb(tr.models_lgb[0], test_X)
pred.predict_xgb(tr.models_xgb[0], test_X)

preds_ans1 = pred.preds_lgb[0] * 0.5 + pred.preds_xgb[0] * 0.5
submission['SalePrice'] = preds_ans1
submission.to_csv(dir_path +'/submit/sample_submit.csv', index=False)