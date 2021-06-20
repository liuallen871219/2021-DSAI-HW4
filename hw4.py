import numpy as np 
import pandas as pd 
import os
import gc
import matplotlib.pylab as plt
all_data=pd.read_csv("train.csv")
data_path='competitive-data-science-predict-future-sales/'
submission = pd.read_csv(data_path + "sample_submission.csv")
# Train data (Features)
X_train = all_data[all_data['date_block_num'] < 33]
X_train = X_train.drop(['item_cnt_month'], axis=1)
# Valid data (Features)
X_valid = all_data[all_data['date_block_num'] == 33]
X_valid = X_valid.drop(['item_cnt_month'], axis=1)
# Test data (Features)
X_test = all_data[all_data['date_block_num'] == 34]
X_test = X_test.drop(['item_cnt_month'], axis=1)

# Train data (Target values)
y_train = all_data[all_data['date_block_num'] < 33]['item_cnt_month']
# Valid data (Target values)
y_valid = all_data[all_data['date_block_num'] == 33]['item_cnt_month']

del all_data
gc.collect()
import lightgbm as lgb
params = {'metric': 'rmse',
          'num_leaves': 376,
          'learning_rate': 0.005,
          'feature_fraction': 0.9,
          'force_col_wise' : True,
          'random_state': 10,
          'boosting_type':'goss'
             }

cat_features = ['shop_id', 'city', 'item_category_id', 'category', 'month']

# lgb train and valid dataset
dtrain = lgb.Dataset(X_train, y_train)
dvalid = lgb.Dataset(X_valid, y_valid)
 
# Train LightGBM model
lgb_model = lgb.train(params=params,
                      train_set=dtrain,
                      num_boost_round=2000,
                      valid_sets=(dtrain, dvalid),
                      early_stopping_rounds=150,
                      categorical_feature=cat_features,
                      verbose_eval=1,
                      )  
# preds = lgb_model.predict(X_test).clip(0,20)
preds = lgb_model.predict(X_test).clip(0,19)
# plt.figure(figsize=(12,6))
# lgb.plot_importance(lgb_model, max_num_features=30)
# plt.title("Featurertances")
# plt.show()
submission['item_cnt_month'] = preds
submission.to_csv('submission.csv', index=False)