import numpy as np 
import pandas as pd 
import os
import gc
def downcast(df,verbose=True):
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        dtype_name=df[col].dtype.name
        if dtype_name == "object":
            pass
        elif dtype_name =="bool":
            df[col]=df[col].astype("int8")
        elif dtype_name.startswith("int") or (df[col].round() == df[col]).all():
            df[col] = pd.to_numeric(df[col], downcast="integer")
        else:
            df[col]=pd.to_numeric(df[col], downcast='float')
        end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(col,dtype_name,'{:.1f}% compressed'.format(100 * (start_mem - end_mem) / start_mem))
    return df

data_path='competitive-data-science-predict-future-sales/'
sales_train = pd.read_csv(data_path + "sales_train.csv")
shops = pd.read_csv(data_path + "shops.csv")
items = pd.read_csv(data_path + "items.csv")
item_categories = pd.read_csv(data_path + "item_categories.csv")
test = pd.read_csv(data_path + "test.csv")
submission = pd.read_csv(data_path + "sample_submission.csv")

all_df = [sales_train, shops, items, item_categories, test]
for df in all_df:
    df = downcast(df)

# delete outliner data
sales_train = sales_train[sales_train['item_price'] > 0]
sales_train = sales_train[sales_train['item_price'] < 50000]
sales_train = sales_train[sales_train['item_cnt_day'] > 0]
sales_train = sales_train[sales_train['item_cnt_day'] < 1000]


sales_train.loc[sales_train['shop_id'] == 0, 'shop_id'] = 57
sales_train.loc[sales_train['shop_id'] == 1, 'shop_id'] = 58
sales_train.loc[sales_train['shop_id'] == 10, 'shop_id'] = 11
sales_train.loc[sales_train['shop_id'] == 39, 'shop_id'] = 40

test.loc[test['shop_id'] == 0, 'shop_id'] = 57
test.loc[test['shop_id'] == 1, 'shop_id'] = 58
test.loc[test['shop_id'] == 10, 'shop_id'] = 11
test.loc[test['shop_id'] == 39, 'shop_id'] = 40

unique_test_shop_id = test['shop_id'].unique()
sales_train = sales_train[sales_train['shop_id'].isin(unique_test_shop_id)]
shops['city'] = shops['shop_name'].apply(lambda x: x.split()[0])
shops.loc[shops['city'] =='!Якутск', 'city'] = 'Якутск'

from sklearn.preprocessing import LabelEncoder

# Create Label Encoder
label_encoder = LabelEncoder()
# City Feature Label Encoding 
shops['city'] = label_encoder.fit_transform(shops['city'])
shops = shops.drop('shop_name', axis=1)
items = items.drop(['item_name'],axis=1)
items['first_sale_date'] = sales_train.groupby("item_id").agg({'date_block_num':"min"})['date_block_num']
items['first_sale_date'] = items['first_sale_date'].fillna(34)
item_categories['category'] = item_categories['item_category_name'].apply(lambda x: x.split()[0])
label_encoder = LabelEncoder()
# Category Feature Label Encoding 
item_categories['category'] = label_encoder.fit_transform(item_categories['category'])
# Remove item_category_name feature
item_categories = item_categories.drop('item_category_name', axis=1)  

from itertools import product
train = []
# Create date_block_num, shop_id, item_id combination
for i in range(34):
    all_shop = sales_train.loc[sales_train['date_block_num']==i, 'shop_id'].unique()
    all_item = sales_train.loc[sales_train['date_block_num']==i, 'item_id'].unique()
    train.append(np.array(list(product([i], all_shop, all_item))))

index_features = ['date_block_num', 'shop_id', 'item_id'] # base features
train = pd.DataFrame(np.vstack(train), columns=index_features)
temp = sales_train.groupby(index_features).agg({'item_cnt_day': 'sum','item_price': 'mean'})
temp = temp.reset_index()
temp = temp.rename(columns={'item_cnt_day': 'item_cnt_month', 'item_price': 'item_price_mean'})
train = train.merge(temp, on=index_features, how='left')

# group variable garbage collection
del temp
gc.collect()

# Add a feature for the number of items sold
group = sales_train.groupby(index_features).agg({'item_cnt_day': 'count'})
group = group.reset_index()
group = group.rename(columns={'item_cnt_day': 'item_count'})

train = train.merge(group, on=index_features, how='left')

# Garbage collection
del group, sales_train
gc.collect()

# Set test data date_block_num to 34
test['date_block_num'] = 34

# Concatenate train and test
all_data = pd.concat([train, test.drop('ID', axis=1)],ignore_index=True,keys=index_features)
# Replace NaN with 0
all_data = all_data.fillna(0)
# Merge other data
all_data = all_data.merge(shops, on='shop_id', how='left')
all_data = all_data.merge(items, on='item_id', how='left')
all_data = all_data.merge(item_categories, on='item_category_id', how='left')

# Data downcasting
all_data = downcast(all_data)

# Garbage collection
del shops, items, item_categories
gc.collect()

def mean_features(df, mean_features, index_features):
    if len(index_features) == 2:
        feature_name = index_features[1] + '_mean_sales'
    else:
        feature_name = index_features[1] + '_' + index_features[2] + '_mean_sales'
    temp = df.groupby(index_features).agg({'item_cnt_month': 'mean'})
    temp = temp.reset_index()
    temp = temp.rename(columns={'item_cnt_month': feature_name}) 
    df = df.merge(temp, on=index_features, how='left')
    df = downcast(df, False)
    mean_features.append(feature_name)
    del temp
    gc.collect()
    return df, mean_features
# List of derived features containing 'item_id' in the grouping base features
item_mean_features = []


# Create month average sales features grouped by ['date_block_num', 'item_id']
all_data, item_mean_features = mean_features(df=all_data,mean_features=item_mean_features,index_features=['date_block_num', 'item_id'])

# Create month average sales features grouped by ['date_block_num', 'item_id', 'city']
all_data, item_mean_features = mean_features(df=all_data,mean_features=item_mean_features,index_features=['date_block_num', 'item_id', 'city'])
# List of derived features containing 'shop_id' in the grouping base features
shop_mean_features = []

# Create monthly average sales derived features grouped by ['date_block_num', 'shop_id', 'item_category_id']
all_data, shop_mean_features = mean_features(df=all_data, mean_features=shop_mean_features,index_features=['date_block_num', 'shop_id', 'item_category_id'])
def lag_features(df, lag_features_to_clip, index_features, lag_feature, nlags=3, clip=False):
    df_temp = df[index_features + [lag_feature]].copy() 

    for i in range(1, nlags+1):
        lag_feature_name = lag_feature +'_lag' + str(i)
        df_temp.columns = index_features + [lag_feature_name]
        df_temp['date_block_num'] += i
        df = df.merge(df_temp.drop_duplicates(), on=index_features, how='left')
        df[lag_feature_name] = df[lag_feature_name].fillna(0)
        if clip: 
            lag_features_to_clip.append(lag_feature_name)  
    df = downcast(df, False)
    del df_temp
    gc.collect()
    return df, lag_features_to_clip

lag_features_to_clip = [] # list of lag features to be clipped to between 0 to 20 
index_features = ['date_block_num', 'shop_id', 'item_id'] # base features

# Create 3 month lag features of item_cnt_month based on index_features
all_data, lag_features_to_clip = lag_features(df=all_data, 
                                                  lag_features_to_clip=lag_features_to_clip,
                                                  index_features=index_features,
                                                  lag_feature='item_cnt_month', 
                                                  nlags=3,
                                                  clip=True)
# Create 3 month lag features of item_count feature based on index_features
all_data, lag_features_to_clip = lag_features(df=all_data, 
                                                  lag_features_to_clip=lag_features_to_clip,
                                                  index_features=index_features,
                                                  lag_feature='item_count', 
                                                  nlags=3)

# Create 3 month lag features of item_price_mean feature based on index_features
all_data, lag_features_to_clip = lag_features(df=all_data, 
                                                  lag_features_to_clip=lag_features_to_clip,
                                                  index_features=index_features,
                                                  lag_feature='item_price_mean', 
                                                  nlags=3)
X_test_temp = all_data[all_data['date_block_num'] == 34]
X_test_temp[item_mean_features].sum()

# Create lag features by item_mean_features element based on dx_features
for item_mean_feature in item_mean_features:
    all_data, lag_features_to_clip = lag_features(df=all_data, 
                                                      lag_features_to_clip=lag_features_to_clip, 
                                                      index_features=index_features, 
                                                      lag_feature=item_mean_feature, 
                                                      nlags=3)
# Remove features in item_mean_features
all_data = all_data.drop(item_mean_features, axis=1)

# Create lag features by shop_mean_features element based on ['date_block_num', 'shop_id', 'item_category_id']
for shop_mean_feature in shop_mean_features:
    all_data, lag_features_to_clip = lag_features(df=all_data,
                                                      lag_features_to_clip=lag_features_to_clip, 
                                                      index_features=['date_block_num', 'shop_id', 'item_category_id'], 
                                                      lag_feature=shop_mean_feature, 
                                                      nlags=3)
# Remove features in shop_mean_features
all_data = all_data.drop(shop_mean_features, axis=1)
# Remove data less than date_block_num 3
all_data = all_data.drop(all_data[all_data['date_block_num'] < 3].index)
all_data['item_cnt_month_lag_mean'] = all_data[['item_cnt_month_lag1','item_cnt_month_lag2', ]].mean(axis=1)

# Clip 0~19
all_data[lag_features_to_clip + ['item_cnt_month', 'item_cnt_month_lag_mean']] = all_data[lag_features_to_clip +['item_cnt_month', 'item_cnt_month_lag_mean']].clip(0, 19)

# all_data['lag_grad1'] = all_data['item_cnt_month_lag2']/all_data['item_cnt_month_lag1']
# all_data['lag_grad1'] = all_data['lag_grad1'].replace([np.inf, -np.inf], np.nan).fillna(0)

# all_data['lag_grad2'] = all_data['item_cnt_month_lag3']/all_data['item_cnt_month_lag2']
# all_data['lag_grad2'] = all_data['lag_grad2'].replace([np.inf, -np.inf],np.nan).fillna(0)


all_data['month'] = all_data['date_block_num']%12
# Remove item_price_mean, item_count features
all_data = all_data.drop(['item_price_mean', 'item_count'], axis=1)
all_data = downcast(all_data, False) # Data downcasting
all_data.info()

all_data.to_csv("train.csv",index=False)