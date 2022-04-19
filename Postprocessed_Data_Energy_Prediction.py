#%% [code]
# v2: 1.102507841834652
# v9 : del area_floor
# 10: remove 1099
# 11: dayweek
# 12 : del bil_median
# 13 : leak data update
# 14 : site-0 unit correction
# sg filter

#v3 : add diff2 (bug)
#v4 : add diff2
#v5 : black 10
black_day = 10
outlier = False
rescale = False

debug=False
num_rounds = 200

clip0=False # minus meter confirmed in test(site0 leak data)

folds = 3 # 3, 6, 12
# 6: 1.1069822104487446
# 3: 1.102507841834652
# 12: 1.1074824417420517

use_ucf=False
ucf_clip=False

ucf_year = [2017, 2018] # ucf data year used in train 

predmode='all' # 'valid', train', 'all'

#%% [code]
import warnings
warnings.filterwarnings('ignore')
import gc
import os
from pathlib import Path
import random
import sys

from tqdm import tqdm_notebook as tqdm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

from IPython.core.display import display, HTML

# --- models ---
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

from sklearn.metrics import mean_squared_error

#%% [code]

os.listdir('/home/joydipb/Documents/CMT307-Coursework-2-Group-19')

#%% [code]
# Original code from https://www.kaggle.com/gemartin/load-data-reduce-memory-usage by @gemartin
# Modified to support timestamp type, categorical type
# Modified to add option to use float16 or not. feather format does not support float16.
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype

def reduce_mem_usage(df, use_float16=False):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        if is_datetime(df[col]) or is_categorical_dtype(df[col]):
            # skip datetime type or categorical type
            continue
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


#%%[code]
zone_dict={0:4,1:0,2:7,3:4,4:7,5:0,6:4,7:4,8:4,9:5,10:7,11:4,12:0,13:5,14:4,15:4} 

def set_local(df):
    for sid, zone in zone_dict.items():
        sids = df.site_id == sid
        df.loc[sids, 'timestamp'] = df[sids].timestamp - pd.offsets.Hour(zone)

#%%[code]
! ls '/home/joydipb/Documents/CMT307-Coursework-2-Group-19'


#%% [code]

root = Path('/home/joydipb/Documents/CMT307-Coursework-2-Group-19')
train_df = pd.read_feather(root/'train.feather')
weather_train_df = pd.read_feather(root/'weather_train.feather')
building_meta_df = pd.read_feather(root/'building_metadata.feather')

building_meta_df = building_meta_df.merge(train_df[['building_id','meter']].drop_duplicates(), on='building_id')

#Set group  (site-meter) for training models

building_meta_df['groupNum_train'] = building_meta_df['site_id'].astype('int')*10 + building_meta_df['meter'].astype('int')

building_meta_df

#%% [code]
print('Shape of Train Data:',train_df.shape)
print('Shape of Building Data:', building_meta_df.shape)
print('Shape of Weather Train Data:', weather_train_df.shape)



#%% [code]

#remove buildings

train_df = train_df [ train_df['building_id'] != 1099 ]

building_meta_df['floor_area'] = building_meta_df.square_feet / building_meta_df.floor_count

#%% [code]
print('Shape of Train Data:',train_df.shape)
print('Shape of Building Data:', building_meta_df.shape)
print('Shape of Weather Train Data:', weather_train_df.shape)

# %%
# Site Specific Holiday

import holidays

en_holidays = holidays.England()
ir_holidays = holidays.Ireland()
ca_holidays = holidays.Canada()
us_holidays = holidays.UnitedStates()

def add_holiyday(df_weather):
    en_idx = df_weather.query('site_id == 1 or site_id == 5').index
    ir_idx = df_weather.query('site_id == 12').index
    ca_idx = df_weather.query('site_id == 7 or site_id == 11').index
    us_idx = df_weather.query('site_id == 0 or site_id == 2 or site_id == 3 or site_id == 4 or site_id == 6 or site_id == 8 or site_id == 9 or site_id == 10 or site_id == 13 or site_id == 14 or site_id == 15').index

    df_weather['IsHoliday'] = 0
    df_weather.loc[en_idx, 'IsHoliday'] = df_weather.loc[en_idx, 'timestamp'].apply(lambda x: en_holidays.get(x, default=0))
    df_weather.loc[ir_idx, 'IsHoliday'] = df_weather.loc[ir_idx, 'timestamp'].apply(lambda x: ir_holidays.get(x, default=0))
    df_weather.loc[ca_idx, 'IsHoliday'] = df_weather.loc[ca_idx, 'timestamp'].apply(lambda x: ca_holidays.get(x, default=0))
    df_weather.loc[us_idx, 'IsHoliday'] = df_weather.loc[us_idx, 'timestamp'].apply(lambda x: us_holidays.get(x, default=0))

    holiday_idx = df_weather['IsHoliday'] != 0
    df_weather.loc[holiday_idx, 'IsHoliday'] = 1
    df_weather['IsHoliday'] = df_weather['IsHoliday'].astype(np.uint8)
# %% [code]

set_local(weather_train_df)
add_holiyday(weather_train_df)

# %% [code]
weather_train_df.head()

# %% [code]
# Threshold By Black day¶

# # Count zero streak
train_df_black = train_df.copy()
train_df_black = train_df_black.merge(building_meta_df, on=['building_id', 'meter'], how='left')
train_df_black = train_df_black.merge(weather_train_df, on=['site_id', 'timestamp'], how='left')

train_df_black['black_count']=0

for bid in train_df_black.building_id.unique():
    df = train_df_black[train_df_black.building_id==bid]
    for meter in df.meter.unique():
        dfm = df[df.meter == meter]
        b = (dfm.meter_reading == 0).astype(int)
        train_df_black.loc[(train_df_black.building_id==bid) & (train_df_black.meter == meter), 'black_count'] = b.groupby((~b.astype(bool)).cumsum()).cumsum()

# %% [code]
train_df_black[train_df_black.building_id == 954].black_count.plot()


# %% [code]
train_df = train_df.merge(train_df_black[['timestamp','building_id','meter','black_count']], on=['timestamp','building_id','meter'])

# %% [code]
train_df = train_df[train_df['black_count'] < 24*black_day].drop('black_count', axis=1)
# %% [code]
train_df.shape

# %% [code]
del train_df_black
gc.collect()
# %% [code]
## Removing weired data on site_id 0
#building_meta_df[building_meta_df.site_id == 0]
train_df = train_df.query('not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")')

# %% [code]
train_df[train_df.building_id == 954].meter_reading.plot()

# %% [code]
(train_df[train_df.building_id == 954].meter_reading==0).astype(int).plot()

# %% [code]
train_df[train_df.building_id == 1221].meter_reading.plot()

# %% [code]
train_df = train_df.query('not (building_id == 954 & meter_reading == 0)')
train_df = train_df.query('not (building_id == 1221 & meter_reading == 0)')

# %% [code]
train_df[train_df.building_id == 954].meter_reading.plot()

# %% [code]
train_df[train_df.building_id == 1221].meter_reading.plot()

# %% [code]
##Delete Outliear¶

funny_bids = [993, 1168,  904,  954,  778, 1021]

print ('before', len(train_df))

if outlier:
    #993
    # or delete
    train_df.loc[(train_df.building_id == 993) & (train_df.meter == 0) & (train_df.meter_reading > 30000), 'meter_reading'] = 31921
    train_df.loc[(train_df.building_id == 993) & (train_df.meter == 1) & (train_df.meter_reading > 90000), 'meter_reading'] =  96545.5

    #1168
    train_df = train_df[((train_df.building_id == 1168) & (train_df.meter == 0) & (train_df.meter_reading >10000)) == False]

    #904
    train_df.loc[(train_df.building_id == 904) & (train_df.meter == 0)& (train_df.meter_reading >10000), 'meter_reading'] = 11306

    #954
    train_df = train_df[((train_df.building_id == 954) & (train_df.meter_reading >10000))==False]

if rescale:
    #778 rescale ?
    train_df.loc[(train_df.building_id == 778) & (train_df.meter == 1), 'meter_reading'] = train_df.loc[(train_df.building_id == 778) & (train_df.meter == 1), 'meter_reading']/ 100
    #
    #1021 rescale ?
    train_df.loc[(train_df.building_id == 1021) & (train_df.meter == 3), 'meter_reading'] = train_df.loc[(train_df.building_id == 1021) & (train_df.meter == 3), 'meter_reading']/ 1000
    #plt.plot(np.log1p(train_df.loc[(train_df.building_id == 1021) & (train_df.meter == 3), 'meter_reading'] ))

train_df = train_df.reset_index()


print ('after', len(train_df))
gc.collect()


# %% [code]
for bid in funny_bids:
    plt.figure(figsize=[20,3])
    plt.subplot(141)
    plt.plot(train_df[(train_df.building_id == bid) & (train_df.meter == 0)].meter_reading)
    plt.subplot(142)
    plt.plot(train_df[(train_df.building_id == bid) & (train_df.meter == 1)].meter_reading)
    plt.subplot(143)
    plt.plot(train_df[(train_df.building_id == bid) & (train_df.meter == 2)].meter_reading)
    plt.subplot(144)
    plt.plot(train_df[(train_df.building_id == bid) & (train_df.meter == 3)].meter_reading)
    plt.title(bid)
# %% [code]
## Site-0 Correction¶
# https://www.kaggle.com/c/ashrae-energy-prediction/discussion/119261#latest-684102
site_0_bids = building_meta_df[building_meta_df.site_id == 0].building_id.unique()
print (len(site_0_bids), len(train_df[train_df.building_id.isin(site_0_bids)].building_id.unique()))
train_df[train_df.building_id.isin(site_0_bids) & (train_df.meter==0)].head(50)

# %% [code]
train_df.loc[(train_df.building_id.isin(site_0_bids)) & (train_df.meter==0), 'meter_reading'] = train_df[(train_df.building_id.isin(site_0_bids)) & (train_df.meter==0) ]['meter_reading'] * 0.2931

# %% [code]
train_df[(train_df.building_id.isin(site_0_bids)) & (train_df.meter==0)].head(50)

# %% [code]
## Data preprocessing

train_df['date'] = train_df['timestamp'].dt.date
train_df['meter_reading_log1p'] = np.log1p(train_df['meter_reading'])

# %% [markdown]
""" # Add time feature¶

Some features introduced in https://www.kaggle.com/ryches/simple-lgbm-solution by @ryches
Features that are likely predictive:

Buildings:-

primary_use,
square_feet,
year_built,
floor_count (may be too sparse to use),
Weather,

time of day :-
holiday,
weekend,
cloud_coverage + lags,
dew_temperature + lags,
precip_depth + lags,
sea_level_pressure + lags,
wind_direction + lags,
wind_speed + lags,

Train:-

max, mean, min, std of the specific building historically,
number of meters,
number of buildings at a siteid.
 """
# %% [code]

def preprocess(df):
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["weekend"] = df["timestamp"].dt.weekday
    df["month"] = df["timestamp"].dt.month
    df["dayofweek"] = df["timestamp"].dt.dayofweek

#     hour_rad = df["hour"].values / 24. * 2 * np.pi
#     df["hour_sin"] = np.sin(hour_rad)
#     df["hour_cos"] = np.cos(hour_rad)


# %% [code]
preprocess(train_df)

# %% [code]
## sort train
if use_ucf:
    train_df = train_df.sort_values('month')
    train_df = train_df.reset_index()
# %% [code]
#df_group = train_df.groupby('building_id')['meter_reading_log1p']
#building_median = df_group.median().astype(np.float16)
#train_df['building_median'] = train_df['building_id'].map(building_median)

# %% [code]
## Fill Nan value in weather dataframe by interpolation¶
weather_train_df.head()

# %% [code]
weather_train_df.describe()

# %% [code]
weather_train_df.isna().sum()

# %% [code]
weather_train_df.shape

# %% [code]
weather_train_df.groupby('site_id').apply(lambda group: group.isna().sum())

# %% [code]
weather_train_df = weather_train_df.groupby('site_id').apply(lambda group: group.interpolate(method ='ffill', limit_direction ='forward'))
#weather_train_df.interpolate(method='ffill', axis=0, limit=None, inplace=False, limit_direction='forward', limit_area=None, downcast=None)

# %% [code]
weather_train_df.groupby('site_id').apply(lambda group: group.isna().sum())

# %% [code]
## Adding some lag feature

def add_lag_feature(weather_df, window=3):
    group_df = weather_df.groupby('site_id')
    cols = ['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction', 'wind_speed']
    rolled = group_df[cols].rolling(window=window, min_periods=0)
    lag_mean = rolled.mean().reset_index().astype(np.float16)
    lag_max = rolled.max().reset_index().astype(np.float16)
    lag_min = rolled.min().reset_index().astype(np.float16)
    lag_std = rolled.std().reset_index().astype(np.float16)
    for col in cols:
        weather_df[f'{col}_mean_lag{window}'] = lag_mean[col]
        weather_df[f'{col}_max_lag{window}'] = lag_max[col]
        weather_df[f'{col}_min_lag{window}'] = lag_min[col]
        weather_df[f'{col}_std_lag{window}'] = lag_std[col]


# %% [code]
add_lag_feature(weather_train_df, window=3)
add_lag_feature(weather_train_df, window=72)

# %% [code]
weather_train_df.head()

# %% [code]
weather_train_df.columns

# %% [code]
weather_train_df.groupby('site_id').apply(lambda group: group.isna().sum())

# %% [code]
## count encoding

year_map = building_meta_df.year_built.value_counts()
building_meta_df['year_cnt'] = building_meta_df.year_built.map(year_map)

bid_map = train_df.building_id.value_counts()
train_df['bid_cnt'] = train_df.building_id.map(bid_map)
# %% [code]
# categorize primary_use column to reduce memory on merge...

primary_use_list = building_meta_df['primary_use'].unique()
primary_use_dict = {key: value for value, key in enumerate(primary_use_list)} 
print('primary_use_dict: ', primary_use_dict)
building_meta_df['primary_use'] = building_meta_df['primary_use'].map(primary_use_dict)

gc.collect()
# %% [code]
train_df = reduce_mem_usage(train_df, use_float16=True)
building_meta_df = reduce_mem_usage(building_meta_df, use_float16=True)
weather_train_df = reduce_mem_usage(weather_train_df, use_float16=True)
# %% [code]
building_meta_df.head()

# %% [code]
print('Shape of Train Data:',train_df.shape)
print('Shape of Building Data:', building_meta_df.shape)
print('Shape of Weather Train Data:', weather_train_df.shape)


# %% [code]
## SG Filter for Weather

from scipy.signal import savgol_filter as sg

def add_sg(df):
    w = 11
    p = 2
    for si in df.site_id.unique():
        index = df.site_id == si
        df.loc[index, 'air_smooth'] = sg(df[index].air_temperature, w, p)
        df.loc[index, 'dew_smooth'] = sg(df[index].dew_temperature, w, p)
        
        df.loc[index, 'air_diff'] = sg(df[index].air_temperature, w, p, 1)
        df.loc[index, 'dew_diff'] = sg(df[index].dew_temperature, w, p, 1)
        
        df.loc[index, 'air_diff2'] = sg(df[index].air_temperature, w, p, 2)
        df.loc[index, 'dew_diff2'] = sg(df[index].dew_temperature, w, p, 2)


# %% [code]

add_sg(weather_train_df)


# %% [code]
weather_train_df[weather_train_df.site_id==1].air_temperature[:100].plot()
weather_train_df[weather_train_df.site_id==1].air_smooth[:100].plot()

# %% [code]
weather_train_df[weather_train_df.site_id==2].dew_temperature[:100].plot()
weather_train_df[weather_train_df.site_id==2].dew_smooth[:100].plot()
# %% [code]
weather_train_df[weather_train_df.site_id==0].dew_diff[:100].plot()
weather_train_df[weather_train_df.site_id==0].air_diff[:100].plot()

# %% [markdown]
 ### For time series data, it is better to consider time-splitting.
 ## However just to keep it simple, I am using K-fold cross validation

# %% [code]
## Train Model 

category_cols = ['building_id', 'site_id', 'primary_use', 'IsHoliday', 'groupNum_train']  # , 'meter'
feature_cols = ['square_feet', 'year_built'] + [
    'hour', 'weekend',
#    'day', # 'month' ,
#    'dayofweek',
#    'building_median'
    ] + [
    'air_temperature', 'cloud_coverage',
    'dew_temperature', 'precip_depth_1_hr',
    'sea_level_pressure',
#'wind_direction', 'wind_speed',
    'air_temperature_mean_lag72',
    'air_temperature_max_lag72', 'air_temperature_min_lag72',
    'air_temperature_std_lag72', 'cloud_coverage_mean_lag72',
    'dew_temperature_mean_lag72', 'precip_depth_1_hr_mean_lag72',
    'sea_level_pressure_mean_lag72',
#'wind_direction_mean_lag72',
    'wind_speed_mean_lag72', 
    'air_temperature_mean_lag3',
    'air_temperature_max_lag3',
    'air_temperature_min_lag3', 'cloud_coverage_mean_lag3',
    'dew_temperature_mean_lag3',
    'precip_depth_1_hr_mean_lag3',
    'sea_level_pressure_mean_lag3',
#    'wind_direction_mean_lag3', 'wind_speed_mean_lag3',
#    'floor_area',
    'year_cnt', 'bid_cnt',
    'dew_smooth', 'air_smooth',
    'dew_diff', 'air_diff',
    'dew_diff2', 'air_diff2'
] #+ list(df_groupNum_median.drop('timestamp',axis=1).columns)

# %% [code]
train_df = train_df.merge(building_meta_df, on=['building_id','meter'], how='left')

# %% [code]

train_df = reduce_mem_usage(train_df, use_float16=True)
weather_train_df = reduce_mem_usage(weather_train_df, use_float16=True)
# %% [code]
train_df = train_df.merge(weather_train_df, on=['site_id','timestamp'], how='left')

# %% [code]
train_df = reduce_mem_usage(train_df, use_float16=True)

del weather_train_df
gc.collect()

# %% [code]
def create_X_y(train_df, groupNum_train):
    
    target_train_df = train_df[train_df['groupNum_train'] == groupNum_train].copy()        
    # target_train_df = target_train_df.merge(df_groupNum_median, on=['timestamp'], how='left')
    # target_train_df['group_median_'+str(groupNum_train)] = np.nan
    
    X_train = target_train_df[feature_cols + category_cols]
    y_train = target_train_df['meter_reading_log1p'].values
    
    del target_train_df
    return X_train, y_train

# %% [code]

def fit_lgbm(train, val, devices=(-1,), seed=None, cat_features=None, num_rounds=1500, lr=0.1, bf=0.1):
    """Train Light GBM model"""
    X_train, y_train = train
    X_valid, y_valid = val
    metric = 'l2'
    params = {'num_leaves': 31,
              'objective': 'regression',
#               'max_depth': -1,
              'learning_rate': lr,
              "boosting": "gbdt",
              "bagging_freq": 5,
              "bagging_fraction": bf,
              "feature_fraction": 0.9,
              "metric": metric,
#               "verbosity": -1,
#               'reg_alpha': 0.1,
#               'reg_lambda': 0.3
              }
    device = devices[0]
    if device == -1:
        # use cpu
        pass
    else:
        # use gpu
        print(f'using gpu device_id {device}...')
        params.update({'device': 'gpu', 'gpu_device_id': device})

    params['seed'] = seed

    early_stop = 20
    verbose_eval = 50

    d_train = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
    d_valid = lgb.Dataset(X_valid, label=y_valid, categorical_feature=cat_features)
    watchlist = [d_train, d_valid]

    print('training LGB:')
    model = lgb.train(params,
                      train_set=d_train,
                      num_boost_round=num_rounds,
                      valid_sets=watchlist,
                      verbose_eval=verbose_eval,
                      early_stopping_rounds=early_stop)

    # predictions
    y_pred_valid = model.predict(X_valid, num_iteration=model.best_iteration)
    
    print('best_score', model.best_score)
    log = {'train/mae': model.best_score['training']['l2'],
           'valid/mae': model.best_score['valid_1']['l2']}
    return model, y_pred_valid, log

# %% [code]
from sklearn.model_selection import GroupKFold, StratifiedKFold

seed = 666
shuffle = False
#kf = KFold(n_splits=folds, shuffle=shuffle, random_state=seed)
#kf = GroupKFold(n_splits=folds)
kf = StratifiedKFold(n_splits=folds)

# %% [markdown]
### Train model by each group # (site-meter)

# %% [code]

def plot_feature_importance(model):
    importance_df = pd.DataFrame(model[1].feature_importance(),
                                 index=feature_cols + category_cols,
                                 columns=['importance']).sort_values('importance')
    fig, ax = plt.subplots(figsize=(8, 8))
    importance_df.plot.barh(ax=ax)
    fig.show()

# %% [code]

## Exporting Train Data to use in other models
train_df.to_feather('train_df_processed.feather')

# %% [code]
## Traning the Light GBM Model
gc.collect()

for groupNum_train in building_meta_df['groupNum_train'].unique():
    X_train, y_train = create_X_y(train_df, groupNum_train=groupNum_train)
    y_valid_pred_total = np.zeros(X_train.shape[0])
    gc.collect()
    print('groupNum_train', groupNum_train, X_train.shape)

    cat_features = [X_train.columns.get_loc(cat_col) for cat_col in category_cols]
    print('cat_features', cat_features)

    exec('models' +str(groupNum_train)+ '=[]')

    train_df_site = train_df[train_df['groupNum_train']==groupNum_train].copy()
    
    #for train_idx, valid_idx in kf.split(X_train, y_train):
    #for train_idx, valid_idx in kf.split(X_train, y_train, groups=get_groups(train_df, groupNum_train)):    
    for train_idx, valid_idx in kf.split(train_df_site, train_df_site['building_id']):
        train_data = X_train.iloc[train_idx,:], y_train[train_idx]
        valid_data = X_train.iloc[valid_idx,:], y_train[valid_idx]

        mindex = train_df_site.iloc[valid_idx,:].month.unique()
        print (mindex)

        print('train', len(train_idx), 'valid', len(valid_idx))
    #     model, y_pred_valid, log = fit_cb(train_data, valid_data, cat_features=cat_features, devices=[0,])
        model, y_pred_valid, log = fit_lgbm(train_data, valid_data, cat_features=category_cols,
                                            num_rounds=num_rounds, lr=0.05, bf=0.7)
        y_valid_pred_total[valid_idx] = y_pred_valid
        exec('models' +str(groupNum_train)+ '.append([mindex, model])')        
        gc.collect()
        if debug:
            break

    try:
        sns.distplot(y_train)
        sns.distplot(y_valid_pred_total)
        plt.show()
    except:
        pass

    del X_train, y_train
    gc.collect()
    
    print('-------------------------------------------------------------')

# %% [code]

# Prediction on test data¶

print('loading...')
test_df = pd.read_feather(root/'test.feather')
weather_test_df = pd.read_feather(root/'weather_test.feather')

print('Before Preprocessing ....')
print('Shape of test data: ', test_df.shape)
print('Shape of Weather test data: ',weather_test_df.shape)

weather_test_df = weather_test_df.drop_duplicates(['timestamp', 'site_id'])
set_local(weather_test_df)
add_holiyday(weather_test_df)

print('preprocessing building...')
test_df['date'] = test_df['timestamp'].dt.date
preprocess(test_df)
#test_df['building_median'] = test_df['building_id'].map(building_median)

print('preprocessing weather...')
weather_test_df = weather_test_df.groupby('site_id').apply(lambda group: group.interpolate(method ='ffill', limit_direction ='forward'))
weather_test_df.groupby('site_id').apply(lambda group: group.isna().sum())

add_sg(weather_test_df)

add_lag_feature(weather_test_df, window=3)
add_lag_feature(weather_test_df, window=72)

#bid_map = train_df.building_id.value_counts()
test_df['bid_cnt'] = test_df.building_id.map(bid_map)

test_df = test_df.merge(building_meta_df[['building_id','meter','groupNum_train']], on=['building_id','meter'], how='left')
              
print('reduce mem usage...')
test_df = reduce_mem_usage(test_df, use_float16=True)
weather_test_df = reduce_mem_usage(weather_test_df, use_float16=True)

gc.collect()

print('After Preprocessing ....')
print('Shape of test data: ', test_df.shape)
print('Shape of Weather test data: ',weather_test_df.shape)



# %% [code]
sample_submission = pd.read_feather(os.path.join(root, 'sample_submission.feather'))
reduce_mem_usage(sample_submission)

print(sample_submission.shape)

# %% [code]

def create_X(test_df, groupNum_train):
    
    target_test_df = test_df[test_df['groupNum_train'] == groupNum_train].copy()        
    # target_test_df = target_test_df.merge(df_groupNum_median, on=['timestamp'], how='left')
    target_test_df = target_test_df.merge(building_meta_df, on=['building_id','meter','groupNum_train'], how='left')
    target_test_df = target_test_df.merge(weather_test_df, on=['site_id', 'timestamp'], how='left')
    # target_test_df['group_median_'+str(groupNum_train)] = np.nan

    X_test = target_test_df[feature_cols + category_cols]
    
    return X_test

# %% [code]

def pred_all(X_test, models, batch_size=1000000):
    iterations = (X_test.shape[0] + batch_size -1) // batch_size
    print('iterations', iterations)

    y_test_pred_total = np.zeros(X_test.shape[0])
    for i, (mindex, model) in enumerate(models):
        print(f'predicting {i}-th model')
        for k in tqdm(range(iterations)):
            y_pred_test = model.predict(X_test[k*batch_size:(k+1)*batch_size], num_iteration=model.best_iteration)
            y_test_pred_total[k*batch_size:(k+1)*batch_size] += y_pred_test

    y_test_pred_total /= len(models)
    return y_test_pred_total


def pred(X_test, models, batch_size=1000000):
    if predmode == 'valid':
        print ('valid pred')
        return pred_valid(X_test, models, batch_size=1000000)
    elif predmode == 'train':
        print ('train pred')
        return pred_train(X_test, models, batch_size=1000000)
    else:
        print ('all pred')
        return pred_all(X_test, models, batch_size=1000000)

# %% [code]
for groupNum_train in building_meta_df['groupNum_train'].unique():
    print('groupNum_train: ', groupNum_train)
    X_test = create_X(test_df, groupNum_train=groupNum_train)
    gc.collect()

    exec('y_test= pred(X_test, models' +str(groupNum_train)+ ')')

    sns.distplot(y_test)
    plt.show()

    print(X_test.shape, y_test.shape)
    sample_submission.loc[test_df["groupNum_train"] == groupNum_train,"meter_reading"] = np.expm1(y_test)
    
    del X_test, y_test
    gc.collect()

# %% [code]
# Exporting Test data, building metadata, and weather data after preprocessing 
# To be used in other models.
test_df.to_feather('test_df_processed.feather')
weather_test_df.to_feather('weather_test_df_processed.feather')
building_meta_df.to_feather('building_meta_df_processed.feather')

# %% [markdown]
### site-0 correction 

# %% [code]

# https://www.kaggle.com/c/ashrae-energy-prediction/discussion/119261#latest-684102
sample_submission.loc[(test_df.building_id.isin(site_0_bids)) & (test_df.meter==0), 'meter_reading'] = sample_submission[(test_df.building_id.isin(site_0_bids)) & (test_df.meter==0)]['meter_reading'] * 3.4118




# %% [code]
if rescale:   
    sample_submission.loc[(test_df.building_id == 778) & (test_df.meter == 1), 'meter_reading'] = 100 * sample_submission.loc[(test_df.building_id == 778) & (test_df.meter == 1), 'meter_reading']
    sample_submission.loc[(test_df.building_id == 1021) & (test_df.meter == 3), 'meter_reading'] = 1000 * sample_submission.loc[(test_df.building_id == 1021) & (test_df.meter == 3), 'meter_reading']
    
    plt.figure()
    plt.subplot(211)
    sample_submission.loc[(test_df.building_id == 778) & (test_df.meter == 1), 'meter_reading'].plot()
    plt.subplot(212)    
    sample_submission.loc[(test_df.building_id == 1021) & (test_df.meter == 3), 'meter_reading'].plot()

# %% [code]
if clip0:
    sample_submission.loc[sample_submission.meter_reading < 0, 'meter_reading'] = 0


# %% [code]
sample_submission.head()

# %% [code]
sample_submission.tail()

# %% [code]
print('Shape of Sample Submission', sample_submission.shape)

# %% [code]
if not debug:
    sample_submission.to_csv('submission.csv', index=False, float_format='%.4f')

# %% [code]
np.log1p(sample_submission['meter_reading']).hist(bins=100)
# %%
