#%%

print("----- Importing Libraries -----")
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/home/joydipb/Documents/CMT307-Coursework-2-Group-19/Keras Neural Network'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

from keras.models import Model, load_model
from keras.layers import Input, Dropout, Dense, Embedding, SpatialDropout1D, concatenate, BatchNormalization, Flatten
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from keras import backend as K
from keras.models import Model
from keras.losses import mean_squared_error as mse_loss

from keras import optimizers
from tensorflow.keras.optimizers import RMSprop, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import holidays
from pathlib import Path
import gc

from pandas.api.types import is_categorical_dtype
from pandas.api.types import is_datetime64_any_dtype as is_datetime

from scipy.signal import savgol_filter as sg

import warnings
warnings.filterwarnings('ignore')

print("----- Setting the Path -----")

root = Path('/home/joydipb/Documents/CMT307-Coursework-2-Group-19')


# %%

# Reading the Dataset

print("----- Importing Datasets -----")

building_meta_df = pd.read_feather(root/"building_metadata.feather")
weather_train_df = pd.read_feather(root/"weather_train.feather")
train_df = pd.read_feather(root/"train.feather")

# %%
print("----- Creating Methods for Imputation -----")
def average_imputation(df, column_name):
    imputation = df.groupby(['timestamp'])[column_name].mean()
    
    df.loc[df[column_name].isnull(), column_name] = df[df[column_name].isnull()][[column_name]].apply(lambda x: imputation[df['timestamp'][x.index]].values)
    del imputation
    return df
# %% [code]
# Original code from https://www.kaggle.com/gemartin/load-data-reduce-memory-usage by @gemartin
# Modified to support timestamp type, categorical type
# Modified to add option to use float16 or not. feather format does not support float16.

print("----- Creating Methods to reduce memory -----")
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
    print('Decreased by {:.1f}%'.format(
        100 * (start_mem - end_mem) / start_mem))

    return df

# %% [code]

print("----- Processing Buidling Data -----")

building_meta_df = building_meta_df.merge(
    train_df[['building_id', 'meter']].drop_duplicates(), on='building_id')

# Set group  (site-meter) for training models

building_meta_df['groupNum_train'] = building_meta_df['site_id'].astype(
    'int')*10 + building_meta_df['meter'].astype('int')

building_meta_df['floor_area'] = building_meta_df.square_feet / \
    building_meta_df.floor_count

print(building_meta_df)

# remove buildings
print("----- Processing train Data -----")

train_df = train_df[train_df['building_id'] != 1099]

print(train_df)



# %% [code]

print("----- Shape of the Dataset -----")

print('Shape of Train Data:', train_df.shape)
print('Shape of Building Data:', building_meta_df.shape)
print('Shape of Weather Train Data:', weather_train_df.shape)

# %%[code]
print("----- Setting Methods for time-zone -----")
''' Time zones are determined by the Site Analysis python file '''
zone_dict = {0: 4, 1: 0, 2: 7, 3: 4, 4: 7, 5: 0, 6: 4, 7: 4,
             8: 4, 9: 5, 10: 7, 11: 4, 12: 0, 13: 5, 14: 4, 15: 4}


def set_local(df):
    for sid, zone in zone_dict.items():
        sids = df.site_id == sid
        df.loc[sids, 'timestamp'] = df[sids].timestamp - pd.offsets.Hour(zone)

# %% [code]

print("----- Setting Methods for holiday flag -----")

'''The holiday flag has been determined using time zone and site analysis python file'''
# Site Specific Holiday
en_holidays = holidays.England()
ir_holidays = holidays.Ireland()
ca_holidays = holidays.Canada()
us_holidays = holidays.UnitedStates()


def add_holiyday(df_weather):
    en_idx = df_weather.query('site_id == 1 or site_id == 5').index
    ir_idx = df_weather.query('site_id == 12').index
    ca_idx = df_weather.query('site_id == 7 or site_id == 11').index
    us_idx = df_weather.query(
        'site_id == 0 or site_id == 2 or site_id == 3 or site_id == 4 or site_id == 6 or site_id == 8 or site_id == 9 or site_id == 10 or site_id == 13 or site_id == 14 or site_id == 15').index

    df_weather['IsHoliday'] = 0
    df_weather.loc[en_idx, 'IsHoliday'] = df_weather.loc[en_idx,
                                                         'timestamp'].apply(lambda x: en_holidays.get(x, default=0))
    df_weather.loc[ir_idx, 'IsHoliday'] = df_weather.loc[ir_idx,
                                                         'timestamp'].apply(lambda x: ir_holidays.get(x, default=0))
    df_weather.loc[ca_idx, 'IsHoliday'] = df_weather.loc[ca_idx,
                                                         'timestamp'].apply(lambda x: ca_holidays.get(x, default=0))
    df_weather.loc[us_idx, 'IsHoliday'] = df_weather.loc[us_idx,
                                                         'timestamp'].apply(lambda x: us_holidays.get(x, default=0))

    holiday_idx = df_weather['IsHoliday'] != 0
    df_weather.loc[holiday_idx, 'IsHoliday'] = 1
    df_weather['IsHoliday'] = df_weather['IsHoliday'].astype(np.uint8)

# %% [code]

print("----- Processing Weather Data  -----")

set_local(weather_train_df)
add_holiyday(weather_train_df)

# %% [code]
# Threshold By Black day
black_day = 10
print("----- Processing train Data and removing black days -----")

'''Some building ID's have zero meter reading, hence removing those,
some meters have zero reading before certain dates, hence removing those,
resetting the index, not sure but that works well in the Model'''

# # Count zero streak
train_df_black = train_df.copy()
train_df_black = train_df_black.merge(
    building_meta_df, on=['building_id', 'meter'], how='left')
train_df_black = train_df_black.merge(
    weather_train_df, on=['site_id', 'timestamp'], how='left')

train_df_black['black_count'] = 0

for bid in train_df_black.building_id.unique():
    df = train_df_black[train_df_black.building_id == bid]
    for meter in df.meter.unique():
        dfm = df[df.meter == meter]
        b = (dfm.meter_reading == 0).astype(int)
        train_df_black.loc[(train_df_black.building_id == bid) & (
            train_df_black.meter == meter), 'black_count'] = b.groupby((~b.astype(bool)).cumsum()).cumsum()

# train_df_black[train_df_black.building_id == 954].black_count.plot()

train_df = train_df.merge(train_df_black[['timestamp', 'building_id', 'meter', 'black_count']], on=[
                          'timestamp', 'building_id', 'meter'])

train_df = train_df[train_df['black_count'] <
                    24*black_day].drop('black_count', axis=1)

del train_df_black
gc.collect()


train_df = train_df.query(
    'not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")')

train_df = train_df.query('not (building_id == 954 & meter_reading == 0)')
train_df = train_df.query('not (building_id == 1221 & meter_reading == 0)')

train_df = train_df.reset_index()

train_df.shape

gc.collect()
# %% [code]
# Site-0 Correction
# https://www.kaggle.com/c/ashrae-energy-prediction/discussion/119261#latest-684102

print("----- Correction in site-0 Data  -----")

site_0_bids = building_meta_df[building_meta_df.site_id ==
                               0].building_id.unique()
print(len(site_0_bids), len(
    train_df[train_df.building_id.isin(site_0_bids)].building_id.unique()))
train_df[train_df.building_id.isin(
    site_0_bids) & (train_df.meter == 0)].head(10)

train_df.loc[(train_df.building_id.isin(site_0_bids)) & (train_df.meter == 0), 'meter_reading'] = train_df[(
    train_df.building_id.isin(site_0_bids)) & (train_df.meter == 0)]['meter_reading'] * 0.2931

train_df[(train_df.building_id.isin(site_0_bids))
         & (train_df.meter == 0)].head(10)

# %% [code]
# Data preprocessing

train_df['date'] = train_df['timestamp'].dt.date
train_df['meter_reading_log1p'] = np.log1p(train_df['meter_reading'])
# %% [code]

def preprocess(df):
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["weekend"] = df["timestamp"].dt.weekday
    df["month"] = df["timestamp"].dt.month
    df["dayofweek"] = df["timestamp"].dt.dayofweek



preprocess(train_df)
# %% [code]
weather_train_df.groupby('site_id').apply(lambda group: group.isna().sum())

# %% [code]
weather_train_df = weather_train_df.groupby('site_id').apply(
    lambda group: group.interpolate(method='ffill', limit_direction='forward'))


# %% [code]
weather_train_df.groupby('site_id').apply(lambda group: group.isna().sum())
# %%
# Adding some lag feature


def add_lag_feature(weather_df, window=3):
    group_df = weather_df.groupby('site_id')
    cols = ['air_temperature', 'cloud_coverage', 'dew_temperature',
            'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction', 'wind_speed']
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
# %%
add_lag_feature(weather_train_df, window=3)
add_lag_feature(weather_train_df, window=72)
# %%
# count encoding

year_map = building_meta_df.year_built.value_counts()
building_meta_df['year_cnt'] = building_meta_df.year_built.map(year_map)

bid_map = train_df.building_id.value_counts()
train_df['bid_cnt'] = train_df.building_id.map(bid_map)
# %%
# categorize primary_use column to reduce memory on merge...

primary_use_list = building_meta_df['primary_use'].unique()
primary_use_dict = {key: value for value, key in enumerate(primary_use_list)}
print('primary_use_dict: ', primary_use_dict)
building_meta_df['primary_use'] = building_meta_df['primary_use'].map(
    primary_use_dict)

gc.collect()
# %%
train_df = reduce_mem_usage(train_df, use_float16=True)
building_meta_df = reduce_mem_usage(building_meta_df, use_float16=True)
weather_train_df = reduce_mem_usage(weather_train_df, use_float16=True)
# %%
# SG Filter for Weather


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
# %%
# feature Selection

category_cols = ['building_id', 'site_id', 'primary_use',
                 'IsHoliday', 'groupNum_train']  # , 'meter'
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
    # 'wind_direction_mean_lag72',
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
]
# %%
train_df = train_df.merge(building_meta_df, on=[
                          'building_id', 'meter'], how='left')
train_df = train_df.merge(weather_train_df, on=[
                          'site_id', 'timestamp'], how='left')

train_df = reduce_mem_usage(train_df, use_float16=True)

del weather_train_df
gc.collect()

# %%

target = np.log1p(train_df["meter_reading"])

del train_df["meter_reading"] 
del train_df["meter_reading_log1p"] 

# %%
## Modelling 

def model(dense_dim_1=64, dense_dim_2=32, dense_dim_3=32, dense_dim_4=16, 
dropout1=0.2, dropout2=0.1, dropout3=0.1, dropout4=0.1, lr=0.001):

    #Inputs
    site_id = Input(shape=[1], name="site_id")
    building_id = Input(shape=[1], name="building_id")
    groupNum_train = Input(shape=[1], name="groupNum_train")
    primary_use = Input(shape=[1], name="primary_use")
    IsHoliday = Input(shape=[1], name="IsHoliday")
    square_feet = Input(shape=[1], name="square_feet")
    year_built = Input(shape=[1], name="year_built")
    air_temperature = Input(shape=[1], name="air_temperature")
    cloud_coverage = Input(shape=[1], name="cloud_coverage")
    dew_temperature = Input(shape=[1], name="dew_temperature")
    hour = Input(shape=[1], name="hour")
    precip = Input(shape=[1], name="precip_depth_1_hr")
    weekend = Input(shape=[1], name="weekend")
    #beaufort_scale = Input(shape=[1], name="beaufort_scale")
    pressure = Input(shape=[1], name="sea_level_pressure_mean_lag72")
    air_temperature_mean_lag72 = Input(shape=[1], name="air_temperature_mean_lag72")
    air_temperature_max_lag72 = Input(shape=[1], name="air_temperature_max_lag72")
    air_temperature_min_lag72 = Input(shape=[1], name="air_temperature_min_lag72")
    air_temperature_std_lag72 = Input(shape=[1], name="air_temperature_std_lag72")
    air_temperature_mean_lag3 = Input(shape=[1], name="air_temperature_mean_lag3")
    air_temperature_max_lag3 = Input(shape=[1], name="air_temperature_max_lag3")
    air_temperature_min_lag3 = Input(shape=[1], name="air_temperature_min_lag3")
    cloud_coverage_mean_lag72 = Input(shape=[1], name="cloud_coverage_mean_lag72")
    cloud_coverage_mean_lag3 = Input(shape=[1], name="cloud_coverage_mean_lag3")
    dew_temperature_mean_lag72 = Input(shape=[1], name="dew_temperature_mean_lag72")
    dew_temperature_mean_lag3 = Input(shape=[1], name="dew_temperature_mean_lag3")
    precip_depth_1_hr_mean_lag72 = Input(shape=[1], name="precip_depth_1_hr_mean_lag72")
    precip_depth_1_hr_mean_lag3 = Input(shape=[1], name="precip_depth_1_hr_mean_lag3")
    sea_level_pressure_mean_lag72 = Input(shape=[1], name="sea_level_pressure_mean_lag72")
    sea_level_pressure_mean_lag3 = Input(shape=[1], name="sea_level_pressure_mean_lag3")
    wind_speed_mean_lag72 = Input(shape=[1], name="wind_speed_mean_lag72")
    year_cnt = Input(shape=[1], name="year_cnt")
    bid_cnt = Input(shape=[1], name="bid_cnt")
    dew_smooth = Input(shape=[1], name="dew_smooth")
    air_smooth = Input(shape=[1], name="air_smooth")
    dew_diff = Input(shape=[1], name="dew_diff")
    air_diff = Input(shape=[1], name="air_diff")
    dew_diff2 = Input(shape=[1], name="dew_diff2")
    air_diff2 = Input(shape=[1], name="air_diff2")


    #Embeddings layers
    emb_site_id = Embedding(16, 2)(site_id)
    emb_building_id = Embedding(1448, 6)(building_id)
    #emb_meter = Embedding(4, 2)(meter)
    emb_groupNum_train = Embedding(39, 2)(groupNum_train)
    emb_primary_use = Embedding(16, 2)(primary_use)
    emb_hour = Embedding(24, 3)(hour)
    emb_weekend = Embedding(7, 2)(weekend)
    emb_IsHoliday = Embedding(3,1)(IsHoliday)


    concat_emb = concatenate([
           Flatten() (emb_site_id)
         , Flatten() (emb_building_id)
         , Flatten() (emb_groupNum_train)
         , Flatten() (emb_primary_use)
         , Flatten () (emb_hour)
         , Flatten() (emb_weekend)
         , Flatten() (emb_IsHoliday)
         
         
    ])#, Flatten() (emb_meter), Flatten() (emb_hour), Flatten() (emb_weekday)
    
    categ = Dropout(dropout1)(Dense(dense_dim_1,activation='relu') (concat_emb))
    categ = BatchNormalization()(categ)
    categ = Dropout(dropout2)(Dense(dense_dim_2,activation='relu') (categ))
    
    #main layer
    main_l = concatenate([
          categ
        , square_feet
        , year_built
        , air_temperature
        , cloud_coverage
        , dew_temperature
        , precip
        , weekend
        , pressure
        , air_temperature_mean_lag72
        , air_temperature_max_lag72
        , air_temperature_min_lag72
        , air_temperature_std_lag72
        , air_temperature_mean_lag3
        , air_temperature_max_lag3
        , air_temperature_min_lag3
        , cloud_coverage_mean_lag72
        , dew_temperature_mean_lag72
        , precip_depth_1_hr_mean_lag72
        , sea_level_pressure_mean_lag72
        , cloud_coverage_mean_lag3
        , dew_temperature_mean_lag3
        , precip_depth_1_hr_mean_lag3
        , sea_level_pressure_mean_lag3
        , wind_speed_mean_lag72
        , year_cnt
        , bid_cnt
        , dew_smooth
        , air_smooth
        , dew_diff
        , air_diff
        , dew_diff2
        , air_diff2
        , hour
    ])#beaufort_scale
    
    main_l = Dropout(dropout3)(Dense(dense_dim_3,activation='relu') (main_l))
    main_l = BatchNormalization()(main_l)
    main_l = Dropout(dropout4)(Dense(dense_dim_4,activation='relu') (main_l))
    
    #output
    output = Dense(1) (main_l)

    model = Model([ site_id,
                    building_id, 
                    groupNum_train, 
                    primary_use,
                    IsHoliday 
                    , square_feet
                    , year_built
                    , air_temperature
                    , cloud_coverage
                    , dew_temperature
                    , precip
                    , weekend
                    , pressure
                    , air_temperature_mean_lag72
                    , air_temperature_max_lag72
                    , air_temperature_min_lag72
                    , air_temperature_std_lag72
                    , air_temperature_mean_lag3
                    , air_temperature_max_lag3
                    , air_temperature_min_lag3
                    , cloud_coverage_mean_lag72
                    , dew_temperature_mean_lag72
                    , precip_depth_1_hr_mean_lag72
                    , sea_level_pressure_mean_lag72
                    , cloud_coverage_mean_lag3
                    , dew_temperature_mean_lag3
                    , precip_depth_1_hr_mean_lag3
                    , sea_level_pressure_mean_lag3
                    , wind_speed_mean_lag72
                    , year_cnt
                    , bid_cnt
                    , dew_smooth
                    , air_smooth
                    , dew_diff
                    , air_diff
                    , dew_diff2
                    , air_diff2
                    ], output) #beaufort_scale, meter

    model.compile(optimizer = Adam(lr=lr),
                  loss= mse_loss,
                  metrics=[root_mean_squared_error])
    return model

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=0))
# %%
# Method for Train and Validation
def get_keras_data(df, num_cols, cat_cols):
    cols = num_cols + cat_cols
    X = {col: np.array(df[col]) for col in cols}
    return X

def train_model(keras_model, X_t, y_train, batch_size, epochs, X_v, y_valid, fold, patience=3):
    early_stopping = EarlyStopping(patience=patience, verbose=1)
    model_checkpoint = ModelCheckpoint("model_" + str(fold) + ".hdf5",
                                       save_best_only=True, verbose=1, monitor='val_root_mean_squared_error', mode='min')

    hist = keras_model.fit(X_t, y_train, batch_size=batch_size, epochs=epochs,
                            validation_data=(X_v, y_valid), verbose=1,
                            callbacks=[early_stopping, model_checkpoint])

    keras_model = load_model("model_" + str(fold) + ".hdf5", custom_objects={'root_mean_squared_error': root_mean_squared_error})
    
    return keras_model
# %%
from sklearn.model_selection import KFold, StratifiedKFold

oof = np.zeros(len(train_df))
batch_size = 1024
epochs = 10
models = []

folds = 3
seed = 666

kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

for fold_n, (train_index, valid_index) in enumerate(kf.split(train_df, train_df['building_id'])):
    print('Fold:', fold_n)
    X_train, X_valid = train_df.iloc[train_index], train_df.iloc[valid_index]
    y_train, y_valid = target.iloc[train_index], target.iloc[valid_index]
    X_t = get_keras_data(X_train, feature_cols, category_cols)
    X_v = get_keras_data(X_valid, feature_cols, category_cols)
    
    keras_model = model(dense_dim_1=64, dense_dim_2=32, dense_dim_3=32, dense_dim_4=16, 
                        dropout1=0.2, dropout2=0.1, dropout3=0.1, dropout4=0.1, lr=0.001)
    mod = train_model(keras_model, X_t, y_train, batch_size, epochs, X_v, y_valid, fold_n, patience=3)
    models.append(mod)
    print('*'* 50)
# %%
