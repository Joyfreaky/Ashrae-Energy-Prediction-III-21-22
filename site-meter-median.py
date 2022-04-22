# %% [code]

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pathlib import Path
import os

# %% [code]
root_data = Path('/home/joydipb/Documents/CMT307-Coursework-2-Group-19')

sample_submission = pd.read_feather(root_data/'sample_submission.feather')
df_train = pd.read_feather(root_data/'train.feather')
df_test = pd.read_feather(root_data/'test.feather')
df_meta = pd.read_feather(root_data/'building_metadata.feather')

sample_submission = pd.read_csv(root_data/'submission.csv')

# %% [code]
df_meta = df_meta.merge(df_train[['building_id','meter']].drop_duplicates(), on='building_id')
df_meta['groupNum_train'] = df_meta['site_id'].astype('int')*10 + df_meta['meter'].astype('int')
df_meta

# %% [code]

df_train['meter_reading_log1p'] = np.log1p(df_train['meter_reading'])

# %% [code]

sample_submission = sample_submission.merge(df_test, on=['row_id'])
sample_submission = sample_submission.merge(df_meta, on=['building_id','meter'], how='left')
sample_submission['meter_reading_log1p'] = np.log1p(sample_submission['meter_reading'])
sample_submission

# %% [code]

df_allData = pd.concat([df_train.merge(df_meta, on=['building_id','meter'], how='left')[['building_id','meter','groupNum_train','timestamp','meter_reading_log1p']], 
                       sample_submission[['building_id','meter','groupNum_train','timestamp','meter_reading_log1p']]],axis=0)
df_allData

# %% [code]

df_median = df_allData.groupby(["groupNum_train","timestamp"])\
                                .agg(median_building_meter=("meter_reading_log1p", "median")).reset_index()
df_median = df_median.pivot_table(values='median_building_meter',columns='groupNum_train', index='timestamp')
df_median.columns = 'group_median_' + (df_median.columns).astype('str')
df_median = df_median.reset_index()
df_median

# %% [code]

df_median.to_pickle('df_groupNum_median.pickle')

# %%
