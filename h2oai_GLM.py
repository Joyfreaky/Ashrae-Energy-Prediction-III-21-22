# %%
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

# --- plotly ---
from plotly import tools, subplots
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff

# --- models ---
from sklearn import preprocessing
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

from sklearn.metrics import mean_squared_error


import warnings
warnings.filterwarnings('ignore')
# %%
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
# %%
# H2O ML MODEL ======================================================================================================================
# preproc ===========================
import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator

root = Path('/home/joydipb/Documents/CMT307-Coursework-2-Group-19')

# init ==============================
h2o.init(max_mem_size='16G')

# import data =======================
train = h2o.import_file("/home/joydipb/Documents/CMT307-Coursework-2-Group-19/train_df_processed.feather", header=1)

#%%
y = "meter_reading"
x = train.columns[0:2]

# fit model =========================
glm_model = H2OGeneralizedLinearEstimator(
    family="gaussian", 
    solver='AUTO', 
    alpha=0.5,
    #lambda=0.0,
    link='Family_Default',
    intercept=True,
    lambda_search=True, 
    nlambdas=100, 
    missing_values_handling='MeanImputation',
    standardize=True,
    #nfolds = 5, 
    seed = 1333
)
glm_model.train(x=x, y=y, training_frame=train)
# %%
# Eval mod ==========================
glm_model.rmse(xval=True)
# %%
# release memory and load test ========
h2o.remove(train)
del train
gc.collect()
# %%
# Model pred ========================
test = h2o.import_file("/home/joydipb/Documents/CMT307-Coursework-2-Group-19/test.csv", header=1)
preds = glm_model.predict(test).as_data_frame()

# release memory and load test ========
h2o.remove(test)
del test
gc.collect()

h2o.cluster().shutdown()
# %%
#preds.to_csv(preds, 'preds.csv', index=False, float_format='%.4f')
# %% [markdown]
#Link and out subm

# %%

test_df = pd.read_csv(root/'test.csv', parse_dates=["timestamp"])
building_meta_df = pd.read_csv(root/'building_metadata.csv')
# %%
test_df['pred'] = np.expm1(preds['predict'])
test_df.loc[test_df.pred<0, 'pred'] = 0

test_df = reduce_mem_usage(test_df)

# %%
sample_submission = pd.read_csv(root/'sample_submission.csv')
sample_submission['meter_reading'] = test_df.pred
sample_submission.loc[sample_submission.meter_reading < 0, 'meter_reading'] = 0
# %%
sample_submission.head()
# %%
sns.distplot(np.log1p(sample_submission.meter_reading))
# %%
sample_submission.to_csv('submission_h20ai_GLM.csv', index=False, float_format='%.4f')
# %%
## Submission
! mkdir -p ~/.kaggle/ && \
  echo '{"username":"joydipbhowmick","key":"5bd4e6a1fec9fc7f8a93def26785a6d2"}' > ~/.kaggle/kaggle.json && \
  chmod 600 ~/.kaggle/kaggle.json # Create a new direcory use the kaggle token key in that and make it read only to current user.
! kaggle competitions submit -c ashrae-energy-prediction -f submission_h20ai_GLM.csv -m "trail h20ai GLM using gussian"
# %%
