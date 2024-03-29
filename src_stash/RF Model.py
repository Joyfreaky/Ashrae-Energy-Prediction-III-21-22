# %%[markdown]
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 20:49:23 2022

@author: Ed
Random Forest
"""
# %% [code]
# Basic imports
import numpy as np 
import pandas as pd 

# %% [code]
# Reading in processed data
train_df = pd.read_feather("train_df_processed.feather")
test_df = pd.read_feather("test_df_processed.feather")
weather_test_df = pd.read_feather("weather_test_df_processed.feather")
building_meta_df = pd.read_feather("building_meta_df_processed.feather")



# %% [code]
# # Dropping unnecessary columns
# # Timestamp, weekend and date will already be processed in the hour, day, month columns
# test_data.drop("timestamp", inplace=True, axis = 1)
# test_data.drop("weekend", inplace=True, axis = 1)
# test_data.drop("date", inplace=True, axis = 1)
# # Row ID is just the index
# test_data.drop("row_id", inplace=True, axis = 1)

print('Shape of the Train data after processing',train_df.shape)
print('Shape of the Test data after processing',test_df.shape)
print('Shape of the weather test data after processing',weather_test_df.shape)
print('Shape of the building metadata after processing',building_meta_df.shape)

# %%[code]
# Reducing memory usage func

# Modified to support timestamp type, categorical type
# Modified to add option to use float16 or not. feather format does not support float16.
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype

def reduce_mem_usage(df):
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
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
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
# %%[code]

# Implementing memory reduction func
train_df = reduce_mem_usage(train_df)
test_df = reduce_mem_usage(test_df)
weather_test_df = reduce_mem_usage(weather_test_df)
building_meta_df = reduce_mem_usage(building_meta_df)

# %%[code]
# Imports for encoding data before model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer



# Transformer for categorical features such as objects and boolean.
categorical_features = []
categorical_transformer = Pipeline(
    [
        ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
    ]
)
# Transformer for numerical features such as floats and integers
numeric_features = ['building_id', 'meter','hour', 'day','dayofweek', 'month',
						      'bid_cnt', 'groupNum_train']
                    
numeric_transformer = Pipeline(
    [
        ('scaler', StandardScaler())
    ]
)

# Combine them in a single ColumnTransformer
preprocessor = ColumnTransformer(
    [
        ('categoricals', categorical_transformer, categorical_features),
        ('numericals', numeric_transformer, numeric_features)
    ],
    remainder = 'drop'
)


# Producing the training values
X_train = train_data[['building_id', 'meter','hour', 'day','dayofweek', 'month',
						      'bid_cnt', 'groupNum_train']]

y_train = train_data['meter_reading_log1p'].astype(int)

# Deleting as no longer necessary to reduce memory
del train_data


# Imports for classifier 
from sklearn.ensemble import RandomForestClassifier

# RF classifier
RFClassifier = Pipeline(
    [
     ('preprocessing', numeric_transformer),
     ('classifier', RandomForestClassifier( 
                                           random_state=0) )
    ]
)


# Inputting the training data through the classifications
RFClassifier.fit(X_train, y_train)
del X_train
del y_train

# Using the test data to produce predicted meter readings as log1p values
RF_pred = RFClassifier.predict(test_data)

# Inversing the logarithm to give true predictions
RF_pred = np.expm1(RF_pred)

# Creating row_id (index)
row_id = np.linspace(0,len(RF_pred)-1, len(RF_pred))

# Combine into a dataframe for exporting
df = pd.DataFrame({"row_id":row_id, "meter_reading":RF_pred})

# Exporting as csv
df.to_csv("Random_Forest_Model")
