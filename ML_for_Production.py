#!/usr/bin/env python
# coding: utf-8

# In[41]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


# In[42]:


def load_data():
    sales_df = pd.read_csv(r"C:\Users\aghay\Downloads\sales.csv")
    sales_df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
    sales_df['year'] = pd.to_datetime(sales_df['timestamp']).dt.year
    sales_df['month'] = pd.to_datetime(sales_df['timestamp']).dt.month
    sales_df['day'] = pd.to_datetime(sales_df['timestamp']).dt.day
    sales_df['hour'] = pd.to_datetime(sales_df['timestamp']).dt.hour
    sales_df['minute'] = pd.to_datetime(sales_df['timestamp']).dt.minute
    sales_df = sales_df.drop(['transaction_id','timestamp','month','year'],axis = 1)
    stock_df = pd.read_csv(r"C:\Users\aghay\Downloads\sensor_stock_levels.csv")
    stock_df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
    stock_df['year'] = pd.to_datetime(stock_df['timestamp']).dt.year
    stock_df['month'] = pd.to_datetime(stock_df['timestamp']).dt.month
    stock_df['day'] = pd.to_datetime(stock_df['timestamp']).dt.day
    stock_df['hour'] = pd.to_datetime(stock_df['timestamp']).dt.hour
    stock_df['minute'] = pd.to_datetime(stock_df['timestamp']).dt.minute
    stock_df = stock_df.groupby(['product_id','day','hour','minute']).agg({'estimated_stock_pct' : 'mean'}).reset_index()
    temp_df = pd.read_csv(r"C:\Users\aghay\Downloads\sensor_storage_temperature.csv")
    temp_df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
    temp_df['year'] = pd.to_datetime(temp_df['timestamp']).dt.year
    temp_df['month'] = pd.to_datetime(temp_df['timestamp']).dt.month
    temp_df['day'] = pd.to_datetime(temp_df['timestamp']).dt.day
    temp_df['hour'] = pd.to_datetime(temp_df['timestamp']).dt.hour
    temp_df['minute'] = pd.to_datetime(temp_df['timestamp']).dt.minute
    temp_df = pd.DataFrame(temp_df.groupby(['day','hour','minute']).agg({'temperature' : 'mean'})).reset_index()
    df_init = stock_df.merge(sales_df, left_on='product_id', right_on='product_id',how='left').drop(['minute_y','hour_y','day_y'],axis = 1).drop_duplicates()
    df_init = df_init.rename(columns = {'day_x':'day','hour_x':'hour','minute_x':'minute'})
    df = df_init.merge(temp_df,on = ['day','hour','minute'])
    df = df.drop(['product_id','minute'],axis = 1)
    return df


# In[43]:


def preprocess_cat_num_variable(data: pd.DataFrame = None):
    cats = data.select_dtypes(include = ['object'])
    nums = data.select_dtypes(include = ['float64','int64']).drop('estimated_stock_pct',axis = 1)
    scaler = StandardScaler()
    scaled_nums = scaler.fit_transform(nums)
    data[nums.columns] = scaled_nums
    for cat in cats.columns:
        data[cat] = data[cat].map((data[cat].value_counts())/len(data))
    return data


# In[44]:


preprocess_cat_num_variable(load_data())


# In[45]:


def split_test_train(data: pd.DataFrame = None):
    target = data['estimated_stock_pct']
    features = data.drop('estimated_stock_pct',axis = 1)
    return features,target


# In[46]:


def train_algorithm_with_cross_validation(
    X: pd.DataFrame = None, 
    y: pd.Series = None
):
    accuracy_mae = []
    accuracy_r2 = []
    for fold in range(0,5):
        model = RandomForestRegressor()
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=42)
        model.fit(X_train,y_train)
        # Generate predictions on test sample
        y_pred = model.predict(X_test)

        # Compute accuracy, using mean absolute error
        mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
        r2 = r2_score(y_true=y_test, y_pred=y_pred)
        accuracy_mae.append(mae)
        accuracy_r2.append(r2)
        print(f"Fold {fold + 1}: MAE = {mae:.3f}")
        print(f"Fold {fold + 1}: R2_score = {r2:.3f}")
    


# In[47]:


train_algorithm_with_cross_validation(split_test_train(preprocess_cat_num_variable(load_data()))[0],
                                     split_test_train(preprocess_cat_num_variable(load_data()))[1])


# In[ ]:




