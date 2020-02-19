### Helper function for simulation model

#### Author: Yiran Jing
#### Date: Feb 2020

import pandas as pd
import numpy as np
import operator
import matplotlib.pyplot as plt
import pandas as pd
import pandas
import datetime
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

########################
## dataset help function
########################

def get_province_df(df, provinceName: str) -> pandas.core.frame.DataFrame:
    """
    Return time series data of given province
    """
    return df[(df['province']==provinceName) & (df['city'].isnull())]


def get_China_total(df) -> pandas.core.frame.DataFrame:
    """
    Return time series data of China total (including HK and Taiwan)
    """
    return df[(df['countryCode']=='CN') & (df['province'].isnull())]

##################
## Clean data
##################
def add_days(DXYArea: pandas.core.frame.DataFrame) -> pandas.core.frame.DataFrame:
    """
    Create a new column: Days, number of days after 2019-12-08 (detect the first case)
    """
    DXYArea['date'] = pd.to_datetime(DXYArea['date'])
    first_day = datetime.datetime(2019, 12, 8) # the time when detected the first case (2019-12-08)
    DXYArea['Days'] = (DXYArea['date'] - first_day).dt.days
    return DXYArea

def split_train_test_by_date(df: pandas.core.frame.DataFrame, splicer):
    """
    Separate Train and Test dataset in time series
    """
    if type(splicer) == float: ## check if splicer is a float
        ndays = 3 if (df['date'].max() - df['date'].min()).days < 3 else splicer * (df['date'].max() - df['date'].min()).days
        ndays = np.ceil(ndays)
    elif type(splicer) == int:
        ndays = splicer 
    else:
        raise Exception('split value should not be greater than length of data')

    split_date = df['date'].max() - datetime.timedelta(days=ndays)
    
    ## Separate Train and Test dataset
    Train = df[df['date'] < split_date]
    Test = df[df['date'] >= split_date]
    print("Train dataset: data before {} \nTest dataset: the last {} days".format(split_date,ndays))
    
    return Train, Test

def data_processing(df, splicer, features_to_engineer):

    overall_df = pd.DataFrame(df.groupby(['date']).agg({'confirmed': "sum",
                                                        'suspected':'sum',
                                                        'cured': "sum",
                                                        'dead': 'sum',
                                                        'Days': 'mean'})).reset_index()
    
    
    
    Train, Test = split_train_test_by_date(overall_df, splicer)
    print(Train)

    X_train = Train.loc[:,['Days']+[x+'_lag1' for x in features_to_engineer]]
    y_train = Train['confirmed']
    X_test =  Test.loc[:,['Days']+[x+'_lag1' for x in features_to_engineer]]
    y_test = Test['confirmed']
    
    return X_train, X_test, y_train, y_test

##################
###      feature engineering
##################

def feature_engineering(df:pandas.core.frame.DataFrame, features_to_engineer):
    for feature in features_to_engineer:
        df[f'{feature}_lag1'] = df[f'{feature}'].shift()
        df[f'{feature}_lag1'].fillna(0, inplace = True)
    return df

##################
###           EDA
##################

def tsplot_conf_dead_cured(df, title_prefix, figsize=(13,10), fontsize=18, logy=False):
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    plot_df = df.groupby('date').agg('sum')
    plot_df.plot(y=['confirmed'], style='-*', ax=ax1, grid=True, figsize=figsize, logy=logy, color='black', marker='o')
    if logy:
        ax1.set_ylabel("log(confirmed)", color="black", fontsize=14)
    else:
        ax1.set_ylabel("confirmed", color="black", fontsize=14)
    if 'dailyNew_confirmed' in df.columns:
        ax11 = ax1.twinx()
        ax11.bar(x=plot_df.index, height=plot_df['dailyNew_confirmed'], alpha=0.3, color='blue')
        ax11.set_ylabel('dailyNew_confirmed', color='blue', fontsize=14)
    ax2 = fig.add_subplot(212)
    plot_df.plot(y=['dead', 'cured'], style=':*', grid=True, ax=ax2, figsize=figsize, sharex=False, logy=logy)
    ax2.set_ylabel("count")
    title = title_prefix + ' Cumulative Confirmed, Death, Cure'
    fig.suptitle(title, fontsize=fontsize)
    
def draw_province_trend(title_prefix: str, df: pandas.core.frame.DataFrame):
    """
    df is the daily dataset from DXY
    """
    sub_df = df[df['province'] == title_prefix]
    tsplot_conf_dead_cured(sub_df, title_prefix)
    
def draw_city_trend(title_prefix: str, df: pandas.core.frame.DataFrame):
    """
    df is the daily dataset from DXY
    """
    sub_df = df[df['city'] == title_prefix]
    tsplot_conf_dead_cured(sub_df, title_prefix)


###################
##  Modelling
###################

### general additive model

def draw_fit_plot(degree: int, area: str, X_train, X_test, y_train, y_test, y_train_predicted, y_test_predict, df):
    if len(y_test)>0:
        x = pd.Series(np.concatenate((X_train, X_test)))
        y = pd.Series(np.concatenate((y_train, y_test)))
    else:
        x = X_train; y = y_train
    
    fig, ax = plt.subplots()
    #fig.canvas.draw()
    plt.scatter(x, y, s=10, c = 'black')
    plt.plot(X_train, y_train_predicted, color='green')
    plt.plot(X_test, y_test_predict, color = 'blue')
    plt.title("Polynomial Regression {} with degree = {}".format(area, degree))
    plt.ylabel('Confirmed cases')
    plt.xlabel('2020 Date')
    
    datemin = df['date'].min()
    numdays = len(X_train) + len(X_test)
    labels = list((datemin + datetime.timedelta(days=x)).strftime('%m-%d') for x in range(numdays))
    
    x = pd.Series(np.concatenate((X_train, X_test)))
    plt.xticks(x, labels,rotation=60)
    #fig.autofmt_xdate() # axes up to make room for them
    
    plt.show()
    

def fit_pygam_model(X_train: pandas.core.frame.DataFrame, 
                   X_test: pandas.core.frame.DataFrame,
                   y_train: pandas.core.frame.DataFrame, 
                   y_test: pandas.core.frame.DataFrame):
    '''
    Creates a general additive model LinearGAM (normally distributed errors)
    with grid search. Returns the best model with given hyperparameters.
    hyperparameters: n_splines and lam regularization parameter.
    '''
    from pygam import LinearGAM
    gam = LinearGAM().gridsearch(X_train.values, y_train, n_splines=np.arange(3,20), lam = np.logspace(-3, 3, 11))
    print(gam.summary())
    
    y_train_predicted = gam.predict(X_train)
    y_test_predicted = np.floor(gam.predict(X_test))
    
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_predicted))
    mae_train = mean_absolute_error(y_train, y_train_predicted)
    r2_train = r2_score(y_train, y_train_predicted)
    print("RMSE of training set is {}".format(rmse_train))
    print("MAE of testing set is {}".format(mae_train))
    print("R2 score of training set is {}\n".format(r2_train))
    
    if len(y_test)>0:
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_predicted))
        mae_test = mean_absolute_error(y_test, y_test_predicted)
        r2_test = r2_score(y_test, y_test_predicted)
        print("RMSE of testing set is {}".format(rmse_test))
        print("MAE of testing set is {}".format(mae_test))
        print("R2 score of testing set is {}\n".format(r2_test))
    
    '''
    Visualize the feature significance and confidence intervals
    '''
    num_features = len(X_train.columns)
    fig = plt.figure(figsize=(18,12))
    fig.subplots_adjust(hspace=0.4)

    cnt = 1
    p_values = gam.statistics_['p_values']

    for i in range(num_features):
        axs = fig.add_subplot(num_features,1,cnt)
        m = gam.generate_X_grid(term=i)
        axs.plot(m[:,i], gam.partial_dependence(term=i, X=m)) # this is the actual coefficents
        axs.plot(m[:,i], gam.partial_dependence(term=i, X=m, width=.95)[1],c='r',ls='--') # this plots the confidence intervals
        axs.set_title(X_train.columns[i] + ('*' if p_values[cnt]<0.05 else ''))
        cnt += 1

    