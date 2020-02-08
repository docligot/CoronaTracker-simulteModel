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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

##################
## Clean data
##################
def add_days(DXYArea: pandas.core.frame.DataFrame) -> pandas.core.frame.DataFrame:
    """
    Create a new column: Days, number of days after 2019-12-08 (detect the first case)
    """
    DXYArea['updateDate'] = pd.to_datetime(DXYArea['updateDate'])
    first_day = datetime.datetime(2019, 12, 8) # the time when detected the first case (2019-12-08)
    DXYArea['Days'] = (DXYArea['updateDate'] - first_day).dt.days
    return DXYArea

def split_train_test_by_date(df: pandas.core.frame.DataFrame):
    """
    Separate Train and Test dataset in time series
    """
    # we use the last 3 days as test data
    split_date = df['updateDate'].max() - datetime.timedelta(days=2)
    
    ## Separate Train and Test dataset
    Train = df[df['updateDate'] < split_date]
    Test = df[df['updateDate'] >= split_date]
    print("Train dataset: data before {} \nTest dataset: the last 3 days".format(split_date))
    
    return Train, Test

##################
###           EDA
##################

def tsplot_conf_dead_cured(df, title_prefix, figsize=(13,10), fontsize=18, logy=False):
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    plot_df = df.groupby('updateDate').agg('sum')
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
    sub_df = df[df['provinceName'] == title_prefix]
    tsplot_conf_dead_cured(sub_df, title_prefix)
    
def draw_city_trend(title_prefix: str, df: pandas.core.frame.DataFrame):
    """
    df is the daily dataset from DXY
    """
    sub_df = df[df['cityName'] == title_prefix]
    tsplot_conf_dead_cured(sub_df, title_prefix)


###################
##  Modelling
###################

### polynomial_regression
def as_arrary(x):
    return [np.asarray(x)]

def draw_fit_plot(degree: int, X_train, X_test, y_train, y_test, y_train_predicted, y_test_predict, df):
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
    plt.title("Polynomial Regression with degree = {}".format(degree))
    plt.ylabel('Confirmed cases')
    plt.xlabel('2020 Date')
    
    datemin = df['updateDate'].min()
    numdays = len(X_train) + len(X_test)
    labels = list((datemin + datetime.timedelta(days=x)).strftime('%m-%d') for x in range(numdays))
    
    x = pd.Series(np.concatenate((X_train, X_test)))
    plt.xticks(x, labels,rotation=60)
    #fig.autofmt_xdate() # axes up to make room for them
    
    plt.show()
    

def create_polynomial_regression_model(degree:int, df,
                                       X_train: pandas.core.frame.DataFrame, 
                                       X_test: pandas.core.frame.DataFrame,
                                       y_train: pandas.core.frame.DataFrame, 
                                       y_test: pandas.core.frame.DataFrame,
                                       draw_plot = False):
    "Creates a polynomial regression model for the given degree"
    
    poly_features = PolynomialFeatures(degree=degree)
      
    # transforms the existing features to higher degree features.
    X_train_array = tuple(map(as_arrary, list(X_train)))
    X_test_array = tuple(map(as_arrary, list(X_test)))
    
    # Normalize input data
    X_train_poly = poly_features.fit_transform(X_train_array)
    
    # fit the transformed features to Linear Regression
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
      
    # predicting on training data-set
    y_train_predicted = poly_model.predict(X_train_poly)
      
    # predicting on test data-set
    y_test_predict = poly_model.predict(poly_features.fit_transform(X_test_array))
      
    # evaluating the model on training dataset
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_predicted))
    r2_train = r2_score(y_train, y_train_predicted)
    print("Degree {}:".format(degree))
    print("RMSE of training set is {}".format(rmse_train))
    print("R2 score of training set is {}\n".format(r2_train))
    
    if len(y_test)>0:
        # evaluating the model on test dataset
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_predict))
        r2_test = r2_score(y_test, y_test_predict)
        print("RMSE of test set is {}".format(rmse_test))
        print("R2 score of test set is {}".format(r2_test))
        
    print('---------------------------------------\n')
    
    # Draw fit plot
    if draw_plot == True: 
        draw_fit_plot(degree, X_train, X_test, y_train, y_test, y_train_predicted, y_test_predict, df)


def forecast_next_4_days(degree: int, df: pandas.core.frame.DataFrame):
    """
    Use all observations to train, based on the 
    """
    X_train = df['Days']
    y_train = df['confirmed']
    
    # Create dataset for next 4 days
    X_test = [df['Days'].max() + 1, df['Days'].max() + 2, df['Days'].max() + 3, df['Days'].max() + 4]
    y_test = []
    
    create_polynomial_regression_model(2, df, X_train, X_test, y_train, y_test, draw_plot = True)
    
    


    