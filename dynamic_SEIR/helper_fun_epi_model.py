## Help function for SIR 2019-nCoV estimation
## Author: Yran Jing
## Date: 2020-02-02

import numpy as np
import scipy.optimize as optimization
import pandas as pd
import pandas


#############################
## Data processing
##############################

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


def get_China_exclude_province(df, provinceName: str)-> pandas.core.frame.DataFrame:
    """
    Return time series data of China total exclude the given province
    """
    Hubei= get_province_df(df, provinceName)
    China_total = get_China_total(df)
    
    NotHubei = China_total.reset_index(drop= True)
    Hubei = Hubei.reset_index(drop= True)
    
    NotHubei['E'] = NotHubei['E'] - Hubei['E']
    NotHubei['R'] = NotHubei['R'] - Hubei['R']
    NotHubei['I'] = NotHubei['I'] - Hubei['I']
    
    return NotHubei

#######################################################
### function for SEIR model using MCMC simulated result
#######################################################

def run_SEIR(Est_beta: float, econ: int, E0:float, R0:int, I0:int, population: int,
             rateIR:float, rateAl:float,
             title:str, death_rate: float, show_Sus = True) -> pandas.core.frame.DataFrame:
        """
        Run SEIR model
        """
        Est_beta = Est_beta
        seir = SEIR(eons=econ, Susceptible=population-E0-I0-R0, Exposed = E0, 
                    Infected=I0, Resistant=R0, rateSI=Est_beta, rateIR=rateIR, 
                    rateAl = rateAl)
        result = seir.run(death_rate)
        # Draw plot
        if show_Sus:
            seir.plot(title, 'population', "2020 Date")
        else:
            seir.plot_noSuscep(title, 'population', "2020 Date")
            
        return result