'''
This script is loads and visualize data,save aligned data into csv files
'''

import copy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import glob
from datetime import datetime
import pymc as pm
import pytensor
import pytensor.tensor as pt
import arviz as az
from numba import njit
from pymc.ode import DifferentialEquation
from pytensor.compile.ops import as_op
from scipy.integrate import odeint
from scipy.optimize import least_squares,minimize
import math
import dill

import os
os.environ["PATH"] += os.pathsep + 'C:/ProgramData/Anaconda3/Library/bin/graphviz/'
#constant value, sowing date from their paper https://www.biorxiv.org/content/10.1101/2024.10.04.616624v2.full.pdf
sowing_date_dictionary = pd.DataFrame({2016:'2015-10-13',2017:'2016-11-01',2018:'2017-11-02',2019:'2018-10-07',2021:'2020-10-21',2022:'2021-10-25'},index=[0])
sowing_date_dictionary =  sowing_date_dictionary.T
sowing_date_dictionary[0]= pd.to_datetime(sowing_date_dictionary[0]).dt.date
sowing_date_dictionary=sowing_date_dictionary.T
def convert_date_to_namber(date=''):
    #convert date to number for count time length later

    number = date.toordinal()-736998
    return number

def data_load(file_name,year:int|list = 2019):
    """
    load the selected year data

    :param file_name: str, name of the file with longitudinal trauts value
    :param year: the experiment year
    :return: pd.Dataframe
    """
    #read data
    df = pd.read_csv(file_name,index_col=0,header=0)
    #print(df.columns)
    #convert timestamp type to date
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.date
    if type(year) is int:
        single_year_experiment = df[df['year_site.harvest_year'] == year][['timestamp','genotype.id','plot.UID','value',
                                                                            'Air_temperature_0.1_m',
                                                                           'Air_temperature_2_m','Relative_air_humidity_2_m',
                                                                           'Short_wavelenght_solar_irradiance_2_m','Soil_temperature_-0.05_m',
                                                                           'plot.row_global', 'plot.range_global','lot','year_site.harvest_year']]
        return single_year_experiment
    elif type(year) is list:
        multiple_year_experiment = df[df['year_site.harvest_year'].isin(year)][['timestamp','genotype.id','plot.UID','value',
                                                                                 'Air_temperature_0.1_m',
                                                     'Air_temperature_2_m','Relative_air_humidity_2_m',
                                                     'Short_wavelenght_solar_irradiance_2_m','Soil_temperature_-0.05_m',
                                                     'plot.row_global', 'plot.range_global','lot','year_site.harvest_year']]

        return multiple_year_experiment

    '''
    # count number of genotypes without replicates in lot
    single_year_experiment = single_year_experiment[single_year_experiment['lot']=='lot4']
    print(single_year_experiment['lot'].unique())
    print(len(single_year_experiment['genotype.id'].unique()))
    groups_object = single_year_experiment.groupby('timestamp')
    for group in groups_object.groups:
        one_day_df = groups_object.get_group(group)
        print()
        print(one_day_df['genotype.id'].duplicated(keep=False).value_counts())
    '''

def load_multiple_year_data(height_file="C:/data_from_paper/ETH/olivia_WUR/process_data/Canopy coverage.csv",
                                canopy_coverage_file="C:/data_from_paper/ETH/olivia_WUR/process_data/Plant height.csv",env_save_directory="C:/data_from_paper/ETH/olivia_WUR/process_data/",
                                years=[2018, 2019, 2021, 2022], keep_environment_covariant=False,same_length='_same_length'):

    '''
    read multiple year data, align time series
    parameters: years: list of year
    '''
    print('data for year {}'.format(years))
    canopy_df = data_load(canopy_coverage_file, years)
    height_df = data_load(height_file, years)

    if same_length=='':
        #read env files
        files = glob.glob("{}*_m.csv".format( env_save_directory))
        print(files)
        files = [str(x).split('\\')[1] for x in files]
        env_df_full = pd.DataFrame()
        for environment_factors_file in files:
            env_df = pd.read_csv("{}{}".format(env_save_directory,environment_factors_file))
            env_df.rename(columns={"value": "{}".format(environment_factors_file.split('.csv')[0])}, inplace=True)
            env_df.drop(columns='year', inplace=True)
            env_df['timestamp'] = pd.to_datetime(env_df['timestamp']).dt.date
            env_df_full = pd.concat([env_df_full,env_df],axis=1)
        else:
            env_df_full = env_df_full.loc[:, ~env_df_full.columns.duplicated()].copy()
            print(env_df_full)
        # print(canopy_df['corrected_value'].dropna(inplace=True))
    if keep_environment_covariant:
        print('Keep environment')
    else:
        print('Do not keep environment')
        canopy_df = canopy_df[
            ['timestamp', 'plot.UID', 'value', 'genotype.id', 'plot.row_global', 'plot.range_global', 'lot',
             'year_site.harvest_year']]
        height_df = height_df[
            ['timestamp', 'plot.UID', 'value', 'genotype.id', 'plot.row_global', 'plot.range_global', 'lot',
             'year_site.harvest_year']]
        print(canopy_df['value'])
        print(height_df)


    #group based on date, and find the start measuring date canopy coverage
    n_row = len(canopy_df.index)
    group_year = canopy_df.groupby('year_site.harvest_year')
    canopy_df_multiple_year = pd.DataFrame()
    start_date_list_canopy_coverage = [] #save the start date, use to align time stamp
    end_date_list_canopy_coverage = []
    for group in group_year.groups:
        #lopp through each year
        group_df = group_year.get_group(group)
        # start_date_canopy = group_df['timestamp'].min()
        year = group_df['year_site.harvest_year'].unique()[0]
        start_date_canopy = sowing_date_dictionary[year].values
        print('canopy coverage start day for year:{} is {}'.format(year,start_date_canopy))
        # set start dates as 0, add column'day_after_start_measure'
        group_df.loc[:,'day_after_start_measure'] = (group_df['timestamp'] - start_date_canopy).apply(lambda x: x.days)
        canopy_df_multiple_year = pd.concat([canopy_df_multiple_year,group_df])
        print('start measuring date canopy')
        print(start_date_canopy)
        end_date_canopy = group_df['day_after_start_measure'].max()
        print(group_df['timestamp'].max())
        end_date_list_canopy_coverage.append(end_date_canopy)
        start_date_list_canopy_coverage.append(start_date_canopy)
    else:
        # print(canopy_df_multiple_year)
        assert n_row == len(canopy_df_multiple_year)
    # raise EOFError
    #group based on date, and find the start measuring date plant height
    n_row = len(height_df.index)
    group_year = height_df.groupby('year_site.harvest_year')
    height_df_multiple_year = pd.DataFrame()
    start_date_list_plant_height=[]
    end_date_list_plant_height = []
    for group in group_year.groups:
        group_df = group_year.get_group(group)
        # start_date_plant_height = group_df['timestamp'].min()
        year = group_df['year_site.harvest_year'].unique()[0]
        start_date_plant_height = sowing_date_dictionary[year].values
        print('plant height start day for year:{} is {}'.format(year,start_date_plant_height))
        # set start dates as 0, add column'day_after_start_measure'
        group_df.loc[:,'day_after_start_measure'] = (group_df.loc[:,'timestamp'] - start_date_plant_height).apply(lambda x: x.days)
        height_df_multiple_year = pd.concat([height_df_multiple_year,group_df])
        # print('start date plant height')
        # print(start_date_plant_height) #pd.date value
        end_date_height = group_df['day_after_start_measure'].max()
        end_date_list_plant_height.append(end_date_height)
        start_date_list_plant_height.append(start_date_plant_height)
    else:

        # drop_na_corrected= height_df_multiple_year[['value']].dropna()
        # print(drop_na_corrected)
        assert n_row==len(height_df_multiple_year)

    #full time_stamps
    group_year_height = height_df_multiple_year.groupby('year_site.harvest_year')
    group_year_canopy = canopy_df_multiple_year.groupby('year_site.harvest_year')
    new_canopy_df = pd.DataFrame()
    new_height_df = pd.DataFrame()
    for start,end1,end2,group_canopy,group_height,start_date_plant_height_value,start_date_canopy_value in zip([0]*len(years),end_date_list_canopy_coverage,
                                                         end_date_list_plant_height,group_year_canopy.groups,
                                                         group_year_height.groups,start_date_list_plant_height,start_date_list_plant_height):
        # loop through year
        end = max(end1,end2)
        full_growing_season = list(range(start,end+1))
        print(start_date_plant_height_value)
        full_growing_season_df = pd.DataFrame({'day_after_start_measure':full_growing_season})
        # print(full_growing_season_df['timestamp'])
        if same_length=='':

            full_growing_season_df['timestamp'] = (full_growing_season_df['day_after_start_measure'].apply(
                lambda x: start_date_plant_height_value + pd.DateOffset(days=x))).apply(lambda x: x[0])#.dt.date
            print(full_growing_season_df['timestamp'])
            env_df_full['timestamp']=pd.to_datetime(env_df_full['timestamp'])
            print(env_df_full['timestamp'])
            full_growing_season_df = pd.merge(full_growing_season_df,env_df_full,on='timestamp')

        group_canopy_df = group_year_canopy.get_group(group_canopy)
        group_height_df = group_year_height.get_group(group_height)

        #group by plotUID and merge with full time steps
        group_based_on_plot_canopy = group_canopy_df.groupby('plot.UID')
        group_based_on_plot_height = group_height_df.groupby('plot.UID')
        # print(group_canopy_df)
        for group_plot_canopy,group_plot_height in zip(group_based_on_plot_canopy.groups,group_based_on_plot_height.groups):
            plot_canopy_df = group_based_on_plot_canopy.get_group(group_plot_canopy)
            extend_df_canopy = pd.merge(plot_canopy_df,full_growing_season_df,on='day_after_start_measure',how='outer')
            if same_length=='':
                # keep environment factors and time stamp even the plant phenotype value is nan
                extend_df_canopy['timestamp'] = full_growing_season_df['timestamp']
                extend_df_canopy['Air_temperature_0.1_m'] = full_growing_season_df['Air_temperature_0.1_m']
                extend_df_canopy['Air_temperature_2_m'] = full_growing_season_df['Air_temperature_2_m']
                extend_df_canopy['Relative_air_humidity_2_m'] = full_growing_season_df['Relative_air_humidity_2_m']
                extend_df_canopy['Short_wavelenght_solar_irradiance_2_m'] = full_growing_season_df['Short_wavelenght_solar_irradiance_2_m']
                extend_df_canopy['Soil_temperature_-0.05_m'] = full_growing_season_df['Soil_temperature_-0.05_m']
                extend_df_canopy.drop(columns=['timestamp_x', 'timestamp_y'],inplace=True)
                extend_df_canopy.drop(columns=['Air_temperature_0.1_m_x','Air_temperature_0.1_m_y'],inplace=True)
                extend_df_canopy.drop(columns=['Air_temperature_2_m_x', 'Air_temperature_2_m_y'], inplace=True)
                extend_df_canopy.drop(columns=['Relative_air_humidity_2_m_x', 'Relative_air_humidity_2_m_y'],inplace=True)
                extend_df_canopy.drop(columns=['Short_wavelenght_solar_irradiance_2_m_x', 'Short_wavelenght_solar_irradiance_2_m_y'],inplace=True)
                extend_df_canopy.drop(
                    columns=['Soil_temperature_-0.05_m_x', 'Soil_temperature_-0.05_m_y'],inplace=True)

            extend_df_canopy['plot.UID'] = plot_canopy_df['plot.UID'].unique()[0]
            extend_df_canopy['year_site.harvest_year'] = plot_canopy_df['year_site.harvest_year'].unique()[0]
            extend_df_canopy['genotype.id'] = plot_canopy_df['genotype.id'].unique()[0]
            extend_df_canopy['plot.row_global'] = plot_canopy_df['plot.row_global'].unique()[0]
            extend_df_canopy['plot.range_global'] = plot_canopy_df['plot.range_global'].unique()[0]
            extend_df_canopy['lot'] = plot_canopy_df['lot'].unique()[0]


            plot_height_df = group_based_on_plot_height.get_group(group_plot_height)
            extend_df_height = pd.merge(plot_height_df, full_growing_season_df, on='day_after_start_measure',
                                        how='outer')
            extend_df_height['plot.UID'] = plot_height_df['plot.UID'].unique()[0]
            extend_df_height['year_site.harvest_year'] = plot_height_df['year_site.harvest_year'].unique()[0]
            extend_df_height['genotype.id'] = plot_height_df['genotype.id'].unique()[0]
            extend_df_height['plot.row_global'] = plot_height_df['plot.row_global'].unique()[0]
            extend_df_height['plot.range_global'] = plot_height_df['plot.range_global'].unique()[0]
            extend_df_height['lot'] = plot_height_df['lot'].unique()[0]
            if same_length =='':
                extend_df_height['timestamp'] = full_growing_season_df['timestamp']
                extend_df_height['Air_temperature_0.1_m'] = full_growing_season_df['Air_temperature_0.1_m']
                extend_df_height['Air_temperature_2_m'] = full_growing_season_df['Air_temperature_2_m']
                extend_df_height['Relative_air_humidity_2_m'] = full_growing_season_df['Relative_air_humidity_2_m']
                extend_df_height['Short_wavelenght_solar_irradiance_2_m'] = full_growing_season_df[
                    'Short_wavelenght_solar_irradiance_2_m']
                extend_df_height['Soil_temperature_-0.05_m'] = full_growing_season_df['Soil_temperature_-0.05_m']
                extend_df_height.drop(columns=['timestamp_x', 'timestamp_y'], inplace=True)
                extend_df_height.drop(columns=['Air_temperature_0.1_m_x', 'Air_temperature_0.1_m_y'], inplace=True)
                extend_df_height.drop(columns=['Air_temperature_2_m_x', 'Air_temperature_2_m_y'], inplace=True)
                extend_df_height.drop(columns=['Relative_air_humidity_2_m_x', 'Relative_air_humidity_2_m_y'], inplace=True)
                extend_df_height.drop(
                    columns=['Short_wavelenght_solar_irradiance_2_m_x', 'Short_wavelenght_solar_irradiance_2_m_y'],
                    inplace=True)
                extend_df_height.drop(
                    columns=['Soil_temperature_-0.05_m_x', 'Soil_temperature_-0.05_m_y'], inplace=True)

            new_canopy_df = pd.concat([new_canopy_df,extend_df_canopy])
            # print(new_canopy_df)
            new_height_df = pd.concat([new_height_df, extend_df_height])
    else:
        # print(new_height_df['value'])
        print(height_df_multiple_year)
        # print(len(copy.deepcopy(new_height_df).dropna(subset='value').index))
        # print(len(canopy_df_multiple_year.index))
        # print(len(copy.deepcopy(new_canopy_df).dropna(subset='value').index))
        assert len(height_df_multiple_year.dropna(subset='value').index) == len(copy.deepcopy(new_height_df).dropna(subset='value').index)
        with open('rest_timestamps_dfs.dill', 'wb') as file:
            dill.dump([new_canopy_df,new_height_df], file)
        file.close()
        # print(new_height_df.groupby(['plot.UID']).first()['day_after_start_measure'].unique())
        if keep_environment_covariant:
            new_height_df.to_csv('../processed_data/align_height_env{}.csv'.format(same_length))
            new_canopy_df.to_csv('../processed_data/align_canopy_env{}.csv'.format(same_length))
        else:
            new_height_df.to_csv('../processed_data/align_height{}.csv'.format(same_length))
            new_canopy_df.to_csv('../processed_data/align_canopy{}.csv'.format(same_length))


        group1_object=new_height_df.groupby(['year_site.harvest_year','day_after_start_measure'])
        group2_object =new_canopy_df.groupby(['year_site.harvest_year','day_after_start_measure'])
        # if in a day values for all genotypes for all of the features is na, then drop it
        h_df = pd.DataFrame()
        c_df = pd.DataFrame()
        for (group1,group2) in zip(group1_object.groups,group2_object.groups):
            print(group1)
            df1 = group1_object.get_group(group1)
            df2 = group2_object.get_group(group2)
            print(df1.columns)
            if df1['value'].isna().all() and df2['value'].isna().all():
                print(df1['value'])
            else:
                h_df = pd.concat([h_df,df1],ignore_index=True)
                c_df = pd.concat([c_df, df2],ignore_index=True)
                print(h_df)
        h_df['timestamp'] = h_df['day_after_start_measure']
        c_df['timestamp'] = c_df['day_after_start_measure']
        if keep_environment_covariant:
            h_df.to_csv('../processed_data/align_height_drop_na_env{}.csv'.format(same_length))
            c_df.to_csv('../processed_data/align_canopy_drop_na_env{}.csv'.format(same_length))
        else:

            h_df.to_csv('../processed_data/align_height_drop_na{}.csv'.format(same_length))
            c_df.to_csv('../processed_data/align_canopy_drop_na{}.csv'.format(same_length))
        return c_df,h_df
def average_based_on_days(data_dfs,window_size:int=1):

    print('{} feature dataframe for average'.format(len(data_dfs)))
    if window_size<1:
        #check window size
        raise ValueError(
            'Please input valid window size (days), which need to be equal or larger than 1, the input window size is {}'.format(
                window_size))
    dfs=[]
    for df in data_dfs:
        #loop trough two features: canopy coverage and plant height
        # print(len(df[df['year_site.harvest_year']==2019]['plot.UID'].unique()))
        print(len(df[df['year_site.harvest_year'] == 2019]['day_after_start_measure'].unique()))
        print(df[df['plot.UID']=='FPWW0240001']['day_after_start_measure'].unique())
        print(df.groupby(['plot.UID',df.day_after_start_measure // window_size])['value'].mean().reset_index()['day_after_start_measure'].unique())


        average_df = df.groupby(['plot.UID','year_site.harvest_year','genotype.id', 'plot.row_global', 'plot.range_global', 'lot',df.day_after_start_measure // window_size])['value'].mean().reset_index()
        average_df['timestamp']=average_df['day_after_start_measure']
        average_df.rename(columns={'day_after_start_measure':'window_size_{}*day_after_start_measure'.format(window_size)})

        print(average_df)
        print(len(df[df['year_site.harvest_year']==2019].index))

        dfs.append(average_df)
    else:
        print(
            'Average based on window size {} days finish, save temporal file and return dataframes in list'.format(window_size)
        )
        with open('average_dfs.dill', 'wb') as file:
            dill.dump(dfs, file)
        file.close()
        for year in list(average_df['year_site.harvest_year'].unique()):
            print(year)
            print('after average with window size{}, time stamps length:{}'.format(window_size,len(average_df[average_df['year_site.harvest_year']==year]['timestamp'].unique())))
        return dfs

def average_based_on_genotype(df:pd.DataFrame,average_value='value'):
    '''
    for plants with same genotype, calculate average of traits value at the same time stamp
    '''
    print('dataframe before average:')
    print(df)
    print('_____________________________________')
    average_df = df.groupby(['genotype.id','timestamp'])[average_value].mean().to_frame()

    # print(average_df.index)
    average_df.reset_index(inplace=True)
    average_df.loc[:, 'timestamp'] = average_df.loc[:,'timestamp'].apply(convert_date_to_namber)
    average_df = average_df.astype('float64')
    #print(average_df)
    # sns.lineplot(x=average_df['timestamp'],y= average_df['value'],hue=average_df['genotype.id'])
    # plt.show()

    ################################################
    print('after average::::::::::')
    print(average_df)
    #convert to the same time length
    groups_object = average_df.groupby('genotype.id')
    length = 0
    time_list = []
    genotypeid_list=[]
    for i,item in enumerate(groups_object.groups):

        # group based on location
        plant_df = groups_object.get_group(item)
        plant_df.set_index("timestamp",inplace=True)
        new_time_set = set(list(plant_df.index))
        time_list = set(time_list)
        time_list.update(new_time_set)
        time_list = list(time_list)
        genotypeid_list.append(plant_df['genotype.id'].unique().item())
    # because there are some day missing for the whole dataset, add those days
    time_list.sort()

    start_day = int(time_list[0])
    end_day = int(time_list[-1])
    # print(start_day)
    # print(end_day)
    time_list = list(range(start_day,end_day+1))
    print(len(time_list))
    print(time_list)
    #creat new empty dataframe with the time list

    average_df.set_index(['timestamp', 'genotype.id'], inplace=True)
    idx = pd.MultiIndex.from_product([time_list,genotypeid_list],
                                 names=['timestamp', 'genotype.id'])

    new_df = pd.DataFrame(index=idx,columns=["na"]).astype(float)
    new_df= new_df.merge(average_df, how='left', left_index=True, right_index=True)

    new_df.reset_index(inplace=True)

    new_df.drop(columns='na',inplace=True)

    return new_df


def logistic_ode_model(X, t, theta):
    #logistic ode
    # unpack parameters
    y = X
    r, y_max, yt0 = theta
    # equations

    dy_dt = r * y *(1-(y/y_max))
    return dy_dt

def irradiance_ode_model(X, t, theta):
    #irradiance ode
    # unpack parameters
    y,irradiance_value = X
    r, a,phi, y_max, yt0 = theta
    # equations
    dy_dt = (r+(a*math.sin((2*math.pi/365)*t+phi))) * y *(1-(y/y_max))

    return dy_dt
def plot_data(ax, time,data, lw=2,title="Plant height Data"):
    '''plot data, use together with plot model to show how the ode models fit with the data'''
    # height_df = data_load("C:/data_from_paper/ETH/olivia_WUR/process_data/Plant height.csv",  2019)
    # height_df = average_based_on_genotype(height_df)
    # data = copy.deepcopy(height_df)
    # data = data[data['genotype.id']==537]
    data = data.drop(columns='genotype.id')
    # print(len(data.timestamp))
    # print(len(time))
    ax.plot(time, data.value, color="g", lw=lw, marker="+", markersize=14, label="Height (Data)")
    ax.legend(fontsize=14, loc="center left", bbox_to_anchor=(1, 0.5))

    ax.set_ylim(0)
    ax.set_xlabel("Date", fontsize=14)
    ax.set_ylabel("Value", fontsize=14)
    #ax.set_xticks(range(int(list(data.timestamp)[0]),int(list(data.timestamp)[-1])))
    #ax.set_xticklabels(ax.get_xticks(), rotation=45)
    ax.set_title(title, fontsize=16)
    return ax
def plot_model(
    ax,
    x_y,
    time=np.arange(0,64, 1),
    alpha=1,
    lw=3,
    title="plot model",
):
    ax.plot(time, x_y[:, 0], color="g", alpha=alpha, lw=lw, )#label="Height (Model)"
    ax.legend(fontsize=14)
    ax.set_title(title, fontsize=16)
    return ax
def solve_logistic_ode(ode_model:callable, data:pd.DataFrame, theta):
    """

    :param ode_model:
    :param t: time array
    :param data: real data
    :param theta: np.array, parameters
    :param ax,ax for plot
    :return:
    """

    # call Scipy's odeint function
    import numpy as np
    from scipy.stats import norm
    #fit ode for every sample
    groups_object = data.groupby("genotype.id")
    df = pd.DataFrame()
    residual_df = pd.DataFrame()
    res_list = []
    for i, item in enumerate(groups_object.groups):
        input_data = groups_object.get_group(item)
        input_data.dropna(subset=['value'], inplace=True)
        #print(input_data)
        #plot_data
        time = input_data['timestamp'].unique()
        fig, ax = plt.subplots(figsize=(12, 4))
        ax = plot_data(ax, time,data=input_data, lw=0)

        # function that calculates residuals based on a given theta
        def ode_model_resid(theta):

            return (
                    input_data[["value"]] - odeint(func=ode_model, y0=theta[-1:], t=time, args=(theta,))
            ).values.flatten()


        results = least_squares(ode_model_resid, x0=theta,bounds=(0,np.inf))

        # put the results in a dataframe for presentation and convenience

        parameter_names = ["r", "y_max",  "y0"]
        newline=pd.DataFrame()
        newline["Parameter"] = parameter_names
        newline["Least Squares Solution"] = results.x
        newline.set_index(keys='Parameter',drop=True,inplace=True)
        newline=newline.round(3)

        # print(newline)
        # print(results)
        parameters = results.x
        print(parameters)
        x_y = odeint(func=logistic_ode_model, y0=parameters[-1:], t=time, args=(parameters,))
        plot_model(ax, x_y, time=time)
        fig.autofmt_xdate()
        # plt.show()
        # plot residule
        # print(np.squeeze(input_data[["value"]].T.to_numpy()))
        # print(np.squeeze(x_y[:, 0]))
        residual = (x_y[:, 0]-np.squeeze(input_data[["value"]].T.to_numpy()))
        sns.scatterplot(x=time,y=residual)
        # plt.show()
        # print(residual)
        residual_row= pd.DataFrame({'residual':residual})
        import statsmodels

        residual_df = pd.concat([residual_df,residual_row])
        sns.histplot(residual_row)
        # plt.show()
        import statsmodels.api as sm
        result_residual = sm.stats.stattools.durbin_watson(residual, axis=0) #https://www.statsmodels.org/dev/generated/statsmodels.stats.stattools.durbin_watson.html
        print(result_residual)
        res_list.append(result_residual)
        fig = sm.qqplot(np.array(sorted(residual)))
        plt.show()
        # sns.barplot(residual)
        # plt.show()
        df = pd.concat([df,newline.T])
    res_list_df = pd.DataFrame({'test':res_list})
    res_list_df.to_csv('residual_correlation_test.csv')
    df.to_csv("parameters.csv")
    residual_df.to_csv('residual.csv')
    return df


def posterior_predict_check(input_data, pytensor_forward_model_matrix, theta):
    with pm.Model() as model:
        # Priors
        r = pm.TruncatedNormal("r", mu=theta[0], sigma=0.1, lower=0, initval=theta[0])
        y_max = pm.TruncatedNormal("y_max", mu=theta[1], sigma=0.01, lower=0, initval=theta[1])
        yt0 = pm.TruncatedNormal("yto", mu=theta[2], sigma=1, lower=0, initval=theta[2])
        sigma = pm.HalfNormal("sigma", 10)

        # Ode solution function
        ode_solution = pytensor_forward_model_matrix(
            pm.math.stack([r, y_max, yt0])
        )

        # Likelihood
        print(input_data[["value"]])
        pm.Normal("value_obs", mu=ode_solution, sigma=sigma, observed=input_data[["value"]].values)

        # Sample from the posterior
        trace1 = pm.sample(200, tune=50)
        print(trace1)

        trace_df = az.extract(trace1, num_samples=200)
        print(trace_df)


def main():
    load_multiple_year_data(height_file="C:/data_from_paper/ETH/olivia_WUR/process_data/Plant height.csv",
                            canopy_coverage_file="C:/data_from_paper/ETH/olivia_WUR/process_data/Canopy coverage.csv",
                            env_save_directory="C:/data_from_paper/ETH/olivia_WUR/process_data/",
                            years=[2018,2019,2021,2022],keep_environment_covariant=True,same_length='')
    # load_multiple_year_data(height_file="C:/data_from_paper/ETH/trait_repository_clone/process_data/Plant heights.csv",
    #                         canopy_coverage_file="C:/data_from_paper/ETH/trait_repository_clone/process_data/Canopy coverages.csv",
    #                         env_save_directory="C:/data_from_paper/ETH/trait_repository_clone/process_data/",
    #                         years=[2016,2017,2018,2019,2021,2022],keep_environment_covariant=True)
    # canopy,height = load_multiple_year_data()
    # average_based_on_days([canopy,height],5)
    # 'load rf model'
    # with open('model/rf_model/best_validation{}_rf_after_fit.dill', 'rb') as file:
    #     best_rf=dill.load(file)
    # file.close()
    # best_rf.predict()
    '''
    canopy_df = data_load("C:/data_from_paper/ETH/olivia_WUR/process_data/Canopy coverage.csv",  2019)
    av_data_canopy = average_based_on_genotype(canopy_df)
    height_df = data_load("C:/data_from_paper/ETH/olivia_WUR/process_data/Plant height.csv",  2019)
    av_data_height = average_based_on_genotype(height_df)
    # save temporary data
    # av_data_canopy.to_csv("C:/data_from_paper/ETH/olivia_WUR/process_data/Canopy_coverage_average_2019.csv")
    # av_data_height.to_csv("C:/data_from_paper/ETH/olivia_WUR/process_data/Plant_height_average_2019.csv")

    time = av_data_height['timestamp'].unique() #there is a around 100 days time gap in the middle of time series
    # plot

    # note theta =  gamma, y_max, yt0
    theta = np.array([0.05, 1.0, 0.006])


    fit_para_df = solve_logistic_ode(logistic_ode_model, av_data_height, theta)
    print(fit_para_df)

    residual = pd.read_csv('residual.csv',header=0,index_col=0)
    from scipy import stats
    result,p =stats.normaltest(residual)
    print(result,p)
    '''
if __name__ == "__main__":
    main()