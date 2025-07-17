'''
This script is to combine traits from multiple years experiment together, merge environment factors.
Notice this is the script that process the raw data from ETH, other proccessing is basd on this
author:Yingjie shao
Usage: python3 raw_data_merge_visualize.py
Currently, the directory of files are specified in the code
'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import glob




def read_file_environment_factors(file="C:/data_from_paper/ETH/olivia_WUR/covariates.csv"):
    #read and calculated daily average and plot
    raw_df = pd.read_csv(file)
    environment_factors = list(raw_df['covariate.name'].unique())

    environment_dictionary = dict()
    for environment_factor in environment_factors:
        print(environment_factor)
        environment_dictionary[environment_factor] = (
            raw_df)[raw_df['covariate.name'] == environment_factor][['timestamp', 'value']]
        environment_dictionary[environment_factor].set_index('timestamp',inplace=True)

        environment_dictionary[environment_factor].index = environment_dictionary[environment_factor].index.map(lambda x: str(str(x).split("T")[0]))
        #calculated mean value
        environment_dictionary[environment_factor]["value"] = environment_dictionary[environment_factor].\
        groupby(environment_dictionary[environment_factor].index)['value'].mean()

        environment_dictionary[environment_factor]["year"] = environment_dictionary[environment_factor].index.map(lambda x: str(x).split("-")[0])

        """convert index type to datetime"""
        environment_dictionary[environment_factor].index = pd.to_datetime(environment_dictionary[environment_factor].index)
        environment_dictionary[environment_factor]['time_stamp'] = environment_dictionary[environment_factor].index
        environment_dictionary[environment_factor] = environment_dictionary[environment_factor].drop_duplicates()
        environment_dictionary[environment_factor] = environment_dictionary[environment_factor].drop(columns='time_stamp')

        file_name = str(environment_factor).replace(" ", "_")
        dir_name = file.split("covariates.csv")[0]
        print(dir_name)
        print("save to file: {}/process_data/{}.csv".format(dir_name,file_name))

        environment_dictionary[environment_factor].to_csv("{}/process_data/{}.csv".format(dir_name,file_name))

        #plot 2018
        #df_2018 = environment_dictionary[environment_factor][environment_dictionary[environment_factor]["year"] == '2018'].sort_values(by='timestamp')
        #print(df_2018.index)
        sns.lineplot(x=list(environment_dictionary[environment_factor].index), y=environment_dictionary[environment_factor]['value'], label=environment_factor)

    plt.legend()
    plt.show()

def read_file_without_genotype():
    import glob
    files = glob.glob("C:/data_from_paper/ETH/olivia_WUR/trait_data*.csv")
    files = [str(x).split('\\')[1] for x in files]
    print(files)
    trait_id_name_df = pd.DataFrame(columns =['id','name'])
    for file in files:
        df = pd.read_csv('C:/data_from_paper/ETH/olivia_WUR/{}'.format(file),nrows=2)
        trait=df['trait.name'].unique()
        id = [file.split('_')[6]]
        print('trait id:{} name:{}'.format(id,trait))
        new_row=pd.DataFrame.from_dict({'id':id,'name':trait},orient='columns')
        trait_id_name_df = pd.concat([trait_id_name_df,new_row],axis=0)
    trait_id_name_df.set_index('id',inplace=True)
    trait_id_name_df.sort_index(inplace=True)
    trait_id_name_df = trait_id_name_df.drop_duplicates()
    print(trait_id_name_df)

def combine_multiple_year_experiments_save_files_for_plant_height_and_canopy_coverge(directory="C:/data_from_paper/ETH/olivia_WUR/"):
    """
    selcet files based on trait id, combine those csv files and save
    :return: None
    """
    trait_id = {'Plant height':5,'Canopy coverage':38,'yield':8}
    df_dict = {}

    for key in trait_id.keys():
        files = glob.glob("{}trait_data*_trait_id_{}_*.csv".format(directory,trait_id[key]))
        print(files)
        files = [str(x).split('\\')[1] for x in files]
        # print(files)
        df_dict[key] = pd.DataFrame()

        for file in files:
            print('read trait value file....')
            print(file)
            new_df = pd.read_csv("{}{}".format(directory,file),header=0)
            print(new_df)
            plot_id = file.split("_")[2]
            lot = file.split("_")[3]
            #read design matrix and merge
            new_df['lot'] = lot
            new_df['timestamp'] = pd.to_datetime(new_df['timestamp']).dt.date
            print('duplicate values:')
            if new_df.duplicated(['timestamp', 'plot.UID','lot', 'trait.name']).sum()>0:
                print('drop duplicate by average')
                new_df = new_df.groupby(['timestamp', 'plot.UID','lot', 'trait.name'])['value'].mean().reset_index()
                print(new_df)
            design_df = pd.read_csv("{}design_{}_{}.csv".format(directory,plot_id,lot))

            new_df = pd.merge(new_df,design_df,left_on=['plot.UID'],right_on= ['plot.UID'],how='left')
            #do not use joint! https://stackoverflow.com/questions/22676081/what-is-the-difference-between-join-and-merge-in-pandas
            df_dict[key] = pd.concat([df_dict[key],new_df])
            df_dict[key]['timestamp'] = pd.to_datetime(df_dict[key]['timestamp']).dt.date
            # print(df_dict[key])
        else:
            # drop columns only have same values
            for col in df_dict[key].columns:
                if len(df_dict[key][col].unique()) == 1 and str(col) != 'trait.name':
                    df_dict[key] = df_dict[key].drop(columns=col)
            # df_dict[key]['timestamp'] = pd.to_datetime(df_dict[key]['timestamp']).dt.date
            # read environment factors and merge
            files = glob.glob("{}process_data/*_m.csv".format(directory))
            print(files)
            files = [str(x).split('\\')[1] for x in files]
            for environment_factors_file in files:

                env_df = pd.read_csv("{}process_data/{}".format(directory,environment_factors_file))
                env_df.rename(columns={"value":"{}".format(environment_factors_file.split('.csv')[0])},inplace=True)
                env_df.drop(columns='year',inplace=True)
                env_df['timestamp'] = pd.to_datetime(env_df['timestamp']).dt.date
                print(len(env_df['timestamp'].unique()))
                df_dict[key] = pd.merge(df_dict[key], env_df, on='timestamp',how='left')

                # print(df_dict[key][df_dict[key]['timestamp'] == datetime.strptime('2019-05-25', '%Y-%m-%d').date()])
                # print(len(df_dict[key]['timestamp'].unique()))
            #only keep useful columns
            try:
                df_dict[key] = df_dict[key][
                    ['plot.UID', 'timestamp', 'value', 'genotype.id', 'lot', 'year_site.harvest_year', 'plot.replication',
                     'Air_temperature_0.1_m', 'Air_temperature_2_m', 'Precipitation_2_m', 'Relative_air_humidity_2_m',
                     'Short_wavelenght_solar_irradiance_2_m', 'Soil_temperature_-0.05_m','trait.name',
                     'value_json','plot.row_global', 'plot.range_global']]
            except:
                # if static data, may without value jason
                df_dict[key] = df_dict[key][
                    ['plot.UID', 'timestamp', 'value', 'genotype.id', 'lot', 'year_site.harvest_year', 'plot.replication',
                     'Air_temperature_0.1_m', 'Air_temperature_2_m', 'Precipitation_2_m', 'Relative_air_humidity_2_m',
                     'Short_wavelenght_solar_irradiance_2_m', 'Soil_temperature_-0.05_m','trait.name',
                     'plot.row_global', 'plot.range_global']]

            #chek duplicate because some year are measured with two methods, will take average for now

            df_dict[key].dropna(axis='columns')
            print(df_dict[key].nunique())
            df_dict[key].to_csv("{}process_data/{}_6years.csv".format(directory,key))
            print(df_dict[key]['plot.replication'].unique())

            # plot_multiple_environment_factors_lineplot(df_dict[key])

    #group based on year and plot
    print(df_dict)

def plot_height_coverage_together(height_file,coverage_file):
    """
    plot height and coverage for a example genotype
    :param height_file:
    :param coverage_file:
    :return:
    """
    height_df = pd.read_csv(height_file,index_col=0,header=0)
    coverage_df = pd.read_csv(coverage_file, index_col=0, header=0)
    #convert type to date for plot
    height_df['timestamp'] = pd.to_datetime(height_df['timestamp']).dt.date
    df = pd.concat([height_df],axis=0)#,coverage_df

    fig, ax = plt.subplots()
    # plot an example
    harvest_years = list(df['year_site.harvest_year'].unique())

    print(harvest_years)

    yearly_experiment = df[df['year_site.harvest_year'] == 2019]
    df_print = yearly_experiment[yearly_experiment['genotype.id'] == 335]
    print(df_print)
    sns.lineplot(df_print, x='timestamp', y='value', hue='year_site.harvest_year',style='plot.UID', markers=True,color='b',
             ax=ax)  # plot.UID #hue="trait.name"
    # plt.ylabel("coverage")
    # ax2 = plt.twinx()
    # sns.lineplot(df_print[df_print['trait.name'] == 'Plant height'], x='timestamp', y='value', hue="plot.UID",color='g', markers=True,
    #              ax=ax)  # plot.UID
    # plt.ylabel("height")
    plt.show()


def plot_multiple_environment_factors_lineplot(df):
    yearly_experiment = df[df['year_site.harvest_year'] == 2019]

    df_print = yearly_experiment[yearly_experiment['genotype.id'] == 335]
    #drop_columns
    for col in df_print.columns:

        if len(df_print[col].unique()) <= 1:
            df_print.drop(columns=col,inplace=True)
    print(df_print)
    df_print.drop(columns=['value','plot.replication','plot.row_global','plot.range_global'],inplace=True) #,'corrected_value'
    df_print['timestamp'] = pd.to_datetime(df_print['timestamp']).dt.date
    df_print.set_index('timestamp', inplace=True)
    sns.lineplot(df_print)
    plt.show()

def find_genotype_present_at_multiple_years(df_name="../processed_data/align_height_env.csv",number_of_years=4)->list:
    df = pd.read_csv(df_name,header=0,index_col=0)
    gen_list = []
    for genotyep in df['genotype.id'].unique():
        present_year = len(df[df['genotype.id']==genotyep]['year_site.harvest_year'].unique())
        if present_year== number_of_years:
            gen_list.append(genotyep)
    else:
        print('following genotype present in all {} years: \n years:'.format(number_of_years))
        print(df[df['genotype.id']==gen_list[0]]['year_site.harvest_year'].unique())
        print('genotypes id:')
        print(gen_list)
        print(len(gen_list))
        return gen_list

def genotype_present_in_at_least_in_one_of_selected_years(df_path="../processed_data/align_height_env.csv",year_list:list=[2018,2019])->list:
    df = pd.read_csv(df_path, header=0, index_col=0)
    gen_list = []
    for genotype in df['genotype.id'].unique():
        select_genotype_df_years = list(df[df['genotype.id']==genotype]['year_site.harvest_year'].unique())
        # print(select_genotype_df_years)
        if bool(set(year_list) &set(select_genotype_df_years)):
            print('add genotype:{}'.format(genotype))
            gen_list.append(genotype)
    else:
        print('{} genotypes present in at least one of given years {}'.format(len(gen_list),year_list))
        print(gen_list)
        return gen_list

def genotype_present_in_specific_years(df_path="../processed_data/align_height_env.csv",year_list:list=[2018,2019,2021,2022])->list:
    df = pd.read_csv(df_path, header=0, index_col=0)
    gen_list = []
    for genotype in df['genotype.id'].unique():
        select_genotype_df_years = list(df[df['genotype.id']==genotype]['year_site.harvest_year'].unique())
        # print(select_genotype_df_years)
        if set(year_list).issubset(set(select_genotype_df_years)):
            print('add genotype:{}'.format(genotype))
            gen_list.append(genotype)
    else:
        print('{} genotypes present in year {}'.format(len(gen_list),year_list))
        print(gen_list)
        return gen_list
def main():
    combine_multiple_year_experiments_save_files_for_plant_height_and_canopy_coverge(
        directory="C:/data_from_paper/ETH/trait_repository_clone/")
    plot_height_coverage_together("C:/data_from_paper/ETH/olivia_WUR/process_data/Canopy coverage.csv",
                                  "C:/data_from_paper/ETH/olivia_WUR/process_data/Plant height.csv")

    genotype_present_in_specific_years(year_list=[2018,2019])
    genotype_present_in_at_least_in_one_of_selected_years(year_list=[2018,2019,2021,2022])

if __name__ == "__main__":
    main()