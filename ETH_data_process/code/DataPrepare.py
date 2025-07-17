# correct spatial trend with NN model compared base on MSE by using time series for yield prediction, genotype specific growth curve(NA is removd when calculated MAPE)
import copy

from operator import itemgetter

import numpy as np  # for maths
import pandas as pd  # for data manipulation
import torch
from prettytable import PrettyTable
import random
import LoadData
from sklearn.model_selection import GroupShuffleSplit


def count_parameters(model):
    print(model)

    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():

        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param

    print(table)
    print(f"Total Trainable Params: {total_params}")

    return total_params


def convert_inputx_to_tesor_list(datasets):
    """
    input: datasets: list of feature dataframe,
    return: 3 dimension torch.tensor for inputX with multiple features
    """
    tensor_datasets = []
    for df in datasets:
        df = df.T

        sequences = df.astype(np.float32).to_numpy().tolist()
        dataset = torch.stack([torch.tensor(s).unsqueeze(1).float() for s in sequences])

        tensor_datasets.append(dataset)
    # remove the last dimention

    tensor_dataset = torch.squeeze(torch.stack(tensor_datasets, dim=0), dim=-1)
    tensor_dataset = torch.permute(tensor_dataset, (2, 1, 0))
    print("shape of created dataset [seq_length,num_seq,feature_size]: {}".format(tensor_dataset.shape))

    return tensor_dataset


def train_test_split_based_on_group(df, group_column: pd.DataFrame, group_name: str | list = 'genotype.id',
                                    test_size: float = 0.1,
                                    random_seeds: int = 0, n_split=5) -> dict:
    '''
    This method is to split the training test and validation group based on group_column(usually, genotype), so no overlap genotypes between
    dfs in three sets. test set will be the same for all train test split in returned dictionary
    parameters:
        df: input dataframe which will be split into train validation and test set, the split dim should be at first
        group_column: pd.Dataframe, which are used as group in train_test_split
        self.n_split(default=5): the number of split is defined in class create_tensor_dataset
    return: splitted_dictionary: dictionary which key is the number of split, value is a list '[train_index, validation_index, test_index]'
    '''
    try:
        group_column = group_column.drop(columns='index').reset_index()
    except:
        print(group_column)
    from sklearn.model_selection import GroupShuffleSplit
    splitted_dictionary = {}
    n_splits = n_split

    for i in range(n_splits):
        splitted_dictionary['splits_{}'.format(i)] = []
    if isinstance(group_name, str):
        group_list = group_column[group_name].to_list()
    else:
        # print(group_column.columns)
        group_column['new_group_list'] = group_column[group_name].astype(str).apply(lambda row: '_'.join(row), axis=1)
        group_list = group_column['new_group_list'].to_list()
        # print(group_list)
    splitter = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state=random_seeds)  # 10%validation
    split = splitter.split(df, group_column, groups=group_list)
    train_and_test_index, test_index = next(split)  # test is the same for all train test split
    # split into train validate and test , save the index for each split
    # print(train_and_test_index, test_index)
    train_test_genotype = group_column.iloc[train_and_test_index]
    train_test_df = df.iloc[train_and_test_index, :]
    train_test_group_column = itemgetter(*train_and_test_index)(group_list)
    splitter = GroupShuffleSplit(train_size=0.8, n_splits=n_splits, random_state=random_seeds)
    for i, (train_index, validation_index) in enumerate(
            splitter.split(train_test_df, train_test_genotype, groups=train_test_group_column)):
        print('n_split {}'.format(i))
        # print(train_index)
        # print(validation_index)
        # the returned index is based on order, not the true index, thus will cause over lap with the first split
        # get the original index number
        train_index = train_test_genotype.iloc[train_index, :].index.to_list()
        validation_index = train_test_genotype.iloc[validation_index, :].index.to_list()
        # assert no overlap between train test and validation
        assert set(validation_index).isdisjoint(set(test_index))
        assert set(train_index).isdisjoint(set(test_index))
        assert set(train_index).isdisjoint(set(validation_index))
        print(':::validation index and test index is print below:::::')
        print(validation_index)
        print(test_index)
        splitted_dictionary['splits_{}'.format(i)] = [train_index, validation_index, test_index]
    return splitted_dictionary


def manually_data_split_based_on_one_group(group_df,split_group = 'year_site.harvest_year',n_splits=None):
    """
    perform cross validation based on all possible combination of training year
    """
    from itertools import combinations
    from sklearn.model_selection import GroupKFold

    if split_group=='year_site.harvest_year':
        # manually split
        print('years in data:{}'.format(group_df['year_site.harvest_year'].unique()))
        print(group_df)
        full_year_list = group_df['year_site.harvest_year'].unique().tolist()

        train_year_combination = list(combinations(full_year_list, 2))
        train_test_validation_dictionary = dict()
        for rand_seed in range(len(train_year_combination)):
            list_years = group_df['year_site.harvest_year'].unique().tolist()
            random.seed(rand_seed)
            train_years = train_year_combination[rand_seed]
            list_years = list(set(list_years) - set(train_years))
            print('train years')
            print(train_years)
            val_year = random.choice(list_years)
            print('validation year')
            print(val_year)
            list_years.remove(val_year)
            test_year = random.choice(list_years)
            print('test year')
            print(test_year)
            train_index = group_df[group_df['year_site.harvest_year'].isin(train_years)].index.tolist()
            val_index = group_df[group_df['year_site.harvest_year'] == val_year].index.tolist()
            test_index = group_df[group_df['year_site.harvest_year'] == test_year].index.tolist()
            print('{} split :'.format(rand_seed))
            print('train index:')
            print(train_index)
            print('val index:')
            print(val_index)
            print('test index:')
            print(test_index)
            assert_no_overlap(train_index, val_index, test_index)
            train_test_validation_dictionary['splits_{}'.format(rand_seed)] = [train_index, val_index, test_index]
        else:
            return train_test_validation_dictionary,train_year_combination
    else:
        print(group_df[split_group])
        if not n_splits:
            raise ValueError("n_split should be an number")
        group_kfold = GroupKFold(n_splits=n_splits)
        group_kfold.get_n_splits(group_df, group_df, group_df[split_group])
        train_test_validation_dictionary={}
        for i, (train_index, val_test_index) in enumerate(group_kfold.split(group_df, group_df, group_df[split_group])):
            val_test_group = group_df.iloc[val_test_index,:]

            genotype_splitter = GroupShuffleSplit(test_size=0.5, n_splits=1, random_state=i)  # 15%validation
            genotype_split = genotype_splitter.split(val_test_group, groups=val_test_group['genotype.id'])
            val_index, test_index = next(genotype_split)
            val_index = val_test_group.iloc[val_index].index.tolist()
            test_index = val_test_group.iloc[test_index].index.tolist()
            print('{} split :'.format(i))
            print('train index:')
            print(train_index)
            print(group_df.iloc[train_index,:]['genotype.id'].unique().tolist())
            print('val index:')
            print(val_index)
            print(group_df.iloc[val_index, :]['genotype.id'].unique().tolist())
            print('test index:')
            print(test_index)
            print(group_df.iloc[test_index, :]['genotype.id'].unique().tolist())
            #check on overlap between each pair of the three lists
            assert_no_overlap(train_index, val_index, test_index)
            train_test_validation_dictionary['splits_{}'.format(i)] = [train_index, val_index, test_index]


def assert_no_overlap(list1, list2, list3):
    # Convert lists to sets for efficient intersection checks
    set1, set2, set3 = set(list1), set(list2), set(list3)

    # Check for overlaps
    assert not (set1 & set2), f"Lists 1 and 2 overlap: {set1 & set2}"
    assert not (set1 & set3), f"Lists 1 and 3 overlap: {set1 & set3}"
    assert not (set2 & set3), f"Lists 2 and 3 overlap: {set2 & set3}"

def manually_split_on_two_groups(group_df):
    """
    First split based on year, then randomly assign 70% genotype to training set, then 15% for val and 15% test
    """
    train_test_validate_dictionary,train_years = manually_data_split_based_on_one_group(group_df,split_group='year_site.harvest_year')
    new_train_val_test_dictionary = {}
    random_seeds =0
    for split in train_test_validate_dictionary.keys():
        train_index_year,val_index_year,test_index_year = train_test_validate_dictionary[split]
        unique_genotype_list = group_df['genotype.id'].unique().tolist()
        random.seed(random_seeds)
        random.shuffle(unique_genotype_list)
        train_g = unique_genotype_list[: int(len(unique_genotype_list)*0.7)]
        val_g = unique_genotype_list[int(len(unique_genotype_list)*0.7):int(0.85*len(unique_genotype_list))]
        test_g = unique_genotype_list[int(0.85*len(unique_genotype_list)):]
        print('train,val,test g split')
        print(train_g, val_g, test_g)


        train_group = group_df.iloc[train_index_year,:]
        val_group = group_df.iloc[val_index_year, :]
        test_group = group_df.iloc[test_index_year, :]
        train_index = train_group[train_group['genotype.id'].isin(train_g)].index.to_list()
        val_index = val_group[val_group['genotype.id'].isin(val_g)].index.to_list()
        test_index = test_group[test_group['genotype.id'].isin(test_g)].index.to_list()
        print(train_index,val_index,test_index)
        assert_no_overlap(train_index, val_index, test_index)
        new_train_val_test_dictionary[f'splits_{random_seeds}'] = [train_index, val_index, test_index]
        random_seeds += 1
    else:
        return new_train_val_test_dictionary



def train_test_split_based_on_two_groups(df: pd.DataFrame, group_column: pd.DataFrame,
                                         test_size: float = 0.1, random_seeds: int = 0, n_split: int = 5) -> dict:
    '''
    Split DataFrame into training, validation, and test sets based on exclusive groups of genotype and year.
    If either the genotype or the year of any sample is present in one year_split, that genotype or year is excluded
    from other splits.

    Parameters:
        df: Input DataFrame to be year_split.
        group_column: DataFrame containing columns used for group exclusivity (e.g., genotype and year).
        group_name: List of column names to enforce as exclusive (e.g., ['genotype.id', 'year_site.harvest_year']).
        test_size: Proportion of the dataset to use as the test set.
        random_seeds: Random seed for reproducibility.
        n_split: Number of train-validation splits to generate.

    Returns:
        splitted_dictionary: Dictionary where each key represents a year_split (e.g., 'splits_0') and the value is a list
                             containing `[train_index, validation_index, test_index]`.
    '''

    splitted_dictionary = {}

    # Step 1: Split based on Year (2 years for training, 1 year for validation, 1 year for test)

    year_splitter = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state=random_seeds)  # 10%validation
    year_split = year_splitter.split(df, group_column, groups=group_column['year_site.harvest_year'])
    train_and_test_index, test_index = next(year_split)  # test is the same for all train test year_split
    # year_split into train validate and test , save the index for each year_split
    print('test index')
    print(test_index)
    # print(train_and_test_index, test_index)
    train_test_genotype = group_column.iloc[train_and_test_index]
    train_test_df = df.iloc[train_and_test_index, :]
    train_test_group_column = itemgetter(*train_and_test_index)(group_column['year_site.harvest_year'].to_list())
    year_splitter = GroupShuffleSplit(train_size=0.8, n_splits=n_split, random_state=random_seeds)
    for i, (train_index, validation_index) in enumerate(
            year_splitter.split(train_test_df, train_test_genotype, groups=train_test_group_column)):
        # print('n_split {}'.format(i))
        # print(train_index)
        # print(validation_index)
        # raise EOFError
        # the returned index is based on order, not the true index, thus will cause over lap with the first year_split
        # get the original index number
        train_index = train_test_genotype.iloc[train_index, :].index.to_list()
        validation_index = train_test_genotype.iloc[validation_index, :].index.to_list()
        # print('train index after year split:')
        # print(train_index)

        # get dataframe after year split
        train_group_df = df.iloc[train_index, :]
        validation_group_df = df.iloc[validation_index, :]
        test_group_df = df.iloc[test_index, :]
        # print(test_group_df)

        # then split based on genotype, and only keep genotype not overlap
        unique_genotype_list = group_column['genotype.id'].unique().tolist()
        genotype_splitter = GroupShuffleSplit(test_size=0.3, n_splits=1, random_state=random_seeds)  # 10%validation
        genotype_split = genotype_splitter.split(unique_genotype_list, groups=unique_genotype_list)
        train_and_test_g_index, test_g_index = next(genotype_split)  # test is the same for all train test year_split
        train_test_genotype_list = itemgetter(*train_and_test_g_index)(unique_genotype_list)
        genotype_splitter = GroupShuffleSplit(test_size=0.3, n_splits=1, random_state=random_seeds)  # 10%validation
        genotype_split = genotype_splitter.split(train_test_genotype_list, groups=train_test_genotype_list)
        train_g_index, validation_g_index = next(genotype_split)

        train_g = itemgetter(*train_g_index)(train_test_genotype_list)
        val_g = itemgetter(*validation_g_index)(train_test_genotype_list)
        test_g = itemgetter(*test_g_index)(
            unique_genotype_list)  # test is first split, so the this index is refer to the all genotype list
        print('train,val,test g split')
        print(train_g, val_g, test_g)
        # only get rows with corresponding genotype to avoid overlap
        train_group_df = train_group_df[train_group_df['genotype.id'].isin(train_g)]
        validation_group_df = validation_group_df[validation_group_df['genotype.id'].isin(val_g)]
        test_group_df = test_group_df[test_group_df['genotype.id'].isin(test_g)]
        print(train_group_df)
        print(test_group_df)

        train_index_final = train_group_df.index.to_list()
        validation_index_final = validation_group_df.index.to_list()
        test_index_final = test_group_df.index.to_list()
        print(':::validation index and test index is print below:::::')
        print(train_index_final)
        print(validation_index_final)
        print(test_index_final)
        # raise EOFError
        # Store the indices in the dictionary
        assert_no_overlap(train_index_final, validation_index_final, test_index_final)
        splitted_dictionary[f'splits_{i}'] = [train_index_final, validation_index_final, test_index_final]
    else:
        return splitted_dictionary


class create_tensor_dataset():
    """
    create 3-dimensions tensor from a list of dataframes
    :param dfs: list, the row of df is sequence length
    :return: X_shape(features,seq_length,samples_number);Y_shape(samples_number,class_number)
    """

    def __init__(self, dfs, use='yield_predict', year: int | list = 2019, average_endpoint_value=True, n_split=5,
                 random_seed=0):
        self.dfs = dfs
        self.use = use
        self.year = year
        self.average_endpoint_value = average_endpoint_value
        self.n_split = n_split
        self.seed = random_seed

    def one_hot_encoding_genotype(self, genotype, select_index=None):
        genotype = genotype.astype(int).to_numpy()
        # print(genotype)
        # one hot encoding genotype
        from torch.nn.functional import one_hot
        indices = np.squeeze(genotype)
        depth = len(np.unique(genotype))  # number of classes
        gene_code = list(range(0, depth))  # map genotype to an continous number: gene_code
        gene_map = {list(np.unique(genotype))[i]: gene_code[i] for i in range(depth)}
        mp = np.arange(0, max(indices) + 1)  # create a new dataframe from
        mp[list(gene_map.keys())] = list(
            gene_map.values())  # map genotype_id to gene_code, for python >3.7, dictionary is ordered
        maped_data = np.array(mp[indices])
        if select_index != None:
            maped_data = maped_data[select_index]

        print('unique genotype:{}'.format(len(set(maped_data))))
        genotype_tensor = one_hot(torch.from_numpy(maped_data).to(torch.int64), depth)
        # print(genotype_tensor.shape)
        label_number = genotype_tensor.shape[1]
        return genotype_tensor

    def keep_overlap_time_stamps_between_multiple_features_dfs(self):
        """
        check if number of time steps for two features are equal, only keep overlap

        return: new_dfs: list of dataframe with different features but same row and col, plot.UID as row and time steps as col
                genotype_df: dataframe use to map plot.UID with genotype.id, same order as features dataframe
                env_reshape_dictionary: dictionary which saves dataframe for different environment  factors with same row and col as df in new_dfs, plot.UID as row and time steps as col
        """
        environment_factors = ['Air_temperature_0.1_m',
                               'Air_temperature_2_m', 'Relative_air_humidity_2_m',
                               'Short_wavelenght_solar_irradiance_2_m', 'Soil_temperature_-0.05_m']

        # create dictionary for saving environment tensor
        env_reshape_dictionary = {}
        for env_factor in environment_factors:
            env_reshape_dictionary[env_factor] = None

        dfs = copy.deepcopy(self.dfs)  # dfs is list of df which for different features
        genotype_plot_mapper = pd.DataFrame()
        new_time_steps = pd.DataFrame()
        start = True
        # reset time stamp make sure the two feature df has the same order of genotypes and time
        ## save new time stamp
        for df in dfs:
            try:
                df = df[df['year_site.harvest_year'].isin(self.year)]
            except:
                df = df[df['year_site.harvest_year'] == self.year]
            if start:
                # print(df[['timestamp','year_site.harvest_year']]['timestamp'].unique())
                new_time_steps = df[['timestamp', 'year_site.harvest_year']].drop_duplicates()
                start = False
            else:
                new_time_steps = pd.merge(new_time_steps, df[['timestamp', 'year_site.harvest_year']], how='inner')
            genotype_plot_mapper = pd.concat(
                [genotype_plot_mapper, df[['plot.UID', 'genotype.id', 'plot.row_global', 'plot.range_global', 'lot',
                                           'year_site.harvest_year']]],
                axis=0)
        else:
            new_time_steps.drop_duplicates(inplace=True)
            print('new time stamps in keep overlap time stamp method:\n{}'.format(new_time_steps))

        genotype_plot_mapper = genotype_plot_mapper.drop_duplicates()
        # print(genotype_plot_mapper[genotype_plot_mapper['plot.UID']=='FPWW0240082'])
        # make sure the two feature df has the same order of genotypes and time
        merge_df = pd.DataFrame(df[['plot.UID', 'timestamp', 'year_site.harvest_year']])
        new_dfs = []
        print('before merge---')

        for index, df in enumerate(dfs):
            # loop and merge df with different features, to make sure the order of values are the same(same time, same genotype)
            df = pd.merge(new_time_steps, df, how='left',
                          on=['timestamp', 'year_site.harvest_year'])  # keep the same time steps
            print(merge_df[['plot.UID', 'timestamp', 'year_site.harvest_year']])
            print('df merge with time-----------------------')
            # there are some plot has two value on same day from different methods
            df = df.rename(columns={'value': 'value_{}'.format(index)})
            print(df)
            columns = df.columns
            # df11 = df[['plot.UID', 'timestamp', 'year_site.harvest_year']].sort_values(by=df[['plot.UID', 'timestamp', 'year_site.harvest_year']].columns.tolist()).reset_index(drop=True)
            # df21 = merge_df[['plot.UID', 'timestamp', 'year_site.harvest_year']].sort_values(by=merge_df[['plot.UID', 'timestamp', 'year_site.harvest_year']].columns.tolist()).reset_index(drop=True)
            # print(df[['plot.UID', 'timestamp', 'year_site.harvest_year']])
            # print("check equal after sort")
            # print(df11.equals(df21))
            # print(df11.drop_duplicates())
            # print(columns)
            merge_df = pd.merge(merge_df, df, how='outer', suffixes=('_x', ''),
                                right_on=['plot.UID', 'timestamp', 'year_site.harvest_year'],
                                left_on=['plot.UID', 'timestamp', 'year_site.harvest_year'])
            print('herecescdcccccccccccccccccccc')
            print(merge_df)
            new_df = merge_df[columns]
            print('dataset after merge')
            print(new_df)
            new_df = new_df.rename(columns={'value_{}'.format(index): 'value'})
            print('na number:{}'.format(new_df['genotype.id'].isna().sum()))
            for index in new_df[new_df['genotype.id'].isna()].index:
                # fill na causes by some values are missing in one dataframe
                plotid = new_df.loc[index, 'plot.UID']
                fill_na = genotype_plot_mapper.loc[
                    genotype_plot_mapper['plot.UID'] == plotid, ['genotype.id', 'plot.row_global', 'plot.range_global',
                                                                 'lot',
                                                                 'year_site.harvest_year']].values.squeeze().tolist()
                new_df.at[index, 'genotype.id'] = fill_na[0]
                new_df.at[index, 'plot.row_global'] = fill_na[1]
                new_df.at[index, 'plot.range_global'] = fill_na[2]
                new_df.at[index, 'lot'] = fill_na[3]
            print(new_df[new_df['genotype.id'].isna()])
            new_df['genotype.id'] = new_df['genotype.id'].astype(int)
            if 'index' in new_df.columns:
                print('drop index')
                new_df = new_df.drop(columns='index')
            new_dfs.append(new_df)
        reshape_dfs = []
        genotype_df = new_dfs[0][['plot.UID', 'genotype.id', 'lot', 'plot.row_global', 'plot.range_global',
                                  'year_site.harvest_year']].astype(
            object)
        create_env_tensor = False

        for df in new_dfs:
            assert genotype_df.equals(
                df[['plot.UID', 'genotype.id', 'lot', 'plot.row_global', 'plot.range_global',
                    'year_site.harvest_year']].astype(object))
            # print(
            #     'genotype_df__________'
            # )
            # print(genotype_df)
            genotype_df = df[['plot.UID', 'genotype.id', 'lot', 'plot.row_global', 'plot.range_global',
                              'year_site.harvest_year']].astype(object)
            # print(genotype_df)
            # reshap to time steps as index, plot.UID as columns
            df.drop(columns=['genotype.id', 'plot.row_global', 'plot.range_global'], inplace=True)
            print(df)
            df[df[
                ['plot.UID', 'timestamp',
                 'year_site.harvest_year']].duplicated()].to_csv('reshape_df.csv')
            print(df[df[
                ['plot.UID', 'timestamp',
                 'year_site.harvest_year']].duplicated()])
            assert len(df.index[df[
                ['plot.UID', 'timestamp',
                 'year_site.harvest_year']].duplicated()].unique()) == 0, 'The number of duplicate rows should be zero!'

            # df.drop_duplicates(inplace=True)
            df.reset_index(inplace=True, drop=True)
            # df = df.fillna(0.0)

            if create_env_tensor == False:
                # if we haven't create environment tensor
                # env_air_temperature_df = df.pivot(index='timestamp', columns='plot.UID', values='Air_temperature_2_m')
                for key in env_reshape_dictionary.keys():
                    print(df.columns)
                    env_reshape_dictionary[key] = df.pivot(index='timestamp', columns='plot.UID', values=key)

            df = df.pivot(index='timestamp', columns='plot.UID', values='value')

            # print('order df')
            mapper = copy.deepcopy(genotype_df).drop_duplicates()
            order_genotype = mapper.set_index('plot.UID').loc[
                df.columns.tolist()].reset_index()  # return genotype with location asthe order as plot.UID
            # print(order_genotype)
            assert order_genotype[
                       'plot.UID'].tolist() == df.columns.tolist()  # check the genotype dataframe should has the same order as original plot.UID for feature df
            reshape_dfs.append(df.T)
            # print(df.T)

        return reshape_dfs, order_genotype, env_reshape_dictionary

    def create_train_test_validation_index(self, year):
        dfs, genotype_df, env_dfs = self.keep_overlap_time_stamps_between_multiple_features_dfs()
        keep_index_env_dfs_list = []
        # features df in dfs should have the same order in time and genotype as row and col
        yield_tensor, na_index = self.read_yield_convert_to_tensor(genotype_df, year=year, average=True)
        keep_index = sorted(list(set(range(0, len(genotype_df.index))) - set(na_index)))
        # remove na before train test split
        genotype_df = genotype_df.iloc[keep_index, :].reset_index()
        for item in env_dfs.values():
            keep_index_env_dfs_list.append(item.iloc[keep_index, :].reset_index())

        print('unique genotypes number after drop na after merge with yield{}'.format(
            len(genotype_df['genotype.id'].unique())))
        remove_na_dfs = []
        for df in dfs:
            df = df.iloc[keep_index, :]
            remove_na_dfs.append(df)
        dictionary_of_n_splits_index = train_test_split_based_on_group(remove_na_dfs[0], group_column=genotype_df,
                                                                       test_size=0.15, random_seeds=self.seed,
                                                                       n_split=self.n_split)
        return remove_na_dfs, genotype_df, dictionary_of_n_splits_index, yield_tensor, keep_index_env_dfs_list

    def read_yield_convert_to_tensor(self, genotype,
                                     yield_file="C:/data_from_paper/ETH/olivia_WUR/process_data/yield.csv",
                                     year=2019, average=True):
        '''
        filter the yield for selected genotype, convert to tensor
        :param genotype: dataframe of genotype with the same order as the input X, the index of which need to be keep
        :param yield_file:
        :return:
        '''
        print('genotype for yield_____________+++++++++++++')
        print(genotype)
        print('unique genotypes number in genotype df{}'.format(len(genotype['genotype.id'].unique())))
        yield_df = LoadData.data_load(yield_file, year=year)
        yield_df.rename(columns={'value': 'yield'}, inplace=True)
        yield_df = yield_df[['genotype.id', 'yield', 'lot', 'year_site.harvest_year']]
        print('unique genotypes number in yield df{}'.format(len(yield_df['genotype.id'].unique())))
        select_year_yield = pd.merge(genotype, yield_df, how='left',
                                     on=['genotype.id', 'lot', 'year_site.harvest_year'])
        print(select_year_yield)
        if average:
            # calculate average yield between 2 lots
            num_index_before_average = len(genotype.index)
            # .mean() will skip na, which means calculate mean between a number with na, it will be the number
            average_df = select_year_yield[['genotype.id', 'yield']].groupby(['genotype.id'])['yield'].mean().to_frame()
            average_df.reset_index(inplace=True)
            # print(average_df)
            # print(select_year_yield)
            select_year_yield = genotype[['genotype.id']].merge(average_df, how='left', on='genotype.id')  # ,'lot'
            num_index_after_average = len(select_year_yield.index)
            print('unique genotypes number{}'.format(len(select_year_yield['genotype.id'].unique())))
            print(select_year_yield['genotype.id'].unique())
            print(select_year_yield)
            # select_year_yield = select_year_yield.drop_duplicates()#drop duplicate row with same genotype
            assert num_index_before_average == num_index_after_average, 'the number of index should not change after average'
        # return the list of index with yield is nan, and drop them
        na_index = list(select_year_yield.loc[pd.isna(select_year_yield["yield"]), :].index)
        # print(na_index)
        # self.yield_tensor_before_remove_na = copy.deepcopy(select_year_yield)
        # self.yield_tensor_before_remove_na  = self.yield_tensor_before_remove_na [["yield"]].to_numpy()
        # self.yield_tensor_before_remove_na  = torch.from_numpy(self.yield_tensor_before_remove_na ).to(torch.float32)

        select_year_yield.dropna(inplace=True)
        select_year_yield = select_year_yield[["yield"]].to_numpy()
        tensor_yield = torch.from_numpy(select_year_yield).to(torch.float32)
        print('select year(s) yield shape')
        print(tensor_yield.shape)

        return tensor_yield, na_index

    def train_test_validation_tensor(self):

        use = self.use
        dfs, genotype_df, dictionary_of_n_splits_index, yield_tensor, env_dfs = self.create_train_test_validation_index(
            year=self.year)

        n = 1
        data_dictionary = {}
        for key in dictionary_of_n_splits_index.keys():
            # loop to create dataset based on split with different random seeds
            train, test, validation = dictionary_of_n_splits_index[key]
            # pack train tets and validation set in a list
            inputX_list = []
            yield_tensor_list = []
            genotype_tensor_list = []
            position_tensor_list = []
            name = ['train', 'test', 'validation']
            for name_index, data_index in enumerate([train, test, validation]):
                split_dfs = []
                for df in dfs:
                    split_df = df.iloc[data_index]
                    # print('split_df___________________________')
                    # print(split_df)
                    split_dfs.append(split_df)
                else:
                    tensor_dataset = convert_inputx_to_tesor_list(split_dfs)  # create inptX

                genotype = genotype_df[['genotype.id', 'lot']].iloc[
                    data_index]  # lot is for find lot specific yield loop from train, test and split index
                if use == 'yield_predict':
                    yield_tensor_split = yield_tensor[data_index, :]
                    inputX_list.append(
                        tensor_dataset)  # pack inputX tensor after remove seq without corresponding yield
                    print('inputX shape, yield tensor shape')
                    print(tensor_dataset.shape, yield_tensor_split.shape)
                    yield_tensor_list.append(yield_tensor_split)  # pack yield tensor

                    # plot_X(genotype, tensor_dataset, n, name[name_index])
                    # convert position as tensor
                    location_df = genotype_df[['plot.row_global', 'plot.range_global']]
                    location_numpy = location_df.astype(int).to_numpy()
                    position_tensor = torch.from_numpy(location_numpy).to(torch.int64)
                    position_tensor = position_tensor[data_index, :]

                    # print(position_tensor)
                    print(position_tensor.shape)
                    position_tensor_list.append(position_tensor)  # pack position tensor

                    # drop lot from genotype
                    genotype = genotype[['genotype.id']]
                    genotype_tensor = self.one_hot_encoding_genotype(genotype)
                    print(genotype_tensor.shape)
                    genotype_tensor_list.append(genotype_tensor)  # pack genotype tensor

                    n_seq, seq_len, n_features = tensor_dataset.shape
                    # check shape
                    assert genotype_tensor.shape[0] == n_seq
                elif use == 'genotype_predict':
                    '''x is the time seriers for height and canopy coverage, Y is one hot encoded genotype'''
                    inputX_list.append(tensor_dataset)  # pack inputX tensor
                    genotype = genotype[['genotype.id']]
                    Y_tensor = self.one_hot_encoding_genotype(genotype)
                    genotype_tensor_list.append(Y_tensor)  # pack Y tensor
                    print(Y_tensor.shape)
                    label_number = Y_tensor.shape[1]
                    n_seq, seq_len, n_features = tensor_dataset.shape
                    assert label_number == n_seq
            else:
                n = n + 1
            if use == 'yield_predict':
                data_dictionary[key] = (inputX_list, yield_tensor_list, position_tensor_list, genotype_tensor_list)
                # yield inputX_list, yield_tensor_list, position_tensor_list, genotype_tensor_list
            elif use == 'genotype_predict':
                data_dictionary[key] = (inputX_list, genotype_tensor_list)
                # yield inputX_list, genotype_tensor_list

        self.dictionary_of_n_splits_index_remove_na = dictionary_of_n_splits_index
        self.data_dictionary = data_dictionary
        return data_dictionary

    def train_test_validation_tensor_iterator(self):
        self.train_test_validation_tensor()
        for key in self.data_dictionary.keys():
            if self.use == 'yield_predict':
                print('yield prediction')
                inputX_list = self.data_dictionary[key][0]
                yield_tensor_list = self.data_dictionary[key][1]
                position_tensor_list = self.data_dictionary[key][2]
                genotype_tensor_list = self.data_dictionary[key][3]
                yield inputX_list, yield_tensor_list, position_tensor_list, genotype_tensor_list
            elif self.use == 'genotype_predict':
                inputX_list = self.data_dictionary[key][0]
                genotype_tensor_list = self.data_dictionary[key][1]
                yield inputX_list, genotype_tensor_list

def convert_one_hot_endoding_to_label(y: torch.tensor):
    label_list = []
    for y_label in y:
        for index, label in enumerate(y_label.numpy()):

            if label == 1:
                break
        label_list.append(index)
    # print(label_list)
    return label_list

def standard_scalar(input_tensor: torch.tensor) -> torch.tensor:
    """standardize scalar x=(x-mean)/std"""
    import torch
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaled_data = torch.tensor(scaler.fit_transform(input_tensor))
    assert torch.allclose(input_tensor, torch.tensor(scaler.inverse_transform(scaled_data)))
    return scaled_data, scaler

    input_tensor = input_tensor.float()
    std = torch.std(input_tensor)
    mean = input_tensor.mean()
    scaled_tensor = (input_tensor - mean) / std
    return scaled_tensor, std, mean


def minmax_scaler(input_tensor: torch.tensor, min=0, max=1) -> tuple[torch.tensor, list]:
    """
    Apply minmax scaler to torch.tensor which shape[seq_length,n_seq,n_feature]
    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X_scaled = X_std * (max - min) + min

    input_tensor: tensor which need to be scaled, shape = (seq_length,n_samples, n_features)
    feature_position: specify which dimension is the feature dimension, default -1
    """

    from sklearn.preprocessing import MinMaxScaler
    # print('input shape:{}'.format(input_tensor.shape))
    np_input = input_tensor.numpy()
    np_input[np_input == np.inf] = np.nan
    np_input[np_input == -np.inf] = np.nan

    data_min = np.nanmin(np_input)

    # print('data minimum:{}'.format(data_min))
    data_max = np.nanmax(np_input)
    # print('data max:{}'.format(data_max))
    input_std = (np_input - data_min) / (data_max - data_min)
    scaled_data = torch.tensor(input_std * (max - min) + min)

    # check if this function works correctly
    inverse_scale = torch.tensor(
        reverse_min_max_scaling(data=scaled_data.numpy(), data_min=data_min, data_max=data_max, min=min, max=max,
                                set_nan_tozero=False)).type(torch.float64)

    assert torch.allclose(input_tensor.type(torch.float64), inverse_scale, equal_nan=True,
                          rtol=0.001)  # the precision of float make affect the inverse result not excatlly equal

    scaler_list = [np.nanmin(np_input), np.nanmax(np_input)]

    print(scaled_data.shape)
    return scaled_data, scaler_list


def reverse_min_max_scaling(data: torch.tensor, data_min, data_max, min: float = 0.0, max: float = 1.0,
                            set_nan_tozero=True):
    try:
        data = data.detach()
        data = data.numpy()
        data = data.astype(float)
    except:
        print('numpy input')
        data = data.astype(float)
    data_unscale_middle_step = (data - min) / (max - min)
    unscaled_data = torch.tensor(data_unscale_middle_step * (data_max - data_min) + data_min)
    if set_nan_tozero:
        unscaled_data = torch.nan_to_num(unscaled_data, posinf=0.0, neginf=0.0)
    return unscaled_data





