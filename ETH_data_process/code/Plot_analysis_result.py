"""
This script is to visualize results
"""
import copy
from pathlib import Path
from soupsieve.util import lower
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import platform
import glob
n_split_year_map = pd.DataFrame()
def check_code_running_system():
    system = platform.system()
    print(system)
    return lower(system)
def plot_saved_result(file='rf_mean_input_validation_result.csv'):
    result_csv = pd.read_csv(file,header=0,index_col=0)
    result = result_csv[['validation_MSE','validation_MAPE','pearson_coef','window_size','spearman_coef']]
    plt.rc('xtick', labelsize=8)
    fig,axs = plt.subplots(2,2)
    sns.boxplot(data=result[['validation_MSE','window_size']],y= 'validation_MSE', x='window_size',palette="Blues",hue='window_size',legend=False,ax=axs[0,0])
    sns.boxplot(data=result[['validation_MAPE','window_size']],y='validation_MAPE', x='window_size',palette="Blues",hue='window_size',legend=False,ax=axs[0,1])
    sns.boxplot(data=result[['pearson_coef','window_size']],y='pearson_coef', x='window_size',palette="Blues",hue='window_size',legend=False,ax=axs[1,0])
    sns.boxplot(data=result[['spearman_coef','window_size']],y='spearman_coef', x='window_size',palette="Blues",hue='window_size',legend=False,ax=axs[1,1])

    plt.tight_layout()
    # plt.savefig('{}.svg'.format(file.split('.')[0]),dpi=2400)
    # plt.show()
def plot_train_result(file='../samples_visualization/rf_plots/rf_mean_input_test_result_multiple_year_test.csv'):
    result_csv = pd.read_csv(file,header=0,index_col=0)
    result = result_csv[['mean_test_score','param_max_depth','param_max_features','param_n_estimators']].reset_index().fillna('NA')
    result.drop_duplicates(inplace=True)
    print(result[['mean_test_score','param_max_depth']])
    plt.rc('xtick', labelsize=8)
    fig,axs = plt.subplots(2,2)
    sns.boxplot(data=result[['mean_test_score','param_max_depth']],y= 'mean_test_score', x='param_max_depth',palette="Blues",hue='param_max_depth',legend=False,ax=axs[0,0])
    sns.boxplot(data=result[['mean_test_score','param_max_features']],y='mean_test_score', x='param_max_features',palette="Blues",hue='param_max_features',legend=False,ax=axs[0,1])
    sns.boxplot(data=result[['mean_test_score','param_n_estimators']],y='mean_test_score', x='param_n_estimators',palette="Blues",hue='param_n_estimators',legend=False,ax=axs[1,0])

    plt.tight_layout()
    # plt.savefig('{}.svg'.format(file.split('.')[0]),dpi=2400)
    # plt.show()

def plot_NN_result(files=['best_test_one_layer_spearmanr_align.csv','best_test_2_spearmanr_align.csv','best_test_1_spearmanr_align.csv']):
    result_list=[]
    for file in files:
        result_csv = pd.read_csv(file,header=0,index_col=0)
        print(result_csv)
        result_csv['model']=''.join(file.split('_')[2:4])

        result = result_csv[['validation_MSE','spearmanr_validation','model']]
        result_list.append(result)
    result = pd.concat(result_list)
    plt.rc('xtick', labelsize=8)
    fig,axs = plt.subplots(2)
    sns.boxplot(data=result[['validation_MSE','model']],y= 'validation_MSE', x='model',palette="Blues",hue='model',legend=False,ax=axs[0])
    sns.boxplot(data=result[['spearmanr_validation','model']],y='spearmanr_validation', x='model',palette="Blues",hue='model',legend=False,ax=axs[1])

    plt.tight_layout()
    plt.savefig('{}.svg'.format(file.split('.')[0]),dpi=2400)
    # plt.show()

def plot_PINN_result(file='pinn_result/PINN_mask_loss_best_ remove_one_outlier.csv'):
    """
    This is the function take the result.csv(before averge) to calculate average result and order based on validation rmse
    """

    result_csv = pd.read_csv(file,header=0,index_col=0,sep='[;,]')
    print(result_csv)
    len_row_1= len(result_csv.index)
    result_csv.drop_duplicates(inplace=True)
    len_row_2 = len(result_csv.index)
    if len_row_1 !=len_row_2:
        result_csv.to_csv(file)
    # result_csv = result_csv[result_csv.l2 ==1.0]
    # print(result_csv)
    result_csv['model']=''.join(file.split('_')[2:4])
    # try:
    result = result_csv[['validation_rMSE','train_rMSE','test_rMSE','train_shapeDTW','validation_shapeDTW','test_shapeDTW',
                         'weight_physic','predicted_r','predicted_y_max','Trainable_Params','l2','ode_int','lr',"hidden_size",
                         "num_layer","n_split"]]

    result = result[result['ode_int']==False]
    result.reset_index(inplace=True,drop=True)
    result['x_axis']=0

    #create a new column as the x axis
    for ind in result.index:
        # print(result.loc[ind,'weight_physic'])
        x_axis = 'W_{}_TrP_{}_l2_{}_lr_{}'.format(result.loc[ind,'weight_physic'],result.loc[ind,'Trainable_Params'],
                                                         result.loc[ind,'l2'],result.loc[ind,'lr'])
        result.loc[ind,'x_axis'] = x_axis
    print(result)
    # sns.boxplot(data=result[['test_rMSE','n_split']],y= 'test_rMSE', x='n_split',palette="Blues")
    # plt.show()
    calculate_std_for_different_seed_result(result, file)


def plot_mean_error_bar_multiple_genotype(file_name='pinn',name='PINN_mask_loss_pinn_lstm_same_length',
                                          result_file_dir='pinn_result/result_summary/single_genotype/pure_ml/smooth_temp/'):
    """
    This function is to plot different genotype result together, for single genotype model
    name: str, need to be unique to use for identify a run for 19 genotyps
    """
    import re
    system_name = check_code_running_system()
    order_g,genotype_similarity = order_genotype_based_on_their_similarity(first_genotype='106')

    #find mean result files
    if file_name=='pinn_penalize_r':
        print("pinn_result/result_summary/mean_result_from5_seed/single_g/{}*115*.csv".format(name))
        files = glob.glob("pinn_result/result_summary/mean_result_from5_seed/single_g/*{}*115*.csv".format(name))
        print(len(files))
        assert len(files)==19
        print(files)
        if 'windows' in system_name:
            files = [str(x).split('\\')[-1] for x in files]
        else:
            files = [str(x).split('/')[-1] for x in files]
        pattern = re.compile(r'genotype(\d*)penalize_rpinnmode_True')
    elif file_name == 'pinn':
        files = glob.glob("pinn_result/result_summary/mean_result_from5_seed/single_g/{}*genotype*pinnmode_True*.csv".format(name))
        print(files)
        if 'windows' in system_name:
            files = [str(x).split('\\')[-1] for x in files]
        else:
            files = [str(x).split('/')[-1] for x in files]
        pattern = re.compile(r'genotype(\d*)pinnmode_True')
    elif file_name == 'ml':
        files = glob.glob("pinn_result/result_summary/mean_result_from5_seed/single_g/{}*genotype*pinnmode_False*rescaleFalse*.csv".format(name))
        print('search for ML files: {}'.format("pinn_result/result_summary/mean_result_from5_seed/single_g/{}*genotype*pinnmode_False*.csv".format(name)))
        print(len(files))
        print(files)
        assert len(files)==19
        print(files)
        if 'windows' in system_name:
            files = [str(x).split('\\')[-1] for x in files]
        else:
            files = [str(x).split('/')[-1] for x in files]
        pattern = re.compile(r'genotype(\d*)pinnmode_False')

    df_multi_g = pd.DataFrame()
    df_ode_parameters=pd.DataFrame()
    df_model_hyperparameters = pd.DataFrame()
    # Loop through each file
    for file in files:
        # Search for the genotype value using the regex pattern
        match = pattern.search(file)
        if match:
            genotype = match.group(1)  # Extract the genotype part
            print(f"File: {file}")
            print(f"Genotype: {genotype}")
            df_param = pd.read_csv("pinn_result/result_summary/mean_result_from5_seed/single_g/{}".format(file), header=0, index_col=0).iloc[:1,
            :][['predicted_r','predicted_y_max']]
            df_param.index=['{}'.format(genotype)]
            df_ode_parameters = pd.concat([df_ode_parameters,df_param])
            #
            df_hyper = pd.read_csv("pinn_result/result_summary/mean_result_from5_seed/single_g/{}".format(file), header=0, index_col=0).iloc[:1,
            :][['weight_physic','Trainable_Params','l2','ode_int','lr','hidden_size','num_layer']] #'genetics_embedding_size','last_fc_hidden_size'
            df_hyper.index=['{}'.format(genotype)] #genotype as index
            print(df_hyper)
            #read full result before average
            if file_name == 'pinn_penalize_r':
                print('search for files:{}{}*genotype{}penalize_r*115.csv'.format(result_file_dir,name,genotype))
                original_file_names = glob.glob('{}{}*genotype{}penalize_r*115.csv'.format(result_file_dir,name,genotype))
                assert len(original_file_names)==1
                if 'windows' in system_name:
                    original_file_names = [str(x).split('\\')[-1] for x in original_file_names]
                else:
                    original_file_names = [str(x).split('/')[-1] for x in original_file_names]
                print('original_file:{}'.format(original_file_names))
                orginal_df = pd.read_csv("{}{}".format(result_file_dir,original_file_names[0]),
                                         header=0, index_col=0)
            elif file_name == 'pinn':
                original_file_names = glob.glob(
                    '{}{}*genotype{}pinnmode_True*.csv'.format(result_file_dir,name,
                        genotype))
                if 'windows' in system_name:
                    original_file_names = [str(x).split('\\')[-1] for x in original_file_names]
                else:
                    original_file_names = [str(x).split('/')[-1] for x in original_file_names]
                # print('original_file:{}'.format(original_file_names))
                orginal_df = pd.read_csv("{}{}".format(result_file_dir,original_file_names[0]),
                                         header=0, index_col=0)
            elif file_name == 'ml':
                print('seach: {}{}*genotype{}*.csv'.format(result_file_dir,name,
                        genotype))
                original_file_names = glob.glob(
                    '{}{}*genotype{}*rescaleFalse*.csv'.format(result_file_dir,name,
                        genotype))
                if 'windows' in system_name:
                    original_file_names = [str(x).split('\\')[-1] for x in original_file_names]
                else:
                    original_file_names = [str(x).split('/')[-1] for x in original_file_names]
                print('original_file:{}'.format(original_file_names))
                orginal_df = pd.read_csv("{}{}".format(result_file_dir,original_file_names[0]),
                                         header=0, index_col=0,sep='[;,]')
                # print(orginal_df)
            orginal_df.drop_duplicates(inplace=True)

            best_result_df =  orginal_df.loc[
                                (orginal_df['weight_physic'] == df_hyper.loc[genotype,'weight_physic']) &
                                (orginal_df['Trainable_Params'] == df_hyper.loc[genotype,'Trainable_Params']) &
                                (orginal_df['l2'] == df_hyper.loc[genotype,'l2']) &
                                (orginal_df['ode_int'] == df_hyper.loc[genotype,'ode_int']) &
                                (orginal_df['lr'] == df_hyper.loc[genotype,'lr']) &
                                (orginal_df['hidden_size'] == df_hyper.loc[genotype,'hidden_size']) &
                                # (orginal_df['genetics_embedding_size'] == df_hyper.loc[genotype, 'genetics_embedding_size']) &
                                # (orginal_df['last_fc_hidden_size'] == df_hyper.loc[genotype, 'last_fc_hidden_size']) &
                                (orginal_df['num_layer'] == df_hyper.loc[genotype,'num_layer'])
                                ]
            best_result_df['genotype']=genotype
            df_model_hyperparameters = pd.concat([df_model_hyperparameters, best_result_df])
            # print('df_model_hyperparameters')
            # print(df_model_hyperparameters)

            #read from the first line, which is the lowest validation rmse
            df = \
            pd.read_csv("pinn_result/result_summary/mean_result_from5_seed/single_g/{}".format(file), header=0, index_col=0).iloc[:1,
            :][['validation_rMSE', 'train_rMSE', 'test_rMSE', 'train_rMSE_std', 'test_rMSE_std',
                'validation_rMSE_std','train_shapeDTW','validation_shapeDTW','test_shapeDTW','train_shapeDTW_std',
                'validation_shapeDTW_std','test_shapeDTW_std']]
            df = df.rename(columns={"validation_rMSE_std":"validation_std","test_rMSE_std":"test_std","train_rMSE_std":"train_std"})
            genotype_df = df.T
            genotype_df.columns=['{}'.format(str(genotype))]
            # print(genotype_df)
            df_multi_g = pd.concat([df_multi_g,genotype_df],axis=1)
            # print(df_multi_g)
    else:
        # print(df_ode_parameters)
        # print(df_model_hyperparameters)
        df_model_hyperparameters.to_csv('best_model_result_summary/{}_best_hyperparameters_result.csv'.format(name))
        # df_ode_parameters.to_csv('pinn_parameters.csv')
        # print(df_ode_parameters.index)
        # print(df_ode_parameters.index.unique())
        df_ode_parameters = df_ode_parameters.reindex(order_g)
        df_ode_parameters['genotype']= df_ode_parameters.index
        # print(df_ode_parameters)
        # print(df_ode_parameters['genotype'].unique())
        # raise EOFError

        g_similarity= torch.tanh(torch.tensor(genotype_similarity))
        # print(g_similarity)
        # g_similarity[0] = 0.2
        simlarity_df = pd.DataFrame({'genotype': order_g, 'similarity_compard_with_106': g_similarity},
                                    index=range(len(order_g)))
        simlarity_df['genotype'] = simlarity_df['genotype'].astype(int)
        df_ode_parameters['genotype'] = df_ode_parameters['genotype'].astype(int)
        df_ode_parameters = pd.merge(df_ode_parameters, simlarity_df, 'left')
        # print(df_ode_parameters)

        plt.figure(figsize=(10, 6))
        # palette = sns.color_palette("viridis", len(df_ode_parameters['similarity_compard_with_106'].values))
        scatter = plt.scatter(data=df_ode_parameters, x='predicted_r', y='predicted_y_max',
                              c=df_ode_parameters['similarity_compard_with_106'].values, cmap='viridis', s=50)

        colormap = plt.cm.viridis
        norm = plt.Normalize(df_ode_parameters['similarity_compard_with_106'].min(),
                             df_ode_parameters['similarity_compard_with_106'].max())

        # Add genotype IDs as annotations
        for i in range(len(df_ode_parameters)):
            plt.text(
                df_ode_parameters['predicted_r'][i],
                df_ode_parameters['predicted_y_max'][i],
                df_ode_parameters['genotype'][i],
                horizontalalignment='left',
                size='medium',
                color=colormap(norm(df_ode_parameters['similarity_compard_with_106'][i])),  # Match node color
                weight='semibold'
            )

        plt.colorbar(scatter, label='Similarity')

        # Customize the plot
        plt.title(' r vs. ymax {}'.format(file_name))
        plt.xlabel('predicted_r')
        plt.ylabel('predicted_y_max')
        plt.grid()
        # plt.show()

        df= df_multi_g
        # df.to_csv('multipl_g_result_summary.csv')
        print("average train rmse:{}".format(round(df.T['train_rMSE'].mean(),3)))
        print("average validation rmse:{}".format(round(df.T['validation_rMSE'].mean(),3)))
        print("average test rmse:{}".format(round(df.T['test_rMSE'].mean(),3)))
        print("average train std:{}".format(round(df.T['train_std'].mean(),3)))
        print("average validation std:{}".format(round(df.T['validation_std'].mean(),3)))
        print("average test std:{}".format(round(df.T['test_std'].mean(),3)))

        print("average train dtw:{}".format(round(df.T['train_shapeDTW'].mean(),3)))
        print("average validation dtw:{}".format(round(df.T['validation_shapeDTW'].mean(),3)))
        print("average test dtw:{}".format(round(df.T['test_shapeDTW'].mean(),3)))
        print("average train dtw std:{}".format(round(df.T['train_shapeDTW_std'].mean(),3)))
        print("average validation dtw std:{}".format(round(df.T['validation_shapeDTW_std'].mean(),3)))
        print("average test dtw std:{}".format(round(df.T['test_shapeDTW_std'].mean(),3)))
        # Reshape the DataFrame into long form for easier plotting
        df_long = pd.melt(
                            df_model_hyperparameters ,
                            id_vars=["genotype","n_split","random_sees"],
                            value_vars=["train_rMSE", "validation_rMSE", "test_rMSE"],
                            var_name="metric",
                            value_name="value"
                            )
        # df_long = df.T.reset_index().melt(id_vars='index', var_name='metric', value_name='value')
        # print(df_long)
        # # Split the 'metric' column into 'set_type' (train/val/test) and 'metric_type' (rmse/std)
        # df_long['set_type'] = df_long['metric'].apply(lambda x: x.split('_')[0])  # train, val, test
        # df_long['metric_type'] = df_long['metric'].apply(lambda x: x.split('_')[1])  # rmse, std
        #
        # # Pivot to get RMSE and STD separately
        # df_rmse = df_long[df_long['metric_type'] == 'rMSE']
        # df_std = df_long[df_long['metric_type'] == 'std']

        # Merge RMSE and std dataframes based on Genotype.id (index) and set_type
        # df_final = pd.merge(df_rmse, df_std, on=['index', 'set_type'], suffixes=('_rMSE', '_std'))
        # print(df_final['value_rMSE'])
        # Plot the data using seaborn
        plt.figure(figsize=(10, 6))
        plt.ylim(0,0.3)
        sns.boxplot(x='genotype', y='value', hue='metric', data=df_long, dodge=True,order=order_g)
        plt.legend(title='Set Type',loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()


        #plot
        if file_name == 'pinn_penalize_r':
            plt.title('RMSE and std by Genotype for single genotype model PINN (penalize_r)')
        elif file_name == 'pinn':
            plt.title('RMSE and std by Genotype for single genotype model PINN')
        elif file_name == 'ml':
            plt.title('RMSE and std by Genotype for single genotype model (pure ML)')
        plt.xlabel('Genotype')
        plt.ylabel('RMSE')
        # plt.show()
        # plt.clf()

        # print(df_long)
        # print(n_split_year_map)
        df_long=df_long.merge(n_split_year_map,on='n_split',how='left')
        plt.figure(figsize=(10, 6))
        plt.ylim(0,0.3)
        sns.boxplot(x='train_year', y='value', hue='metric', data=df_long, dodge=True)
        plt.legend(title='Set Type',loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        # plt.show()
        # plt.clf()
        return df_long,order_g

def plot_loss(model,x,y,position,genotype,with_position=''):
    learning_rate = 0.001  #
    num_epochs = 500

    # Convert input and target data to PyTorch datasets
    print(x.shape)
    print(y.shape)
    print(position.shape)
    print(genotype.shape)
    dataset = TensorDataset(x, y, position, genotype)

    # Create data loader for iterating over dataset in batches during training
    dataloader = DataLoader(dataset, batch_size=30, shuffle=False)

    # initialize model
    model.init_network()

    num_sequences = x.shape[0]
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    loss_list = []

    x_axis = []
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for epoch in range(num_epochs):
        running_loss = 0.0  # running loss for every epoch
        running_loss_test = 0.0

        for i, (inputs, targets, position_batch, genotype_batch) in enumerate(dataloader):
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass

            # print(inputs.shape)
            yield_pre = model(inputs.float(),position_batch, genotype_batch)  # batch first, the input is (110,46,2) for 2019

            # print("target:{}".format(predict_genotype.shape))
            # print("output_fc{}".format(output_fc.shape))
            targets = targets.float()
            # print(targets)
            # print(yield_pre.shape)
            loss = model.spearman_rank_loss(true_label=targets, predict_Y=yield_pre)
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.shape[1]  # loss.item() is the mean loss of the total batch

        if (epoch + 1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs,
                                                       running_loss / (num_sequences)))
            x_axis.append(epoch + 1)
            # plot loss

            loss_list.append(running_loss/(num_sequences))
            #     # curve_loss_list.append(running_loss_curve/(num_sequences)-previous_curve_loss)
            #     # gene_loss_list.append(running_loss_gene/(num_sequences)-previous_genoype_loss)
            #     #
            #     #
            line1, = ax.plot(x_axis, loss_list, c='blue')
            #     # line2, = ax.plot(x_axis, curve_loss_list, c='red')
            #     # line3, = ax.plot(x_axis, gene_loss_list, c='orange')
            #     ax.set_ylabel("MSE loss")
            #     # line1.set_ydata(loss_list)
            #     # line1.set_xdata(x_axis)
            #     # line2.set_ydata(curve_loss_list)
            #     # line2.set_xdata(x_axis)
            #     # line3.set_ydata(gene_loss_list)
            #     # line3.set_xdata(x_axis)
            #     # plt.axhline(y=0, color='g', linestyle='-')
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.show(block=False)

def merge_result_csv_file():
    '''This function is to concat seperate dataframe which used for save result for different weights'''
    files = glob.glob("pinn_result/result_summary/single_genotype/PINN_mask_loss_na_as_0_same_lengthgenotype335penalize_rpinnmode_Truefit_ml_first_FalsecorrectedFalsesmoothFalsefill_in_naTruerescaleFalsestart_date_91w*_.csv") #read all files for seperate weights
    print(files)
    try:
        files = [str(x).split('\\')[1] for x in files]
    except:
        files = [str(x).split('/')[-1] for x in files]
    # print(files)
    print(len(files))
    files_name = list(set([x.split('91w')[0] for x in files]))
    print(len(files))
    for file in files_name:
        # print(file)
        sub_files = glob.glob('pinn_result/result_summary/single_genotype/{}*_.csv'.format(file))
        # print(sub_files)
        try:
            #this works when run from pycharm
            sub_files = [str(x).split('\\')[1] for x in sub_files]
        except:
            #this works when run through cmd from linux
            sub_files = [str(x).split('/')[-1] for x in sub_files]
        df_merge = pd.DataFrame()
        for sub_file in sub_files:
            print(sub_file)
            df = pd.read_csv('pinn_result/result_summary/single_genotype/{}'.format(sub_file),header=0,index_col=0)
            df_merge = pd.concat([df_merge,df],axis=0)
            df_merge.drop_duplicates(inplace=True)
            print(df_merge)
        else:
            df_merge.to_csv('pinn_result/result_summary/{}91merge.csv'.format(file))
            # print('rm pinn_result/result_summary/{}*_.csv'.format(file))
            # os.system('rm pinn_result/result_summary/{}*_.csv'.format(file)) # rm seperate files after merge
def calculate_std_for_different_seed_result(result,file):

    result_group_std = result.groupby(by=['weight_physic','Trainable_Params', 'l2', 'ode_int', 'lr',"hidden_size","num_layer",'x_axis',]).std().reset_index()
    result_group_std.rename(columns={'validation_rMSE':'validation_rMSE_std', 'train_rMSE':'train_rMSE_std', 'test_rMSE':'test_rMSE_std',
                                     'train_shapeDTW':'train_shapeDTW_std','validation_shapeDTW':'validation_shapeDTW_std','test_shapeDTW':'test_shapeDTW_std'},inplace=True)
    result_group_mean = result.groupby(by=['weight_physic', 'Trainable_Params', 'l2', 'ode_int', 'lr',"hidden_size","num_layer",'x_axis',]).mean().reset_index()
    print(result_group_std['validation_rMSE_std'])
    print(result_group_std)#calculate std for each hyperparameters combination for differnt random seed
    result_group = pd.concat([result_group_mean,result_group_std[['validation_rMSE_std', 'train_rMSE_std', 'test_rMSE_std','train_shapeDTW_std','validation_shapeDTW_std','test_shapeDTW_std']]],axis=1)
    result_group = result_group.sort_values(by=['validation_rMSE'])  #
    print(result_group)
    file_name = file.split('.csv')[0].split("/")[-1]
    print(file_name)
    result_group.to_csv('pinn_result/result_summary/mean_result_from5_seed/single_g/{}_mean_reult.csv'.format(file_name))
    return result_group

def calculate_std_for_different_seed_result_multiple_g(file):
    file_name= str(file)
    # print(file_name)
    result = pd.read_csv('{}'.format(file_name),header=0,index_col=0)
    result = result.drop_duplicates()
    # result.to_csv('{}'.format(file_name))
    print(result.columns)
    #for gpu, physics_weight is na, will replace with 0
    result.loc[:, result.isna().all()] = 0

    file = str(file.name).split('.csv')[0]
    print(file)
    result_group_std = result.groupby(by=['learning_rate','hidden_size','l2','num_layer','physics_weight','y_max_bound','smooth_loss','genetics_embedding_size','last_fc_hidden_size']).std().reset_index().round(3)
    result_group_std.rename(columns={'validation_rMSE':'validation_rMSE_std', 'train_rMSE':'train_rMSE_std', 'test_rMSE':'test_rMSE_std',
                                     'train_shapeDTW':'train_shapeDTW_std','validation_shapeDTW':'validation_shapeDTW_std','test_shapeDTW':'test_shapeDTW_std'},inplace=True)
    result_group_mean = result.groupby(by=['learning_rate','hidden_size','l2','num_layer','physics_weight','y_max_bound','smooth_loss','genetics_embedding_size','last_fc_hidden_size']).mean().reset_index().round(3)
    # print(result_group_std['validation_rMSE_std']) #calculate std for each hyperparameters combination for differnt random seed
    result_group = pd.concat([result_group_mean,result_group_std[['validation_rMSE_std', 'train_rMSE_std', 'test_rMSE_std','train_shapeDTW_std','validation_shapeDTW_std','test_shapeDTW_std']]],axis=1)
    result_group = result_group.sort_values(by=['validation_rMSE'])  #
    print(result_group)
    file_name = file.split('.csv')[0].split("/")[-1]
    # print(file_name)
    result_group.to_csv('pinn_result/result_summary/mean_result_from5_seed/multiple_g/{}_mean_reult.csv'.format(file_name))

    df_hyper = result_group.iloc[:1,
               :][['learning_rate','hidden_size','l2','num_layer','physics_weight','y_max_bound','smooth_loss','genetics_embedding_size','last_fc_hidden_size']]
    df_best_hyperparameters_result = pd.merge(result,df_hyper,how='inner')
    print(df_best_hyperparameters_result)

    df_best_hyperparameters_result.to_csv("best_model_result_summary/{}_best_hyperparameters_result.csv".format(file_name))
    return result_group

def smooth_result(model, data_df):
    """
    Only take the predicition from days with env input, and apply spline on it?
    """
def tensor_to_float(tensor_str):
    import re
    if isinstance(tensor_str, str) and 'tensor(' in tensor_str:
        # Use regex to extract the numeric value before any ', grad_fn=' part
        match = re.search(r'tensor\(([\d\.\-e]+)', tensor_str)
        if match:
            # Convert the extracted value to a float
            return float(match.group(1))
    return tensor_str  # Return None if not a valid tensor string

def anova_test(test_value='test_rMSE'):
    import scipy.stats as stats
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from statsmodels.formula.api import ols
    import statsmodels.api as sm
    ml_df = pd.read_csv('best_model_result_summary/ml_best_hyperparameters_result.csv',header=0,index_col=0)
    pinn_df = pd.read_csv('best_model_result_summary/pinn_best_hyperparameters_result.csv',header=0,index_col=0)
    pinn_penalize_r_df = pd.read_csv('best_model_result_summary/pinn_penalize_r_best_hyperparameters_result.csv',header=0,index_col=0)
    logistic_ode_df = pd.read_csv('best_model_result_summary/logistic_ode_fit_multiple_genotype.csv',header=0,index_col=0)
    rf_df = pd.read_csv('best_model_result_summary/rf_model_result_summary.csv',header=0,index_col=0)
    test_df_rf = rf_df[[test_value,'genotype']]
    test_df_rf['group']='group_rf'

    test_df_ode = logistic_ode_df[[test_value,'genotype']]
    test_df_ode['group']='group_logistic_ode'

    test_df_ml = ml_df[[test_value,'genotype']]
    test_df_ml['group']='group_ml'
    test_df_pinn = pinn_df[[test_value,'genotype']]
    test_df_pinn['group']='group_pinn'
    test_df_pinn_penalize_r = pinn_penalize_r_df[[test_value,'genotype']]
    test_df_pinn_penalize_r['group']='group_pinn_penalize_r'
    combined_group = pd.concat([test_df_ml,test_df_pinn_penalize_r,test_df_ode,test_df_rf], ignore_index=True)
    # Perform ANOVA
    grouped_data = [combined_group[combined_group['group'] == group][test_value] for group in combined_group['group'].unique()]
    print(grouped_data)
    # ANOVA
    f_stat, p_value = stats.f_oneway(*grouped_data)
    print(f'ANOVA F-statistic: {f_stat}, p-value: {p_value}')
    tukey = pairwise_tukeyhsd(endog=combined_group[test_value], groups=combined_group['group'], alpha=0.05)
    print(tukey)
    # raise EOFError
    #two-way ANOVA
    model = ols('{} ~ C(genotype) + C(group) + C(genotype):C(group)'.format(test_value), data=combined_group).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)
    # For genotype comparison
    tukey_genotype = pairwise_tukeyhsd(endog=combined_group[test_value], groups=combined_group['genotype'], alpha=0.05)
    print(tukey_genotype)
    # For group comparison
    tukey_group = pairwise_tukeyhsd(endog=combined_group[test_value], groups=combined_group['group'], alpha=0.05)
    print(tukey_group)

def plot_curve_witherror_bar_for_single_genotype_model_result(name='test',best_hyperparameters_df='pinn_penalize_r_best_hyperparameters_result.csv',if_pinn=True):
    """plot for test, did not svave preicted curev for traina and val, thus this function doesn't work"""
    from NNmodel_training import mask_rmse_loss
    #read save best df
    best_df = pd.read_csv(best_hyperparameters_df,header=0,index_col=0)
    groups_g_object = best_df.groupby(by='genotype')
    print(len(groups_g_object))

    for genotype in groups_g_object.groups:
        print('genotype:{}'.format(genotype))
        group_g = groups_g_object.get_group(genotype)
        print(group_g)
        hidden= group_g['hidden_size'].unique()[0]
        num_layer = group_g['num_layer'].unique()[0]
        lr = group_g['lr'].unique()[0]
        weight_physic = group_g['weight_physic'].unique()[0]
        ode_int_loss = group_g['ode_int'].unique()[0]
        L2 = group_g['l2'].unique()[0]
        print('hidden size')
        print(hidden)
        if if_pinn:
            #pinn
            col_name = '^predict_{}split_hidden{}_env{}_ts{}_lr_{}_w_ph{}_ode_int_{}_l2_{}_{}_rs_\d+_seq\d+$'.format(
                0, hidden, num_layer, num_layer, lr, weight_physic, ode_int_loss, L2,
                'cpu_logi_pinn_save_model__same_lengthgenotype{}penalize_rpinnmode_Truefit_ml_first_FalsecorrectedFalsesmoothFalsefill_in_naTruerescaleFalsestart_date_115'.format(genotype))

            f=glob.glob('pinn_result/{}_predict_curve_cpu_logi_pinn_save_model__same_lengthgenotype{}penalize_rpinnmode_Truefit_ml_first_FalsecorrectedFalsesmoothFalsefill_in_naTruerescaleFalsestart_date_115.csv'.format(
                    name,genotype))
            print(f)
            assert len(f)==1
            predict_curve_df = pd.read_csv(
                'pinn_result/{}_predict_curve_cpu_logi_pinn_save_model__same_lengthgenotype{}penalize_rpinnmode_Truefit_ml_first_FalsecorrectedFalsesmoothFalsefill_in_naTruerescaleFalsestart_date_115.csv'.format(
                    name,genotype), header=0, index_col=0)
            df_true = pd.read_csv(
                "pinn_result/{}_true_curves_cpu_logi_pinn_save_model__same_lengthgenotype{}penalize_rpinnmode_Truefit_ml_first_FalsecorrectedFalsesmoothFalsefill_in_naTruerescaleFalsestart_date_115.csv".format(
                    name, genotype), header=0, index_col=0)
            c='red'
        else:
            #ml
            col_name = '^predict_{}split_hidden{}_env{}_ts{}_lr_{}_w_ph{}_ode_int_{}_l2_{}_{}_rs_\d+_seq\d+$'.format(
                0, hidden, num_layer, num_layer, lr, weight_physic, ode_int_loss, L2,
                'gpu_lstm_NN_save_model_same_lengthgenotype{}pinnmode_Falsefit_ml_first_FalsecorrectedFalsesmoothFalsefill_in_naTruerescaleFalsestart_date_115w0_'.format(
                    genotype))
            f = glob.glob('pinn_result/{}_predict_curve_gpu_lstm_NN_save_model_same_lengthgenotype{}pinnmode_Falsefit_ml_first_FalsecorrectedFalsesmoothFalsefill_in_naTruerescaleFalsestart_date_115w0_.csv'.format(name,
                    genotype))
            # print(f)
            assert len(f)==1
            predict_curve_df = pd.read_csv("pinn_result/{}_predict_curve_gpu_lstm_NN_save_model_same_lengthgenotype{}pinnmode_Falsefit_ml_first_FalsecorrectedFalsesmoothFalsefill_in_naTruerescaleFalsestart_date_115w0_.csv".format(name,
                    genotype), header=0, index_col=0)
            # print(predict_curve_df)
            f = glob.glob("pinn_result/{}_true_curves_gpu_lstm_NN_save_model_same_lengthgenotype{}pinnmode_Falsefit_ml_first_FalsecorrectedFalsesmoothFalsefill_in_naTruerescaleFalsestart_date_115w0_.csv".format(
                    name, genotype))
            # print(predict_curve_df.columns)
            # print(f[0])
            assert len(f) == 1
            df_true = pd.read_csv("pinn_result/{}_true_curves_gpu_lstm_NN_save_model_same_lengthgenotype{}pinnmode_Falsefit_ml_first_FalsecorrectedFalsesmoothFalsefill_in_naTruerescaleFalsestart_date_115w0_.csv".format(
                    name, genotype)
                , header=0, index_col=0)
            print(col_name)
            c='orange'

        df_pred = predict_curve_df.filter(regex=col_name)
        #'print filter with col name'
        print('predicted data frame')
        print(df_pred)
        if df_pred.empty:
            print(col_name)
            raise ValueError("The DataFrame is empty. Please provide a valid input.")

        df_mean = df_pred.mean(axis=1)
        df_std = df_pred.std(axis=1)

        # Convert 0.0 to NaN in true_df
        df_true.replace(0.0, np.nan, inplace=True)
        # df_mean[np.isnan(df_true)] = np.nan
        # df_std[np.isnan(df_true)] = np.nan

        fig, ax = plt.subplots(figsize=(8, 6))
        plt.ylim(0,1.5)
        # Create a new figure for each genotype


        # Get all columns in true_df corresponding to the current genotype (including different plots)
        true_columns = df_true.columns.to_list()
        # print(true_columns)
        print(type(df_mean))
        df_mean=pd.DataFrame(data=df_mean,columns=true_columns)
        df_std = pd.DataFrame(data=df_std, columns=true_columns)
        # df_std.columns = true_columns

        print(df_mean)
        # print(df_true.columns)
        # raise EOFError
        rmse = 0.0
        for i in true_columns:
            sns.scatterplot(df_true[i],ax=ax,legend=False)
            pred_vals = df_mean[i]
            # print(torch.tensor(df_true[i]).shape)
            loss_seq = torch.tensor(df_true[i])
            loss_seq_true = torch.nan_to_num(loss_seq, 0.0)
            loss_seq = torch.tensor(pred_vals)
            loss_seq_pred = torch.nan_to_num(loss_seq, 0.0)
            rmse += mask_rmse_loss(true_y=loss_seq_true, predict_y=loss_seq_pred).item()
            # print(rmse)
            pred_std = df_std[i]
            print(df_mean[i])
            sns.lineplot(df_mean[i], ax=ax, legend=False, color=c)  # [true_columns]
            ax.fill_between(x=list(range(170)), y1=pred_vals + pred_std, y2=pred_vals - pred_std, color=c,
                            alpha=0.3,
                            interpolate=True)

        rmse = round(rmse / len(true_columns), 2)
        print(rmse)

        # Label the plot
        ax.set_title(f'Genotype: {genotype} test RMSE: {rmse}')
        ax.set_xlabel('Time step')
        ax.set_ylabel('Plant Height')
        ax.legend()

        plt.tight_layout()
        # plt.show()
        if if_pinn:
            plt.savefig(f'../figure/single_g/{name}_genotype_{genotype}_plot_pinn_new.png')
        else:
            plt.savefig(f'../figure/single_g/{name}_genotype_{genotype}_plot_pure_lstm_new.png')
        # Close the figure after saving to avoid memory issues
        plt.clf()

def plot_curve_witherror_bar_for_multiple_genotype_model_result(data_split='val',file_name='pinn_result_lstm_corr_rescale_gkinship_matrix_encodingyear_split',if_pinn='True'):

    best_hyperparameters = pd.read_csv('best_model_result_summary/multiple_g_result_pinn_{}_{}_best_hyperparameters_result.csv'.format(if_pinn,file_name),header=0,index_col=0)
    if if_pinn =='True':
        c='red'
    else:
        c='orange'
    lr = best_hyperparameters['learning_rate'].unique()[0]
    hidden_size  = best_hyperparameters['hidden_size'].unique()[0]
    num_layer = best_hyperparameters['num_layer'].unique()[0]
    weight_physic = best_hyperparameters['physics_weight'].unique()[0]
    if (weight_physic==0) or (weight_physic==0.0):
        weight_physic=None
    L2 = best_hyperparameters['l2'].unique()[0]
    last_fc_hidden_size = best_hyperparameters['last_fc_hidden_size'].unique()[0]
    genetics_embedding_size = best_hyperparameters['genetics_embedding_size'].unique()[0]
    y_max_bound = best_hyperparameters['y_max_bound'].unique()[0]


    files = glob.glob(
        'pinn_result/multiple_g/pinn_weight_{9}_{0}_curves_{1}_train_lr_{2}_hidden_{3}_fc_hidden_{4}_g_embed{5}_num_layers_{6}_l2_{7}_ymax_bound_{8}_smooth_lossFalse_seed_*'.format(
            data_split, file_name, lr, hidden_size, last_fc_hidden_size, genetics_embedding_size, num_layer, L2, y_max_bound,weight_physic))
    print("pinn_result/multiple_g/{}_true_curves_{}.csv".format(data_split,file_name))
    df_true = pd.read_csv("pinn_result/multiple_g/{}_true_curves_{}.csv".format(data_split,file_name), header=0, index_col=0)
    sub_files = [str(x).split('\\')[1] for x in files]
    print(sub_files)
    df_pred_list =[]
    for file in sub_files:
        print(file)
        df = pd.read_csv('pinn_result/multiple_g/{}'.format(file),header=0,index_col=0)
        df_pred_list.append(df)
    df_mean = pd.concat(df_pred_list, axis=0).groupby(level=0).mean()

    df_std = pd.concat(df_pred_list, axis=0).groupby(level=0).std()

    print(df_mean.columns)
    print(df_true.columns)
    # List of genotypes
    genotypes = df_true.columns.str.split('.').str[0].unique()  # Get unique genotype names

    # Convert 0.0 to NaN in true_df
    df_true.replace(0.0, np.nan, inplace=True)


    colors_list = sns.color_palette('dark',n_colors=len(genotypes))
    for genotype in genotypes:
        print(genotype)
        fig, ax = plt.subplots(figsize=(8, 6))
        color = colors_list.pop()
        # Create a sub-figure for each genotype
        # Get all columns in true_df corresponding to the current genotype (including different plots)
        true_columns = [col for col in df_true.columns if col.split('.')[0] == genotype]
        rmse=0.0
        for i in true_columns:
            sns.scatterplot(df_true[i],ax=ax,legend=False,color=c)
            pred_vals = df_mean[i]
            # print(torch.tensor(df_true[i]).shape)
            loss_seq = torch.tensor(df_true[i])
            loss_seq_true=torch.nan_to_num(loss_seq,0.0)
            loss_seq=torch.tensor(pred_vals)
            loss_seq_pred = torch.nan_to_num(loss_seq,0.0)
            rmse += mask_rmse_loss(true_y=loss_seq_true,predict_y=loss_seq_pred).item()
            # print(rmse)
            pred_std = df_std[i]
            sns.lineplot(df_mean[i],ax=ax,legend=False,color=c)#[true_columns]
            ax.fill_between(x=list(range(170)), y1=pred_vals + pred_std, y2=pred_vals - pred_std, color=c, alpha=0.3,
                             interpolate=True)
        rmse = round(rmse/len(true_columns),2)
        print(rmse)
        ax.set_title(f'Multiple genotype model {data_split} Genotype {genotype} test RMSE:{rmse}')
        ax.set_xlabel('Time step')
        ax.set_ylabel('Plant Height')
        plt.ylim(0,1.2)
        ax.legend()

        # Save the figure as a separate file
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'../figure/multiple_g/{data_split}_genotype_{genotype}_plot_multiple_genotype_{file_name}.png')

        # Close the figure after saving to avoid memory issues
        plt.clf()

def average_temperature_ode_result(file):
    temperatue_ode_df = pd.read_csv(file,header=0,index_col=0,sep='[;,]')
    print(temperatue_ode_df)
    genotype_list  = temperatue_ode_df['genotype'].unique()
    result_df_all = pd.DataFrame()
    for g in genotype_list:
        average_df = temperatue_ode_df.groupby(by=['genotype']).mean()
        std_df = temperatue_ode_df.groupby(by=['genotype']).std()
        std_df.rename(columns={'validation_rMSE':'validation_std', 'train_rMSE':'train_std', 'test_rMSE':'test_std',
                               'dtw_std_val':'dtw_std_val_std', 'dtw_std_train':'dtw_std_train_std', 'dtw_std_test':'dtw_std_test_std'},inplace=True)
        print(average_df)
        print(std_df)
        result_df = pd.concat([average_df,std_df],axis=1)
        result_df_all = pd.concat([result_df,result_df_all],axis=0)
    result_df_all.to_csv('temperature_ode_fit_result_mean.csv')

    print("average train rmse:{}".format(round(result_df_all['train_rMSE'].mean(),3)))
    print("average validation rmse:{}".format(round(result_df_all['validation_rMSE'].mean(),3)))
    print("average test rmse:{}".format(round(result_df_all['test_rMSE'].mean(),3)))
    print("average train std:{}".format(round(result_df_all['train_std'].mean(),3)))
    print("average validation std:{}".format(round(result_df_all['validation_std'].mean(),3)))
    print("average test std:{}".format(round(result_df_all['test_std'].mean(),3)))
    print(f"\nAverage train shapeDTW across all genotypes: {round(result_df_all['dtw_std_train'].mean(),3)} std:{round(result_df_all['dtw_std_train_std'].mean(),3)}")
    print(f"\nAverage Validation shapeDTW across all genotypes: {round(result_df_all['dtw_std_val'].mean(),3)} std:{round(result_df_all['dtw_std_val_std'].mean(),3)}")
    print(f"\nAverage test shapeDTW across all genotypes: {round(result_df_all['dtw_std_test'].mean(),3)} std:{round(result_df_all['dtw_std_test_std'].mean(),3)}")

def boxplot_multiple_genotype():
    pinn_df = pd.read_csv('best_model_result_summary/pinn_penalize_r_best_hyperparameters_result.csv')
    pinn_df['model']='PINN'
    ml_df = pd.read_csv('best_model_result_summary/ml_best_hyperparameters_result.csv')
    ml_df['model'] = 'NN'
    df_ml_melted = pd.melt(ml_df, id_vars=['genotype', 'model'],
                           value_vars=['test_rMSE', 'train_rMSE', 'validation_rMSE'],
                           var_name='group', value_name='rmse')

    df_pinn_melted = pd.melt(pinn_df, id_vars=['genotype', 'model'],
                             value_vars=['test_rMSE', 'train_rMSE', 'validation_rMSE'],
                             var_name='group', value_name='rmse')
    print(df_ml_melted)
    df_combined = pd.concat([df_ml_melted, df_pinn_melted], axis=0)

    # 4. Rename 'group' values from 'testrmse', 'trainrmse', 'valrmse' to 'test', 'train', 'validation'
    df_combined['group'] = df_combined['group'].replace({
        'testrmse': 'test',
        'trainrmse': 'train',
        'valrmse': 'validation'
    })
    custom_palette = {
        'NN_train_rMSE': '#1f77b4',  # dark blue
        'NN_test_rMSE': '#2ca02c',  # dark green
        'NN_validation_rMSE': '#ff7f0e',  # dark orange
        'PINN_train_rMSE': '#aec7e8',  # light blue
        'PINN_test_rMSE': '#98df8a',  # light green
        'PINN_validation_rMSE': '#ffbb78'  # light orange
    }
    df_combined['model_group'] = df_combined['model'] + '_' + df_combined['group']
    # df_combined=df_combined[df_combined['genotype'].isin([335,133,362])]
    sns.boxplot(x='genotype', y='rmse', hue='model_group', data=df_combined, palette=custom_palette, dodge=True)

    plt.title('RMSE per Genotype with Separate Boxes for NN and PINN (Train/Test/Validation)')
    plt.ylabel('RMSE')

    # Adjust legend to avoid overlap
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    # plt.show()

def order_genotype_based_on_their_similarity(first_genotype:str):

    kinship_pd = pd.read_csv('../temporary/kinship_matrix_astle.csv',header=0,index_col=0)
    # kinship_pd = pd.read_csv('../processed_data/distance_encoding_matrix_full.csv', header=0,index_col=0)
    # kinship_pd.index= kinship_pd['genotype']
    # kinship_pd =kinship_pd.drop(columns='genotype')
    print(kinship_pd)
    #kinship matrix should be a square matrix with the same order
    kinship_pd.index = kinship_pd.columns
    # g_similarity,_ = minmax_scaler(torch.tensor(kinship_pd.to_numpy()),min=-1,max=1)
    # g_similarity = np.tanh(kinship_pd.to_numpy())
    # kinship_pd = pd.DataFrame(data=g_similarity,index=kinship_pd.index,columns=kinship_pd.columns)
    # print(kinship_pd.sum())
    # kinship_pd = kinship_pd.abs() do not need this because negetive value does not mean negetive correlation,
    # but means it has lower correlation than random
    print(kinship_pd)
    #descending, the larger the value the similar the two genotypes are
    similarity_row = kinship_pd.sort_values(by=first_genotype,ascending=False)
    sorted_df = kinship_pd.sort_values(by=first_genotype,ascending=False)
    sorted_columns = sorted_df.index.tolist()
    sorted_df = sorted_df[sorted_columns]
    sns.heatmap(sorted_df, vmin=-2, vmax=2)
    plt.title('19 genotypes kinship matrix')
    # plt.show()
    print(similarity_row)
    ordered_genotype_list = list(similarity_row.index)
    # raise EOFError
    return ordered_genotype_list,similarity_row[first_genotype].to_list()
def average_and_plot_result_single_g(name='gpu_cv_final'):

    files = glob.glob("pinn_result/result_summary/*{}_same_length*rescaleTrue*115*.csv".format(name))
    print(files)
    assert len(files)==19
    files = [str(x).split('\\')[1] for x in files]
    for file in files:
        print(file)
        plot_PINN_result("pinn_result/result_summary/{}".format(file))
    plot_mean_error_bar_multiple_genotype(file_name='ml',
                                          name='PINN_mask_loss_{}'.format(name),
                                          result_file_dir='pinn_result/result_summary/')
def single_g_cv_result_for_significant_test():
    """
    test if PINN significantly perform better than LSTM use paired t test
    """
    # df_ml = pd.read_csv("pinn_result/result_summary/best_model_cv/PINN_mask_loss_ml__cv.csv", header=0, index_col=0)
    # df_ml = df_ml.drop_duplicates()

    df_ml = pd.DataFrame()
    file_paths = [Path(path) for path in
                  glob.glob("pinn_result/result_summary/best_model_cv/PINN_mask_loss_*PINN_mask_loss_gpu_lstm_corr_same_length_smooth_tempFalserescaleFalse_cv.csv")]
    print(file_paths)
    for file in file_paths:
        single_g_df_ml = pd.read_csv(file, header=0, index_col=0)
        df_ml = pd.concat([df_ml, single_g_df_ml])
    df_ml = df_ml.drop_duplicates()
    print(df_ml.mean())
    grouped_std = copy.deepcopy(df_ml).groupby(['genotype', 'n_split'])['train_rMSE'].std()
    # Then take the mean of all stds
    average_std = grouped_std.mean()
    print("ml train std:{}".format(average_std))

    grouped_std = copy.deepcopy(df_ml).groupby(['genotype', 'n_split'])['validation_rMSE'].std()

    # Then take the mean of all stds
    average_std = grouped_std.mean()
    print("ml val std:{}".format(average_std))

    grouped_std = copy.deepcopy(df_ml).groupby(['genotype', 'n_split'])['test_rMSE'].std()

    # Then take the mean of all stds
    average_std = grouped_std.mean()
    print("ml test std:{}".format(average_std))
    df_cv_merge = pd.DataFrame()
    file_paths = [Path(path) for path in
                  glob.glob("pinn_result/result_summary/best_model_cv/*penalize_r*rescaleFalse*")]
    print(file_paths)
    for file in file_paths:
        single_g_df = pd.read_csv(file, header=0, index_col=0)
        df_cv_merge = pd.concat([df_cv_merge, single_g_df])

    print(df_cv_merge.mean())
    print(df_cv_merge.columns)

    grouped_std = copy.deepcopy(df_cv_merge).groupby(['genotype',  'n_split'])['train_rMSE'].std()
    # Then take the mean of all stds
    average_std = grouped_std.mean()
    print("pinn train std:{}".format(average_std))

    grouped_std = copy.deepcopy(df_cv_merge).groupby(['genotype',  'n_split'])['validation_rMSE'].std()
    # Then take the mean of all stds
    average_std = grouped_std.mean()
    print("pinn val std:{}".format(average_std))

    grouped_std = copy.deepcopy(df_cv_merge).groupby(['genotype',  'n_split'])['test_rMSE'].std()
    # Then take the mean of all stds
    average_std = grouped_std.mean()
    print("pinn test std:{}".format(average_std))

    df_cv_merge['model'] = 'logi-PINN'
    df_ml['model'] = 'LSTM-NN'
    # print(df_cv_merge)
    # print(df_ml)
    from scipy.stats import ttest_rel, wilcoxon, shapiro

    custom_palette = {
        'LSTM-NN': 'orange', 'logi-PINN': 'red'
    }
    # print(merged_df)
    plot_df = pd.concat([df_ml, df_cv_merge],axis=0)
    plot_df =plot_df.merge(n_split_year_map, on='n_split', how='left')
    print(plot_df)
    ax=sns.boxplot(data =plot_df,hue='model',x='train_year',y='test_rMSE',dodge=True,palette=custom_palette)
    ax.set_axisbelow(True)
    ax.legend(title='Model Type', loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig('../figure/6year_data_split_comparision.png')

def only_plot_test_box_plot_single_g():
    """
    make box plot comparision for test rmse result with 5 different models
    """
    #PINN
    files = glob.glob("pinn_result/result_summary/single_g/PINN_mask_loss_cpu_logi_pinn_save_model__same_lengthgenotype*penalize_rpinnmode_Truefit_ml_first_FalsecorrectedFalsesmoothFalsefill_in_naTruerescaleFalsestart_date_115.csv")
    print(files)
    files = [str(x).split('\\')[-1] for x in files]
    for file in files:
        print(file)
        plot_PINN_result("pinn_result/result_summary/single_g/{}".format(file))

    pinn_df, order_g =plot_mean_error_bar_multiple_genotype(file_name='pinn_penalize_r', name='PINN_mask_loss_cpu_logi_pinn_save_model__same_length',
                                          result_file_dir='pinn_result/result_summary/single_g/')
    print('pinn df for plot')
    pinn_df = pinn_df[pinn_df['metric']=='test_rMSE']
    pinn_df['model']='PINN'
    print(pinn_df.columns)
    pinn_df.rename({"random_seed":"random_sees"})

    #LSTM
    files = glob.glob(
        "pinn_result/result_summary/single_g/PINN_mask_loss_gpu_lstm_NN_save_model_same_lengthgenotype*pinnmode_Falsefit_ml_first_FalsecorrectedFalsesmoothFalsefill_in_naTruerescaleFalsestart_date_115w0_.csv")
    print(files)
    files = [str(x).split('\\')[-1] for x in files]
    for file in files:
        print(file)
        plot_PINN_result("pinn_result/result_summary/single_g/{}".format(file))
    lstm_df, order_g =plot_mean_error_bar_multiple_genotype(file_name='ml', name='PINN_mask_loss_gpu_lstm_NN_save_model_same_length',
                                          result_file_dir='pinn_result/result_summary/single_g/')
    lstm_df = lstm_df[lstm_df['metric'] == 'test_rMSE']
    lstm_df['model'] = 'LSTM'
    print(lstm_df)
    plot_long_df = pd.concat([pinn_df,lstm_df],axis=0)
    # plot_long_df = plot_long_df.merge(n_split_year_map, on='n_split', how='left')

    #logistic_ode
    logistic_ode_df = pd.read_csv("../figure/logistic_ODE_fit/single_genotype_model/logistic_ode_predict_multiple_genotype_raw.csv",header=0,index_col=0)

    df_ode_parameters = copy.deepcopy(logistic_ode_df)
    _, genotype_similarity = order_genotype_based_on_their_similarity(first_genotype='106')
    simlarity_df = pd.DataFrame({'genotype': order_g, 'similarity_compard_with_106': genotype_similarity},
                                index=range(len(order_g)))
    simlarity_df['genotype'] = simlarity_df['genotype'].astype(int)
    df_ode_parameters['genotype'] = df_ode_parameters['genotype'].astype(int)
    df_ode_parameters = pd.merge(logistic_ode_df, simlarity_df, 'left')
    plt.clf()
    # palette = sns.color_palette("viridis", len(df_ode_parameters['similarity_compard_with_106'].values))
    scatter=plt.scatter(data=df_ode_parameters, x='predicted_r', y='predicted_y_max',
                          c=df_ode_parameters['similarity_compard_with_106'].values, cmap='viridis', s=30)

    colormap = plt.cm.viridis
    norm = plt.Normalize(df_ode_parameters['similarity_compard_with_106'].min(),
                         df_ode_parameters['similarity_compard_with_106'].max())
    # Add genotype IDs as annotations
    for i in range(len(df_ode_parameters)):
        plt.text(
            df_ode_parameters['predicted_r'][i],
            df_ode_parameters['predicted_y_max'][i],
            df_ode_parameters['genotype'][i],
            horizontalalignment='left',
            size='medium',
            color=colormap(norm(df_ode_parameters['similarity_compard_with_106'][i])),  # Match node color
            weight='semibold'
        )

    plt.colorbar(scatter, label='Similarity')
    # Customize the plot
    plt.title(' r vs. ymax')
    plt.xlabel('predicted_r')
    plt.ylabel('predicted_y_max')
    plt.grid()
    plt.tight_layout()
    plt.savefig('../figure/logistic_ode_vs_kinship.svg', bbox_inches='tight',dpi=1000)
    plt.clf()

    logistic_ode_df = pd.melt(
        logistic_ode_df,
        id_vars=["genotype",'random_sees'],
        value_vars=["train_rMSE", "validation_rMSE", "test_rMSE"],
        var_name="metric",
        value_name="value"
    )
    logistic_ode_df = logistic_ode_df[logistic_ode_df['metric'] == 'test_rMSE']
    logistic_ode_df['model'] = 'logistic_ode'
    plot_long_df = pd.concat([plot_long_df, logistic_ode_df], axis=0)


    # temperature_ode
    temperature_ode_df = pd.read_csv("best_model_result_summary/temperature_ode_fit_result_new_april.csv", header=0, index_col=0)
    print(temperature_ode_df)
    temperature_ode_df = pd.melt(
        temperature_ode_df,
        id_vars=["genotype",'random_sees'],
        value_vars=["train_rMSE", "validation_rMSE", "test_rMSE"],
        var_name="metric",
        value_name="value"
    )
    temperature_ode_df = temperature_ode_df[temperature_ode_df['metric'] == 'test_rMSE']
    temperature_ode_df['model'] = 'temperature_ode'
    plot_long_df = pd.concat([plot_long_df, temperature_ode_df], axis=0)

    #random forest
    rf_df = pd.read_csv('best_model_result_summary/rf_model_result_summary.csv',header=0,index_col=0)
    rf_df = pd.melt(
        rf_df,
        id_vars=["genotype",'random_sees'],
        value_vars=["train_rMSE", "validation_rMSE", "test_rMSE"],
        var_name="metric",
        value_name="value"
    )
    rf_df = rf_df[rf_df['metric'] == 'test_rMSE']
    rf_df['model'] = 'RF'
    plot_long_df = pd.concat([plot_long_df, rf_df], axis=0)

    print(plot_long_df)


    order_g = pinn_df.groupby("genotype")["value"].mean().sort_values().index.tolist()
    plot_long_df['genotype'] = pd.Categorical(plot_long_df['genotype'].astype(str), categories=order_g, ordered=True)

    plt.figure(figsize=(10, 6))
    # plt.ylim(0, 0.3)
    ax=sns.stripplot(data=plot_long_df, x="genotype", y="value", hue="model",dodge=True,jitter=True,alpha=0.8,
                       hue_order=['logistic_ode','temperature_ode','RF','LSTM','PINN'],palette=["yellowgreen","purple","cyan", 'orange','red'])
    ax.legend(title='Model Type', loc="upper left")
    # ax= sns.boxplot(x='genotype', y='value', hue='model', data=plot_long_df, dodge=True, order=order_g,
    #             palette=["orange","pink","cyan", 'yellowgreen','darkgreen'],showfliers=False,hue_order=['logistic_ode','temperature_ode','RF','LSTM','PINN'])
    ax.legend(title='Model Type', loc="upper left", bbox_to_anchor=(1, 1))
    # # Set axis below so grid doesn't overlay the boxes
    ax.set_axisbelow(True)
    ax.set_ylabel('test RMSE')
    # Remove default x-grid if any
    ax.xaxis.grid(False)

    # Get the tick positions for the x-axis categories (the groups)
    xticks = ax.get_xticks()

    # Calculate midpoints between x-group ticks for vertical gridlines
    xlines = xticks[:-1] + np.diff(xticks) / 2

    # Draw vertical grid lines between x-groups
    for x in xlines:
        ax.axvline(x=x, color='gray', linestyle='--', linewidth=0.5)

    # Optional: show y-grid as usual
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5)
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('../figure/5_models_test_result_compare_stripplot_new.png',dpi=2000, bbox_inches='tight')
    # plt.show()
    return order_g

    # plt.figure(figsize=(10, 6))
    # plt.ylim(0, 0.2)
    # sns.boxplot(x='train_year', y='value', hue='model', data=plot_long_df, dodge=True,
    #             palette=['darkgreen', 'yellowgreen'])
    # plt.legend(title='Model Type', loc="upper left", bbox_to_anchor=(1, 1))
    # plt.tight_layout()
    # plt.show()
def plot_the_first_layer_weight_lstm(filename='gpu_lstm_corr_same_length',order_g=None):
    """
    heatmap, generate by AI and modified by me.
    remark: you can copy paste unicode subscript as comment or print
    """
    from NNmodel_training import LSTM_ts_and_temperature_predict_height
    order_g = [int(g) for g in order_g]
    best_hyperparameter_df = pd.read_csv('best_model_result_summary/PINN_mask_loss_{}_best_hyperparameters_result.csv'.format(filename), header=0, index_col=0)

    print(best_hyperparameter_df.columns)

    # Prepare heatmap storage: rows = genotypes, columns = 4 gates * 2 (ih and hh)
    columns = ['ih_input', 'ih_forget', 'ih_cell', 'ih_output', 'hh_input', 'hh_forget', 'hh_cell', 'hh_output']
    heatmap_data = {genotype: np.zeros(len(columns)) for genotype in order_g}
    print(order_g)
    counts = {genotype: 0 for genotype in order_g}
    for genotype in order_g:
        result_df = best_hyperparameter_df[best_hyperparameter_df['genotype'] == genotype]
        print(result_df)
        hidden = result_df['hidden_size'].unique().item()
        num_layer = result_df['num_layer'].unique().item()
        lr = result_df['lr'].unique().item()
        weight_physic = result_df['weight_physic'].unique().item()
        L2 = result_df['l2'].unique().item()

        for seed in [1,2,3,4,5]:

            model_name="pinn_result/model/trained_model_0split_hidden{0}_env{1}_ts{2}_lr_{3}_w_ph{4}_ode_int_False_l2_{5}_{6}genotype{7}penalize_rpinnmode_Truefit_ml_first_FalsecorrectedFalsesmoothFalsefill_in_naTruerescaleFalsestart_date_115_rs_{8}.pt".format(hidden,num_layer,num_layer,lr,weight_physic,L2,filename,genotype,seed)
            print(model_name)
            # model_name = 'pinn_result/model/trained_model_0split_hidden5_env2_ts2_lr_0.001_w_ph9_ode_int_False_l2_1.0_cpu_pinn_save_model_same_lengthgenotype5penalize_rpinnmode_Truefit_ml_first_FalsecorrectedFalsesmoothFalsefill_in_naTruerescaleTruestart_date_115_rs_1.pt'
            with open(
                    model_name, 'rb') as file:
                # model1 = torch.load(model_name, map_location=torch.device('cpu'),weights_only=False)

                model1 = LSTM_ts_and_temperature_predict_height(hidden_size=hidden, input_size=1,
                                                                    num_layer=num_layer,
                                                                    smooth_out=False,
                                                                    temperature_pinn=False,
                                                                    genetics=None)
                model1.load_state_dict(torch.load(model_name, weights_only=True))
                model1.eval()
                # model1 = dill.load(file).cpu()
            for name, param in model1.named_parameters():
                print(name)
                print(param)

            weight_ih = model1.lstm1.weight_ih_l0  # [4*hidden_size, input_size]
            weight_hh = model1.lstm1.weight_hh_l0  # [4*hidden_size, hidden_size]

            hidden_size = weight_hh.shape[1]
            print(hidden_size)

            # Split into 4 gates
            w_ih_i, w_ih_f, w_ih_g, w_ih_o = weight_ih.chunk(4, dim=0)
            w_hh_i, w_hh_f, w_hh_g, w_hh_o = weight_hh.chunk(4, dim=0)

            # Calculate L2 norm per gate
            ih_norm = torch.norm(w_ih_i) + torch.norm(w_ih_f) + torch.norm(w_ih_g) + torch.norm(w_ih_o)
            hh_norm = torch.norm(w_hh_i) + torch.norm(w_hh_f) + torch.norm(w_hh_g) + torch.norm(w_hh_o)

            summary_ratio = hh_norm / (ih_norm + hh_norm)

            print(f"Contribution from Hₜ₋₁: {summary_ratio:.3f} vs. Xₜ: {(1 - summary_ratio):.3f}")

            values = [
                torch.norm(w_ih_i).item(),
                torch.norm(w_ih_f).item(),
                torch.norm(w_ih_g).item(),
                torch.norm(w_ih_o).item(),
                torch.norm(w_hh_i).item(),
                torch.norm(w_hh_f).item(),
                torch.norm(w_hh_g).item(),
                torch.norm(w_hh_o).item(),
            ]

            heatmap_data[genotype] += np.array(values)
            counts[genotype] += 1
    else:
        # Average over seeds
        for genotype in order_g:
            if counts[genotype] > 0:
                heatmap_data[genotype] /= counts[genotype]

        # Create DataFrame and plot
        print(order_g)
        heatmap_df = pd.DataFrame.from_dict(heatmap_data, orient='index', columns=columns)
        heatmap_df=heatmap_df.reindex(index=order_g)
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_df, annot=True, cmap="viridis")
        plt.title("Averaged of LSTM Gates (ih & hh) per Genotype")
        plt.xlabel("LSTM Gates")
        plt.ylabel("Genotype")
        plt.tight_layout()
        plt.savefig('weight_heatmap.png',dpi=2000)
        # plt.show()
def process_multiple_g_result():
    #LSTM-NN multiple g
    file_paths = [Path(path) for path in glob.glob(
        "pinn_result/result_summary/multiple_g/*multiple_g_result_*rerun_final*smooth_temp_False*.csv")]
    print('processing {} files'.format(len(file_paths)))
    for file_path in file_paths:
        calculate_std_for_different_seed_result_multiple_g(file=file_path)
    #logi-PINN multiple g
    file_paths = [Path(path) for path in glob.glob(
        "pinn_result/result_summary/multiple_g/*multiple_g_result_*PINN_retun_final*.csv")]
    print('processing {} files'.format(len(file_paths)))
    for file_path in file_paths:
        calculate_std_for_different_seed_result_multiple_g(file=file_path)

    #plot

    plot_curve_witherror_bar_for_multiple_genotype_model_result(data_split='test',file_name='PINN_retun_finalkinship_matrix_encodingsmooth_temp_Falseyear_split')
    plot_curve_witherror_bar_for_multiple_genotype_model_result(data_split='test',
                                                                file_name='PINN_retun_finalgenotype_one_hot_encodingsmooth_temp_Falseyear_split')
    plot_curve_witherror_bar_for_multiple_genotype_model_result(data_split='test',
                                                                file_name='NN_rerun_finalgenotype_one_hot_encodingsmooth_temp_Falseyear_split',if_pinn='False')
    plot_curve_witherror_bar_for_multiple_genotype_model_result(data_split='test',
                                                                file_name='NN_rerun_finalkinship_matrix_encodingsmooth_temp_Falseyear_split',if_pinn='False')
def main():
    global n_split_year_map
    plt.rcParams["font.family"] = "Times New Roman"
    #multiple genotype
    # process_multiple_g_result()

    #single genotype
    n_split_year_map = pd.read_csv("../temporary/n_split_map_df.csv",header=0,sep=',')
    print(n_split_year_map)
    n_split_year_map['n_split']=n_split_year_map['n_split'].astype(int)
    single_g_cv_result_for_significant_test()
    only_plot_test_box_plot_single_g()
    plot_curve_witherror_bar_for_single_genotype_model_result(name='test',best_hyperparameters_df='best_model_result_summary/PINN_mask_loss_cpu_logi_pinn_save_model__same_length_best_hyperparameters_result.csv',if_pinn=True)
    plot_curve_witherror_bar_for_single_genotype_model_result(name='test',best_hyperparameters_df='best_model_result_summary/PINN_mask_loss_gpu_lstm_NN_save_model_same_length_best_hyperparameters_result.csv',if_pinn=False)
    order_g = only_plot_test_box_plot_single_g()
    plot_the_first_layer_weight_lstm(filename='cpu_logi_pinn_save_model__same_length', order_g=order_g)
if __name__ == '__main__':
    main()
