"""This script is to run Random forest, which use temperature and genotype """
from networkx.algorithms.bipartite import color
from scipy.interpolate import interp1d
from sklearn.model_selection import BaseCrossValidator
import numpy as np
import torch
import torch.nn as nn
import random
import pandas as pd
import dill
import torch.optim as optim

import wandb
import copy
from DataPrepare import reverse_min_max_scaling,train_test_split_based_on_group, count_parameters
from NNmodel_training import average_based_on_group_df,mask_dtw_loss,mask_rmse_loss,smooth_tensor_ignore_nan
from MultipleGenotypeModel import smoothing_spline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.optimize import least_squares

def read_tensor_files(genotype):
    with open("../temporary/plant_height_tensor_{}.dill".format(genotype), 'rb') as f:
        plant_height_tensor = dill.load(f)
    f.close()
    with open("../temporary/group_list_df_{}.dill".format(genotype), 'rb') as f:
        group_df = dill.load(f)
    f.close()
    with open("../temporary/temperature_tensor_{}.dill".format(genotype), 'rb') as f:
        temperature_same_length_tensor = dill.load(f)
        # temperature_same_length_tensor=smooth_tensor_ignore_nan(temperature_same_length_tensor,20)
        temperature_same_length_tensor = torch.nan_to_num(temperature_same_length_tensor, nan=0.0, posinf=0.0,
                        neginf=0.0)
        # print('temperature input tensor shape:{}'.format(temperature_same_length_tensor.shape))
    f.close()

    group_df_replicates = group_df[['genotype.id','year_site.harvest_year','new_group_list']]
    # check shape to perform split
    plant_height_tensor = torch.squeeze(plant_height_tensor)
    temperature_same_length_tensor = torch.squeeze(temperature_same_length_tensor)
    # print(plant_height_tensor.shape)
    # print(temperature_same_length_tensor.shape)
    # print(temperature_same_length_tensor)
    # print(group_df)
    genotype_list = group_df['genotype.id'].to_list()
    tensor_list = []
    for genotype in genotype_list:
        with open('../temporary/{}_{}.dill'.format(genotype, 'binary_encoding'), 'rb') as file1:
            genotype_tensor = dill.load(file1)
            tensor_list.append(genotype_tensor.unsqueeze(dim=0))
    genetics_input_tensor = torch.cat(tensor_list, dim=0).float()
    genetics_input_tensor = torch.squeeze(torch.permute(genetics_input_tensor,(2,0,1)))

    train_test_validation_dictionary = train_test_split_based_on_group(group_df, group_df,
                                                                       group_name=[
                                                                           'year_site.harvest_year',
                                                                       ], n_split=1)
    # print(train_test_validation_dictionary)
    return train_test_validation_dictionary,plant_height_tensor,temperature_same_length_tensor,genetics_input_tensor,group_df[['year_site.harvest_year']],group_df_replicates

def train_rf(train_test_validation_dictionary,plant_height_tensor,temperature_same_length_tensor,genetics_input_tensor,result_df,genotype,group_df_replicates):
    """The main issue here is sklearn does not support customize criteria, so i can not mask NA during traning; while it
    also does not support NA input"""

    n_split = len(train_test_validation_dictionary.keys())
    val_errors=[]
    train_errors=[]
    test_errors=[]
    training_dtw_losses = []
    validation_dtw_losses = []
    test_dtw_losses = []
    for n in range(5):
        rf_model = RandomForestRegressor(n_estimators=100, random_state=n)
        train_index, validation_index, test_index = train_test_validation_dictionary['splits_{}'.format(0)]
        train_y = plant_height_tensor[115:, train_index].permute(1,0)


        train_env = temperature_same_length_tensor[115:, train_index].permute(1,0)
        validation_y = plant_height_tensor[115:, validation_index].permute(1,0)
        validation_env = temperature_same_length_tensor[115:, validation_index].permute(1,0)

        test_y = plant_height_tensor[115:, test_index].permute(1,0)
        test_env = temperature_same_length_tensor[115:, test_index].permute(1,0)
        print("genotype code shape:{}".format(genetics_input_tensor.shape))

        train_g = genetics_input_tensor[train_index,:]
        validation_g = genetics_input_tensor[validation_index,:]
        test_g = genetics_input_tensor[test_index,:]
        group_df_replicates_train = group_df_replicates.iloc[train_index,:]
        group_df_replicates_validation = group_df_replicates.iloc[validation_index, :]
        group_df_replicates_test = group_df_replicates.iloc[test_index, :]
        # creat time sequence, same as input shape
        ts_train = torch.linspace(0.0, 284, 285).unsqueeze(dim=1).repeat(1, train_y.shape[0]
                                                                                          )[:-115, :].permute(1,0)  # time sequences steps
        ts_validation = torch.linspace(0.0, 284, 285).unsqueeze(dim=1).repeat(1, validation_y.shape[0]
                                                                                               )[:-115, :].permute(1,0) # time sequences steps
        ts_test = torch.linspace(0.0, 284, 285).unsqueeze(dim=1).repeat(1, test_y.shape[0]
                                                                                         )[:-115, :].permute(1,0)  # time sequences steps

        #smooth y to get daily result to avoid spike in prediction cause by na
        # train_y_smoothed = smoothing_spline(train_y.permute(1,0).unsqueeze(-1),num_knots=8).squeeze().permute(1,0)
        # sns.lineplot(train_y.T)
        # plt.show()
        # train_input = torch.cat([train_env,train_g],dim=-1)
        # val_input = torch.cat([validation_env, validation_g],dim=-1)
        # test_input = torch.cat([test_env, test_g],dim=-1)
        print(train_env.shape,ts_train.shape)
        train_input = torch.cat([train_env,ts_train],dim=-1)
        val_input = torch.cat([validation_env, ts_validation],dim=-1)
        test_input = torch.cat([test_env, ts_test],dim=-1)
        print('input and output shape for rf regression:')
        print(train_input.shape, train_y.shape)
        # mask = (~torch.isin(train_y, torch.tensor(0.0))).float()
        rf_model.fit(train_input, train_y) # (88,170*2) trait 170 time step as features

        #average across replicates: same as other models

        train_input_avg,_ = average_based_on_group_df(train_input.permute(1,0).unsqueeze(dim=-1), df=group_df_replicates_train)

        val_input_avg,_ = average_based_on_group_df(val_input.permute(1, 0).unsqueeze(dim=-1),
                                                  df=group_df_replicates_validation)
        test_input_avg,_ = average_based_on_group_df(test_input.permute(1, 0).unsqueeze(dim=-1),
                                                    df=group_df_replicates_test)

        train_y_avg,_ = average_based_on_group_df(train_y.permute(1,0).unsqueeze(dim=-1), df=group_df_replicates_train)
        validation_y_avg,_ = average_based_on_group_df(validation_y.permute(1, 0).unsqueeze(dim=-1),
                                                  df=group_df_replicates_validation)
        test_y_avg,_ = average_based_on_group_df(test_y.permute(1, 0).unsqueeze(dim=-1),
                                                    df=group_df_replicates_test)
        train_input_avg = torch.squeeze(train_input_avg,dim=-1).permute(1, 0)
        val_input_avg = torch.squeeze(val_input_avg, dim=-1).permute(1, 0)
        test_input_avg = torch.squeeze(test_input_avg, dim=-1).permute(1, 0)
        train_y_avg = torch.squeeze(train_y_avg,dim=-1).permute(1, 0)
        validation_y_avg = torch.squeeze(validation_y_avg, dim=-1).permute(1, 0)
        test_y_avg = torch.squeeze(test_y_avg, dim=-1).permute(1, 0)
        print(train_input_avg.shape,train_y_avg.shape)
        # raise EOFError
        y_train_pred = rf_model.predict(train_input_avg)

        train_error = mask_rmse_loss(train_y_avg.T, y_train_pred.T)
        shape_dtw_loss_train = mask_dtw_loss(true_y=train_y_avg.T,predict_y=y_train_pred.T)
        training_dtw_losses.append(shape_dtw_loss_train)
        train_errors.append(train_error)
        print("train rmse: {}".format(train_error))
        #run model on validation set
        y_val_pred = rf_model.predict(val_input_avg)
        val_error = mask_rmse_loss(true_y=validation_y_avg.T, predict_y=y_val_pred.T)
        shape_dtw_loss_val = mask_dtw_loss(true_y=validation_y_avg.T, predict_y=y_val_pred.T)
        val_errors.append(val_error)
        validation_dtw_losses.append(shape_dtw_loss_val)
        print(f"Validation MSE for fold {n + 1}: {val_error}")
        y_test_pred = rf_model.predict(test_input_avg)
        #  test RMSE error
        test_error = mask_rmse_loss(test_y_avg.T, y_test_pred.T)
        shape_dtw_loss_test = mask_dtw_loss(true_y=test_y_avg.T, predict_y=y_test_pred.T)
        print("test error: {}".format(test_error))
        test_errors.append(test_error)
        test_dtw_losses.append(shape_dtw_loss_test)
        # sns.scatterplot(test_y_avg.T)
        # sns.lineplot(y_test_pred.T)
        # plt.show()
        # sns.lineplot(test_input_avg[:,:170].T)
        # plt.show()
        new_row = pd.DataFrame({'random_sees':n,'train_rMSE':train_error,'validation_rMSE':val_error,'test_rMSE':test_error,
                                'dtw_std_train': shape_dtw_loss_train,
                'dtw_std_val': shape_dtw_loss_val, 'dtw_std_test': shape_dtw_loss_test,'genotype':genotype},index=[0])
        result_df = pd.concat([result_df,new_row])
    else:
        average_train_error = np.mean(train_errors)
        std_train = np.std(train_errors)
        average_val_error = np.mean(val_errors)
        std_val = np.std(val_errors)
        average_test_error = np.mean(test_errors)
        std_test = np.std(test_errors)
        print(f"\nAverage train RMSE across 5 seeds: {average_train_error} std:{std_train}")
        print(f"\nAverage Validation RMSE across 5 seeds: {average_val_error} std:{std_val}")
        print(f"\nAverage test RMSE across 5 seeds: {average_test_error} std:{std_test}")

        average_train_dtw = np.mean(training_dtw_losses)
        std_train_dtw = np.std(training_dtw_losses)
        average_val_dtw = np.mean(validation_dtw_losses)
        std_val_dtw = np.std(validation_dtw_losses)
        average_test_dtw = np.mean(test_dtw_losses)
        std_test_dtw = np.std(test_dtw_losses)
        print(f"\nAverage train shapeDTW across 5 seeds: {average_train_dtw} std:{std_train_dtw}")
        print(f"\nAverage Validation shapeDTW across 5 seeds: {average_val_dtw} std:{std_val_dtw}")
        print(f"\nAverage test shapeDTW across 5 seeds: {average_test_dtw} std:{std_test_dtw}")
        return average_train_error,std_train,average_val_error,std_val,average_test_error,std_test,average_train_dtw,\
        std_train_dtw,average_val_dtw,std_val_dtw,average_test_dtw,std_test_dtw,result_df

def fit_temperature_ode_with_spicy():
    """
    This is the funtion to fit temperature ODE to 2018 and 2019 data and generate curve based on temperature from 2021 and 2022
    """
    from scipy.integrate import solve_ivp

    def temperature_logistic_growth_ode(t,y, temperature, r, y_max,  tl,  th):

        # # print(temperature.shape)
        # print(y.shape)
        # print(int(t))
        # print(temperature)
        temperature_t= temperature(t)
        # print('temperature at t')
        # print(temperature_t)
        # tl =292
        # th=303
        tal =2000
        tah=60000
        f_t = (1.0 + np.exp((tal / (temperature_t + 273.15)) - (tal /tl)) +
                         np.exp((tah / th) - (tah / (temperature_t + 273.15)))) ** -1
        # print(f_t)
        dY_dT = r * f_t * y * (1.0 - y / y_max).flatten()
        # print(dY_dT)
        # print(dY_dT.shape)
        return dY_dT

    def solve_ode(y0, t, temperature, theta):
        # print()
        sol = solve_ivp(temperature_logistic_growth_ode,[t[0],t[-1]],[y0], args=(temperature, *theta), t_eval=t,method='LSODA')
        return sol.y[0]
        # return odeint(temperature_logistic_growth_ode, y0, t, args=(temperature, *theta))

    def mask_rmse(y_pred, y_true):
        mask = (y_true != 0.0).astype(float)
        loss = (np.sum(((y_true - y_pred) * mask) ** 2) / np.count_nonzero(mask))**0.5
        # print('loss :{}'.format(loss))
        return loss
    def residual(theta, t, temperature, y_true, y0):
        mask = (y_true != 0.0).astype(float)
        y_pred_list=[]
        for seq in range(y_true.shape[1]):

            y_obs = y_true[:, seq]
            temperature_seq= temperature[:,seq]
            temperature_interp = interp1d(t, temperature_seq, fill_value='extrapolate',kind='nearest')

            y_pred = solve_ode(y0, t, temperature_interp, theta)
            y_pred_list.append(y_pred)
        pred_y= np.vstack(y_pred_list).T
        # print( np.mean((pred_y-y_true)*mask,axis=0).shape)
        return np.mean((pred_y-y_true)*mask,axis=0)

    def objective_function(theta, t, temperature, y_true, y0):
        total_sse = 0.0
        # print(temperature.shape)
        for seq in range(y_true.shape[1]):
            # print(seq)
            y_obs = y_true[:, seq]
            temperature_seq= temperature[:,seq]
            temperature_interp = interp1d(t, temperature_seq, fill_value='extrapolate',kind='nearest')

            y_pred = solve_ode(y0, t, temperature_interp, theta)

            total_sse += mask_rmse(y_pred=y_pred, y_true=y_obs)
        total_sse = total_sse/y_true.shape[1]
        # print(total_sse)
        return total_sse

    def fit_ode_model_with_mask_rmse(true_y, temperature, t, y0):

        # base_initial_param = [0.1, 1.0, 2000, 292, 60000, 303]
        base_initial_param = [0.0, 0.0,  292, 303]
        randomness = np.array([
            np.random.uniform(0.1, 0.2),
            np.random.uniform(0.7, 0.8),
            # np.random.normal(0, 0.05),  # For the first parameter (small variance)
            # np.random.normal(0, 0.1),  # For the second parameter (small variance)
            # np.random.normal(0, 20),  # For the third parameter (higher variance)
            np.random.normal(0, 10),  # For temperature lower bound
            # np.random.normal(0, 200),  # For tal (higher variance)
            np.random.normal(0, 10)  # For temperature upper bound
        ])

        # Create the new initial guess with randomness
        initial_param = np.array(base_initial_param) + randomness
        print('inital parameter')
        print(initial_param)
        # Set options for the optimizer
        options = {'maxiter': 3000, 'disp': True, 'adaptive': True}
        result = minimize(objective_function, initial_param, args=(t, temperature, true_y, y0), method='Nelder-Mead',options=options)#
        print(result)
        # fit_parameter= result.x
        # result = least_squares(residual, fit_parameter, args=(t, temperature, true_y, y0), method='lm')
        # result = minimize(objective_function,fit_parameter, args=(t, temperature, true_y, y0),method='L-BFGS-B',options=options)
        # print('after further optimize')
        # print(result)
        return result.x  # Fitted parameters

    def load_data_fit_temperature_ode(train_test_validation_dictionary, plant_height_tensor, temperature_tensor,result_df,
                                      genotype,group_df_replicates):
        plant_height_tensor = torch.squeeze(plant_height_tensor)#.numpy()
        temperature_tensor = torch.squeeze(temperature_tensor)#.numpy()

        train_index, validation_index, test_index = train_test_validation_dictionary['splits_{}'.format(0)]
        train_y = plant_height_tensor[115:, train_index]
        train_env = temperature_tensor[115:, train_index]
        validation_y = plant_height_tensor[115:, validation_index]
        validation_env = temperature_tensor[115:, validation_index]
        test_y = plant_height_tensor[115:, test_index]
        test_env = temperature_tensor[115:, test_index]
        # plt.plot(train_input_avg)
        # plt.show()
        ts_train = np.linspace(0.0, 284, 285)[:-115]
        ts_validation = np.linspace(0.0, 284, 285)[:-115]
        ts_test = np.linspace(0.0, 284, 285)[:-115]
        group_df_replicates_train = group_df_replicates.iloc[train_index,:]
        group_df_replicates_validation = group_df_replicates.iloc[validation_index, :]
        group_df_replicates_test = group_df_replicates.iloc[test_index, :]

        train_input_avg,_ = average_based_on_group_df(train_env.unsqueeze(dim=-1), df=group_df_replicates_train)

        val_input_avg,_ = average_based_on_group_df(validation_env.unsqueeze(dim=-1),
                                                  df=group_df_replicates_validation)
        test_input_avg,_ = average_based_on_group_df(test_env.unsqueeze(dim=-1),
                                                    df=group_df_replicates_test)

        train_y_avg,_ = average_based_on_group_df(train_y.unsqueeze(dim=-1), df=group_df_replicates_train)
        validation_y_avg,_ = average_based_on_group_df(validation_y.unsqueeze(dim=-1),
                                                  df=group_df_replicates_validation)
        test_y_avg,_ = average_based_on_group_df(test_y.unsqueeze(dim=-1),
                                                    df=group_df_replicates_test)
        train_input_avg = torch.squeeze(train_input_avg,dim=-1).numpy()
        val_input_avg = torch.squeeze(val_input_avg, dim=-1).numpy()
        test_input_avg = torch.squeeze(test_input_avg, dim=-1).numpy()
        train_y_avg = torch.squeeze(train_y_avg,dim=-1).numpy()
        validation_y_avg = torch.squeeze(validation_y_avg, dim=-1).numpy()
        test_y_avg = torch.squeeze(test_y_avg, dim=-1).numpy()

        # print(train_input_avg.shape,train_y_avg.shape)

        training_losses = []
        validation_losses = []
        test_losses = []
        training_dtw_losses = []
        validation_dtw_losses = []
        test_dtw_losses = []

        for random_seed in range(5):
            np.random.seed(random_seed)
            random.seed(random_seed)
            print(random_seed)
            # Fit the model to the training data
            # print(train_y)
            params_fitted = fit_ode_model_with_mask_rmse(train_y.numpy(), train_env.numpy(), ts_train, y0=0.0001)
            training_rmse=0.0
            validation_rmse=0.0
            test_rmse=0.0
            shape_dtw_loss_train =0.0
            shape_dtw_loss_val =0.0
            shape_dtw_loss_test = 0.0
            #calculate rmse based on fitted parameters
            for seq in range(train_y_avg.shape[1]):
                # print(seq)
                # print(train_input_avg[:,seq].shape)

                train_env_interp = interp1d(ts_train, train_input_avg[:,seq], fill_value='extrapolate')
                # plt.scatter(x=ts_train,y=train_input_avg[:,seq])
                # plt.show()
                predict_train = solve_ode(0.0001, t=ts_train, temperature=train_env_interp, theta=params_fitted)
                training_rmse += mask_rmse(y_pred=predict_train, y_true=train_y_avg[:,seq])
                shape_dtw_loss_train += mask_dtw_loss(true_y=torch.tensor(train_y_avg[:,seq]).unsqueeze(dim=-1), predict_y=torch.tensor(predict_train).unsqueeze(dim=-1))
                # print('shape_dtw_loss_train{}'.format(shape_dtw_loss_train))
                print('train rmse')
                print(training_rmse)
                plt.plot(predict_train, color='r')
            # raise EOFError
            train_y_avg[train_y_avg == 0.0] = np.nan
            plt.ylim(0, 1.5)
            sns.scatterplot(train_y_avg)
            plt.title('Genotype {} Train RMSE:{} shapeDTW {}'.format(genotype,round(training_rmse/train_y_avg.shape[1],3),round(shape_dtw_loss_train/train_y_avg.shape[1],3)))
            # plt.savefig('../figure/temperature_ODE/Genotype {} temperature_ODE_train_.png'.format(genotype))
            # plt.show()
            plt.clf()
            train_y_avg = np.nan_to_num(train_y_avg, nan=0.0)

            for seq in range(validation_y_avg.shape[1]):
                validation_env_interp = interp1d(ts_validation, val_input_avg[:,seq], fill_value='extrapolate')
                predict_validation = solve_ode(0.0001, t=ts_validation, temperature=validation_env_interp,
                                               theta=params_fitted)
                validation_rmse += mask_rmse(y_pred=predict_validation, y_true=validation_y_avg[:,seq])
                shape_dtw_loss_val += mask_dtw_loss(torch.tensor(validation_y_avg[:,seq]).unsqueeze(dim=-1), torch.tensor(predict_validation).unsqueeze(dim=-1))
                print('shape_dtw_loss_val{}'.format(shape_dtw_loss_val))
                plt.plot(predict_validation, color='r')
            validation_y_avg[validation_y_avg == 0.0] = np.nan
            sns.scatterplot(validation_y_avg)
            plt.ylim(0, 1.5)
            plt.title('Genotype {} validation RMSE:{} shapeDTW {}'.format(genotype,round(validation_rmse/validation_y_avg.shape[1],3),round(shape_dtw_loss_val/validation_y_avg.shape[1],3)))
            # plt.savefig('../figure/temperature_ODE/Genotype {} temperature_ODE_validation_.png'.format(genotype))
            # plt.show()
            plt.clf()
            validation_y_avg = np.nan_to_num(validation_y_avg, nan=0.0)
            assert test_y_avg.shape[1] ==1
            for seq in range(test_y_avg.shape[1]):
                test_env_interp = interp1d(ts_test, test_input_avg[:,seq], fill_value='extrapolate')
                predict_test = solve_ode(0.0001, ts_test, test_env_interp, params_fitted)
                test_rmse += mask_rmse(y_pred=predict_test, y_true=test_y_avg[:,seq])
                shape_dtw_loss_test += mask_dtw_loss(torch.tensor(test_y_avg[:,seq]).unsqueeze(dim=-1), torch.tensor(predict_test).unsqueeze(dim=-1))
                # print('shape_dtw_loss_test{}'.format(shape_dtw_loss_test))
                plt.plot(predict_test, color='r')
            test_y_avg[test_y_avg == 0.0] = np.nan
            sns.scatterplot(test_y_avg)
            plt.ylim(0,1.5)
            plt.title('Genotype {} Test RMSE:{} shapeDTW {}'.format(genotype,round(test_rmse/test_y_avg.shape[1],3),round(shape_dtw_loss_test/test_y_avg.shape[1],3)))
            # plt.savefig('../figure/temperature_ODE/Genotype {} temperature_ODE_test_.png'.format(genotype))
            # plt.show()
            plt.clf()
            test_y_avg = np.nan_to_num(test_y_avg, nan=0.0)

            training_rmse = training_rmse/train_y_avg.shape[1]
            validation_rmse = validation_rmse/validation_y_avg.shape[1]
            test_rmse = test_rmse/test_y_avg.shape[1]
            training_losses.append(training_rmse)
            validation_losses.append(validation_rmse)
            test_losses.append(test_rmse)
            print(f"\nTrain RMSE: {training_rmse}")
            print(f"Validation RMSE: {validation_rmse}")
            print(f"Test RMSE: {test_rmse}")

            shape_dtw_loss_train = shape_dtw_loss_train/train_y_avg.shape[1]
            shape_dtw_loss_val = shape_dtw_loss_val/validation_y_avg.shape[1]
            shape_dtw_loss_test = shape_dtw_loss_test/test_y_avg.shape[1]
            training_dtw_losses.append(shape_dtw_loss_train)
            validation_dtw_losses.append(shape_dtw_loss_val)
            test_dtw_losses.append(shape_dtw_loss_test)
            print(f"\nTrain DTW: {shape_dtw_loss_train}")
            print(f"Validation DTW: {shape_dtw_loss_val}")
            print(f"Test DTW: {shape_dtw_loss_test}")

            new_row = pd.DataFrame(
                {'random_sees': random_seed, 'train_rMSE':training_rmse , 'validation_rMSE': validation_rmse,
                 'test_rMSE': test_rmse, 'dtw_std_train': shape_dtw_loss_train,
                'dtw_std_val': shape_dtw_loss_val, 'dtw_std_test': shape_dtw_loss_test,
                 'genotype': genotype}, index=[0])
            result_df = pd.concat([result_df, new_row])
        else:
            # Calculate average and standard deviation for RMSEs
            avg_train_rmse = np.mean(training_losses)
            std_train_rmse = np.std(training_losses)
            avg_val_rmse = np.mean(validation_losses)
            std_val_rmse = np.std(validation_losses)
            avg_test_rmse = np.mean(test_losses)
            std_test_rmse = np.std(test_losses)

            avg_train_dtw = np.mean(training_dtw_losses)
            dtw_std_train = np.std(training_dtw_losses)
            avg_val_dtw = np.mean(validation_dtw_losses)
            dtw_std_val = np.std(validation_dtw_losses)
            avg_test_dtw = np.mean(test_dtw_losses)
            dtw_std_test = np.std(test_dtw_losses)

            print(f"\nAverage Train DTW: {avg_train_dtw}, Std: {dtw_std_train}")
            print(f"Average Validation DTW: {avg_val_dtw}, Std: {dtw_std_val}")
            print(f"Average Test DTW: {avg_test_dtw}, Std: {dtw_std_test}")

            print(f"\nAverage Train RMSE: {avg_train_rmse}, Std: {std_train_rmse}")
            print(f"Average Validation RMSE: {avg_val_rmse}, Std: {std_val_rmse}")
            print(f"Average Test RMSE: {avg_test_rmse}, Std: {std_test_rmse}")
            return (avg_train_rmse,std_train_rmse,avg_val_rmse,std_val_rmse,avg_test_rmse,std_test_rmse,\
                    avg_train_dtw,dtw_std_train,avg_val_dtw,dtw_std_val,avg_test_dtw,dtw_std_test,result_df)
    def loop_through_g_fit_ode():
        average_train_errors, std_trains, average_val_errors, std_vals, average_test_errors, std_tests = [[] for _ in
                                                                                                       range(6)]
        average_train_dtws, std_train_dtws, average_val_dtws, std_val_dtws, average_test_dtws, std_test_dtws = [[] for _
                                                                                                                in
                                                                                                                range(
                                                                                                                    6)]
        result_df = pd.DataFrame()
        for g in [33, 106, 122, 133, 5, 30, 218, 2, 17, 254, 282, 294, 301, 302, 335, 339, 341, 6, 362]:
            (train_test_validation_dictionary, plant_height_tensor, temperature_same_length_tensor,
             genetics_input_tensor,group_df,group_df_replicates) = read_tensor_files(
                g)
            print(g)
            (average_train_error, std_train, average_val_error, std_val, average_test_error, std_test,
             avg_train_dtw,dtw_std_train,avg_val_dtw,dtw_std_val,avg_test_dtw,dtw_std_test,result_df) = load_data_fit_temperature_ode(
                train_test_validation_dictionary, plant_height_tensor, temperature_same_length_tensor,result_df,g,group_df_replicates)
            average_train_errors.append(average_train_error)
            std_trains.append(std_train)
            average_val_errors.append(average_val_error)
            std_vals.append(std_val)
            average_test_errors.append(average_test_error)
            std_tests.append(std_test)
            average_train_dtws.append(avg_train_dtw)
            std_train_dtws.append(dtw_std_train)
            average_val_dtws.append(avg_val_dtw)
            std_val_dtws.append(dtw_std_val)
            average_test_dtws.append(avg_test_dtw)
            std_test_dtws.append(dtw_std_test)
        else:
            result_df.to_csv('temperature_ode_fit_result_new_april.csv')
            average_train_error = np.mean(average_train_errors)
            std_train = np.mean(std_trains)
            average_val_error = np.mean(average_val_errors)
            std_val = np.mean(std_vals)
            average_test_error = np.mean(average_test_errors)
            std_test = np.mean(std_tests)
            print(f"\nAverage train RMSE across all folds: {round(average_train_error,3)} std:{std_train}")
            print(f"\nAverage Validation RMSE across all folds: {round(average_val_error,3)} std:{std_val}")
            print(f"\nAverage test RMSE across all folds: {round(average_test_error,3)} std:{std_test}")

            average_train_dtw = np.mean(average_train_dtws)
            std_train_dtw = np.mean(std_train_dtws)
            average_val_dtw = np.mean(average_val_dtws)
            std_val_dtw = np.mean(std_val_dtws)
            average_test_dtw = np.mean(average_test_dtws)
            std_test_dtw = np.mean(std_test_dtws)
            print(f"\nAverage train shapeDTW across all folds: {average_train_dtw} std:{std_train_dtw}")
            print(f"\nAverage Validation shapeDTW across all folds: {average_val_dtw} std:{std_val_dtw}")
            print(f"\nAverage test shapeDTW across all folds: {average_test_dtw} std:{std_test_dtw}")
    loop_through_g_fit_ode()


def rf_result_summary():
    average_train_errors, std_trains, average_val_errors, std_vals, average_test_errors, std_tests = [[] for _ in
                                                                                                      range(6)]
    average_train_dtws,std_train_dtws, average_val_dtws, std_val_dtws, average_test_dtws, std_test_dtws = [[] for _ in
                                                                                                      range(6)]
    result_df = pd.DataFrame()
    for g in [33, 106, 122, 133, 5, 30, 218, 2, 17, 254, 282, 294, 301, 302, 335, 339, 341, 6, 362]:
        print(g)
        train_test_validation_dictionary, plant_height_tensor, temperature_same_length_tensor, genetics_input_tensor,group_df,group_df_replicates = read_tensor_files(
            g)
        average_train_error, std_train, average_val_error, std_val, average_test_error, std_test,average_train_dtw,\
        std_train_dtw,average_val_dtw,std_val_dtw,average_test_dtw,std_test_dtw,result_df = train_rf(
            train_test_validation_dictionary, plant_height_tensor, temperature_same_length_tensor,
            genetics_input_tensor,result_df,genotype=g,group_df_replicates=group_df_replicates)

        # print(result_df)
        average_train_errors.append(average_train_error)
        std_trains.append(std_train)
        average_val_errors.append(average_val_error)
        std_vals.append(std_val)
        average_test_errors.append(average_test_error)
        std_tests.append(std_test)
        average_train_dtws.append(average_train_dtw)
        std_train_dtws.append(std_train_dtw)
        average_val_dtws.append(average_val_dtw)
        std_val_dtws.append(std_val_dtw)
        average_test_dtws.append(average_test_dtw)
        std_test_dtws.append(std_test_dtw)
    else:
        # result_df.to_csv('rf_model_result_summary.csv')
        print(len(average_train_errors))
        print(average_train_errors)
        print(std_trains)
        #something wrong, memory error?
        average_train_error = np.mean(average_train_errors)
        std_train = np.mean(std_trains)
        average_val_error = np.mean(average_val_errors)
        std_val = np.mean(std_vals)
        average_test_error = np.mean(average_test_errors)
        print("fgggggggggggggggggggggggggggggggggggg")
        print(result_df)
        # raise EOFError
        std_test = np.mean(std_tests)

        print(f"\nAverage train RMSE across all folds: {average_train_error} std:{std_train}")
        print(f"\nAverage Validation RMSE across all folds: {average_val_error} std:{std_val}")
        print(f"\nAverage test RMSE across all folds: {average_test_error} std:{std_test}")

        average_train_dtw = np.mean(average_train_dtws)
        std_train_dtw = np.mean(std_train_dtws)
        average_val_dtw = np.mean(average_val_dtws)
        std_val_dtw = np.mean(std_val_dtws)
        average_test_dtw = np.mean(average_test_dtws)
        std_test_dtw = np.mean(std_test_dtws)
        print(f"\nAverage train shapeDTW across all folds: {average_train_dtw} std:{std_train_dtw}")
        print(f"\nAverage Validation shapeDTW across all folds: {average_val_dtw} std:{std_val_dtw}")
        print(f"\nAverage test shapeDTW across all folds: {average_test_dtw} std:{std_test_dtw}")

def main():
    fit_temperature_ode_with_spicy()
    # rf_result_summary()
if __name__ == "__main__":
    main()