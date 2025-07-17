import copy
import os
import time
import glob
import numpy as np
import torch
import torch.nn as nn
import random
import pandas as pd
import torch.optim as optim
import re

import matplotlib.pyplot as plt
import seaborn as sns
import colorsys
from scipy.integrate import odeint
from scipy.optimize import least_squares
from torch.utils.data import TensorDataset, DataLoader
from matplotlib.backends.backend_pdf import PdfPages

import wandb
from DataPrepare import (create_tensor_dataset, convert_inputx_to_tesor_list, count_parameters, minmax_scaler,
                         reverse_min_max_scaling, train_test_split_based_on_group, manually_data_split_based_on_one_group)
from torchviz import make_dot
import itertools
import dill

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  #device cup or gpu
current_pwd = os.getcwd()
class LSTM_ts_and_temperature_predict_height(torch.nn.Module):
    def __init__(self, hidden_size=3, input_size=1, num_layer=3, temperature_pinn=False, smooth_out=False,
                 seq_len=None, genetics:bool|nn.Module=False):
        """only fill in seq_length when want to get different r at different time step,otherwise leave it as None"""
        super().__init__()
        self.genetics = genetics
        if genetics is None:
            if seq_len is not None:
                self.r = nn.Parameter(data=torch.full((seq_len,1,1),0.1), requires_grad=True)
            else:
                self.r = nn.Parameter(data=torch.tensor([0.1]), requires_grad=True) #define parameter r, which will be update during training
            self.y_max = nn.Parameter(data=torch.tensor([0.8]), requires_grad=True)

        self.smooth_output = smooth_out
        self.alpha = nn.Parameter(data=torch.tensor([0.8]), requires_grad=True)  # parameter for control smoothness

        self.lstm1 = nn.LSTM(input_size=(1+input_size),hidden_size=hidden_size,num_layers=num_layer).to(DEVICE) #rnn layer for time index input
        self.lstm2 = nn.LSTM(input_size=hidden_size,hidden_size=1,num_layers=1).to(DEVICE) #fully connected layer to connect output of two rnn for ts and temperature
        self.leakyrelu = nn.LeakyReLU() #active function
        if temperature_pinn:
            self.tal = nn.Parameter(data=torch.tensor([0.1]), requires_grad=True)
            self.tl = nn.Parameter(data=torch.tensor([0.1]), requires_grad=True)
            self.tah = nn.Parameter(data=torch.tensor([0.1]), requires_grad=True)
            self.th = nn.Parameter(data=torch.tensor([0.1]), requires_grad=True)
    def smooth_layer(self, x):

        time_steps = x.shape[0]
        s = x[0,:,:].unsqueeze(0)
        sx=[s]
        for t in range(1,time_steps):
            x_t = x[t,:,:].unsqueeze(0)
            s=self.alpha*x_t +(1-self.alpha)*s
            # print(s.shape)
            sx.append(s)
        else:
            smoothed_x = torch.cat(sx, dim=0)  # shape[seq_length,n_seq,feature_size]
            # print(smoothed_x.shape)
            return smoothed_x
    def forward(self,x,ts,genetics_model=None):
        #this function will automatically run when you call the object
        # print(x.shape)
        if genetics_model:
            self.genetics = genetics_model
        if self.genetics:  # if we do not link genetics information to ODE parameters
            # print('running model with genetics input')
            # print(self.genetics.r.shape)
            self.r = self.genetics.r.view(1, x.shape[1], 1)
            self.y_max = self.genetics.y_max.view(1, x.shape[1], 1)
            # print(self.r.shape)
        input_x = torch.cat([x,ts],dim=-1).float()
        out_put,c = self.lstm1(input_x) #(shape:[205,1])
        # out_put = self.leakyrelu(out_put)
        out_put,c = self.lstm2(out_put) #(shape:[205,1])

        #covert to all positive, seem does not helpped a lot
        out = self.leakyrelu(out_put)
        if self.smooth_output:
            out=self.smooth = self.smooth_layer(out)
        return out
    def init_network(self):
        # initialize weight and bias(use xavier and 0 for weight and bias separately)
        for name, param in self.named_parameters():
            if 'weight' in name:
                #https://pytorch.org/cppdocs/api/function_namespacetorch_1_1nn_1_1init_1ace282f75916a862c9678343dfd4d5ffe.html
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0001)
            #print(name, param)

class RNN_ts_predict_growth_curve(torch.nn.Module):
    '''
    this class is to define RNN model which use time index to predict biomass
    '''
    def __init__(self,num_layer_ts =3, initial_r=torch.tensor([0.29759]),initial_ymax=torch.tensor([0.59477])): #3
        super().__init__()
        # set initial value for ode parameters
        self.r = nn.Parameter(data=initial_r, requires_grad=True)
        self.y_max = nn.Parameter(data=initial_ymax, requires_grad=True)
        self.ts = nn.RNN(input_size=1,hidden_size=3,num_layers=1)
        self.ts2 = nn.RNN(input_size=3, hidden_size=1, num_layers=num_layer_ts)
        self.relu = nn.LeakyReLU()
    def forward(self,ts):

        # print(ts.shape)
        ts_out,c = self.ts(ts) #(shape:[120,3])
        ts_out, c = self.ts2(ts_out)  # (shape:[120,3])
        #covert to all positive
        ts_out = self.relu(ts_out)
        return ts_out
    def init_network(self):
        # initialize weight and bias(use xavier and 0 for weight and bias separately)
        for name, param in self.named_parameters():
            if 'weight' in name:
                # nn.init.constant_(param, 0.0046)
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
            #print(name, param)

def smoothing_spline(y,t):
    from scipy.interpolate import splrep, BSpline
    assert len(y) == len(t)
    #drop where y = na

    tck = splrep(t, y, s=0,k=3)
    spline = BSpline(*tck)
    smoothed_y = spline(t)
    smoothed_y = torch.tensor(smoothed_y)
    return smoothed_y

def smooth_tensor_ignore_nan(tensor, window_size):
    """
    Smooth a PyTorch tensor along the first dimension using a moving average,
    ignoring NaN values.

    Parameters:
        tensor (torch.Tensor): Input tensor of shape (N, *, *).
        window_size (int): Size of the smoothing window.

    Returns:
        torch.Tensor: Smoothed tensor with the same shape as the input tensor.
    """
    # Convert to numpy for processing
    tensor_np = tensor.numpy()
    # print(tensor_np)
    # print(tensor_np.shape)
    seq_length, seq_count, feature_size = tensor_np.shape

    # Padding to handle edge cases
    padding = window_size // 2
    padded_data = np.pad(
        tensor_np,
        ((padding, padding), (0, 0), (0, 0)),
        mode='constant',
        constant_values=np.nan,
    )

    # Output array
    smoothed_np = np.empty_like(tensor_np)

    # Perform moving average along the first dimension
    for t in range(seq_length):
        start_idx = max(0, t - window_size + 1)
        end_idx = t + 1  # Include till the current time step
        window = padded_data[start_idx:end_idx]  # Extract window
        valid_mask = ~np.isnan(window)  # Mask valid (non-NaN) values
        valid_counts = valid_mask.sum(axis=0)  # Count valid values in each feature
        valid_sums = np.nansum(window, axis=0)  # Sum valid values (ignoring NaN)
        smoothed_np[t] = np.where(valid_counts > 0, valid_sums / valid_counts, np.nan)  # Avoid division by zero

    # Convert back to tensor
    return torch.tensor(smoothed_np, dtype=tensor.dtype, device=tensor.device)

def train_logistic_model_simulated(model, Y, ts_x, epoches, lr=0.01, weight=2,parameter_boundary=False):
    '''
    This is the function to train neural network for simulated biomass prediction, the real-time plotting is comment out
    parameters:
        model: object, neural network model
        Y:torch.tensor, simulated time seriers(biomass) data
        ts_x: torch.tensor, time index input range(0.1,12)
        parameter_boundary: boolean, define whether add extra penalty to negative r value or not
    return:
        model_return, trained model
        epoch+1, the numebr of epochs used for training
        loss_ratio: pd.Datafrome of dataloss and physics loss
        parameter_changing_during_training:pd.Datafrome of r and ymax changing_during_training
    '''
    print(Y.shape)
    samples_num = Y.shape[1] #the second demension is number of sequences(sample size)
    print('number of seq {}'.format(samples_num))
    #define optimizer for model training
    optimiser = optim.Adam(model.parameters(), lr=lr)
    #set to train mode
    model.train()
    #assign to a very large starting value, which is useful for ealy stop to calculate loss changes at every epoch
    losses = [999]

    model_dict = {}#a dictionary which temproally save trained model, thus we can return previous model when training stop

    '''
    #define plot, which can automatically update during training
    plt.ion()
    fig = plt.figure()
    plt.ylim(-1,2)
    plt.xlim(0, 284)
    plt.show()

    colors = _get_colors(samples_num) #get same number of unique colors(rgb value) as sample number)
    print(colors)
    '''
    #define list to save loss and parameter changes during training
    physic_loss_list = []
    data_loss_list = []
    y_max_list = []
    r_list=[]
    print('shape of ts x:{}'.format(ts_x.shape)) #this is inputX

    for epoch in range(epoches):
        #print(y.shape)
        # print(x.shape)
        #with torch.backends.cudnn.flags(enabled=False):
        predict_y = model(ts_x)#only use time serie as input (time steps)
        # print('Y shape:{}'.format(Y.shape))

        dYpre_dt = grad_logistic_ode(predict_y, ts_x)[0]  # get the gradient (predict_y with respect to ts_x) tensor out of tuple

        y_max_list.append(model.y_max.item())
        r_list.append(model.r.item())
        # print('y max:{}'.format(model.y_max))

        #derivative calculate based on equation
        dY_dt = model.r * predict_y * (1 - (predict_y / model.y_max))  # dY/dt = r*Yt*(1-Yt/Ymax) logistic ODE
        # print('derivative shape:{}'.format(dY_dt.shape))
        physics_loss = torch.mean((dYpre_dt - dY_dt) ** 2)  # logistic ode
        data_loss = mask_rmse_loss(true_y=Y, predict_y=predict_y)

        if parameter_boundary:
            loss =data_loss + weight*physics_loss + (0.001**copy.deepcopy(model.r.detach()))*0.1
            #weigted physic loss
            physics_loss_weighted = (copy.deepcopy(data_loss.item()) * physics_loss) / copy.deepcopy(physics_loss.item())
            # print(physics_loss_weighted)
        else:
            physics_loss_weighted = (copy.deepcopy(data_loss.item()) * physics_loss) / copy.deepcopy(physics_loss.item())
            # print(physics_loss_weighted)
            loss = data_loss + weight*physics_loss

            # if data_loss.item()<0.0001: #this part of the code is becuase i tried to change the weight during training
            #     loss = weight *data_loss + physics_loss
            # else:
            #     loss = data_loss + weight * physics_loss

        loss.backward() #backward propergation
        optimiser.step() #Performs a single optimization step (parameter update).
        physic_loss_list.append(physics_loss_weighted.item()) #take value from the tensor
        data_loss_list.append(data_loss.item())

        parameter_changing_during_training = pd.DataFrame(
            {'ymax': y_max_list, 'r': r_list, 'data_loss': data_loss_list, 'physic_loss': physic_loss_list},
            index=range(len(r_list)))#save in dataframe


        loss_ratio = pd.DataFrame({'data_loss': data_loss_list, 'physic_loss': physic_loss_list},
                                  index=range(len(data_loss_list)))

        if (epoch + 1) % 10 == 0:
            #print at every 10 epochs
            print(f"Epoch {epoch+1}/{epoches}, last loss: {losses[-1]:.4f}")
            print('current loss:{}'.format(loss))
            print_parameters_from_ode(model) #print current r and ymax
            losses.append(loss.item())
            model_dict['{:.6f}'.format(loss.item())] = copy.deepcopy(model)

        '''
        #plot predicted growth curve
        plt.clf()
        for seq_id, color in zip(range(samples_num), colors):
            # convert 0 to nan, so it will not be plot
            # some explaination about detach() https://www.d2l.ai/chapter_preliminaries/autograd.html#detaching-computation
            # Even we detached pred_y, the computational graph leading to y persists, thus we can still calculate the gradient of y with respect to x
            true_y_np = copy.deepcopy(Y.detach()).numpy()


            sns.scatterplot(true_y_np,color=color)
            pred_y_np = copy.deepcopy(predict_y.detach()).numpy()
            sns.lineplot(pred_y_np, linestyle='dashed', color=color)

        # plt.ylim(-1,2)
        # plt.xlim(60, 285)
        plt.show()
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        '''
    else:
        model_return = model_dict['{:.6f}'.format(min(losses))]
        #figure =plot_loss_change(losses, losses,test_losses_plot,name='log_transformed_loss_change')
        return model_return,epoch+1,loss_ratio,parameter_changing_during_training
def _get_colors(n):
    '''
    function of return n colors in list (for plot)
    '''
    num_colors = n
    colors = []
    random.seed(0)
    for i in np.arange(0.0, 360.0, 360.0 / num_colors):
        hue = i / 360.0
        lightness = (50 + np.random.rand() * 10) / 100.0
        saturation = (90 + np.random.rand() * 10) / 100.0
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors
def evidenvce_based_early_stop():
    #based on paper: https://arxiv.org/abs/1703.09580
    """
    This method is to implement the evidence based early stopping cirterira, it may give slightly worse performance for
    multilayer neural networks, when the convergance speed of parameters are different.
    """

def train_model(model, train_env, Y, ts_X, epochs, validation_env, validation_y, ts_validation, lr=0.01,
                physic_loss_weight=2, parameter_boundary='', pinn=True, penalize_negetive_y='penalize_neg_y',
                fit_ml_first: bool | int = False, multiple_r=False, smooth_training=False,
                l2: bool | float = False, ode_intergration_loss=False, temperature_ode=False, test_env=None,
                test_y=None,
                ts_test=None, genetics_model: nn.Module = None, genetics_Train_year: torch.tensor = None,
                genetics_validation_year: torch.tensor = None, genetics_test_year: torch.tensor = None,
                scaler_env: dict = None, genotype_list=[],batch_size=88):

    """
    ML model training, similar to 'train_logistic_model_simulated', but for ETH data
    parameters:
        model: object, neural network model
        X: torch.tensor, training set temperature time seriers
        Y:torch.tensor, training set time seriers(plant height) data
        ts_x: torch.tensor, training set time index input [205,n_samples,1], value from 0 to 204
        epochs: int, number of epochs used for training
        validation_env: torch.tensor, validation set temperature time series
        validation_y: validation set plant height
        ts_validation: training set time index input [205,n_samples,1]
        lr: learning rate, when use Adam, it is the starting learning rate
        physic_loss_weight: assigned extra weight to physics loss (multiplication)
        parameter_boundary: boolean, define whether add extra penalty to negative r value or not
        pinn: boolean,define wether use physics loss or not
        fit_ml_first: boolean or int. if False, the run with normal pinn mode, if int, run pinn after <int> epoch, first train with pure ml
        l2: boolean or float, wheter apply l2 regulization or not, if yes, then pass an float number as lambda
    return:
        model_return, trained model
        epoch+1, the numebr of epochs used for training
        figure: loss changing figure, later save in pdf
        loss_ratio: pd.Datafrome of dataloss and physics loss
    """
    from genetic_embedding import distance_criterion
    samples_num = Y.shape[1]

    print('number of seq {}'.format(samples_num))
    # if smooth_training:
    #     smooth_y_list = []
    #     for i in range(samples_num):
    #         seq = y[:,i,:]
    #         ts = ts_x[:i,:]
    #         smooth_y = smoothing_spline(seq,ts)
    #         smooth_y_list.append(smooth_y)
    #     else:
    #         y = torch.cat(smooth_y_list,dim=1)
    #         print()
    model.init_network()
    optimiser_time = optim.Adam(model.parameters(), lr=lr)

    #optimiser_time = optim.SGD(model.parameters(), lr=lr, weight_decay=l2)
    model.train()
    if genetics_model:
        optimiser_snp = optim.Adam(genetics_model.parameters(), lr=lr)
        genetics_model.train()
        Train_id, validation_id, test_id = genotype_list[0],genotype_list[1],genotype_list[2]
    losses = []
    validation_losses = []#start with a very large value, not useful anymore, 999 is remove from the first place
    validation_losses_plot = []
    train_losses_plot=[]
    test_losses_plot = []
    model_dict = {}

    # plt.ion()
    # fig = plt.figure()

    """
    #draw plot
    plt.ion()
    fig = plt.figure()
    plt.ylim(-1,2)
    plt.xlim(0, 284)
    plt.show()
    colors = _get_colors(samples_num)
    print(colors)
    """
    physic_loss_list = []
    data_loss_list = []
    #dataloader
    if genetics_model:
        print(Train_id.shape)
        # print(X.shape, y.shape, ts_x.shape, genetics_train_year.shape)
        # Train_id = torch.tensor(Train_id.values.astype(float))
        # print(Train_id.shape)
        dataset = TensorDataset(train_env.permute(1, 0, 2), Y.permute(1, 0, 2), ts_X.permute(1, 0, 2), genetics_Train_year,
                                Train_id)
        loader = DataLoader(
            dataset,
            batch_size=batch_size
        )
    else:
        # print(X.shape, y.shape, ts_x.shape, genetics_train_year.shape)
        dataset = TensorDataset(train_env.permute(1, 0, 2), Y.permute(1, 0, 2), ts_X.permute(1, 0, 2))
        loader = DataLoader(
            dataset,
            batch_size=batch_size
        )
    # time1=time.time()
    wandb.watch(model, log="all",log_freq=10)
    # print('watch time:{}'.format(time.time()-time1))
    # train_year,validation_year,test_year = year_list[0],year_list[1],year_list[2]
    for epoch in range(epochs):
        # for seq in range(samples_num):
        if genetics_model:
            loss =0.0
            physics_loss =0.0
            data_loss=0.0
            for batch_idx, (X, y, ts_x, genetics_train_year,train_id) in enumerate(loader):
                x = X.permute(1,0,2).requires_grad_(True)
                y = y.permute(1, 0, 2)
                ts_x = ts_x.permute(1, 0, 2).requires_grad_(True)
                # print(X.shape, y.shape, ts_x.shape, genetics_train_year.shape,train_id)
                optimiser_time.zero_grad()
                optimiser_snp.zero_grad()
                out_vector = genetics_model(genetics_train_year)
                genotype_seperation_loss_train = distance_criterion(out_vector, train_id)
                loss_batch, pred_y, (physic_loss_batch, data_loss_batch) = combined_with_physics_loss(y, model, x, ts_x,
                                                                                    weight=physic_loss_weight,
                                                                                    parameter_boundary=parameter_boundary,
                                                                                    combine_physics_loss=pinn,
                                                                                    penalize_negetive_y=penalize_negetive_y,
                                                                                    multiple_r=multiple_r,
                                                                                    l2_regulization=l2,
                                                                                    ode_intergration_loss=ode_intergration_loss,
                                                                                    temperature_ode=temperature_ode,
                                                                                    scaler_env=scaler_env)#,fig=fig)

                total_loss = 0.5*genotype_seperation_loss_train + loss_batch
                # print('genetics loss:{}'.format(genotype_seperation_loss_train))
                # # print(time.time())
                total_loss.backward()
                # print(time.time())
                # print(loss_batch)
                # optimiser_snp.step()
                # loss_batch.backward()
                optimiser_time.step()  # Performs a single optimization step (parameter update).
                # print(time.time())
                loss = loss+loss_batch
                physic_loss = physic_loss_batch+physics_loss
                data_loss = data_loss_batch+data_loss
            else:
                loss_average = loss.item()/(batch_idx+1)
                # print(batch_idx)
                # print(loss_average)
                losses.append(loss_average)
                physic_loss_average = physic_loss.item()/(batch_idx+1)
                physic_loss_list.append(physic_loss_average)
                data_loss_average = data_loss.item()/(batch_idx+1)
                data_loss_list.append(data_loss_average)
                loss_ratio = pd.DataFrame({'data_loss': data_loss_list, 'physic_loss': physic_loss_list},
                                          index=range(len(data_loss_list)))
                with torch.no_grad():
                    model.eval()
                    genetics_model.eval()
                    genetics_model(genetics_Train_year)
                    train_pred_Y = model(train_env, ts_X)
                    genetics_model(genetics_validation_year)
                    # with torch.backends.cudnn.flags(enabled=False): #this is because of an error: NotImplementedError: the derivative for '_cudnn_rnn_backward' is not implemented.
                    validation_pred_y = model(validation_env, ts_validation)

                    genetics_model(genetics_test_year)
                    test_pred_y = model(test_env, ts_test)
                    model.train()
                    genetics_model.train()
                # print('validation y shape:{}'.format(validation_y.shape))
                validation_loss = mask_rmse_loss(true_y=validation_y, predict_y=validation_pred_y).item()
                validation_losses_plot.append(validation_loss)
                test_loss = mask_rmse_loss(true_y=test_y, predict_y=test_pred_y).item()
                test_losses_plot.append(test_loss)
                train_loss = mask_rmse_loss(true_y=Y, predict_y=train_pred_Y).item()
                train_losses_plot.append(train_loss)

                if (epoch + 1) % 10 == 0:
                    # save log every 10 epochs
                    if epoch>=1500:
                        validation_losses.append(validation_loss)
                        model_dict[str(validation_loss)] = copy.deepcopy(model)
                    time1 = time.time()
                    wandb.log(
                        {"physics_loss": physic_loss_average, "data_loss": data_loss_average,
                         "validation_loss": validation_loss,
                         "test_loss": test_loss,'model.r':model.r,'model.y_max':model.y_max})
                    print(time.time()-time1)
                    # print loss every 10 epochs
                    print(f"Epoch {epoch + 1}/{epochs}, loss: {losses[-1]:.4f}")
                    # print_parameters_from_ode(model)
                    print('validation loss:{}'.format(validation_loss))
        else:
            for batch_idx, (X, y, ts_x) in enumerate(loader):
                x = X.permute(1, 0, 2).requires_grad_(True)
                y = y.permute(1, 0, 2)
                ts_x = ts_x.permute(1, 0, 2).requires_grad_(True)
                # print(X.shape, y.shape, ts_x.shape, genetics_train_year.shape)
                # set gradient to zero after every epoch update parameter
                optimiser_time.zero_grad()

                if (fit_ml_first==False) or (epoch>fit_ml_first):
                    loss, pred_y, (physic_loss, data_loss) = combined_with_physics_loss(y, model, x, ts_x,
                                                                                        weight=physic_loss_weight,
                                                                                        parameter_boundary=parameter_boundary,
                                                                                        combine_physics_loss=pinn,
                                                                                        penalize_negetive_y=penalize_negetive_y,
                                                                                        multiple_r=multiple_r,
                                                                                        l2_regulization=l2,
                                                                                        ode_intergration_loss=ode_intergration_loss,
                                                                                        temperature_ode=temperature_ode,
                                                                                        scaler_env=scaler_env)  # ,fig=fig)
                else:
                    # print('pure ml')
                    loss, pred_y, (physic_loss, data_loss) = combined_with_physics_loss(y, model, x, ts_x,
                                                                                        weight=physic_loss_weight,
                                                                                        parameter_boundary=parameter_boundary,
                                                                                        combine_physics_loss=False, penalize_negetive_y=penalize_negetive_y,
                                                                                        multiple_r=multiple_r, l2_regulization=l2,
                                                                                        ode_intergration_loss=ode_intergration_loss,temperature_ode=temperature_ode)#,fig=fig)
                loss.backward()
                optimiser_time.step()  # Performs a single optimization step (parameter update).
                losses.append(loss.item())
                if pinn and ((epoch>fit_ml_first) or fit_ml_first==False):
                    physic_loss_list.append(physic_loss.item())
                else:
                    physic_loss_list.append(physic_loss)
                data_loss_list.append(data_loss.item())
                # loss_ratio = pd.DataFrame({'data_loss': data_loss_list, 'physic_loss': physic_loss_list},
                #                           index=range(len(data_loss_list)))

                with torch.no_grad():
                    model.eval()
                    if pinn:
                        #with torch.backends.cudnn.flags(enabled=False): #this is because of an error: NotImplementedError: the derivative for '_cudnn_rnn_backward' is not implemented.
                        validation_pred_y = model(validation_env, ts_validation)
                        test_pred_y = model(test_env, ts_test)
                    else:
                        validation_pred_y = model(validation_env, ts_validation)
                        test_pred_y = model(test_env, ts_test)
                    model.train()
                # print('validation y shape:{}'.format(validation_y.shape))
                validation_loss = mask_rmse_loss(true_y=validation_y, predict_y=validation_pred_y).item()
                validation_losses_plot.append(validation_loss)
                test_loss = mask_rmse_loss(true_y=test_y, predict_y=test_pred_y).item()
                test_losses_plot.append(test_loss)
                train_loss = mask_rmse_loss(true_y=y, predict_y=pred_y).item()
                train_losses_plot.append(train_loss)
                wandb.log({"physics_loss": physic_loss, "data_loss": data_loss, "validation_loss": validation_loss,
                           "test_loss": test_loss, 'model.r': model.r.item(), 'model.y_max': model.y_max.item()})
                if (epoch + 1) % 10 == 0:
                    #print every 10 epochs
                    if epoch>=1500:
                        validation_losses.append(validation_loss)
                        model_dict[str(validation_loss)] = copy.deepcopy(model)

                    #print loss every 10 epochs
                    print(f"Epoch {epoch+1}/{epochs}, loss: {losses[-1]:.4f}")
                    # print_parameters_from_ode(model)
                    print('validation loss:{}'.format(validation_loss))
                    # if epoch>=200 and ((validation_loss-min(validation_losses))>0.003) and (min(validation_losses)<=0.01): #0.005
                    #     #early stop, but currently not use
                    #     print('validation loss change:{}'.format((validation_loss-validation_losses[-1])))
                    #     # model_return = model_dict[str(min(validation_losses))]
                    #     print_parameters_from_ode(model)
                    #
                    #     # figure = plot_loss_change(losses,validation_losses_plot,test_losses_plot,name='loss_change')
                    #     print('total loss, physic loss')
                    #     print(loss, loss - mask_rmse_loss(true_y=y, predict_y=pred_y))
                    #     # return model_return,epoch+1,figure,loss_ratio
                    # else:
                    #     continue
                    #     # print_parameters_from_ode(model)


                #plot predicted growth curve
                """
                plt.clf()
                for seq_id, color in zip(range(samples_num), colors):
                    # convert 0 to nan, so it will not be plot
                    # some explaination about detach() https://www.d2l.ai/chapter_preliminaries/autograd.html#detaching-computation
                    # Even we detached pred_y, the computational graph leading to y persists and thus we can calculate the gradient of y with respect to x
                    true_y_np = copy.deepcopy(y.detach())[:, seq_id, 0].numpy()
                    true_y_np[true_y_np== 0.0] = np.nan
                    sns.scatterplot(true_y_np,color=color)
                    pred_y_np = copy.deepcopy(pred_y.detach())[:, seq_id, 0].numpy()
                    # pred_y_np[pred_y_np == 0.0] = np.nan
                    sns.lineplot(pred_y_np, linestyle='dashed', color=color)
                    # sns.lineplot(torch.squeeze(X[:, seq_id, 0].unsqueeze(dim=1)).detach(), linestyle='dashed',
                    #              color=color)
                plt.ylim(-0.2,1)
                plt.xlim(0, 285)
                plt.show()
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                """
    else:
        model_return = model_dict[str(min(validation_losses))]
        figure =plot_loss_change(train_losses_plot, validation_losses_plot,test_losses_plot, name='loss_change')
        if genetics_model:
            genetics_model(genetics_Train_year)
            print(genetics_model.r.shape)
        print(train_env.shape)

        predict_y = model_return(train_env,ts_X,genetics_model)
        dYpre_dt = grad_logistic_ode(predict_y, ts_X)[0] # get the gradient tensor out of tuple [seq_length,n_seq,feature_size]
        dypre_dtemp = grad_logistic_ode(predict_y, train_env)[0] #get gradient with_respect_to_temperature
        dY_dt = model_return.r * predict_y * (1 - (predict_y / model_return.y_max)) #dY/dt = r*Yt*(1-Yt/Ymax) logistic ODE
        mask =(~torch.isin(Y, torch.tensor(0.0).to(DEVICE))).float()
        fig_grad = plot_auto_grad_and_lossagainst_traiining_data(dY_dt=dY_dt,dYpre_dt=dYpre_dt,grad_resp_temp=dypre_dtemp,mask=mask,predict_y=predict_y,true_y=y)#,fig=fig)
        # plt.clf()
        epoch_num = list(model_dict.keys()).index(str(min(validation_losses)))
        if genetics_model:
            return (model_return,genetics_model),epoch_num*10,figure,dypre_dtemp,fig_grad
        else:
            return model_return, (epoch_num * 10 +1500), figure, dypre_dtemp, fig_grad

def mask_rmse_loss(true_y:torch.tensor, predict_y:torch.tensor):
    '''
    calculate mask where (which time step) inout value is zero
    https://discuss.pytorch.org/t/how-does-applying-a-mask-to-the-output-affect-the-gradients/126520/2
    '''
    device = true_y.device
    mask = (~torch.isin(true_y, torch.tensor(0.0).to(device))).float()
    mask_rmse_loss_value = (torch.sum(((true_y - predict_y) * mask) ** 2) / torch.count_nonzero(mask))**0.5
    # mask_rmse_loss_value = ((true_y - predict_y) * mask) ** 2

    # print(mask_rmse_loss_value.shape)
    return mask_rmse_loss_value#,torch.count_nonzero(mask)


# def mask_r2_loss(true_y: torch.tensor, predict_y: torch.tensor):
#     '''
#     Calculate R-squared (R²) for the masked true and predicted values using torchmetrics.
#     '''
#     device = true_y.device
#
#     # Create the mask to ignore values where true_y is zero
#     mask = (~torch.isin(true_y, torch.tensor(0.0).to(device))).float()
#
#     # Mask the true and predicted values
#     true_y_masked = true_y * mask
#     predict_y_masked = predict_y * mask
#
#     r2_metric = R2Score()
#     # Calculate R²
#     r_squared = r2_metric(torch.squeeze(predict_y_masked), torch.squeeze(true_y_masked))
#
#     return r_squared
def mask_MAPE_loss(true:torch.tensor,pred:torch.tensor):
    # Define a small epsilon to prevent division by zero
    epsilon = 1e-8
    device = true.device
    # Create a mask to ignore time points where true values are zero(zero are used to indicated nan
    mask = true != 0.0
    mask = mask.to(device)
    abs_percentage_error = torch.abs((true[mask] - pred[mask]) / (true[mask] + epsilon))

    # Calculate the mean dtw over the selected non-zero values
    mape = torch.mean(abs_percentage_error)
    return mape
def grad_logistic_ode(outputs, inputs)->tuple:
    #  An detailed explaination of torch.autograd.grad https://medium.com/@rizqinur2010/partial-derivatives-chain-rule-using-torch-autograd-grad-a8b5917373fa
    """Computes the partial derivative of tensor outputs with respect to tensor inputs.
        for instance: y=f(x), then inputs is x, outputs is y
    return: dy/dx torch.tensor
    """
    # import traceback
    # try:
    #     print(inputs.grad.shape) #tensor.grad will print gradient with respect to this tensor
    # except AttributeError:
    #     traceback.print_exc()

    # grad_outputs https://discuss.pytorch.org/t/what-is-the-grad-outputs-kwarg-in-autograd-grad/94378
    # i was plan to undo scaling when calculate gradient, while it give an error
    # RuntimeError: One of the differentiated Tensors appears to not have been used in the graph. Set allow_unused=True if this is the desired behavior.
    return torch.autograd.grad(
        outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True,is_grads_batched=False
    ) # create_graph =True, graph of the derivative will be constructed, allowing to compute higher order derivative products.
    # So here it is used later in loss.backward() (or manually_backward(loss, model))for updating parameters
def combined_with_physics_loss(true_y, model, x, ts, weight=2, parameter_boundary='', combine_physics_loss=True,
                                   penalize_negetive_y='', temperature_ode=False, multiple_r=False,
                                   l2_regulization=False, ode_intergration_loss=False, fig=None,scaler_env:dict=None):

    """The physics loss of the model, which are designed to constrain the growth curve follows logistic growth
    parameters:
    -model: neural network model, where y_max and r are parameters from model
    -ts: sequences of time steps in time series: [0,1,2,..]
    -x: environment_factors series
    -combine_physics_loss: boolean, use physics informed loss function when it's Ture, otherwise train pure ML model for 
                           comparison.
    """
    # def logistic_growth_ode(t,y):
    #     # data generate from logistic growth model
    #     r,y_max=model.r,model.y_max
    #     dY_dT = r * y * (1 - (y / y_max))
    #     return dY_dT

    # from torchdiffeq import odeint
    #
    # mask = (~torch.isin(true_y, torch.tensor(0.0).to(DEVICE))).float()
    # print('number of time steps used in loss calculation: {}'.format(torch.count_nonzero(mask)))
    # get the gradient
    if combine_physics_loss:
        # predict y with neural network
        #with torch.backends.cudnn.flags(enabled=False):
        predict_y = model(x, ts)
        # print(predict_y)
        # predict_y.to(DEVICE)
        dYpre_dt = grad_logistic_ode(predict_y, ts)[0] # get the gradient tensor out of tuple
        # print(dYpre_dt)
        # if temperature is rescaled, then multiply it with maximum temeprature

        dypre_dtemp = grad_logistic_ode(predict_y,x)[0] #gradient of y with respect to temperature
        # print(dYpre_dt.shape) #seq_length,n_seq,feature_size

        # compute physical loss MSE(dY_predict/dt - dY/dt), which dY/dt is calculated based on ode function
        # (this is where physics information come into the model).
        if temperature_ode ==False:
            # print(model.r.shape)
            # print(predict_y.shape)
            # print((model.r.repeat(1,predict_y.shape[1],predict_y.shape[2]) * predict_y).shape)
            dY_dt = model.r * predict_y * (1 - (predict_y / model.y_max))
            # dY_dt_true_y = model.r * true_y * (1 - (true_y / model.y_max))*mask #dY/dt = r*Yt*(1-Yt/Ymax) logistic ODE
            # #!!ture of predicted y? true y only has limited data points, while predicted y can be wrong at first and drive the result to be even worse
            # print(dY_dt.shape)
        else:
            # there is one problem, the parameters is this function is too large, the unite for temperature her is k
            #try to fix some parameters like th and tah?
            f_t = (1+ torch.exp((model.tal/(x*scaler_env['Air_temperature_2_m']+273.15)) - (model.tal/model.tl)) + torch.exp((model.tah/model.th) - (model.tah/(x*scaler_env['Air_temperature_2_m']+273.15))))**-1
            dY_dt = model.r * predict_y *f_t* (1 - (predict_y / model.y_max))  # dY/dt = r*f_T(t)*Yt*(1-Yt/Ymax) temperature ODE
            # first dimension should have the same length to perform element-wise multiplication
            # print('temperature ode')
            # print(dY_dt.shape)
        data_loss = mask_rmse_loss(true_y, predict_y)
        # print('dataloss shape:{}'.format(data_loss.shape))
        physics_loss = torch.mean((dYpre_dt - dY_dt) ** 2)  # logistic ode
        # print(physics_loss.shape)
        # plt.plot(((dYpre_dt - dY_dt) ** 2).detach())
        # plt.show()
        # fig_plot = plot_auto_grad_and_lossagainst_traiining_data(dY_dt, dYpre_dt,dypre_dtemp, mask, predict_y, true_y,fig)

        # initial_constrain_loss = torch.mean((predict_y[0,:,:] - torch.full(predict_y[0,:,:].shape, 0.001).to(DEVICE))**2)
        # print(initial_constrain_loss)
        loss = data_loss + weight * physics_loss #+ode_intergrate_loss#+ initial_constrain_loss
        # if ode_intergration_loss:
        #     #intergarate from the minimum value
        #     y0=torch.min(true_y,dim=0).values
        #     # print(y0.shape) #torch.Size([4, 1])
        #     # if any zero (or negetive, which should't in the dataset) value, use 0.0001
        #     y0[y0<=0.0] = torch.tensor([0.0001])
        #     # ts_ode = torch.squeeze(copy.deepcopy(ts))
        #
        #     try:
        #         ode_intergrate = odeint(logistic_growth_ode, y0, torch.squeeze(ts[:, 0, :]))#.repeat(1,4,1)
        #     except:
        #         print(model.r, model.y_max)
        #         raise
        #     # print('intergration loss')
        #     # print(ode_intergrate.shape)
        #     # print('true y shape:{}'.format(true_y.shape))
        #     ode_intergrate_loss = torch.mean(mask*(ode_intergrate - true_y)**2)
        #     #print(ode_intergrate_loss)
        #     loss = loss + ode_intergrate_loss
        # print(loss)
        if parameter_boundary != '':
            # print('penalize negetive r')
            if multiple_r:
                # print('multiple r')
                #for multiple r, penalize average r value
                loss = loss + torch.mean(0.1 * (0.00001 ** model.r))
                # print(loss)
            else:
                loss = loss + torch.mean(0.1*(0.00001**model.r))
            # print(loss)
        if penalize_negetive_y != '':
            p = torch.tensor([200.0]).to(DEVICE)
            # print(loss)
            loss = loss + torch.mean(torch.exp(-p*predict_y + 1/p))
            # print(loss)

        if l2_regulization !=False:
            number_weights = 0
            for name, weights in model.named_parameters():
                if 'bias' not in name:
                    number_weights = number_weights + weights.numel()
            # Calculate L2 term
            L2_term = torch.tensor(0., requires_grad=True)
            for name, weights in model.named_parameters():
                if 'bias' not in name:
                    weights_sum = torch.sum(weights**2)
                    L2_term = L2_term + weights_sum
            # print(number_weights)
            L2_term = L2_term / number_weights # weight**2/number weights
            # print(L2_term * l2_regulization)
            # loss + L2 regularization
            loss = loss + L2_term * l2_regulization
        #get computational graph
        # make_dot(loss).render('cg_loss.png', format="png") # This explain computational graph from torchvis https://weiliu2k.github.io/CITS4012/pytorch/computational_graph.html
        # make_dot(loss).view() #currently work in my sub linux system on windows, has some error for pycharm !! graphviz.backend.execute.ExecutableNotFound: failed to execute WindowsPath('dot'), make sure the Graphviz executables are on your systems' PATH
        # time.sleep(500)
        # raise EOFError
        return loss, predict_y, (weight * physics_loss, mask_rmse_loss(true_y, predict_y))
    else:
        '''if combine_physics_loss=False, become pure ML model, train on masked MSE loss'''
        predict_y = model(x, ts)
        predict_y.to(DEVICE)
        # print('pure ML model')
        loss=mask_rmse_loss(true_y, predict_y)
        return loss,predict_y,(np.nan, mask_rmse_loss(true_y, predict_y))


def plot_auto_grad_and_lossagainst_traiining_data(dY_dt, dYpre_dt,grad_resp_temp, mask, predict_y, true_y,fig=None): #


    fig,ax = plt.subplots(figsize=(12, 10))
    data_loss_plot = copy.deepcopy(((true_y.detach() - predict_y.detach()) * mask) ** 2)[:, 0, 0].cpu()
    plot_true_y = copy.deepcopy(true_y.detach())[:, 0, 0].cpu() / 10
    plot_true_y[plot_true_y == 0.0] = np.nan
    plot_true_y2 = copy.deepcopy(true_y.detach())[:, 2, 0].cpu() / 10
    plot_true_y2[plot_true_y2 == 0.0] = np.nan
    plot_temperature_gradient = copy.deepcopy(grad_resp_temp.detach())[:, 0, 0].cpu()
    plot_temperature_gradient[plot_temperature_gradient== 0.0] = np.nan

    # # print(true_y)
    # # print(predict_y)
    # # print('shape physic loss:{}'.format((mask*(dYpre_dt - dY_dt)**2).shape))
    phy_loss = ((copy.deepcopy(dYpre_dt.detach()) - copy.deepcopy(dY_dt.detach())) ** 2)[:, 0, 0].cpu() * 100 #mask *
    # print(phy_loss.shape)
    # print(predict_y.detach()[:, 0, 0].shape)
    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                      '#f781bf', '#a65628', '#984ea3',
                      '#999999', '#e41a1c', '#dede00']
    ax.plot(copy.deepcopy(copy.deepcopy(dYpre_dt.detach()))[:, 0, 0].cpu(), c='#377eb8', label='auto.grad')
    # plt.plot(copy.deepcopy(ode_int.detach())[:, 0, 0]/10, c='lime', label='ode intergrated')
    ax.plot(plot_temperature_gradient, c='#ff7f00', label=' y gradient with respect to temperature')
    ax.plot(copy.deepcopy(dY_dt.detach())[:, 0, 0].cpu(), c='#4daf4a', label='dy/dt from ode')
    ax.plot(copy.deepcopy(predict_y.detach())[:, 0, 0].cpu() / 10, c='#f781bf', label='predicted y/10 1')
    # plt.plot(copy.deepcopy(predict_y.detach())[:, 2, 0] / 10, c='darkred', label='predicted y/10 2',ax=ax)
    sns.scatterplot(plot_true_y, c='#a65628', label='true y/10 seq1',ax=ax)
    sns.scatterplot(plot_true_y2, c='#984ea3', label='true y/10 seq2',ax=ax)
    ax.plot(phy_loss, c='black', label='physic loss*100')
    ax.plot(data_loss_plot, c='#999999', label='data loss')
    # plt.ylim(-0.1,0.1)
    ax.legend()
    ax.set_ylim(-0.01, 0.15)
    ax.set_title('PINN model gradient and loss after training')
    plt.savefig('derivative_against_predict.svg')

    '''
    #This is for plotting while training
    plt.plot(copy.deepcopy(copy.deepcopy(dYpre_dt.detach()))[:, 0, 0].cpu(), c='#377eb8', label='auto.grad')
    # plt.plot(copy.deepcopy(ode_int.detach())[:, 0, 0]/10, c='lime', label='ode intergrated')
    # plt.plot(plot_temperature_gradient, c='#ff7f00', label=' y gradient with respect to temperature')
    plt.plot(copy.deepcopy(dY_dt.detach())[:, 0, 0].cpu(), c='#4daf4a', label='dy/dt from ode')
    plt.plot(copy.deepcopy(predict_y.detach())[:, 0, 0].cpu() / 10, c='#f781bf', label='predicted y/10 1')
    # plt.plot(copy.deepcopy(predict_y.detach())[:, 2, 0] / 10, c='darkred', label='predicted y/10 2',ax=ax)
    sns.scatterplot(plot_true_y, c='#a65628', label='true y/10 seq1')
    sns.scatterplot(plot_true_y2, c='#984ea3', label='true y/10 seq2')
    plt.plot(phy_loss, c='black', label='physic loss*100')
    plt.plot(data_loss_plot, c='#999999', label='data loss')
    # plt.ylim(-0.1,0.1)
    plt.legend()
    plt.ylim(-0.01, 0.15)
    plt.title('PINN model gradient and loss after training')
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    #
    # # plt.show()
    plt.clf()
    '''
    return fig

def generate_data_fromode(ode_fun):
    t = np.linspace(0, 0.60, 60)
    y0 = [0.001]
    y_max, r = 1.2,0.5
    solution = odeint(ode_fun, y0, t, args=(y_max, r))
    plt.plot(t,solution)
    plt.show()


def print_parameters_from_ode(model):
    '''
    this function is to print r and ymax from pinn model
    '''
    for name, param in model.named_parameters():
        if name == 'r':
            r_value_pred = param.data.item()
            print(name, r_value_pred)
        if name == 'y_max':
            y_max_pred = param.data.item()
            print(name, y_max_pred)
    try:
        #if they are model parameter, it will be create in previous for loop
        return r_value_pred, y_max_pred
    except:
        r_value_pred = model.r
        y_max_pred = model.y_max
        print(y_max_pred)
        return r_value_pred, y_max_pred


def log_transform_tensor(input:torch.tensor)->torch.tensor:
    tensor_dataset = torch.log(input)
    return tensor_dataset


def reverse_log_tensor(input:torch.tensor)->torch.tensor:
    mask = (~torch.isin(input, torch.tensor(0.0))).float() #mask where is 0.0
    out = torch.exp(input)*mask #reverse log, but keep loss where the value is 0.0 as 0.0
    #print(out)
    return out


def plot_loss_change(saved_loss_list_train,saved_loss_list_validation,saved_loss_list_test,name):
    # plt.ioff()
    # plt.clf()
    fig1,ax = plt.subplots()
    sns.lineplot(saved_loss_list_train,color='blue',label='training loss',ax=ax)
    sns.lineplot(saved_loss_list_validation,color='orange',label='validation loss',ax=ax)
    sns.lineplot(saved_loss_list_test, color='red', label='test loss', ax=ax)
    # ax.figure.savefig('pinn_result/figure/{}.svg'.format(name),dpi=1000)
    return fig1

def run_logistic_ode_pinn(data_path='test_seq.csv', save_directory='/pinn_result/figure/', mode: str = '',
                              start_day=None, genotype: int | list = 2, parameter_boundary='', if_pinn=True,
                              penalize_y='', fit_ml_first=False, years: tuple = (2018, 2019, 2021,2022),corrected =False,
                              smooth=False,fill_in_na_at_start=False,multiple_r=False,rescale=False,randomseed=None,
                          weight=None,temperature_pinn=False,environment_input:tuple=('Air_temperature_2_m'),genetics_input:bool=False,
                          snp_encoding_type='one_hot'):
    """
    :param
        data_path: data directory and name, requires .csv file
                ,in this file all values are aligned based on the date start to measure, the time step is one day
        save_directory: str, pth of saving the result files
        mode: string, specify the key word for data processess and model used in training
            -if 'log' in mode, apply log transfrom to input data Y
            -if 'simple_rnn' in mode, use  RNN_ts_and_temperature_predict_height() as NN model
        start_day: int, the date start to use for train model,(days before are dropped)
        genotype: int, genotype.id from file, select a / list of genotypes data for training
        parameter_boundary: string,default: ''. If it's not equal to '', will add add extra penalty to negetive r
        if_pinn: boolean, default:True. If ture run with pinn mode, other wise train model only with masked MSE loss
    """
    from genetic_embedding import pretrain_genetics_embedding
    wand_name= mode
    if if_pinn == False:
        ode_int_loss_list = [False]
        print('pure ml mode')
        penalize_y = ''
        fit_ml_first = False
        multiple_r = False
        if genetics_input is None:
            parameter_boundary = ''
        print('set all related mode: penalize r, penalize y, multiple r, and fit ML as False or \'\'')
        print('weight:{}'.format(weight))
        weight = 0
    else:
        ode_int_loss_list = [False]
    # get current path
    current_pwd = os.getcwd()
    # print(current_pwd)
    # generate_data_fromode(logistic_growth_ode)
    input_seq = pd.read_csv(data_path, header=0, index_col=0)
    # reset timestamp
    input_seq['timestamp'] = input_seq['day_after_start_measure']
    input_seq.drop(columns='day_after_start_measure')
    max_env_dictionary = {}
    # if rescale:
    #     print(((input_seq['Air_temperature_2_m'].abs()).max()))
    #     max_air_temperature = (input_seq['Air_temperature_2_m'].abs()).max()
    #     max_humidity = (input_seq['Relative_air_humidity_2_m'].abs()).max()
    #     max_soil_temperature = (input_seq['Soil_temperature_-0.05_m'].abs()).max()
    #     max_irradiance = (input_seq['Short_wavelenght_solar_irradiance_2_m'].abs()).max()
    #     if not temperature_pinn:
    #         max_env_dictionary['Air_temperature_2_m'] = torch.tensor(max_air_temperature)  # do not scale temperature when use temperature pinn
    #     max_env_dictionary['Relative_air_humidity_2_m'] = torch.tensor(max_humidity)
    #     max_env_dictionary['Soil_temperature_-0.05_m'] = torch.tensor(max_soil_temperature)
    #     max_env_dictionary['Short_wavelenght_solar_irradiance_2_m'] = torch.tensor(max_irradiance)
    #
    #     input_seq['Air_temperature_2_m'] = input_seq['Air_temperature_2_m'] / (
    #         input_seq['Air_temperature_2_m'].abs()).max()
    #     # print('rescale, devided by maximum air humidity:{}'.format((input_seq['Relative_air_humidity_2_m'].abs()).max()))
    #     input_seq['Relative_air_humidity_2_m'] = input_seq['Relative_air_humidity_2_m'] / (
    #         input_seq['Relative_air_humidity_2_m'].abs()).max()
    #     input_seq['Soil_temperature_-0.05_m'] = input_seq['Soil_temperature_-0.05_m'] / (input_seq[
    #                                                                                          'Soil_temperature_-0.05_m'].abs()).max()
    #     input_seq['Short_wavelenght_solar_irradiance_2_m'] = input_seq['Short_wavelenght_solar_irradiance_2_m'] / (
    #         input_seq['Short_wavelenght_solar_irradiance_2_m'].abs()).max()
    if corrected:
        input_seq['value'] = input_seq['corrected_value']
        input_seq.drop(columns='corrected_value')
    # print('genoypes:')
    # print(input_seq['genotype.id'])
    try:  # if one genotype
        input_seq = input_seq[input_seq['genotype.id'] == genotype]
        mode = mode + 'genotype' + str(genotype) + parameter_boundary + 'pinnmode_' + str(if_pinn) + penalize_y + \
               'fit_ml_first_' + str(fit_ml_first) + 'smooth' + str(smooth) + \
               'fill_in_na' + str(fill_in_na_at_start) + 'rescale' + str(rescale)
        print('##model mode:{}'.format(mode))
    except:  # if list of genotypes
        if len(genotype) >= 2:
            print('multiple genotype')
            input_seq = input_seq[input_seq['genotype.id'].isin(genotype)]
            genotype = [str(g) for g in genotype]
            # mode is a string will used in files name later
            mode = mode + 'multi_g' + parameter_boundary + 'pinnmode_' + str(if_pinn) + penalize_y + \
                   'fit_ml_first_' + str(fit_ml_first) + 'corrected' + str(corrected) + 'smooth' + str(smooth) + \
                   'fill_in_na' + str(fill_in_na_at_start) + 'rescale' + str(rescale)
            print('##model mode:{}'.format(mode))
        else:
            input_seq = input_seq[input_seq['genotype.id'].isin(genotype)]
            genotype = [str(g) for g in genotype]
            # mode is a string will used in files name later
            print('gnotype:{}'.format(genotype))
            mode = mode + 'genotype' + genotype[0] + parameter_boundary + 'pinnmode_' + str(if_pinn) + penalize_y + \
                   'fit_ml_first_' + str(fit_ml_first) + 'corrected' + str(corrected) + 'smooth' + str(smooth) + \
                   'fill_in_na' + str(fill_in_na_at_start) + 'rescale' + str(rescale)
            print('##model mode:{}'.format(mode))
    # print(input_seq)
    model_create_data = create_tensor_dataset(year=list(years), dfs=[input_seq])
    # the g_df includes group label(genotype.id, year_site.harvest_year...), will be used for train test split based on group
    dfs, g_df, env_dfs = model_create_data.keep_overlap_time_stamps_between_multiple_features_dfs()
    # env_df_air_temperature = env_dfs['Air_temperature_2_m']
    # env_df_irradiance = env_dfs['Short_wavelenght_solar_irradiance_2_m']
    envdf_list = []
    for key_name in environment_input:
        print(key_name)
        envdf_list.append(env_dfs[key_name])
        if rescale:
            print('{} is scaled: min_max_scaler'.format(key_name))  #
        else:
            max_env_dictionary[key_name] = torch.tensor(1.0, requires_grad=True)
    else:
        envir_tensor_dataset = convert_inputx_to_tesor_list(envdf_list).permute(0, 1, 2)
        print('environment input shape:{}'.format(envir_tensor_dataset.shape))
        env_input_size = envir_tensor_dataset.shape[-1]
    # either same year different genotype or other way around,
    # same year same genotype won't resent in train test together
    if genetics_input:
        group_df = g_df[['year_site.harvest_year', 'genotype.id']]
        train_test_validation_dictionary = train_test_split_based_on_group(dfs[0], group_df,
                                                                           group_name=[
                                                                               'year_site.harvest_year',
                                                                           ], n_split=5)  # 'genotype.id'

    else:
        group_df = g_df[['year_site.harvest_year', 'genotype.id']]
        train_test_validation_dictionary = train_test_split_based_on_group(dfs[0], group_df,
                                                                           group_name=
                                                                           'year_site.harvest_year'
                                                                           , n_split=5)
        # train_test_validation_dictionary,train_years = manually_data_split_based_on_one_group(group_df, split_group='year_site.harvest_year')
    # print(train_years)
    # n_split_df = pd.DataFrame(data={"train_year":train_years},index=range(6))
    # n_split_df.to_csv('n_split_map_df.csv')
    # raise EOFError
    # print(group_df)
    group_df = group_df.copy()
    group_df['new_group_list'] = group_df.astype(str).apply(lambda row: '_'.join(row), axis=1)
    genotype_list = g_df['genotype.id']
    # train test split, save in dictionary
    n_split = len(train_test_validation_dictionary.keys())
    # convet df to torch.tensor
    print('envir_shape{}'.format(envir_tensor_dataset.shape))
    tensor_dataset = convert_inputx_to_tesor_list(dfs).permute(1, 0, 2)
    tensor_dataset = torch.nan_to_num(tensor_dataset, nan=0.0, posinf=0.0, neginf=0.0)
    # set negtive value to 0.0(represent missing)
    tensor_dataset[tensor_dataset <= 0.0] = 0.0
    print('if pinn *************{}'.format(if_pinn))
    # plot_multiple_sequences_colored_based_on_label_df(tensor_dataset[:,:,:], 'plant height', group_df['year_site.harvest_year'])
    #
    # plot_multiple_sequences_colored_based_on_label_df(envir_tensor_dataset[91:,:,0], 'temperature', group_df['year_site.harvest_year'])
    # raise EOFError
    if start_day == None:
        # find minimum value after 50 days(avoid start at november) for each time serier, cut sequence from there
        tensor_dataset[tensor_dataset == 0.0] = 999.0  # set na to 999 to find minimum
        min_position_current = torch.min(torch.argmin(tensor_dataset[50:, :, 0], dim=0))
        start_day = min_position_current.item() + 50
    else:
        tensor_dataset[tensor_dataset == 999.0] = 0.0
    print('start day:{}'.format(start_day))
    mode = mode + 'start_date_' + str(start_day)
    if weight != None:
        mode = mode + 'w{}_'.format(weight)
    # fit_logistic_ode = fit_logistic_ode_to_plant_height()
    # fit_logistic_ode.fit_dataset_and_save(data=tensor_dataset, start_day=start_day, seq_label=group_df)
    # raise EOFError
    # if 'log' in mode:
    #     print('take the natural logarithm of input')
    #     tensor_dataset = log_transform_tensor(tensor_dataset)
    #     #apply minmax scaling to data, as the value increses after log transform
    #     tensor_dataset,scaler = minmax_scaler(tensor_dataset)
    #     tensor_dataset = torch.nan_to_num(tensor_dataset,nan=0.0,posinf=0.0,neginf=0.0)

    # with open('../temporary/plant_height_tensor.dill'.format(genotype),'wb') as file1:
    #     dill.dump(tensor_dataset, file1)
    # file1.close()
    # with open('../temporary/temperature_tensor_same_length.dill'.format(genotype),'wb') as file1:
    #     dill.dump(envir_tensor_dataset, file1)
    # file1.close()
    #
    # with open('../temporary/genotype_list_tensor.dill'.format(genotype),'wb') as file1:
    #     genotype_list_tensor = torch.from_numpy(genotype_list.values.astype(float))
    #     dill.dump(genotype_list_tensor, file1)
    # file1.close()
    # with open('../temporary/group_list_df.dill'.format(genotype),'wb') as file1:
    #     dill.dump(group_df,file1)
    # file1.close()
    # raise EOFError
    # load full environment tensor, not same length
    # with open('full_env_335_no_scaling_env_tensor.dill','rb') as file1:
    #     env_tensor_full = dill.load(file1)
    # print(env_tensor_full.shape)
    # print(tensor_dataset)
    # print(torch.min(tensor_dataset))
    # print(tensor_dataset.shape)
    # print(envir_tensor_dataset.shape)

    # plot_multiple_sequences_colored_based_on_label_df(tensor_dataset[91:, :, :], 'plant height',
    #                                                   group_df['year_site.harvest_year'])
    #
    # raise EOFError
    if genetics_input:
        mode = mode + 'genetics_input'+snp_encoding_type
        if len(genotype)<=2:
            print('Warning: use genetics snps information as input, but only include data from one genotype')

    if rescale:
        #if rescale the order for processing envir_tensor_dataset need to be: minmax scaling -> smoothing ->
        # (drop values based on tensor_dataset) -> replace na with 0.0 -> (fill in small value at start for tensor_dataset)
        print('rescale environment to 0 and 1')
        envir_tensor_dataset, scaler = minmax_scaler(envir_tensor_dataset)

    # #"""moving averge to smooth temperature, use 15 days before measured date to calculate average"""
    # envir_tensor_dataset = smooth_tensor_ignore_nan(envir_tensor_dataset,15)
    #drop to the same length
    # envir_tensor_dataset[tensor_dataset==0.0]=np.nan
    #set na to 0.0

    #"""smooth then set as the same length"""
    envir_tensor_dataset = torch.nan_to_num(envir_tensor_dataset,nan=0.0,posinf=0.0,neginf=0.0)
    # plot_multiple_sequences_colored_based_on_label_df(envir_tensor_dataset[115:, :, 0], 'temperature',
    #                                                   group_df['year_site.harvest_year'])
    # raise EOFError

    if fill_in_na_at_start:
        #find the minimum value position, set all value before as a very small number 0.0001
        tensor_dataset[tensor_dataset == 0.0] = 999.0 #set na to 999 to find minimum
        for seq in range(tensor_dataset.shape[1]):
            tensor_dataset = torch.nan_to_num(tensor_dataset, nan=0.0, posinf=0.0, neginf=0.0)

            min_position = torch.argmin(tensor_dataset[:,seq,:]).item()
            # print(tensor_dataset[:, seq, :])
            # raise EOFError
            if torch.min(tensor_dataset[:,seq,:]).item()>0.0:
                tensor_dataset[:min_position+1, seq,:] = torch.min(tensor_dataset[:,seq,:]).item()
            else:
                tensor_dataset[:min_position+1, seq,:] = 0.0001
        else:
            tensor_dataset[tensor_dataset == 999.0] = 0.0 #set nan back to 0.0

    try:
        #dataframe to same result
        pinn_result = pd.read_csv('pinn_result/result_summary/PINN_mask_loss_{}.csv'.format(mode), header=0,index_col=0)
        print('connect result to existing result.csv file')
    except:
        print('result_summary/PINN_mask_loss_{}.csv Do not exist, create new file'.format(mode))
        pinn_result = pd.DataFrame()

    try:
        #dataframe to same result
        predicte_value_curve_train = pd.read_csv('pinn_result/train_predict_curve_{}.csv'.format(mode), header=0,index_col=0)
        print('connect result to existing predited curve .csv file')
    except:
        print('pinn_result/train_predict_curve_{}.csv Do not exist, create new file'.format(mode))
        predicte_value_curve_train = pd.DataFrame()
    try:
        # dataframe to same result
        predicte_value_curve_val = pd.read_csv('pinn_result/val_predict_curve_{}.csv'.format(mode), header=0, index_col=0)
        print('connect result to existing predited curve .csv file')
    except:
        print('pinn_result/val_predict_curve_{}.csv Do not exist, create new file'.format(mode))
        predicte_value_curve_val = pd.DataFrame()
    try:
        # dataframe to same result
        predicte_value_curve_test = pd.read_csv('pinn_result/test_predict_curve_{}.csv'.format(mode), header=0, index_col=0)
        print('connect result to existing predited curve .csv file')
    except:
        print('pinn_result/test_predict_curve_{}.csv Do not exist, create new file'.format(mode))
        predicte_value_curve_test = pd.DataFrame()

    predicte_value_gradient = pd.DataFrame()
    predicte_value_r = pd.DataFrame()

    for n in [0]:# different split
        train_index, validation_index, test_index = train_test_validation_dictionary['splits_{}'.format(n)]
        # get train test split data
        print('yearly data use for training {}:'.format(len(train_index)))
        train_year_list = group_df.iloc[train_index]['year_site.harvest_year']
        train_genotype_list = torch.from_numpy(genotype_list[train_index].values.astype(float))
        print(train_year_list)
        print('yearly data use for validation {}:'.format(len(validation_index)))
        validation_year_list = group_df.iloc[validation_index]['year_site.harvest_year']
        validation_genotype_list = torch.from_numpy(genotype_list[validation_index].values.astype(float))
        print(validation_year_list)
        print('yearly data use for test {}:'.format(len(test_index)))
        test_year_list = group_df.iloc[test_index]['year_site.harvest_year']
        test_genotype_list = torch.from_numpy(genotype_list[test_index].values.astype(float))
        print(test_year_list)
        # drop the date before (starting date) sping
        train_y = tensor_dataset[start_day:, train_index, :].to(DEVICE)
        # with open('train_y.dill','wb') as file:
        #     dill.dump(train_y,file)
        # file.close()
        print('y train shape')
        print(train_y.shape)
        train_env = envir_tensor_dataset[start_day:, train_index, :].to(DEVICE).requires_grad_(True)
        print('train env shape:{}'.format(train_env.shape))
        # print(train_env)
        if genetics_input:
            # Get unique values and create a mapping for year
            label_to_index = {2018:0,2019:1,2021:2,2022:3}

            # Convert the original list to indices
            train_year_list_id = [label_to_index[label] for label in train_year_list]
            validation_year_list_id = [label_to_index[label] for label in validation_year_list]
            test_list_id = [label_to_index[label] for label in test_year_list]
            # one-hot encod year and concat to genetics input to allow G by E interation
            year_tensor_train = torch.tensor(train_year_list_id).to(DEVICE)
            year_tensor_validation = torch.tensor(validation_year_list_id).to(DEVICE)
            year_tensor_test = torch.tensor(test_list_id).to(DEVICE)
            if snp_encoding_type == 'one_hot':
                year_tensor_train = torch.nn.functional.one_hot(year_tensor_train).unsqueeze(dim=-1).to(DEVICE)
                year_tensor_validation = torch.nn.functional.one_hot(year_tensor_validation).unsqueeze(dim=-1).to(
                    DEVICE)
                year_tensor_test = torch.nn.functional.one_hot(year_tensor_test).unsqueeze(dim=-1).to(DEVICE)

            # the same train test split as PINN input
            year_list = (year_tensor_train,year_tensor_validation,year_tensor_test)
            hyper_parameter,genetics_tensor = pretrain_genetics_embedding((train_genotype_list, validation_genotype_list, test_genotype_list),year_list,
                                                                          file_name='{}'.format(str(mode)),snp_encode_name=snp_encoding_type)
            if snp_encoding_type == '' or snp_encoding_type == 'one_hot':
                out_channel_1, out_channel_2, kernel1, kernel2 = hyper_parameter
            else:
                size = hyper_parameter[0]
            #load best pretrained model
            genetics_train_year =genetics_tensor[0]
            genetics_validation_year = genetics_tensor[1]
            genetics_test_year = genetics_tensor[2]
            # genetics_train_year = copy.deepcopy(torch.cat([genetics_train, year_tensor_train], dim=-1))
            # genetics_validation_year = copy.deepcopy(torch.cat([genetics_validation, year_tensor_validation], dim=-1))
            # genetics_test_year = copy.deepcopy(torch.cat([genetics_test, year_tensor_test], dim=-1))

            genotype_year_tensorlist =[train_genotype_list, validation_genotype_list, test_genotype_list] #genotype id list
            print('train genotype:{}'.format(train_genotype_list))
            print('validation genotype:{}'.format(validation_genotype_list))
            print('test genotype:{}'.format(test_genotype_list))
            # train_label_list = group_df[train_index].to_list()
            # validation_label_list = group_df[validation_index].to_list()
            # test_label_list = group_df[validation_index].to_list()
        else:
            genetics_model = None
            genetics_train = None
            genetics_validation = None
            genetics_test = None
            genetics_train_year=None
            genetics_validation_year=None
            genetics_test_year=None

        #split train validation test
        validation_y = tensor_dataset[start_day:, validation_index, :].to(DEVICE)
        validation_env = envir_tensor_dataset[start_day:, validation_index, :].to(DEVICE)
        test_y = tensor_dataset[start_day:, test_index, :].to(DEVICE)
        test_env = envir_tensor_dataset[start_day:, test_index, :].to(DEVICE)
        train_group_df = copy.deepcopy(group_df).iloc[train_index, :]
        validation_group_df = copy.deepcopy(group_df).iloc[validation_index, :]
        test_group_df = copy.deepcopy(group_df).iloc[test_index, :]

        num_seq = train_y.shape[1]

        # creat time sequence, same as input shape
        ts_train = torch.linspace(0.0, 284, 285).unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, num_seq,
                                                                                          1)[:-start_day, :,
                   :].requires_grad_(True).to(DEVICE)  # time sequences steps
        ts_validation = torch.linspace(0.0, 284, 285).unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, validation_y.shape[1],
                                                                                               1)[:-start_day, :,
                        :].to(DEVICE)  # time sequences steps
        ts_test = torch.linspace(0.0, 284, 285).unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, test_y.shape[1],
                                                                                         1)[:-start_day, :, :].to(
            DEVICE)  # time sequences steps

        # average across replicates predict result, use for result report
        train_env_avg, train_y_avg, ts_train_avg, train_group_df_avg = calculate_average_across_replicates(
            train_env, train_group_df, train_y, ts_train)

        validation_env_avg, validation_y_avg, ts_validation_avg, validation_group_df_avg = calculate_average_across_replicates(
            validation_env, validation_group_df, validation_y,
            ts_validation)

        test_env_avg, test_y_avg, ts_test_avg, test_group_df_avg =calculate_average_across_replicates(
                test_env,test_group_df, test_y,
                ts_test)

        different_genotype_curve_train = pd.DataFrame(
            torch.squeeze(copy.deepcopy(train_y_avg.detach())).cpu().numpy())
        different_genotype_curve_train.columns = train_group_df_avg['genotype.id'].to_list()
        different_genotype_curve_train.to_csv(
            "pinn_result/train_true_curves_{}.csv".format(
                 mode))

        different_genotype_curve_val = pd.DataFrame(
            torch.squeeze(copy.deepcopy(validation_y_avg.detach())).cpu().numpy())
        different_genotype_curve_val.columns = validation_group_df_avg['genotype.id'].to_list()
        different_genotype_curve_val.to_csv(
            "pinn_result/val_true_curves_{}.csv".format(
                 mode))

        different_genotype_curve_test = pd.DataFrame(
            torch.squeeze(copy.deepcopy(test_y_avg.detach())).cpu().numpy())
        different_genotype_curve_test.columns = test_group_df_avg['genotype.id'].to_list()
        different_genotype_curve_test.to_csv(
            "pinn_result/test_true_curves_{}.csv".format(
                mode))

        print('device')
        print(ts_train.device)

        for lr in [0.001]:  # 0.01,0.001,0.0005,
            for hidden in [3,5]:   # 3,5
                    for num_layer in [1,2]:  #1,2
                        if weight!=None:
                            weights = [weight]
                        else:
                            weights = [2,9] # 2,9
                        for weight_physic in weights:  # [2,3,4,5,6,7,8,9,10]:#[2,3,4,5,6,7,8,9,10]: #2,3,4,5,6,7,8,9,10
                            for ode_int_loss in ode_int_loss_list:
                                for L2 in [0.1]: #1.0,0.1
                                    if randomseed:
                                        randomseed_list = [randomseed]
                                    else:
                                        randomseed_list = [1, 2, 3, 4, 5]  # [0,1,2,3,4,5]
                                    for j in randomseed_list:  # different random seed
                                        print('nsplit:{}  random seed:{}'.format(n, j))
                                        random.seed(j)
                                        np.random.seed(j)
                                        torch.manual_seed(j)  # the random seeds works on my laptop,
                                        # while seems the same random seed does not give the same result on server
                                        currenttime = time.time()
                                        if genetics_input:
                                            if snp_encoding_type == '' or snp_encoding_type == 'one_hot':
                                                with open('snps_embedding_model/{0}_{1}_{2}_{3}_{4}_{5}.dill'.format(str(mode),out_channel_1,
                                                                                                         out_channel_2, kernel1,
                                                                                                         kernel2,snp_encoding_type),'rb') as file2:
                                                    genetics_model = dill.load(file2)
                                            else:
                                                with open('snps_embedding_model/{}_{}_{}.dill'.format(str(mode), size, snp_encoding_type),'rb') as file2:
                                                    genetics_model = dill.load(file2)

                                        if multiple_r:
                                            print('get different r at different time')
                                            pinn_model = LSTM_ts_and_temperature_predict_height(hidden, input_size=env_input_size,
                                                                                                num_layer=num_layer,
                                                                                                smooth_out=smooth,
                                                                                                 seq_len=train_y.shape[0], temperature_pinn=temperature_pinn, genetics=genetics_model).to(
                                                DEVICE)
                                        else:
                                            run = wandb.init(
                                                # Set the project where this run will be logged
                                                project="PINN_single_genotype_{}_{}".format(genotype,wand_name),
                                                # id='{0}'.format(mode),
                                                # Track hyperparameters and run metadata
                                                config={
                                                    "learning_rate": lr,
                                                    "epochs": 3000,"n_split": n,
                                              "random_sees": j,
                                              "hidden_size": hidden,
                                              "num_layer": num_layer,
                                              "weight_physic": weight_physic,
                                              'ode_int':ode_int_loss,
                                              'l2':L2,
                                                    'penalize_r':parameter_boundary
                                                },
                                            )
                                            pinn_model = LSTM_ts_and_temperature_predict_height(hidden, input_size=env_input_size,
                                                                                                num_layer=num_layer,
                                                                                                smooth_out=smooth,
                                                                                                temperature_pinn=temperature_pinn, genetics=genetics_model).to(DEVICE)
                                        # pinn_model.cuda() #send model to GPU
                                        # print total number of parameters in model
                                        total_params = count_parameters(pinn_model)
                                        # train pinn on training set
                                        print('train env shape:{}'.format(train_env.shape))
                                        model, stop_num_epochs, loss_fig, y_gradient_temperature, autograd_fig = train_model(
                                            model=pinn_model, train_env=train_env, Y=train_y,
                                            ts_X=ts_train, epochs=3000,
                                            validation_env=validation_env,
                                            validation_y=validation_y,
                                            ts_validation=ts_validation, lr=lr,
                                            physic_loss_weight=weight_physic,
                                            parameter_boundary=parameter_boundary,
                                            pinn=if_pinn, penalize_negetive_y=penalize_y,
                                            fit_ml_first=fit_ml_first,multiple_r=multiple_r,ode_intergration_loss=ode_int_loss,
                                            l2=L2,temperature_ode=temperature_pinn,ts_test=ts_test,test_env=test_env,test_y=test_y,
                                            genetics_model=genetics_model, genetics_Train_year=genetics_train_year,
                                            genetics_validation_year=genetics_validation_year,
                                            genetics_test_year=genetics_test_year,scaler_env=max_env_dictionary,genotype_list=[train_genotype_list, validation_genotype_list, test_genotype_list])

                                        if genetics_model:
                                            genetics_model = model[1]
                                            model = model[0]
                                        print('training stopped due to validation loss increases at epochs:{} '.format(
                                            stop_num_epochs))
                                        if smooth:
                                            smooth_alpha = model.alpha.item()
                                            print('smooth_parameter alpha={}'.format(smooth_alpha))
                                        else:
                                            smooth_alpha = None
                                        # plt.ioff()
                                        # plt.clf()
                                        # loss_ratio_df.to_csv('{}{}/train_predict_{}split_hidden{}_env{}_ts{}_lr_{}_{}_rs_{}_loss_ratio.csv'.format(
                                        #  current_pwd, save_directory, n, hidden, num_layer, ts_layer, lr,mode,j))
                                        # calculate MSE for tain test and validation
                                        r_value_pred, y_max_pred = print_parameters_from_ode(model)
                                        if genetics_model:
                                            #their type will be tensor
                                            r_value_pred = r_value_pred.detach()
                                            y_max_pred = y_max_pred.detach()
                                            #if multiple r is trained, then use mean for plotting and save in result
                                        try:
                                            r_value_pred_mean = torch.mean(r_value_pred).item()
                                            y_max_pred_mean = torch.mean(y_max_pred).item()
                                        except:
                                            #otherwise only have one r value as type float
                                            r_value_pred_mean = r_value_pred
                                            y_max_pred_mean = y_max_pred
                                        # access weight from ts and temperature rnn
                                        # print('temperature rnn weight shape:{}'.format(model.temperature.0.weight_ih_l0.shape))

                                        # https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615
                                        model.eval()  # model will perform different if there are dropout or batchnorm layers
                                        print('train env shape:{}'.format(train_env.shape))

                                        with torch.no_grad():  # will turn of autogradient,
                                            if if_pinn:
                                                #with torch.backends.cudnn.flags(enabled=False):
                                                predict_y_train = model(train_env_avg, ts_train_avg)
                                                # with open('trin_y_pred.dill','wb') as f:
                                                #     dill.dump(predict_y_train,f)
                                                # f.close()
                                                # with open('env_y_pred.dill','wb') as f:
                                                #     dill.dump(train_env,f)
                                                # f.close()
                                                if genetics_model:
                                                    genetics_model(genetics_validation_year)
                                                predict_y_validation = model(validation_env_avg, ts_validation_avg)
                                                # print('predict_y_shape:{}'.format(predict_y_validation.shape))  # [60, 2, 1] =[seq_len,sample_num,feature_size]
                                                if genetics_model:
                                                    genetics_model(genetics_test_year)
                                                predict_y_test = model(test_env_avg, ts_test_avg)
                                            else:
                                                predict_y_train = model(train_env_avg, ts_train_avg)
                                                if genetics_model:
                                                    genetics_model(genetics_validation_year)
                                                predict_y_validation = model(validation_env_avg, ts_validation_avg)
                                                if genetics_model:
                                                    genetics_model(genetics_test_year)
                                                # print('predict_y_shape:{}'.format(predict_y_validation.shape))  # [60, 2, 1] =[seq_len,sample_num,feature_size]
                                                predict_y_test = model(test_env_avg, ts_test_avg)
                                        # copy data later need to modify for plot, if not use deep.copy will modify the original variable
                                        copy_y_train = copy.deepcopy(train_y_avg.detach())
                                        copy_y_test = copy.deepcopy(test_y_avg.detach())

                                        copy_y_validation = copy.deepcopy(validation_y_avg.detach())


                                        predict_y_train_detach = copy.deepcopy(predict_y_train.detach())
                                        # predict_y_train_detach = torch.nan_to_num(predict_y_train_detach, nan=0.0,posinf=0.0,neginf=0.0)
                                        predict_y_validation_detach = copy.deepcopy(predict_y_validation.detach())
                                        # predict_y_validation_detach = torch.nan_to_num(predict_y_validation_detach, nan=0.0,posinf=0.0,neginf=0.0)
                                        predict_y_test_detach = copy.deepcopy(predict_y_test.detach())
                                        # predict_y_test_detach = torch.nan_to_num(predict_y_test_detach,nan=0.0,posinf=0.0,neginf=0.0)
                                        ##print and save mse(reverse scaling and log transform to make it compareable with other model)

                                        # calculate loss
                                        train_masked_rmse = mask_rmse_loss(true_y=copy_y_train, predict_y=predict_y_train_detach)
                                        print('train_rmse: {}'.format(train_masked_rmse))
                                        validation_masked_rmse = mask_rmse_loss(true_y=copy_y_validation,
                                                                                predict_y=predict_y_validation_detach)
                                        print('validation rMSE:{}'.format(validation_masked_rmse))
                                        test_masked_rmse = mask_rmse_loss(true_y=copy_y_test, predict_y=predict_y_test_detach)
                                        print('test rMSE:{}'.format(test_masked_rmse))
                                        corre_train = mask_dtw_loss(true_y=copy_y_train, predict_y=predict_y_train_detach)
                                        print('train shapeDTW')
                                        print(corre_train)
                                        corre_validation = mask_dtw_loss(true_y=copy_y_validation, predict_y=predict_y_validation_detach)
                                        print('validation shapeDTW')
                                        print(corre_validation)
                                        corre_test = mask_dtw_loss(true_y=copy_y_test, predict_y=predict_y_test_detach)
                                        print('test shapeDTW')
                                        print(corre_test)
                                        if genetics_input:
                                            #different genotype different color
                                            fig_residual_train = plot_residual(copy_y_train, predict_y_train_detach,
                                                                               color_label=train_genotype_list,
                                                                               marker_label=train_year_list,
                                                                               title='train')
                                            fig_residual_validation = plot_residual(copy_y_validation,
                                                                                    predict_y_validation_detach,
                                                                                    color_label=validation_genotype_list,
                                                                                    marker_label=validation_year_list,
                                                                                    title='validation')
                                            fig_residual_test = plot_residual(copy_y_test, predict_y_test_detach,
                                                                              color_label=test_genotype_list,
                                                                              marker_label=test_year_list,
                                                                              title='test')
                                        else:
                                            #year is marked by color
                                            fig_residual_train = plot_residual(copy_y_train, predict_y_train_detach,
                                                                               color_label=train_group_df_avg['year_site.harvest_year'],
                                                                               marker_label=train_group_df_avg['genotype.id'],
                                                                               title='train')
                                            fig_residual_validation = plot_residual(copy_y_validation,
                                                                                    predict_y_validation_detach,
                                                                                    color_label=validation_group_df_avg['year_site.harvest_year'],
                                                                                    marker_label=validation_group_df_avg['genotype.id'],
                                                                                    title='validation')
                                            fig_residual_test = plot_residual(copy_y_test, predict_y_test_detach,
                                                                              color_label=test_group_df_avg['year_site.harvest_year'],
                                                                              marker_label=train_group_df_avg['genotype.id'],
                                                                              title='test')
                                        # try:
                                        #     # dataframe to same result
                                        #     predicte_value_curve_test = pd.read_csv(
                                        #         'pinn_result/test_predict_curve_{}.csv'.format(mode), header=0,
                                        #         index_col=0)
                                        #     print('connect result to existing predited curve.csv file')
                                        # except:
                                        #     print('no file exist')
                                        # try:
                                        #     # dataframe to same result
                                        #     predicte_value_curve_val = pd.read_csv(
                                        #         'pinn_result/val_predict_curve_{}.csv'.format(mode), header=0,
                                        #         index_col=0)
                                        #     print('connect result to existing predited curve.csv file')
                                        # except:
                                        #     print('no file exist')
                                        # try:
                                        #     # dataframe to same result
                                        #     predicte_value_curve_train = pd.read_csv(
                                        #         'pinn_result/train_predict_curve_{}.csv'.format(mode), header=0,
                                        #         index_col=0)
                                        #     print('connect result to existing predited curve.csv file')
                                        # except:
                                        #     print('no file exist')
                                        for seq_n in range(predict_y_validation_detach.shape[1]):
                                            col_name = 'predict_{}split_hidden{}_env{}_ts{}_lr_{}_w_ph{}_ode_int_{}_l2_{}_{}_rs_{}_seq{}'.format(
                                                n, hidden, num_layer, num_layer, lr, weight_physic,ode_int_loss,L2, mode, j, seq_n)

                                            new_predict_curve = pd.DataFrame(
                                                data={col_name: torch.squeeze(predict_y_validation_detach[:, seq_n, :]).cpu()},
                                                index=range(predict_y_validation_detach.shape[0]))

                                            predicte_value_curve_val = pd.concat([predicte_value_curve_val, new_predict_curve], axis=1)
                                            predicte_value_curve_val.to_csv('pinn_result/val_predict_curve_{}.csv'.format(mode))

                                        for seq_n in range(predict_y_train_detach.shape[1]):
                                            col_name = 'predict_{}split_hidden{}_env{}_ts{}_lr_{}_w_ph{}_ode_int_{}_l2_{}_{}_rs_{}_seq{}'.format(
                                                n, hidden, num_layer, num_layer, lr, weight_physic, ode_int_loss,
                                                L2, mode, j, seq_n)

                                            new_predict_curve = pd.DataFrame(
                                                data={col_name: torch.squeeze(
                                                    predict_y_train_detach[:, seq_n, :]).cpu()},
                                                index=range(predict_y_train_detach.shape[0]))

                                            predicte_value_curve_train = pd.concat(
                                                [predicte_value_curve_train, new_predict_curve], axis=1)
                                            predicte_value_curve_train.to_csv(
                                                'pinn_result/train_predict_curve_{}.csv'.format(mode))
                                        for seq_n in range(predict_y_test_detach.shape[1]):
                                            col_name = 'predict_{}split_hidden{}_env{}_ts{}_lr_{}_w_ph{}_ode_int_{}_l2_{}_{}_rs_{}_seq{}'.format(
                                                n, hidden, num_layer, num_layer, lr, weight_physic,ode_int_loss,L2, mode, j, seq_n)

                                            new_predict_curve = pd.DataFrame(
                                                data={col_name: torch.squeeze(predict_y_test_detach[:, seq_n, :]).cpu()},
                                                index=range(predict_y_test_detach.shape[0]))
                                            predicte_value_curve_test = pd.concat([predicte_value_curve_test, new_predict_curve], axis=1)
                                            predicte_value_curve_test.to_csv('pinn_result/test_predict_curve_{}.csv'.format(mode))

                                        new_row = pd.DataFrame(
                                            data={"lr": lr,
                                                  "n_split": n,
                                                  "random_sees": j,
                                                  "hidden_size": hidden,
                                                  "num_layer": num_layer,
                                                  "weight_physic": weight_physic,
                                                  "Trainable_Params": total_params,
                                                  'epoch': stop_num_epochs,
                                                  'train_rMSE': round(train_masked_rmse.item(),3),
                                                  'validation_rMSE': round(validation_masked_rmse.item(),3),
                                                  "test_rMSE": round(test_masked_rmse.item(),3),
                                                  'train_shapeDTW': round(corre_train,3),
                                                  'validation_shapeDTW': round(corre_validation,3),
                                                  "test_shapeDTW": round(corre_test,3),
                                                  "predicted_r": r_value_pred_mean,
                                                  "predicted_y_max": y_max_pred_mean,
                                                  "smooth_alpha": smooth_alpha,
                                                  'ode_int':ode_int_loss,
                                                  'l2':L2
                                                  },
                                            index=[0])
                                        try:
                                            # dataframe to same result
                                            pinn_result = pd.read_csv(
                                                'pinn_result/result_summary/PINN_mask_loss_{}.csv'.format(mode),
                                                header=0, index_col=0)
                                            print('connect result to existing result.csv file')
                                        except:
                                            print(
                                                'result_summary/PINN_mask_loss_{}.csv Do not exist, this is the first iteration'.format(
                                                    mode))
                                        pinn_result = pd.concat(
                                            [pinn_result, new_row])
                                        pinn_result.to_csv('pinn_result/result_summary/PINN_mask_loss_{}.csv'.format(mode))
                                        # Save the model in the exchangeable ONNX format
                                        # torch.onnx.export(model, (test_env,ts_test), "model.onnx")
                                        # wandb.save("model.onnx")
                                        # with open(
                                        #         'pinn_result/model/trained_model_{}split_hidden{}_env{}_ts{}_lr_{}_w_ph{}_ode_int_{}_l2_{}_{}_rs_{}.dill'.format(
                                        #                 n, hidden, num_layer, num_layer, lr, weight_physic,ode_int_loss,L2, mode, j), 'wb') as file:
                                        #     dill.dump(model, file)
                                        # file.close()
                                        torch.save(model.state_dict(),
                                                   f"pinn_result/model/trained_model_{n}split_hidden{hidden}_env{num_layer}_ts{num_layer}_lr_{lr}_w_ph{weight_physic}_ode_int_{ode_int_loss}_l2_{L2}_{mode}_rs_{j}.pt")

                                        train_predict_plot_save_name = 'trained_figure_{}split_hidden{}_env{}_ts{}_lr_{}_w_ph{}_ode_int_{}_l2_{}_{}_rs_{}.svg'.format(
                                            n, hidden, num_layer, num_layer, lr, weight_physic,ode_int_loss,L2, mode, j)
                                        print('train env shape:{}'.format(train_env.shape))
                                        validation_predict_plot_save_name = 'validation_figure_{}split_hidden{}_env{}_ts{}_lr_{}_w_ph{}_ode_int_{}_l2_{}_{}_rs_{}.svg'.format(
                                            n, hidden, num_layer, num_layer, lr, weight_physic, ode_int_loss,
                                            L2, mode, j)
                                        test_predict_plot_save_name = 'test_figure_{}split_hidden{}_env{}_ts{}_lr_{}_w_ph{}_ode_int_{}_l2_{}_{}_rs_{}.svg'.format(
                                            n, hidden, num_layer, num_layer, lr, weight_physic,
                                            ode_int_loss, L2, mode, j)
                                        if not if_pinn:
                                            r_value_pred=None
                                            y_max_pred=None
                                        if genetics_input:
                                            train_fig = plot_save_predict_curve(current_pwd, 'train',
                                                                                predict_y_train_detach,
                                                                                r_value_pred,
                                                                                save_directory, copy_y_train, ts_train_avg,
                                                                                y_max_pred,
                                                                                marker_label=train_group_df_avg['year_site.harvest_year'],
                                                                                color_label=train_group_df_avg['genotype.id'],
                                                                                corresponding_environment=train_env,name=train_predict_plot_save_name)

                                            #
                                            validation_fig = plot_save_predict_curve(current_pwd,
                                                                                     'validation',
                                                                                     predict_y_validation_detach,
                                                                                     r_value_pred,
                                                                                     save_directory,
                                                                                     copy_y_validation,
                                                                                     ts_validation_avg, y_max_pred,
                                                                                     marker_label=validation_group_df_avg['year_site.harvest_year'],
                                                                                     color_label=validation_group_df_avg['genotype.id'],
                                                                                     corresponding_environment=validation_env,name=validation_predict_plot_save_name)

                                            test_fig = plot_save_predict_curve(current_pwd, 'test',
                                                                               predict_y_test_detach, r_value_pred,
                                                                               save_directory, copy_y_test, ts_test_avg,
                                                                               y_max_pred,
                                                                               marker_label=test_group_df_avg['year_site.harvest_year'],
                                                                               color_label=test_group_df_avg['genotype.id'],
                                                                               corresponding_environment=test_env
                                                                               , name=test_predict_plot_save_name
                                                                               )
                                        else:
                                            train_fig = plot_save_predict_curve(current_pwd, 'train',
                                                                                predict_y_train_detach,
                                                                                r_value_pred_mean,
                                                                                save_directory, copy_y_train, ts_train_avg,
                                                                                y_max_pred,
                                                                                marker_label=train_group_df_avg['genotype.id'],
                                                                                color_label=train_group_df_avg['year_site.harvest_year'],
                                                                                corresponding_environment=train_env_avg,name=train_predict_plot_save_name)
                                            # validation_predict_plot_save_name = '{}{}/validation_predict_{}split_hidden{}_num_layer{}_ts_layer_{}_lr_{}_weight_physic_{}_dropout_{}_ode_int_{}_{}_rs_{}.png'.format(
                                            #     current_pwd, save_directory, n, hidden, num_layer, ts_layer, lr, weight_physic,drop_out,ode_int_loss, mode, j)
                                            #
                                            validation_fig = plot_save_predict_curve(current_pwd,
                                                                                     'validation',
                                                                                     predict_y_validation_detach,
                                                                                     r_value_pred_mean,
                                                                                     save_directory,
                                                                                     copy_y_validation,
                                                                                     ts_validation_avg, y_max_pred,
                                                                                     marker_label=validation_group_df_avg['genotype.id'],
                                                                                     color_label=validation_group_df_avg['year_site.harvest_year'],
                                                                                     corresponding_environment=validation_env_avg,name=validation_predict_plot_save_name)
                                            # test_predict_plot_save_name = '{}{}/test_predict_{}split_hidden{}_num_layer{}_ts_layer_{}_lr_{}_weight_physic_{}_dropout_{}_ode_int_{}_{}_rs_{}.png'.format(
                                            #     current_pwd, save_directory, n, hidden, num_layer, ts_layer, lr, weight_physic,drop_out,ode_int_loss, mode, j)
                                            #
                                            test_fig = plot_save_predict_curve(current_pwd, 'test',
                                                                               predict_y_test_detach, r_value_pred_mean,
                                                                               save_directory, copy_y_test, ts_test_avg,
                                                                               y_max_pred,
                                                                               marker_label=test_group_df_avg['genotype.id'],
                                                                               color_label=test_group_df_avg['year_site.harvest_year'],
                                                                               corresponding_environment=test_env_avg,name=test_predict_plot_save_name
                                                                               )
                                        wandb.log({'end_epoch':stop_num_epochs})
                                        wandb.log({'train_prediction_plot':wandb.Image(train_fig)})
                                        wandb.log({'val_prediction_plot': wandb.Image(validation_fig)})
                                        wandb.log({'test_prediction_plot': wandb.Image(test_fig)})
                                        with PdfPages(
                                                '{}{}/result_{}_split_hidden{}_env{}_ts{}_lr_{}_w_ph{}_ode_int_{}_l2_{}_{}_rs_{}.pdf'.format(
                                                        current_pwd, save_directory, n, hidden, num_layer, num_layer, lr,
                                                        weight_physic,ode_int_loss,L2, mode, j)) as pdf:
                                            print('save figure')
                                            for fig in [loss_fig, train_fig, autograd_fig, fig_residual_train, validation_fig,
                                                        fig_residual_validation, test_fig, fig_residual_test]:
                                                pdf.savefig(fig, bbox_inches='tight')
                                            else:
                                                plt.close('all')
                                            run.finish()
                                        if if_pinn == False:
                                            print('******************************************************************')
                                            print(weight_physic)
                                            #break
                                        one_runtime = time.time()-currenttime
                                        print('runing time for one run:{}'.format(one_runtime))


def fit_ode_for_seq_seperatelly_plot(data_path,genotype, start_day,years,corrected=False,result_df=None ):



    input_seq = pd.read_csv(data_path, header=0, index_col=0)
    # reset timestamp
    input_seq['timestamp'] = input_seq['day_after_start_measure']
    input_seq.drop(columns='day_after_start_measure')
    max_env_dictionary = {}
    if corrected:
        input_seq['value'] = input_seq['corrected_value']
        input_seq.drop(columns='corrected_value')
    print('genoypes:')
    print(input_seq['genotype.id'])
    # if one genotype
    input_seq = input_seq[input_seq['genotype.id'] == genotype]

    # print(input_seq)
    model_create_data = create_tensor_dataset(year=list(years), dfs=[input_seq])
    # the g_df includes group label(genotype.id, year_site.harvest_year...), will be used for train test split based on group
    dfs, g_df, env_dfs = model_create_data.keep_overlap_time_stamps_between_multiple_features_dfs()


    group_df = g_df[['year_site.harvest_year', 'genotype.id']]

    tensor_dataset = convert_inputx_to_tesor_list(dfs).permute(1, 0, 2)
    tensor_dataset = torch.nan_to_num(tensor_dataset, nan=0.0, posinf=0.0, neginf=0.0)
    # set negtive value to 0.0(represent missing)
    tensor_dataset[tensor_dataset <= 0.0] = 0.0

    if start_day == None:
        # find minimum value after 50 days(avoid start at november) for each time serier, cut sequence from there
        tensor_dataset[tensor_dataset == 0.0] = 999.0  # set na to 999 to find minimum
        min_position_current = torch.min(torch.argmin(tensor_dataset[50:, :, 0], dim=0))
        start_day = min_position_current.item() + 50
    print('start day:{}'.format(start_day))

    tensor_dataset[tensor_dataset == 999.0] = 0.0

    fit_logistic_ode = fit_logistic_ode_to_plant_height()
    genotype_average_rmse =fit_logistic_ode.fit_dataset_and_save(data=tensor_dataset, start_day=start_day, seq_label=group_df)
    new_row = pd.DataFrame({'genotype':genotype,'rmse':genotype_average_rmse},index=[0])
    result_df = pd.concat([result_df,new_row],ignore_index=True)
    result_df.to_csv('logistic_genotype_average_rmse_seq_fit.csv')
    return result_df
def plot_residual(copy_y_train, predict_y_train_detach,color_label,marker_label,title):
    """
    this is the function for plotting residual
    """
    copy_y_train[copy_y_train == 0.0] = np.nan  # set to na so it will not be ploted
    print(copy_y_train.shape)
    residual_y_train = torch.squeeze((predict_y_train_detach - copy_y_train),dim=-1).detach().cpu().numpy()
    print('residual shape')
    print(residual_y_train.shape)
    print(marker_label)
    # residual_y_train[residual_y_train == -0.0] = np.nan
    fig_residual, ax = plt.subplots()
    if isinstance(color_label, torch.Tensor):
        color_label = pd.DataFrame({'COLOR_LABLE':copy.deepcopy(color_label.detach().cpu()).numpy()})
        color_label= color_label['COLOR_LABLE']
    if isinstance(marker_label, torch.Tensor):
        marker_label = pd.DataFrame({'marker_label':copy.deepcopy(marker_label.detach().cpu()).numpy()})
        marker_label= marker_label['marker_label']
    print('color label')
    print(color_label)
    marker_list = ['x', '.', '1', '*', "$\u266B$",'2','3','4','p','<','>','^','v','h','+']  # year is no more tha 5 currently
    # unique markers needed in plot -> correspondiing to year
    unique_markers = marker_label.unique()
    markers_dictionary = {}
    for label in unique_markers:
        # save year and marker type in dictionary
        markers_dictionary[label] = marker_list.pop(0)
    # do the same for color, whihc is corresponding to genotype
    unique_colors = color_label.unique()
    colors_list = sns.color_palette("dark", len(unique_colors))  # _get_colors(len(unique_colors))
    color_dictionary = {}
    for label, color in zip(unique_colors, colors_list):
        color_dictionary[label] = color

    for seq, type, label in zip(range(residual_y_train.shape[1]), marker_label, color_label):
        sns.scatterplot(residual_y_train[:, seq], label=label, ax=ax, marker=markers_dictionary[type],
                        color=color_dictionary[label])

    # sns.scatterplot(residual_y_train, ax=ax)
    ax.axhline(y=0.0, color='black', linestyle='-')
    ax.set_title('{} set residual'.format(title))
    f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
    labels = list(color_dictionary.keys()) + list(markers_dictionary.keys())
    # print(labels)
    handles = [f('s', x) for x in color_dictionary.values()] + [f(x, 'k') for x in list(markers_dictionary.values())]

    plt.legend(handles, labels, title="color:genotype, markers:year")

    # ax.figure.savefig('{}{}/train_residual_{}split_hidden{}_num_layer{}_ts_layer_{}_lr_{}_{}.png'.format(
    #     current_pwd, save_directory, n, hidden, num_layer, ts_layer, lr, mode))
    # plt.show()
    return fig_residual

def run_pinn_with_simulated_data(
            data_x_file: str = '../processed_data/simulated_data/simulated_X_data_logistic_time_dependent_noise_0.2.csv',
            weight=2, parameter_boundary=False,missing_value_precentage=0.6):
    """
    run ml model, similar to run_logistic_ode_pinn(), but with simulated biomass data (orginal data/10). Training set is
    single time serie to get r and ymax for seperate sample comparision with real parameters value
    """

    file_name = data_x_file.split('/')[-1]
    path = "/".join(data_x_file.split('/')[:-1])
    print(file_name,path)
    #read noise free data
    ode_df = pd.read_csv(data_x_file,header=0,index_col=0)

    parameters_df_list = pd.read_csv('{}/parameters_list_{}'.format(path,file_name),header=0,index_col=0).T
    print(parameters_df_list)
    Y_tensor = convert_inputx_to_tesor_list([ode_df])
    # print(Y_tensor)
    shape = Y_tensor.shape
    print(torch.count_nonzero(Y_tensor))

    index_list = [list(range(shape[0])),list(range(shape[1])),list(range(shape[2]))]
    # print(index_list)
    index_list_combination = list(itertools.product(*index_list))
    assign_missing_index = random.sample(index_list_combination, int(len(index_list_combination)*missing_value_precentage))
    # print(assign_missing_index)
    for indexs in assign_missing_index:
        # print(indexs)
        # print(*indexs)
        i,j,p = indexs[0],indexs[1],indexs[2]
        Y_tensor[i,j,p] = 0.0
    # print(Y_tensor)
    print(torch.count_nonzero(Y_tensor))
    # raise ValueError
    # #devided Y by 10, then value is betwen 0 and 1
    # Y_tensor = Y_tensor/10
    print(Y_tensor)
    Y_tensor =torch.squeeze(Y_tensor)#shape [120,300]
    import dill
    with open('scaled_{}.dill'.format("_".join(file_name.split('_')[1:])), 'wb') as file:
        dill.dump(Y_tensor, file)
    file.close()

    print(Y_tensor.shape)
    seq_length = Y_tensor.shape[0]
    num_seq = Y_tensor.shape[1]
    print('seq length {}'.format(seq_length))
    #120 time steps, between 0.1 to 12.0
    ts_X = torch.linspace(0.1, seq_length/10, seq_length).unsqueeze(dim=1).unsqueeze(dim=1).repeat(1,num_seq,1)# time sequences steps
    # print(ts_X.shape)#[120,300,1]
    parameters_df = pd.DataFrame()
    for seq in range(num_seq):

        parameter_df = parameters_df_list.iloc[seq,:]
        real_r = torch.tensor(data =parameter_df['r'])
        mmax = torch.tensor(data =parameter_df['Mmax'])
        # train model for each sequence separately
        model = RNN_ts_predict_growth_curve(num_layer_ts=2,initial_r=real_r,initial_ymax=mmax)

        Y=Y_tensor[:,seq].unsqueeze(dim=1)
        ts = ts_X[:,seq,:].requires_grad_(True)
        # print(Y)
        # print(ts)
        print(count_parameters(model))

        model, epoch, loss_ratiodf, parameters_change = train_logistic_model_simulated(model, Y, ts, epoches=2000,
                                                                                       weight=weight,
                                                                                       parameter_boundary=parameter_boundary,
                                                                                       lr=0.001)
        r,y_max = print_parameters_from_ode(model)
        new_row = pd.DataFrame(data={'r':r,'Mmax':y_max},index=[0])
        parameters_df = pd.concat([parameters_df,new_row])
        save_name = "predict_" + "_".join(file_name.split('_')[1:])
        print('{}/{}'.format(path, save_name))
        parameters_df.to_csv('{}/physic_weight_{}_{}_na_{}_{}'.format(path,weight,parameter_boundary,missing_value_precentage,save_name))
        loss_ratiodf.to_csv('{}/loss_ratio_seq{}_physic_weight_{}_na_{}_{}_{}'.format(path,seq,weight,parameter_boundary,missing_value_precentage,save_name))
        parameters_change.to_csv('{}/parameters_changing_in_training_seq{}_physic_weight_{}_na_{}_{}_{}'.format(path,seq,weight,parameter_boundary,missing_value_precentage,save_name))
        #with torch.backends.cudnn.flags(enabled=False):
        predicted_y = model(ts)

        print('predicted_y{}'.format(predicted_y.shape))
        fig,ax = plt.subplots(figsize=(12, 8))
        plot_y = torch.squeeze(copy.deepcopy(Y)).detach()
        plot_y[plot_y==0.0] = np.nan
        sns.scatterplot(y=plot_y,x= torch.squeeze(copy.deepcopy(ts)).detach(), ax=ax)
        sns.lineplot(y= torch.squeeze(copy.deepcopy(predicted_y.detach())),x= torch.squeeze(copy.deepcopy(ts)).detach(),ax=ax)
        yt0 = torch.squeeze(Y)[0].detach()
        if yt0 <= 0.0 or torch.isnan(yt0):
            yt0 = 0.0001
        parameters = [r, y_max, yt0]

        from LoadData import logistic_ode_model
        ts_plot = torch.squeeze(copy.deepcopy(ts).detach())
        print('ts plot shape:{}'.format(ts_plot.shape))
        x_y = odeint(func=logistic_ode_model, y0=yt0, t=ts_plot, args=(parameters,))[:, 0]
        # print(x_y)
        # x_y[x_y == 0.0] = np.nan
        ax.plot(ts_plot, x_y, color="g", )  # label="Height (Model)"
        ax.set_title('Logistic ODE informed PINN   r:{:.4f}, y_max:{:.4f}'.format(r, y_max),fontsize=14)
        plt.savefig('{}/figure/parameters_changing_in_training_seq{}_physic_weight_{}_na_{}_{}_{}.png'.format(path,seq,weight,parameter_boundary,missing_value_precentage,save_name))
        # plt.show()
    else:
        parameters_df.T.to_csv('{}/physic_weight_{}_na_{}_{}_{}'.format(path,weight,parameter_boundary,missing_value_precentage,save_name))


def plot_multiple_sequences_colored_based_on_label_df(input_data:torch.tensor, plot_name:str, label_df):
    """
    This function is for plot, 0 value is not shown on plot
    """

    fig, ax = plt.subplots(figsize=(12, 10))
    # plt.legend(fontsize=18)
    #plt.ylim(-1, 2)
    plt.xlim(0, 285)
    input_data[input_data == 0.0] = np.nan  # set 0.0 to nan, so it will not plot in the figure
    input_data = torch.squeeze(input_data)
    print(input_data.shape)
    print(label_df)
    colors_list = _get_colors(len(label_df.unique()))
    color_dictionary = {}
    for label,color in zip(label_df.unique(),colors_list):
        color_dictionary[label] = color
    seq_num = input_data.shape[1]
    assert seq_num == len(label_df)
    # if seq_num != torch.unique(input_data, dim=0).shape[1]:
    #     print('drop duplicate')
    #     input_data = torch.unique(input_data, dim=0)m for an initial
    #     label_df.drop_duplicates(inplace=True)
    #     print(len(label_df))
    for seq,label in zip(range(input_data.shape[1]),label_df):
        print(input_data[:,seq])
        sns.scatterplot(input_data[:,seq],label=label,ax=ax,color=color_dictionary[label],marker='o')
    # plt.axvline(x=115)
    # plt.ylim(0,1.2)
    # sns.lineplot(torch.squeeze(input_data),ax=ax)
    plt.title(plot_name)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    # plt.show()
    input_data = torch.nan_to_num(input_data, nan=0.0, posinf=0.0, neginf=0.0)
    return fig


def plot_save_predict_curve(current_pwd, title, predict_y_test, r_value_pred, save_directory,
                            test_y, ts_test, y_max_pred, marker_label, color_label, corresponding_environment,name):
    """
    plot
    """
    if isinstance(color_label, torch.Tensor):
        color_label = pd.DataFrame({'COLOR_LABLE':copy.deepcopy(color_label.detach().cpu()).numpy()})
        color_label= color_label['COLOR_LABLE']
    if isinstance(marker_label, torch.Tensor):
        marker_label = pd.DataFrame({'marker_label':copy.deepcopy(marker_label.detach().cpu()).numpy()})
        marker_label= marker_label['marker_label']
    # Need Python>3.7, which dictionaries are ordered
    from matplotlib.lines import Line2D
    fig, [ax1,ax2] = plt.subplots(nrows=2,ncols=1,figsize=(12, 10))
    ax1.set_ylim(-0.1, 1.2)
    ax1.set_xlim(0, 285)
    ax2.set_xlim(0, 285)
    #detach and change the shape for plotting
    plot_predicted_y = torch.squeeze(predict_y_test.detach(),dim=-1).cpu()
    corresponding_environment_plot_list=[]
    marker_list_env = ['x', '.', '1', '*', "$\u266B$",'2','3','4','p','<','>','^','v','h','+']
    for env_i in range(corresponding_environment.shape[-1]):
        corresponding_environment_plot = torch.squeeze(copy.deepcopy(corresponding_environment.detach()[:,:,env_i])).cpu()
        corresponding_environment_plot = pd.DataFrame(corresponding_environment_plot.numpy())#works for small dataset
        corresponding_environment_plot[corresponding_environment_plot == 0.0] = np.nan
        print('color label:{}'.format(color_label))
        # print(corresponding_environment_plot)
        corresponding_environment_plot.columns=list(color_label)
        # print(corresponding_environment_plot)
        #for multiple environment input, reformat and save df in a list
        corresponding_environment_plot_list.append(corresponding_environment_plot)

    marker_list = ['x', '.', '1', '*', "$\u266B$",'2','3','4','p','<','>','^','v','h','+']  #year is no more tha 5 currently
    #unique markers needed in plot -> correspondiing to year
    unique_markers = marker_label.unique()
    markers_dictionary = {}
    for label in unique_markers:
        #save year and marker type in dictionary
        markers_dictionary[label] =marker_list.pop(0)
    #do the same for color, whihc is corresponding to genotype
    unique_colors = color_label.unique()
    colors_list = sns.color_palette("dark", len(unique_colors)) #_get_colors(len(unique_colors))
    color_dictionary = {}
    for label,color in zip(unique_colors,colors_list):
        color_dictionary[label] = color

    y_test_copy = copy.deepcopy(torch.squeeze(test_y,dim=-1).detach()).cpu().numpy()
    y_test_copy[y_test_copy == 0.0] = np.nan # set 0.0 to nan, so it will not plot in the figure

    print(y_test_copy.shape)
    for seq, type,label in zip(range(y_test_copy.shape[1]),marker_label, color_label):
        sns.scatterplot(y_test_copy[:, seq], label=label, ax=ax1,marker=markers_dictionary[type], color=color_dictionary[label])
        # print("seq number:{}".format(seq))
        # print(label)
    print(plot_predicted_y.shape)
    for seq,type,label in zip(range(plot_predicted_y.shape[1]), marker_label, color_label):
        #ax1.plot(plot_predicted_y,color=color_dictionary[label])
        # print("seq number:{}".format(seq))
        sns.lineplot(plot_predicted_y[:,seq], ax=ax1, color=color_dictionary[label])
    # sns.scatterplot(y_test_copy, ax1=ax1)
    #set legend
    f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
    labels = list(color_dictionary.keys())+list(markers_dictionary.keys())
    # print(labels)
    handles = [f('s',x) for x in color_dictionary.values()] + [f(x,'k') for x in list(markers_dictionary.values())]

    ax1.legend(handles, labels, title="color:genotype, markers:year")
    if (r_value_pred != None) & (y_max_pred != None):
        ax_1 = ax1.twinx()
        from LoadData import logistic_ode_model
        for seq in range(plot_predicted_y.shape[1]):
            yt0 = plot_predicted_y[0, seq]
            if yt0 <= 0.0 or np.isnan(yt0):
                yt0 = 0.0001

            #plot predicted y

            if torch.is_tensor(r_value_pred):
                #if genetics input create multiple r
                r_value_pred_seq= torch.squeeze(r_value_pred)[seq].detach().cpu()
                y_max_pred_seq = torch.squeeze(y_max_pred)[seq].detach().cpu()
                parameters = [r_value_pred_seq, y_max_pred_seq, yt0]
            else:
                #otherwise it will be float
                parameters = [r_value_pred,y_max_pred,yt0]
            # print(parameters)
            #calcualted derivate from logistic ODE
            dy_dt = parameters[0] * plot_predicted_y[:,seq] * (1 - (plot_predicted_y[:,seq] / parameters[1]))
            print(dy_dt)
            # x_y = odeint(func=logistic_ode_model, y0=yt0, t=ts_test[:, 0, 0].detach().cpu(), args=(parameters,))[:, 0]
            # print(x_y)
            # x_y[x_y == 0.0] = np.nan
            # sns.lineplot(dy_dt,ax=ax1, color="g")
            ax_1.plot(ts_test[:, seq, 0].detach().cpu(), dy_dt.squeeze(), color="g", )  # label="Height (Model)"
        ax_1.set_ylim(-0.03,0.08)
    ax1.set_title('{} prediction '.format(title),fontsize=14)
    # plt.show()
    #plot input environment factors along
    for corresponding_environment_plot in corresponding_environment_plot_list:
        sns.lineplot(corresponding_environment_plot,ax=ax2,marker=marker_list_env.pop(0))
        ax2.set_title('environments input curve')
    try:
        print(current_pwd)
        # plt.savefig(plot_save_name)
        plt.savefig("{0}{1}{2}".format(current_pwd, save_directory, name))
        print('plot save in :{}{}'.format(current_pwd, save_directory))
    except:
        os.makedirs(current_pwd + save_directory)
        print("Directory '% s' created" % current_pwd + save_directory)
        plt.savefig("{0}{1}{2}".format(current_pwd, save_directory,name))
    plt.tight_layout()
    # plt.show()
    return fig
def read_full_data(rescale=False,genotype=[335],years=(2018, 2019, 2021, 2022),environment_input=['Air_temperature_2_m']):
    #get full environment input
    # generate_data_fromode(logistic_growth_ode)
    input_seq_full = pd.read_csv("../processed_data/align_height_env.csv", header=0, index_col=0)
    # reset timestamp
    input_seq_full['timestamp'] = input_seq_full['day_after_start_measure']
    input_seq_full.drop(columns='day_after_start_measure')
    if rescale:
        max_temperature= (input_seq_full['Air_temperature_2_m'].abs()).max()
        max_air_humidity = (input_seq_full['Relative_air_humidity_2_m'].abs()).max()
        max_soil_temperature = (input_seq_full['Soil_temperature_-0.05_m'].abs()).max()
        input_seq_full['Air_temperature_2_m'] = input_seq_full['Air_temperature_2_m'] / (
            input_seq_full['Air_temperature_2_m'].abs()).max()
        print(
            'rescale, devided by maximum air humidity:{}'.format((input_seq_full['Relative_air_humidity_2_m'].abs()).max()))
        input_seq_full['Relative_air_humidity_2_m'] = input_seq_full['Relative_air_humidity_2_m'] / (
            input_seq_full['Relative_air_humidity_2_m'].abs()).max()
        input_seq_full['Soil_temperature_-0.05_m'] = input_seq_full['Soil_temperature_-0.05_m'] / (input_seq_full[
                                                                                             'Soil_temperature_-0.05_m'].abs()).max()
        input_seq_full['Short_wavelenght_solar_irradiance_2_m'] = input_seq_full['Short_wavelenght_solar_irradiance_2_m'] / (
            input_seq_full['Short_wavelenght_solar_irradiance_2_m'].abs()).max()

    try:  # if one genotype
        input_seq_full = input_seq_full[input_seq_full['genotype.id'] == genotype]

    except:  # if list of genotypes
        if len(genotype) >= 2:
            print('multiple genotype')
        input_seq_full = input_seq_full[input_seq_full['genotype.id'].isin(genotype)]
        genotype = [str(g) for g in genotype]
        # mode is a string will used in files name later
    print(input_seq_full)
    model_create_data_full = create_tensor_dataset(year=list(years), dfs=[input_seq_full])
    # the g_df includes group label(genotype.id, year_site.harvest_year...), will be used for train test split based on group
    _, _, env_dfs_full = model_create_data_full.keep_overlap_time_stamps_between_multiple_features_dfs()
    # env_df_air_temperature = env_dfs['Air_temperature_2_m']
    # env_df_irradiance = env_dfs['Short_wavelenght_solar_irradiance_2_m']
    envdf_list_full = []
    for key_name in environment_input:
        print(key_name)
        envdf_list_full.append(env_dfs_full[key_name])
    else:
        envir_tensor_dataset_full = convert_inputx_to_tesor_list(envdf_list_full).permute(0, 1, 2)
    #set na to 0.0
    envir_tensor_dataset_full = torch.nan_to_num(envir_tensor_dataset_full,nan=0.0,posinf=0.0,neginf=0.0)
    with open(
            'full_env_335_no_scaling_env_tensor.dill', 'wb') as file:
        dill.dump(envir_tensor_dataset_full, file)
    file.close()
    return envir_tensor_dataset_full
def summarize_result(file='pinn_result/PINN_mask_loss.csv'):
    result_df = pd.read_csv(file,header=0,index_col=0)
    summarize_result = result_df.groupby(['lr','hidden_size','num_layer','ts_layer','weight_physic']).mean()
    print(summarize_result)
    # sns.lineplot(data=summarize_result[['weight_physic','test_MSE']])
    # plt.show()
    summarize_result.to_csv('pinn_result/mean_result_no_early_stop_301_remove_out.csv')
class fit_logistic_ode_to_plant_height():
    """
    This class fit logistic ODE to each replicates sperately
    """
    def logistic_growth_ode(self,y, time,theta):
        # data generate from logistic growth model
        r,y_max,yt0 = theta
        dY_dT = r * y * (1 - (y / y_max))
        return dY_dT
    def fit_ode(self,time:torch.tensor,fit_data,color,ax,label):
        """
        logistic_growth_ode: callable function which define the equation of ode
        :param time: time array
        :param fit_data: real data to fit ode to
        :param theta: np.array, parameters
        """
        # function that calculates residuals based on a given theta
        def ode_model_mse(theta):
            """
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html
            """
            #func: callable(y, t, …) or callable(t, y, …)

            predicted_curve = torch.squeeze(torch.tensor(odeint(func=self.logistic_growth_ode, y0=theta[-1:], t=time, args=(theta,))))
            mask = (~torch.isin(fit_data, torch.tensor(0.0))).float()
            # print('predicted_curve{}'.format(predicted_curve))
            loss = (fit_data - predicted_curve) * mask
            return loss

        # note theta =  gamma, y_max, yt0
        theta = np.array([0.05, 1.0, 0.006])
        results = least_squares(ode_model_mse, x0=theta,bounds=(0,np.inf))
        parameter_names = ["r", "y_max", "y0"]
        result_df = pd.DataFrame()
        result_df["Parameter"] = parameter_names
        parameters = results.x
        result_df["Least Squares Solution"] = parameters
        result_df.set_index(keys='Parameter', drop=True, inplace=True)
        result_df = result_df.round(3).T
        fitted_curve = torch.squeeze(torch.tensor(odeint(func=self.logistic_growth_ode, y0=parameters[-1:], t=time, args=(parameters,))))
        # print(fitted_curve)
        # print(time)

        self.plot_data_and_fit_curve(fit_data, time,fitted_curve,color=color,ax=ax,label=label)

        return result_df, fitted_curve
    def fit_dataset_and_save(self,data:torch.tensor,seq_label:pd.DataFrame,start_day=56):

        fit_ode_result = pd.DataFrame()
        print('data shape for fit ode')
        data = data[start_day:,:,:]
        print(data.shape)
        num_seq = data.shape[1]
        print(num_seq,seq_label)
        ts = torch.linspace(0.0, 284, 285).unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, num_seq,
                                                                                          1)[:-start_day, :,
                   :] # time sequences steps
        color_list=sns.color_palette("dark", num_seq)

        color_list = sns.color_palette('tab10',len(seq_label['year_site.harvest_year'].unique()))
        color_dict = dict(zip(seq_label['year_site.harvest_year'].unique(), color_list))
        fig, ax = plt.subplots(figsize=(12, 10))
        plt.ylim(-0.1, 1.5)
        plt.xlim(0, 285)
        for seq,label,year in zip(range(num_seq),seq_label['genotype.id'],seq_label['year_site.harvest_year']):
            print(seq)
            seq_data = torch.squeeze(data[:,seq,:])#.numpy().astype(float)

            ts_seq = torch.squeeze(ts[:,seq,:])#.numpy().astype(float)
            # print(ts_seq.shape)
            c= color_dict[year]
            result_df, fitted_curve = self.fit_ode(ts_seq,seq_data,color=c,ax=ax,label=year)
            # print(seq_data.shape)
            # print(fitted_curve.shape)

            RMSE = mask_rmse_loss(true_y=torch.tensor(seq_data), predict_y=torch.tensor(fitted_curve)).item()
            print('MSE{}'.format(RMSE))
            new_row = pd.DataFrame({'mask_rmse':RMSE,'label':label},index=['Least Squares Solution'])
            new_row = pd.concat([new_row,result_df],axis=1)
            fit_ode_result = pd.concat([fit_ode_result,new_row])
            # fit_ode_result.to_csv('fit_ode_rmse_g{}.csv'.format(seq_label['genotype.id'].unique()[0]))
        plt.xlabel('Time')
        plt.ylabel('Plant Height')
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        # plt.legend()
        plt.title("Genotype: {}".format(seq_label['genotype.id'].unique()[0]))
        plt.savefig('logistic_ode_{}.jpg'.format(seq_label['genotype.id'].unique()[0]))
        plt.close()
        # plt.show() #colored fitted ODE to data

        print(fit_ode_result)
        return fit_ode_result['mask_rmse'].mean()
    def plot_data_and_fit_curve(self,data,ts,fitted_curve,color='g',ax=None,label=None):
        data = copy.deepcopy(data)
        data[data==0.0]=np.nan
        sns.scatterplot(data,ax=ax,color=color,label=label)
        plt.plot(ts,fitted_curve, color=color)

        # plt.show()


def evaluate_pinn_based_on_parameters(predicted_parameters:pd.DataFrame|str,true_parameters:str,simulated_data_after_scaling:torch.tensor=None):
    from scipy.stats import pearsonr

    def logistic_growth_ode(y, time,theta):
        # data generate from logistic growth model
        r,y_max,yt0 = theta
        dY_dT = r * y * (1 - (y / y_max))
        return dY_dT
    if type(predicted_parameters) is str:
        predicted_parameters = pd.read_csv(predicted_parameters,header=0,index_col=0).T.reset_index()
    if type(true_parameters) is str:
        true_parameters = pd.read_csv(true_parameters,header=0,index_col=0).T.reset_index()
    #check length of two input
    assert len(predicted_parameters.index) == len(true_parameters.index)
    print(predicted_parameters)
    print(true_parameters['Mmax'])
    sns.scatterplot(x=predicted_parameters['Mmax'],y=true_parameters['Mmax'])
    plt.xlabel('predict')
    plt.ylabel('true')
    plt.title('predict Mmax vs true Mmax')
    plt.show()
    print(pearsonr(predicted_parameters['Mmax'], true_parameters['Mmax']))
    print(pearsonr(predicted_parameters['r'], true_parameters['r']))
    sns.scatterplot(x=predicted_parameters['r'],y=true_parameters['r'])
    plt.xlabel('predict')
    plt.ylabel('true')
    plt.title('predict r vs true r')
    plt.show()


    #generate data from predicted parameter, and compare with scaled data
    ts = torch.linspace(0.0, simulated_data_after_scaling.shape[0], simulated_data_after_scaling.shape[0]+1)# time sequences steps
    for i in predicted_parameters.index:
        r = predicted_parameters.loc[i,'r']
        ymax = predicted_parameters.loc[i, 'Mmax']
        real_curve = simulated_data_after_scaling[:,i]

        y0=real_curve[0]
        if y0 <= 0.0:
            print('y0 are asigned to 0.001')
            y0 = 0.0001
        print(y0)
        theta = [r,ymax,y0]
        fitted_curve = torch.squeeze(
            torch.tensor(odeint(func=logistic_growth_ode, y0=theta[-1:], t=ts, args=(theta,))))
        plt.plot(fitted_curve)
        sns.scatterplot(real_curve)
        plt.show()

def calculate_average_across_replicates(train_env, train_group_df, train_y, ts_train):


    train_env_avg,average_group_df = average_based_on_group_df(train_env, df=train_group_df)

    ts_train_avg,_ = average_based_on_group_df(ts_train,
                                                df=train_group_df)
    train_y_avg,_ = average_based_on_group_df(train_y,
                                               df=train_group_df)
    # print(
    #     'average y shape:{}'.format(train_y_avg.shape)
    # )

    return train_env_avg, train_y_avg, ts_train_avg,average_group_df
def same_year_fit_to_one_ode_parameter(ts,fit_data):
    from scipy.integrate import solve_ivp
    from scipy.optimize import minimize

    def logistic_growth_ode(t, y, theta):
        r, y_max = theta
        dY_dT = r * y * (1 - y / y_max)
        return dY_dT
    def solve_ode(theta, t, y0):
        sol = solve_ivp(logistic_growth_ode, [t[0], t[-1]], [y0], t_eval=t, args=(theta,),method='LSODA')#,method='LSODA'
        return sol.y[0]

    # Sum of squared errors
    def cost_function(theta, t, y_obs, y0):
        predicted_curve = solve_ode(theta, t, y0)
        mask = (~torch.isin(y_obs, torch.tensor(0.0))).float()

        if predicted_curve.shape[0]!=170:
            sns.scatterplot(fit_data, c='orange')
            plt.plot(predicted_curve,c='blue')
            plt.ylim(0, 1)
            plt.show()
        # fig.canvas.draw_idle()
        # fig.canvas.flush_events()
        # plt.clf()
        # print('predicted_curve shape{}'.format(predicted_curve.shape))
        # print('observe_y shape:{}'.format(y_obs.shape))
        return torch.sqrt(torch.mean(mask*(y_obs - predicted_curve) ** 2)) #rmse

    def objective_function(theta):
        total_rmse = 0 #sum square error
        for seq in range(fit_data.shape[1]):
            # yt0 = fit_data[0, seq]
            # if yt0 <= 0.0 or np.isnan(yt0):
            yt0 = 0.0001
            t = ts
            # print(yt0)
            y_obs = fit_data[:,seq]
            total_rmse += cost_function(theta, t, y_obs, yt0)
        return total_rmse/fit_data.shape[1]
    r_init = random.uniform(0.1, 0.2)
    ymax_init = random.uniform(0.7, 0.8)
    print('init r {}, init y: {}'.format(r_init,ymax_init))
    # theta_initial = [0.1, 0.7]#r ymax
    theta_initial = [r_init, ymax_init]  # r ymax
    # Minimize the SSE to estimate the best-fit parameters
    result = minimize(objective_function, theta_initial)
    print(result)
    theta_best = result.x
    print(f"Estimated parameters: r = {theta_best[0]}, y_max = {theta_best[1]}")
    return theta_best[0], theta_best[1]
def predict_based_on_ode(fitted_parameters, logistic_growth_ode, train_y, ts):
    precit_y = []
    for seq in range(train_y.shape[1]):
        # yt0 = train_y[0, seq]
        # if yt0 <= 0.0 or np.isnan(yt0):
        yt0 = 0.0001
        x_y = odeint(func=logistic_growth_ode, y0=yt0, t=ts, args=(fitted_parameters,))[:, 0]
        precit_y.append(torch.tensor(x_y).unsqueeze(dim=-1))
    else:
        precit_y = torch.cat(precit_y, dim=1)
    return precit_y

def mask_dtw_loss(true_y:torch.tensor, predict_y:torch.tensor):
    from shapedtw.shapedtw import shape_dtw
    from shapedtw.shapeDescriptors import SlopeDescriptor, PAADescriptor, CompoundDescriptor,DerivativeShapeDescriptor,DWTDescriptor
    from shapedtw.dtwPlot import dtwPlot
    device = true_y.device

    mask = (~torch.isin(true_y, torch.tensor(0.0).to(device=device))).float()
    masked_true_y = mask * true_y
    # print(masked_true_y)
    masked_predict_y = mask * predict_y
    # true_y[true_y==0.0]=np.nan
    shape_dtw_distance = 0.0
    for seq in range(masked_true_y.shape[1]):
        seq1= copy.deepcopy(masked_true_y.detach().cpu())[:,seq].numpy()
        index_not_na = seq1 != 0.0
        seq1 = seq1[index_not_na]
        seq2= copy.deepcopy(masked_predict_y.detach().cpu())[:,seq].numpy()
        seq2 = seq2[index_not_na]

        df = pd.DataFrame({"ts_x": seq1, "ts_y": seq2})
        # df.plot()
        # plt.show()
        slope_descriptor = SlopeDescriptor(slope_window=5)
        derivative_descriptor = DerivativeShapeDescriptor()
        paa_descriptor = PAADescriptor(piecewise_aggregation_window=5)
        dwt_descriptor = DWTDescriptor()
        # raw_descriptor = RawSubsequenceDescriptor()
        compound_descriptor = CompoundDescriptor([derivative_descriptor, slope_descriptor, paa_descriptor,dwt_descriptor],
                                                 descriptors_weights=[1., 1., 1.,1.])
        # print(seq1,seq2)
        shape_dtw_results = shape_dtw(
            x=seq1,
            y=seq2,
            subsequence_width=5, #need to figure out how to choose that
            shape_descriptor=compound_descriptor
        )

        # dtwPlot(shape_dtw_results, plot_type="twoway", yoffset=1)
        shape_dtw_distance += shape_dtw_results.shape_normalized_distance
        #normalized_distance attributes of classes representing shape dtw results.
        # print(round(shape_dtw_results.shape_normalized_distance, 2))
    else:
        return shape_dtw_distance/true_y.shape[1]

def ODE_fit_run():
    """
    This class is to use logistic ODE as single genotype prediction model: fit on two year and predict on another two
    """
    def logistic_growth_ode(y,t, theta):
        r, y_max = theta
        dY_dT = r * y * (1 - y / y_max)
        return dY_dT

    def plot_ODE_fit(g, precit_y_test, rmse_loss_test, shape_dtw_loss_test, test_y_avg, ts_test,add_name=''):
        test_y_avg[test_y_avg == 0.0] = np.nan
        plt.scatter(x=ts_test, y=test_y_avg, c='blue')
        # for seq in range(test_y_avg.shape[1]):
        #     plt.scatter(x=ts_test, y=test_y_avg[:,seq], c='blue')
        plt.plot(precit_y_test, c='orange')
        plt.ylim(0, 1.5)
        plt.title('genotype {} RMSE:{}'.format(g, round(rmse_loss_test, 3)))
        plt.savefig('../figure/logistic_ODE_fit/single_genotype_model/genotype{}{}logistic ODE test predict curves.png'.format(g,add_name))
        plt.clf()
        test_y_avg = torch.nan_to_num(test_y_avg, nan=0.0)
        return test_y_avg
    from rf_plant_height_prediction import read_tensor_files
    ode_df = pd.DataFrame()
    all_ode_result_before_average = pd.DataFrame()
    average_train_errors, std_trains, average_val_errors, std_vals, average_test_errors, std_tests = [[] for _ in
                                                                                                   range(6)]
    dtw_std_trains,dtw_std_vals,dtw_std_tests,average_train_dtw_errors,average_val_dtw_errors,average_test_dtw_errors = [[] for _ in
                                                                                                   range(6)]
    for g in [33, 106, 122, 133, 5, 30, 218, 2, 17, 254, 282, 294, 301, 302, 335, 339, 341, 6, 362]:
        training_losses = []
        validation_losses = []
        test_losses = []
        training_dtw_losses = []
        validation_dtw_losses = []
        test_dtw_losses = []
        train_test_validation_dictionary, plant_height_tensor, temperature_same_length_tensor, genetics_input_tensor,year_df,group_df = read_tensor_files(
            g)
        plant_height_tensor = torch.squeeze(plant_height_tensor)
        temperature_same_length_tensor = torch.squeeze(temperature_same_length_tensor)

        train_index, validation_index, test_index = train_test_validation_dictionary['splits_{}'.format(0)]
        train_y = plant_height_tensor[115:, train_index]
        train_group = copy.deepcopy(group_df).iloc[train_index, :]
        # train_env = temperature_same_length_tensor[91:, train_index]
        validation_y = plant_height_tensor[115:, validation_index]
        validation_group = copy.deepcopy(group_df).iloc[validation_index, :]
        # validation_env = temperature_same_length_tensor[91:, validation_index]
        test_y = plant_height_tensor[115:, test_index]
        test_group = copy.deepcopy(group_df).iloc[test_index, :]
        # test_env = temperature_same_length_tensor[91:, test_index]

        ts_train = np.linspace(0.0, 284, 285)[:-115]
        ts_validation = np.linspace(0.0, 284, 285)[:-115]
        ts_test = np.linspace(0.0, 284, 285)[:-115]

        # print(i)
        index_year1 = year_df[year_df['year_site.harvest_year'] == 2018].index
        index_year2 = year_df[year_df['year_site.harvest_year'] == 2019].index
        train_y1 = train_y[:, index_year1.values]
        train_y2 = train_y[:, index_year2.values]

        # plt.plot(train_y2)
        # plt.show()
        print(train_y1.shape)
        print(ts_train.shape)
        print(train_group)
        train_y_avg, _ = average_based_on_group_df(train_y.unsqueeze(dim=-1),
                                                   df=train_group)
        train_y_avg = train_y_avg.squeeze(dim=-1)
        validation_y_avg, _ = average_based_on_group_df(validation_y.unsqueeze(dim=-1),
                                                   df=validation_group)
        validation_y_avg = validation_y_avg.squeeze(dim=-1)
        test_y_avg, _ = average_based_on_group_df(test_y.unsqueeze(dim=-1),
                                                   df=test_group)
        test_y_avg = test_y_avg.squeeze(dim=-1)
        print(train_y_avg.shape)
        print(validation_y_avg.shape)

        # plt.ion()
        # fig = plt.figure()
        for seed in [1,2,3,4,5]:
            random.seed(seed)
            np.random.seed(seed)
            '''
            train_rmse_seq=0.0
            train_dtw_seq=0.0
            for seq in range(train_y.shape[1]):
                r1, ymax1 = same_year_fit_to_one_ode_parameter(ts_train, train_y[:,seq].unsqueeze(dim=-1))
                fitted_parameters1 = [r1, ymax1]
                precit_y_train1 = predict_based_on_ode(fitted_parameters1, logistic_growth_ode,
                                                       train_y[:,seq].unsqueeze(dim=-1), ts_train)
                rmse_loss_train1 = mask_rmse_loss(train_y[:,seq].unsqueeze(dim=-1), precit_y_train1).item()
                train_rmse_seq +=rmse_loss_train1
                shape_dtw_loss_train1 = mask_dtw_loss(train_y[:,seq].unsqueeze(dim=-1), precit_y_train1)
                train_dtw_seq+=shape_dtw_loss_train1
                # raise EOFError('unfinished code')
            else:
                rmse_loss_train=train_rmse_seq/train_y.shape[1]
                training_dtw_losse = train_dtw_seq/train_y.shape[1]
            '''
            r1,ymax1 = same_year_fit_to_one_ode_parameter(ts_train,train_y1)
            fitted_parameters1=[r1,ymax1]
            # precit_y_train1 = predict_based_on_ode(fitted_parameters1, logistic_growth_ode, train_y1, ts_train)
            # rmse_loss_train1 = mask_rmse_loss(train_y1,precit_y_train1).item()
            precit_y_train1 = predict_based_on_ode(fitted_parameters1, logistic_growth_ode, train_y_avg[:,0].unsqueeze(dim=-1), ts_train)
            rmse_loss_train1 = mask_rmse_loss(train_y_avg[:,0].unsqueeze(dim=-1),precit_y_train1).item()
            shape_dtw_loss_train1 = mask_dtw_loss(train_y_avg[:,0].unsqueeze(dim=-1),precit_y_train1)
            print(rmse_loss_train1)
            # raise EOFError
            r2,ymax2 = same_year_fit_to_one_ode_parameter(ts_train,train_y2)
            fitted_parameters2=[r2,ymax2]
            r= (r1+r2)/2
            y_max= (ymax1+ymax2)/2
            precit_y_train2 = predict_based_on_ode(fitted_parameters2, logistic_growth_ode, train_y_avg[:,1].unsqueeze(dim=-1), ts_train)
            rmse_loss_train2 = mask_rmse_loss(train_y_avg[:,1].unsqueeze(dim=-1),precit_y_train2).item()
            shape_dtw_loss_train2 = mask_dtw_loss(train_y_avg[:,1].unsqueeze(dim=-1),precit_y_train2)
            rmse_loss_train = (rmse_loss_train1+rmse_loss_train2)/2
            training_dtw_losse = (shape_dtw_loss_train1+shape_dtw_loss_train2)/2
            training_losses.append(rmse_loss_train)
            training_dtw_losses.append(training_dtw_losse)
            print('train rmse{}'.format(rmse_loss_train))

            '''
            #this is to fit each replcate seperatelly
            val_rmse_seq=0.0
            val_dtw_seq=0.0
            for seq in range(validation_y.shape[1]):
                r_val, ymax_val = same_year_fit_to_one_ode_parameter(ts_validation, validation_y[:,seq].unsqueeze(dim=-1))
                # r_val, ymax_val = same_year_fit_to_one_ode_parameter(ts_validation, validation_y_avg)
                fitted_parameters = [r_val, ymax_val]
                precit_y_val = predict_based_on_ode(fitted_parameters, logistic_growth_ode, validation_y[:,seq].unsqueeze(dim=-1),
                                                    ts_validation)
                # train_y[train_y == 0.0] = np.nan
                rmse_loss_val = mask_rmse_loss(validation_y[:,seq].unsqueeze(dim=-1), precit_y_val).item()
                val_rmse_seq+=rmse_loss_val
                shape_dtw_loss_val = mask_dtw_loss(validation_y[:,seq].unsqueeze(dim=-1), precit_y_val)
                val_dtw_seq+=shape_dtw_loss_val
            else:
                rmse_loss_val=val_rmse_seq/validation_y.shape[1]
                shape_dtw_loss_val = val_dtw_seq/validation_y.shape[1]
            '''
            # print(validation_y_avg)
            fitted_parameters = [r, y_max]
            # r_val,ymax_val = same_year_fit_to_one_ode_parameter(ts_validation,validation_y)
            # r_val, ymax_val = same_year_fit_to_one_ode_parameter(ts_validation, validation_y)
            # fitted_parameters=[r_val,ymax_val]
            precit_y_val = predict_based_on_ode(fitted_parameters, logistic_growth_ode, validation_y_avg, ts_validation)
            rmse_loss_val = mask_rmse_loss(validation_y_avg,precit_y_val).item()
            shape_dtw_loss_val = mask_dtw_loss(validation_y_avg,precit_y_val)
            print('val rmse{}'.format(rmse_loss_val))
            validation_losses.append(rmse_loss_val)
            validation_dtw_losses.append(shape_dtw_loss_val)

            # r_test,ymax_test = same_year_fit_to_one_ode_parameter(ts_test,test_y)
            # r_test, ymax_test = same_year_fit_to_one_ode_parameter(ts_test, test_y_avg)
            # fitted_parameters=[r_test,ymax_test]
            # precit_y_test_fit = predict_based_on_ode(fitted_parameters, logistic_growth_ode, test_y, ts_test)
            # rmse_loss_testfit = mask_rmse_loss(true_y=test_y,predict_y=precit_y_test_fit).item()
            # print('fit parameter ODE:{}'.format(rmse_loss_testfit))
            precit_y_test = predict_based_on_ode(fitted_parameters, logistic_growth_ode, test_y_avg, ts_test)
            # print(precit_y_val,precit_y_test)
            # assert precit_y_val==precit_y_test
            rmse_loss_test = mask_rmse_loss(true_y=test_y_avg,predict_y=precit_y_test).item()
            shape_dtw_loss_test = mask_dtw_loss(true_y=test_y_avg,predict_y=precit_y_test)
            print('test rmse{}'.format(rmse_loss_test))
            plot_ODE_fit(g, precit_y_test, rmse_loss_test, shape_dtw_loss_test,
                         copy.deepcopy(test_y_avg), ts_test, add_name='')
            # # ax,fig = plt.subplots()
            #


            # test_rmse_seq=0.0
            # test_dtw_seq=0.0
            # for seq in range(test_y.shape[1]):
            #     r_test, ymax_test = same_year_fit_to_one_ode_parameter(ts_validation, test_y[:,seq].unsqueeze(dim=-1))
            #     # r_val, ymax_val = same_year_fit_to_one_ode_parameter(ts_validation, validation_y_avg)
            #     fitted_parameters = [r_test, ymax_test]
            #     precit_y_test = predict_based_on_ode(fitted_parameters, logistic_growth_ode, test_y[:,seq].unsqueeze(dim=-1),
            #                                         ts_validation)
            #     # train_y[train_y == 0.0] = np.nan
            #     rmse_loss_test = mask_rmse_loss(test_y[:,seq].unsqueeze(dim=-1), precit_y_test).item()
            #     test_rmse_seq+=rmse_loss_test
            #     shape_dtw_loss_test = mask_dtw_loss(test_y[:,seq].unsqueeze(dim=-1), precit_y_val)
            #     test_dtw_seq+=shape_dtw_loss_test
            #     plot_ODE_fit(g, precit_y_test, rmse_loss_test, shape_dtw_loss_test, copy.deepcopy(test_y[:,seq].unsqueeze(dim=-1)), ts_test,add_name='{}'.format(seq))
            # else:
            #     rmse_loss_test=val_rmse_seq/test_y.shape[1]
            #     shape_dtw_loss_test = val_dtw_seq/test_y.shape[1]

            test_losses.append(rmse_loss_test)
            test_dtw_losses.append(shape_dtw_loss_test)

            new_line_raw = pd.DataFrame({'genotype': g, 'random_sees':seed,'validation_rMSE': rmse_loss_val, 'train_rMSE': rmse_loss_train,
                                     'test_rMSE': rmse_loss_test, 'validation_shapDTW': shape_dtw_loss_val, 'train_shapDTW': training_dtw_losse,
                                     'test_shapDTW': shape_dtw_loss_test, "predicted_r":r,'predicted_y_max':y_max},index=[g])
            all_ode_result_before_average = pd.concat([all_ode_result_before_average,new_line_raw])
        else:
            avg_train_rmse = np.mean(training_losses)
            std_train_rmse = np.std(training_losses)
            avg_val_rmse = np.mean(validation_losses)
            std_val_rmse = np.std(validation_losses)
            avg_test_rmse = np.mean(test_losses)
            std_test_rmse = np.std(test_losses)
            average_train_errors.append(avg_train_rmse)
            std_trains.append(std_train_rmse)
            average_val_errors.append(avg_val_rmse)
            std_vals.append(std_val_rmse)
            average_test_errors.append(avg_test_rmse)
            std_tests.append(std_test_rmse)
            print(f"\nAverage Train RMSE: {round(avg_train_rmse,3)}, Std: {round(std_train_rmse,3)}")
            print(f"Average Validation RMSE: {round(avg_val_rmse,3)}, Std: {round(std_val_rmse,3)}")
            print(f"Average Test RMSE: {round(avg_test_rmse,3)}, Std: {round(std_test_rmse,3)}")

            avg_train_dtw = np.mean(training_dtw_losses)
            dtw_std_train = np.std(training_dtw_losses)
            avg_val_dtw = np.mean(validation_dtw_losses)
            dtw_std_val = np.std(validation_dtw_losses)
            avg_test_dtw = np.mean(test_dtw_losses)
            dtw_std_test = np.std(test_dtw_losses)
            average_train_dtw_errors.append(avg_train_dtw)
            dtw_std_trains.append(dtw_std_train)
            average_val_dtw_errors.append(avg_val_dtw)
            dtw_std_vals.append(dtw_std_val)
            average_test_dtw_errors.append(avg_test_dtw)
            dtw_std_tests.append(dtw_std_test)
            print(f"\nAverage Train DTW: {round(avg_train_dtw,3)}, Std: {round(dtw_std_train,3)}")
            print(f"Average Validation DTW: {round(avg_val_dtw,3)}, Std: {round(dtw_std_val,3)}")
            print(f"Average Test DTW: {round(avg_test_dtw,3)}, Std: {round(dtw_std_test,3)}")


            new_line = pd.DataFrame({'genotype': g, 'validation_rMSE': avg_val_rmse, 'train_rMSE': avg_train_rmse,
                                     'test_rMSE': avg_test_rmse, 'train_std': std_train_rmse,
                                     'test_std': std_test_rmse, 'validation_std': std_val_rmse, 'validation_shapDTW': avg_val_dtw, 'train_shapDTW': avg_train_dtw,
                                     'test_shapDTW': avg_test_dtw, 'dtw_std_train': dtw_std_train,
                                     'dtw_std_val': dtw_std_val, 'dtw_std_test': dtw_std_test,"predicted_r":r,'predicted_y_max':y_max},index=[g])
            ode_df = pd.concat([ode_df,new_line])
    else:
        all_ode_result_before_average.to_csv('logistic_ode_predict_multiple_genotype_raw.csv')
        average_train_error = np.mean(average_train_errors)

        std_train = np.mean(std_trains)
        average_val_error = np.mean(average_val_errors)
        std_val = np.mean(std_vals)
        average_test_error = np.mean(average_test_errors)
        std_test = np.mean(std_tests)
        print(f"\nAverage train RMSE across all genotypes: {average_train_error} std:{std_train}")
        print(f"\nAverage Validation RMSE across all genotypes: {average_val_error} std:{std_val}")
        print(f"\nAverage test RMSE across all genotypes: {average_test_error} std:{std_test}")

        average_train_error = np.mean(average_train_dtw_errors)
        dtw_std_train = np.mean(dtw_std_trains)
        average_val_error = np.mean(average_val_dtw_errors)
        dtw_std_val = np.mean(dtw_std_vals)
        average_test_error = np.mean(average_test_dtw_errors)
        dtw_std_test = np.mean(dtw_std_tests)
        print(f"\nAverage train shapeDTW across all genotypes: {average_train_error} std:{dtw_std_train}")
        print(f"\nAverage Validation shapeDTW across all genotypes: {average_val_error} std:{dtw_std_val}")
        print(f"\nAverage test shapeDTW across all genotypes: {average_test_error} std:{dtw_std_test}")
        ode_df.to_csv('logistic_ode_predict_multiple_genotype.csv')




def plot_ode_fit_result(file_name):
    # plot
    from Plot_analysis_result import order_genotype_based_on_their_similarity
    df_ode_parameters = pd.read_csv(file_name,header=0,index_col=0).drop_duplicates()
    df_ode_full_result = pd.read_csv('logistic_ode_fit_multiple_genotype_raw.csv', header=0, index_col=0).drop_duplicates()
    # df_ode_parameters['genotype'] = df_ode_parameters.index
    df_ode_parameters.reset_index(inplace=True,drop=True)

    order_g,g_similarity = order_genotype_based_on_their_similarity(first_genotype='106')
    # print(g_similarity)
    g_similarity= torch.tensor(g_similarity)
    print(g_similarity)
    # g_similarity[0]=0.2
    simlarity_df = pd.DataFrame({'genotype':order_g,'similarity_compard_with_106':g_similarity},index=range(len(order_g)))
    simlarity_df['genotype'] =simlarity_df['genotype'].astype(int)
    df_ode_parameters['genotype'] =df_ode_parameters['genotype'].astype(int)
    # print(df_ode_parameters)
    # print(df_ode_parameters['genotype'].unique())
    df_ode_parameters = pd.merge(df_ode_parameters,simlarity_df,'left')
    print(df_ode_parameters)

    plt.figure(figsize=(10, 6))
    # palette = sns.color_palette("viridis", len(df_ode_parameters['similarity_compard_with_106'].values))
    scatter = plt.scatter(data=df_ode_parameters, x='predicted_r', y='predicted_y_max',
                    c=df_ode_parameters['similarity_compard_with_106'].values,cmap='viridis', s=100)

    colormap = plt.cm.viridis
    norm = plt.Normalize(df_ode_parameters['similarity_compard_with_106'].min(), df_ode_parameters['similarity_compard_with_106'].max())

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

    plt.colorbar(scatter, label='Kinship correlation')

    # Customize the plot
    plt.title(' logistic ODE parameter VS kinship matrix'.format(file_name))
    plt.xlabel('predicted_r')
    plt.ylabel('predicted_y_max')
    # plt.legend(title='Genotype',loc='right',bbox_to_anchor=(0.5, 1.05))
    plt.grid()
    plt.show()

    # plt.figure(figsize=(10, 6))
    # palette = sns.color_palette("husl", len(df_ode_parameters['genotype'].unique()))
    # sns.scatterplot(data=df_ode_parameters, x='predicted_r', y='predicted_y_max', hue='genotype', palette=palette,
    #                 s=50)
    #
    # # Add genotype IDs as annotations
    # for i in range(len(df_ode_parameters)):
    #     print(i)
    #     plt.text(df_ode_parameters['predicted_r'][i], df_ode_parameters['predicted_y_max'][i],
    #              df_ode_parameters['genotype'][i],
    #              horizontalalignment='left', size='medium',
    #              color=palette[i], weight='semibold')
    #
    # # Customize the plot
    # plt.title(' r vs. ymax {}'.format(file_name))
    # plt.xlabel('predicted_r')
    # plt.ylabel('predicted_y_max')
    # plt.legend(title='Genotype',loc='upper left',bbox_to_anchor=(1,1))
    # plt.grid()
    # plt.tight_layout()
    # plt.show()

    df = df_ode_full_result
    # df.to_csv('multipl_g_result_summary.csv')
    print("average train rmse:{}".format(df['train_rMSE'].mean()))
    print("average validation rmse:{}".format(df['validation_rMSE'].mean()))
    print("average test rmse:{}".format(df['test_rMSE'].mean()))
    print("average train std:{}".format(df['train_rMSE'].std()))
    print("average validation std:{}".format(df['validation_rMSE'].std()))
    print("average test std:{}".format(df['test_rMSE'].std()))
    print(f"\nAverage train shapeDTW across all genotypes: {df['train_shapDTW'].mean()} std:{df['train_shapDTW'].std()}")
    print(f"\nAverage Validation shapeDTW across all genotypes: {df['validation_shapDTW'].mean()} std:{df['validation_shapDTW'].std()}")
    print(f"\nAverage test shapeDTW across all genotypes: {df['test_shapDTW'].mean()} std:{df['test_shapDTW'].std()}")
    # Reshape the DataFrame into long form for easier plotting

    df_long = pd.melt(
        df_ode_parameters,
        id_vars=["genotype"],
        value_vars=["train_rMSE", "validation_rMSE", "test_rMSE"],
        var_name="metric",
        value_name="value"
    )

    plt.figure(figsize=(10, 6))
    plt.ylim(0, 0.3)
    sns.boxplot(x='genotype', y='value', hue='metric', data=df_long, dodge=True, order=order_g)
    plt.legend(title='Set Type', loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()

    # plot
    if file_name == 'pinn_penalize_r':
        plt.title('RMSE and std by Genotype for single genotype model PINN (penalize_r)')
    elif file_name == 'pinn':
        plt.title('RMSE and std by Genotype for single genotype model PINN')
    elif file_name == 'ml':
        plt.title('RMSE and std by Genotype for single genotype model (pure ML)')
    plt.xlabel('Genotype')
    plt.ylabel('RMSE')
    plt.show()



    # df_ode_parameters['genotype'] = df_ode_parameters['genotype'].astype(int)
    # df_ode_parameters=df_ode_parameters[df_ode_parameters['genotype'].isin([106, 122, 133,17,218,254,  282, 294,2,301, 302,30,335, 339,33, 341, 362, 5, 6])]
    # df_ode_parameters['genotype'] = pd.Categorical(df_ode_parameters['genotype'], categories=order_g, ordered=True)
    # df_ode_parameters = df_ode_parameters.sort_values('genotype')
    # df_ode_parameters = df_ode_parameters.reset_index(drop=True)
    # print(df_ode_parameters['genotype'].unique())
    # plt.figure(figsize=(10, 6))
    # palette = sns.color_palette("husl", len(df_ode_parameters['genotype'].unique()))
    # print(len(palette))
    # sns.scatterplot(data=df_ode_parameters, x='predicted_r', y='predicted_y_max', hue='genotype', palette=palette,
    #                 s=100,hue_order=df_ode_parameters['genotype'].unique())
    #
    # # Add genotype IDs as annotations
    # for i,g in enumerate(df_ode_parameters.index):
    #     print(i)
    #     print(df_ode_parameters['predicted_r'])
    #     plt.text(df_ode_parameters.loc[g,'predicted_r'], df_ode_parameters.loc[g,'predicted_y_max'],
    #              df_ode_parameters.loc[g,'genotype'],
    #              horizontalalignment='left', size='medium',
    #              color=palette[i], weight='semibold')
    #
    # # Customize the plot
    # plt.title(' r vs. ymax ')
    # plt.xlabel('predicted_r')
    # plt.ylabel('predicted_y_max')
    # plt.legend(title='Genotype')
    # plt.grid()
    # plt.show()
    # df_ode_parameters.index = df_ode_parameters['genotype'].to_list()
    # df = df_ode_parameters[
    #     ['validation_rMSE', 'train_rMSE', 'test_rMSE', 'train_std', 'test_std', 'validation_std',
    #      ]]
    # print("average train rmse:{}".format(df['train_rMSE'].mean()))
    # print("average validation rmse:{}".format(df['validation_rMSE'].mean()))
    # print("average test rmse:{}".format(df['test_rMSE'].mean()))
    # print("average train std:{}".format(df['train_std'].mean()))
    # print("average validation std:{}".format(df['validation_std'].mean()))
    # print("average test std:{}".format(df['test_std'].mean()))
    # # Reshape the DataFrame into long form for easier plotting
    # df_long = df.reset_index().melt(id_vars='index', var_name='metric', value_name='value')
    # print(df_long)
    # # Split the 'metric' column into 'set_type' (train/val/test) and 'metric_type' (rmse/std)
    # df_long['set_type'] = df_long['metric'].apply(lambda x: x.split('_')[0])  # train, val, test
    # df_long['metric_type'] = df_long['metric'].apply(lambda x: x.split('_')[1])  # rmse, std
    #
    # # Pivot to get RMSE and STD separately
    # df_rmse = df_long[df_long['metric_type'] == 'rMSE']
    # df_std = df_long[df_long['metric_type'] == 'std']
    #
    # # Merge RMSE and std dataframes based on Genotype.id (index) and set_type
    # df_final = pd.merge(df_rmse, df_std, on=['index', 'set_type'], suffixes=('_rMSE', '_std'))
    # print(df_final)
    # #just to make sure it follows the same order as other two plots
    # df_final['index'] = pd.Categorical(df_final['index'], categories=order_g, ordered=True)
    # # Plot the data using seaborn
    # plt.figure(figsize=(10, 6))
    # sns.pointplot(x='index', y='value_rMSE', hue='set_type', data=df_final,
    #               dodge=True, markers=['o', 's', 'D'], capsize=.1, ci=None, join=False)
    #
    # plt.ylim(0, 0.3)
    # # Add error bars manually
    # for i in range(df_final.shape[0]):
    #     print(df_final['index'].iloc[i])
    #     plt.errorbar(x=df_final['index'].iloc[i],
    #                  y=df_final['value_rMSE'].iloc[i],
    #                  yerr=df_final['value_std'].iloc[i],
    #                  fmt='none', capsize=5, color='gray')
    #
    # # plot
    # plt.title('RMSE and std by Genotype for single genotype model (PINN)')
    # plt.xlabel('Genotype')
    # plt.ylabel('RMSE')
    # plt.legend(title='Set Type')
    # plt.show()

def average_multiple_g_pinn_ode_parameters():
    """
    average across different random seed result from PINN
    """

    files = glob.glob("pinn_multiple_g_parameters_7_train_lr_0.005_hidden_5_seed_*.csv")
    print(files)
    with open('genotyp_mapp_dictionary.dill','rb') as f:
        index_to_genotype=dill.load(f)
    f.close()
    # files = [str(x).split('\\')[1] for x in files]
    df_avg = None
    for f in files:
        print(f)
        df  = pd.read_csv('{}'.format(f),header=0,index_col=0)
        df['genotype']=df['genotype'].map(index_to_genotype)
        df.index = df['genotype']
        try:
            if df_avg == None:
                df_avg = df.copy()
                df_avg[:] = 0.0
        except:
            print('finish_initialize')
        # print(df)
        df_avg += df
        # print(df_avg)
    df_avg /= len(files)
    print(df_avg)
    df_avg['genotype'] = df_avg.index
    df_avg = df_avg.reset_index(drop=True)
    print(df_avg)
    df_avg.to_csv('pinn_multiple_g_parameters_7_train_lr_0.005_hidden_5_predicted_parameters.csv')

def average_based_on_group_df(tensor_data,df):
    # Initialize lists to hold the averaged tensor and reduced DataFrame rows
    averaged_tensors = []
    reduced_rows = []
    #convet masked value to na for calculate average
    data_type = tensor_data.dtype
    req_grad= tensor_data.requires_grad
    device = tensor_data.device
    tensor_data = copy.deepcopy(tensor_data.clone().detach())


    tensor_data = tensor_data.float()
    tensor_data[tensor_data == 0.0] = np.nan
    # print('input tensor shape before averge:{}'.format(tensor_data.shape))
    df = df.reset_index(drop=True)
    # Group DataFrame by 'group' column
    grouped = df.groupby(['new_group_list'])

    # Loop over each group in the DataFrame
    for group_name, indices in grouped.groups.items():
        # Get the indices of the samples belonging to the current group
        sample_indices = list(indices)
        # print('group name used for average:{}'.format(group_name))
        # print(sample_indices)
        if len(tensor_data.shape)==3:
            # Select the corresponding samples from the tensor
            selected_tensor = tensor_data[:, sample_indices, :]
            # print('select tensor shape:{}'.format(selected_tensor.shape))
            average_tensor = selected_tensor.mean(dim=1,keepdim=True)  # Average across time steps
            # print(average_tensor.shape)
            # Append the average tensor for this group
            averaged_tensors.append(average_tensor)  # Further reduce across samples if needed
        else:
            raise EOFError ('check input shape')

        # Append the corresponding group name (one row for each group)
        reduced_rows.append({'group': group_name,'genotype.id':df.iloc[indices,:]['genotype.id'].unique()[0],
                             'year_site.harvest_year':df.iloc[indices,:]['year_site.harvest_year'].unique()[0]})
    if len(tensor_data.shape) == 3:
        # cat the averaged tensors along the new sample dimension
        final_tensor = torch.cat(averaged_tensors, dim=1)
    else:
        raise EOFError ('check input shape')
    # Create a reduced DataFrame with the group names
    reduced_df = pd.DataFrame(reduced_rows)

    # Now final_tensor is the averaged tensor and reduced_df is the corresponding reduced DataFrame
    # print("Averaged Tensor Shape:", final_tensor.shape)
    # print("Reduced DataFrame:")
    # print(len(reduced_df.index))
    #it was convert to float at first, now convert back
    final_tensor = torch.nan_to_num(final_tensor, nan=0.0, posinf=0.0, neginf=0.0).to(data_type)

    if req_grad:
        final_tensor = final_tensor.requires_grad_(True).to(device)
    else:
        final_tensor = final_tensor.to(device)

    return final_tensor,reduced_df

def load_best_hyperparameters_and_cross_validate(if_pinn=True, best_hyperparameter_file='best_model_result_summary/pinn_penalize_r_best_hyperparameters_result.csv',
                                                 smooth_temp=False,rescale=False,fill_in_na_at_start=True,start_day=115,genotype=106):

    mode = best_hyperparameter_file.split('/')[-1].split('best_hyperparameters_result.csv')[0]
    print(mode)
    # raise EOFError
    mode = str(genotype)+mode + 'smooth_temp'+str(smooth_temp) +'rescale' + str(rescale)
    parameter_boundary='penl_r'
    save_directory='pinn_result/result_summary/best_model_cv/'
    best_hyperparameter_df = pd.read_csv(best_hyperparameter_file,header=0,index_col=0)
    print(best_hyperparameter_df.columns)
    # group_object=best_hyperparameter_df.groupby('genotype')
    # for genotype in group_object.groups:
    result_df=best_hyperparameter_df[best_hyperparameter_df['genotype']==genotype]
    print(result_df)
    hidden = result_df['hidden_size'].unique().item()
    num_layer = result_df['num_layer'].unique().item()
    lr = result_df['lr'].unique().item()
    weight_physic = result_df['weight_physic'].unique().item()
    ode_int_loss = result_df['ode_int'].unique().item()
    L2 = result_df['l2'].unique().item()
    trainable_params_num = result_df['Trainable_Params'].unique().item()

    with open('../temporary/plant_height_tensor_{}.dill'.format(genotype), 'rb') as f:
        plant_height_tensor = dill.load(f)
    f.close()
    with open('../temporary/group_list_df_{}.dill'.format(genotype), 'rb') as f:
        group_df = dill.load(f)
    print(group_df)
    f.close()
    with open('../temporary/temperature_tensor_same_length_{}.dill'.format(genotype), 'rb') as f:
        envir_tensor_dataset = dill.load(f)
    f.close()
    genotype_list = copy.deepcopy(group_df['genotype.id'])

    if rescale:
        # if rescale the order for processing envir_tensor_dataset need to be: minmax scaling -> smoothing ->
        # (drop values based on tensor_dataset) -> replace na with 0.0 -> (fill in small value at start for tensor_dataset)
        print('rescale environment to 0 and 1')
        envir_tensor_dataset, scaler = minmax_scaler(envir_tensor_dataset)

    if smooth_temp:
        # #moving averge to smooth temperature, use 15 days before measured date to calculate average
        envir_tensor_dataset = smooth_tensor_ignore_nan(envir_tensor_dataset, 15)
        # drop to the same length
        # envir_tensor_dataset[tensor_dataset==0.0]=np.nan
        # set na to 0.0

    # smooth then set as the same length
    envir_tensor_dataset = torch.nan_to_num(envir_tensor_dataset, nan=0.0, posinf=0.0, neginf=0.0)

    if fill_in_na_at_start:
        # find the minimum value position, set all value before as a very small number 0.0001
        plant_height_tensor[plant_height_tensor == 0.0] = 999.0  # set na to 999 to find minimum
        for seq in range(plant_height_tensor.shape[1]):
            plant_height_tensor = torch.nan_to_num(plant_height_tensor, nan=0.0, posinf=0.0, neginf=0.0)

            min_position = torch.argmin(plant_height_tensor[:, seq, :]).item()
            if torch.min(plant_height_tensor[:, seq, :]).item() > 0.0:
                plant_height_tensor[:min_position + 1, seq, :] = torch.min(plant_height_tensor[:, seq, :]).item()
            else:
                plant_height_tensor[:min_position + 1, seq, :] = 0.0001
        else:
            plant_height_tensor[plant_height_tensor == 999.0] = 0.0  # set nan back to 0.0

    try:
        # dataframe to same result
        pinn_result = pd.read_csv('pinn_result/result_summary/best_model_cv/PINN_mask_loss_{}_cv.csv'.format(mode), header=0,
                                  index_col=0)
        print('connect result to existing pinn_result file')
    except:
        print('result_summary/PINN_mask_loss_{}.csv Do not exist, create new file'.format(mode))
        pinn_result = pd.DataFrame()

    group_df = group_df[['year_site.harvest_year', 'genotype.id','new_group_list']]

    train_test_validation_dictionary,train_years = manually_data_split_based_on_one_group(group_df,split_group='year_site.harvest_year')
    n_split = len(train_test_validation_dictionary.keys())
    # assert n_split==6
    # raise EOFError
    for n in range(n_split):# different split range(n_split)
        train_index, validation_index, test_index = train_test_validation_dictionary['splits_{}'.format(n)]
        # get train test split data
        print('yearly data use for training {}:'.format(len(train_index)))
        train_year_list = group_df.iloc[train_index]['year_site.harvest_year']
        train_genotype_list = torch.from_numpy(genotype_list[train_index].values.astype(float))
        print(train_year_list)
        print('yearly data use for validation {}:'.format(len(validation_index)))
        validation_year_list = group_df.iloc[validation_index]['year_site.harvest_year']
        validation_genotype_list = torch.from_numpy(genotype_list[validation_index].values.astype(float))
        print(validation_year_list)
        print('yearly data use for test {}:'.format(len(test_index)))
        test_year_list = group_df.iloc[test_index]['year_site.harvest_year']
        test_genotype_list = torch.from_numpy(genotype_list[test_index].values.astype(float))
        print(test_year_list)
        # drop the date before (starting date) sping
        train_y = plant_height_tensor[start_day:, train_index, :].to(DEVICE)
        # with open('train_y.dill','wb') as file:
        #     dill.dump(train_y,file)
        # file.close()
        print('y train shape')
        print(train_y.shape)
        train_env = envir_tensor_dataset[start_day:, train_index, :].to(DEVICE).requires_grad_(True)
        print('train env shape:{}'.format(train_env.shape))

        # split train validation test
        validation_y = plant_height_tensor[start_day:, validation_index, :].to(DEVICE)
        validation_env = envir_tensor_dataset[start_day:, validation_index, :].to(DEVICE)
        test_y = plant_height_tensor[start_day:, test_index, :].to(DEVICE)
        test_env = envir_tensor_dataset[start_day:, test_index, :].to(DEVICE)
        train_group_df = copy.deepcopy(group_df).iloc[train_index, :]
        validation_group_df = copy.deepcopy(group_df).iloc[validation_index, :]
        test_group_df = copy.deepcopy(group_df).iloc[test_index, :]

        num_seq = train_y.shape[1]

        # creat time sequence, same as input shape
        ts_train = torch.linspace(0.0, 284, 285).unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, num_seq,
                                                                                          1)[:-start_day, :,
                   :].requires_grad_(True).to(DEVICE)  # time sequences steps
        ts_validation = torch.linspace(0.0, 284, 285).unsqueeze(dim=1).unsqueeze(dim=1).repeat(1,
                                                                                               validation_y.shape[
                                                                                                   1],
                                                                                               1)[:-start_day, :,
                        :].to(DEVICE)  # time sequences steps
        ts_test = torch.linspace(0.0, 284, 285).unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, test_y.shape[1],
                                                                                         1)[:-start_day, :, :].to(
            DEVICE)  # time sequences steps

        # average across replicates predict result, use for result report
        train_env_avg, train_y_avg, ts_train_avg, train_group_df_avg = calculate_average_across_replicates(
            train_env, train_group_df, train_y, ts_train)

        validation_env_avg, validation_y_avg, ts_validation_avg, validation_group_df_avg = calculate_average_across_replicates(
            validation_env, validation_group_df, validation_y,
            ts_validation)

        test_env_avg, test_y_avg, ts_test_avg, test_group_df_avg = calculate_average_across_replicates(
            test_env, test_group_df, test_y,
            ts_test)

        # different_genotype_curve_train = pd.DataFrame(
        #     torch.squeeze(copy.deepcopy(train_y_avg.detach())).cpu().numpy())
        # different_genotype_curve_train.columns = train_group_df_avg['genotype.id'].to_list()
        # different_genotype_curve_train.to_csv(
        #     "pinn_result/train_true_curves_{}.csv".format(
        #         mode))
        #
        # different_genotype_curve_val = pd.DataFrame(
        #     torch.squeeze(copy.deepcopy(validation_y_avg.detach())).cpu().numpy())
        # different_genotype_curve_val.columns = validation_group_df_avg['genotype.id'].to_list()
        # different_genotype_curve_val.to_csv(
        #     "pinn_result/val_true_curves_{}.csv".format(
        #         mode))
        #
        # different_genotype_curve_test = pd.DataFrame(
        #     torch.squeeze(copy.deepcopy(test_y_avg.detach())).cpu().numpy())
        # different_genotype_curve_test.columns = test_group_df_avg['genotype.id'].to_list()
        # different_genotype_curve_test.to_csv(
        #     "pinn_result/test_true_curves_{}.csv".format(
        #         mode))

        print('device')
        print(ts_train.device)

        randomseed_list = [1, 2, 3, 4, 5]  # [0,1,2,3,4,5]
        for j in randomseed_list:  # different random seed

            print('nsplit:{}  random seed:{}'.format(n, j))
            random.seed(j)
            np.random.seed(j)
            torch.manual_seed(j)  # the random seeds works on my laptop,
            # while seems the same random seed does not give the same result on server
            currenttime = time.time()
            run = wandb.init(
                # Set the project where this run will be logged
                project="single_genotype_{}".format(mode),
                # id='{0}'.format(mode),
                # Track hyperparameters and run metadata
                config={
                    "learning_rate": lr,
                    "epochs": 3000, "n_split": n,
                    "random_sees": j,
                    "hidden_size": hidden,
                    "num_layer": num_layer,
                    "weight_physic": weight_physic,
                    'ode_int': ode_int_loss,
                    'l2': L2,
                    'penalize_r': parameter_boundary
                },
            )
            pinn_model = LSTM_ts_and_temperature_predict_height(hidden,
                                                                input_size=train_env.shape[-1],
                                                                num_layer=num_layer,
                                                                smooth_out=False,
                                                                temperature_pinn=False,genetics=None
                                                                ).to(
                DEVICE)
            # pinn_model.cuda() #send model to GPU
            # print total number of parameters in model
            total_params = count_parameters(pinn_model)
            assert total_params == trainable_params_num
            # train pinn on training set
            print('train env shape:{}'.format(train_env.shape))
            model, stop_num_epochs, loss_fig, y_gradient_temperature, autograd_fig = train_model(
                model=pinn_model, train_env=train_env, Y=train_y,
                ts_X=ts_train, epochs=3000,
                validation_env=validation_env,
                validation_y=validation_y,
                ts_validation=ts_validation, lr=lr,
                physic_loss_weight=weight_physic,
                parameter_boundary=parameter_boundary,
                pinn=if_pinn, penalize_negetive_y='',
                fit_ml_first=False, multiple_r=False,
                ode_intergration_loss=ode_int_loss,
                l2=L2, temperature_ode=False, ts_test=ts_test, test_env=test_env,
                test_y=test_y,
                genetics_model=None, genetics_Train_year=None,
                genetics_validation_year=None,
                genetics_test_year=None, scaler_env={},
                genotype_list=[train_genotype_list, validation_genotype_list,
                               test_genotype_list])

            print('training stopped due to validation loss increases at epochs:{} '.format(
                stop_num_epochs))

            smooth_alpha = None
            r_value_pred, y_max_pred = print_parameters_from_ode(model)
            try:
                r_value_pred_mean = torch.mean(r_value_pred).item()
                y_max_pred_mean = torch.mean(y_max_pred).item()
            except:
                # otherwise only have one r value as type float
                r_value_pred_mean = r_value_pred
                y_max_pred_mean = y_max_pred
            model.eval()  # model will perform different if there are dropout or batchnorm layers
            print('train env shape:{}'.format(train_env.shape))

            with torch.no_grad():  # will turn of autogradient,
                if if_pinn:
                    # with torch.backends.cudnn.flags(enabled=False):
                    predict_y_train = model(train_env_avg, ts_train_avg)
                    predict_y_validation = model(validation_env_avg, ts_validation_avg)
                    # print('predict_y_shape:{}'.format(predict_y_validation.shape))  # [60, 2, 1] =[seq_len,sample_num,feature_size]
                    predict_y_test = model(test_env_avg, ts_test_avg)
                else:
                    predict_y_train = model(train_env_avg, ts_train_avg)
                    predict_y_validation = model(validation_env_avg, ts_validation_avg)
                    # print('predict_y_shape:{}'.format(predict_y_validation.shape))  # [60, 2, 1] =[seq_len,sample_num,feature_size]
                    predict_y_test = model(test_env_avg, ts_test_avg)
            # copy data later need to modify for plot, if not use deep.copy will modify the original variable
            copy_y_train = copy.deepcopy(train_y_avg.detach())
            copy_y_test = copy.deepcopy(test_y_avg.detach())

            copy_y_validation = copy.deepcopy(validation_y_avg.detach())


            predict_y_train_detach = copy.deepcopy(predict_y_train.detach())
            # predict_y_train_detach = torch.nan_to_num(predict_y_train_detach, nan=0.0,posinf=0.0,neginf=0.0)
            predict_y_validation_detach = copy.deepcopy(predict_y_validation.detach())
            # predict_y_validation_detach = torch.nan_to_num(predict_y_validation_detach, nan=0.0,posinf=0.0,neginf=0.0)
            predict_y_test_detach = copy.deepcopy(predict_y_test.detach())
            # predict_y_test_detach = torch.nan_to_num(predict_y_test_detach,nan=0.0,posinf=0.0,neginf=0.0)
            ##print and save mse(reverse scaling and log transform to make it compareable with other model)

            # calculate loss
            train_masked_rmse = mask_rmse_loss(true_y=copy_y_train,
                                               predict_y=predict_y_train_detach)
            print('train_rmse: {}'.format(train_masked_rmse))
            validation_masked_rmse = mask_rmse_loss(true_y=copy_y_validation,
                                                    predict_y=predict_y_validation_detach)
            print('validation rMSE:{}'.format(validation_masked_rmse))
            test_masked_rmse = mask_rmse_loss(true_y=copy_y_test,
                                              predict_y=predict_y_test_detach)
            print('test rMSE:{}'.format(test_masked_rmse))
            corre_train = mask_dtw_loss(true_y=copy_y_train,
                                        predict_y=predict_y_train_detach)
            print('train shapeDTW')
            print(corre_train)
            corre_validation = mask_dtw_loss(true_y=copy_y_validation,
                                             predict_y=predict_y_validation_detach)
            print('validation shapeDTW')
            print(corre_validation)
            corre_test = mask_dtw_loss(true_y=copy_y_test, predict_y=predict_y_test_detach)
            print('test shapeDTW')
            print(corre_test)


            new_row = pd.DataFrame(
                data={"lr": lr,
                      "n_split": n,
                      "random_sees": j,
                      "hidden_size": hidden,
                      "num_layer": num_layer,
                      "weight_physic": weight_physic,
                      "Trainable_Params": total_params,
                      'epoch': stop_num_epochs,
                      'train_rMSE': round(train_masked_rmse.item(), 3),
                      'validation_rMSE': round(validation_masked_rmse.item(), 3),
                      "test_rMSE": round(test_masked_rmse.item(), 3),
                      'train_shapeDTW': round(corre_train, 3),
                      'validation_shapeDTW': round(corre_validation, 3),
                      "test_shapeDTW": round(corre_test, 3),
                      "predicted_r": r_value_pred_mean,
                      "predicted_y_max": y_max_pred_mean,
                      "smooth_alpha": smooth_alpha,
                      'ode_int': ode_int_loss,
                      'l2': L2,
                      "genotype":genotype
                      },
                index=[0])
            try:
                # dataframe to same result
                pinn_result = pd.read_csv(
                    'pinn_result/result_summary/best_model_cv/PINN_mask_loss_{}_cv.csv'.format(mode),
                    header=0, index_col=0)
                print('connect result to existing result.csv file')
            except:
                print(
                    'result_summary/PINN_mask_loss_{}.csv Do not exist, this is the first iteration'.format(
                        mode))
            pinn_result = pd.concat(
                [pinn_result, new_row])
            pinn_result.to_csv(
                'pinn_result/result_summary/best_model_cv/PINN_mask_loss_{}_cv.csv'.format(mode))

            torch.save(model.state_dict(),
                    f='pinn_result/result_summary/best_model_cv/model/genotype_{}_trained_model_{}split_hidden{}_env{}_ts{}_lr_{}_w_ph{}_ode_int_{}_l2_{}_{}_rs_{}.dill'.format(
                        genotype,n, hidden, num_layer, num_layer, lr, weight_physic, ode_int_loss,
                        L2, mode, j))


            train_predict_plot_save_name = 'genotype_{}trained_figure_{}split_hidden{}_env{}_ts{}_lr_{}_w_ph{}_ode_int_{}_l2_{}_{}_rs_{}.svg'.format(
                genotype,n, hidden, num_layer, num_layer, lr, weight_physic, ode_int_loss, L2, mode,
                j)
            print('train env shape:{}'.format(train_env.shape))
            validation_predict_plot_save_name = 'genotype_{}validation_figure_{}split_hidden{}_env{}_ts{}_lr_{}_w_ph{}_ode_int_{}_l2_{}_{}_rs_{}.svg'.format(
                genotype,n, hidden, num_layer, num_layer, lr, weight_physic, ode_int_loss,
                L2, mode, j)
            test_predict_plot_save_name = 'genotype_{}test_figure_{}split_hidden{}_env{}_ts{}_lr_{}_w_ph{}_ode_int_{}_l2_{}_{}_rs_{}.svg'.format(
                genotype,n, hidden, num_layer, num_layer, lr, weight_physic,
                ode_int_loss, L2, mode, j)
            if not if_pinn:
                r_value_pred = None
                y_max_pred = None

            train_fig = plot_save_predict_curve(current_pwd, 'train',
                                                predict_y_train_detach,
                                                r_value_pred_mean,
                                                save_directory, copy_y_train,
                                                ts_train_avg,
                                                y_max_pred,
                                                marker_label=train_group_df_avg[
                                                    'genotype.id'],
                                                color_label=train_group_df_avg[
                                                    'year_site.harvest_year'],
                                                corresponding_environment=train_env_avg,
                                                name=train_predict_plot_save_name)

            validation_fig = plot_save_predict_curve(current_pwd,
                                                     'validation',
                                                     predict_y_validation_detach,
                                                     r_value_pred_mean,
                                                     save_directory,
                                                     copy_y_validation,
                                                     ts_validation_avg, y_max_pred,
                                                     marker_label=
                                                     validation_group_df_avg[
                                                         'genotype.id'],
                                                     color_label=
                                                     validation_group_df_avg[
                                                         'year_site.harvest_year'],
                                                     corresponding_environment=validation_env_avg,
                                                     name=validation_predict_plot_save_name)

            test_fig = plot_save_predict_curve(current_pwd, 'test',
                                               predict_y_test_detach, r_value_pred_mean,
                                               save_directory, copy_y_test, ts_test_avg,
                                               y_max_pred,
                                               marker_label=test_group_df_avg[
                                                   'genotype.id'],
                                               color_label=test_group_df_avg[
                                                   'year_site.harvest_year'],
                                               corresponding_environment=test_env_avg,
                                               name=test_predict_plot_save_name
                                               )
            wandb.log({'end_epoch': stop_num_epochs})
            wandb.log({'train_prediction_plot': wandb.Image(train_fig)})
            wandb.log({'val_prediction_plot': wandb.Image(validation_fig)})
            wandb.log({'test_prediction_plot': wandb.Image(test_fig)})
            run.finish()
            one_runtime = time.time() - currenttime
            print('runing time for one run:{}'.format(one_runtime))

def load_model_recalculate_prediction_rmse(file_name:str,genotype=None):
    """
    This function is to load saved model and re-do prediction to calculated genotype specific curve and calculate rmse
    file_name: the .csv file from result summary/ for model (before average different random seed result)
    genotype: int
    """
    with open('../temporary/plant_height_tensor_all.dill', 'rb') as f:
        plant_height_tensor = dill.load(f)
    f.close()
    with open('../temporary/group_list_df_all.dill', 'rb') as f:
        group_df = dill.load(f)
    print(group_df)
    f.close()
    with open('../temporary/temperature_tensor_same_length_all.dill', 'rb') as f:
        temperature_same_length_tensor = dill.load(f)
    f.close()
    start_day=115

    indices = group_df[group_df['genotype.id'] == genotype].index.tolist()
    group_df = group_df[group_df['genotype.id'] == genotype].reset_index()
    # Step 2: Use these indices to filter the second dimension of the tensor
    plant_height_tensor = plant_height_tensor[:, indices, :]
    # print(plant_height_tensor[:,0,:])
    temperature_same_length_tensor = temperature_same_length_tensor[:,indices,:]

    plant_height_tensor = torch.nan_to_num(plant_height_tensor, nan=0.0, posinf=0.0, neginf=0.0)
    plant_height_tensor[plant_height_tensor <= 0.0] = 0.0
    temperature_same_length_tensor = torch.nan_to_num(temperature_same_length_tensor, nan=0.0, posinf=0.0, neginf=0.0)

    plant_height_tensor = plant_height_tensor[start_day:,:,:]
    temperature_same_length_tensor = temperature_same_length_tensor[start_day:, :, :]

    train_test_validation_dictionary = train_test_split_based_on_group(group_df, group_df,
                                                                       group_name=[
                                                                           'year_site.harvest_year',
                                                                       ], n_split=5)
    n=0
    train_index, validation_index, test_index = train_test_validation_dictionary['splits_{}'.format(n)]
    train_y = plant_height_tensor[:, train_index, :].to(DEVICE)
    train_env = temperature_same_length_tensor[:, train_index, :].to(DEVICE)
    validation_y = plant_height_tensor[:, validation_index, :].to(DEVICE)
    validation_env = temperature_same_length_tensor[:, validation_index, :].to(DEVICE)

    test_y = plant_height_tensor[:, test_index, :].to(DEVICE)
    test_env = temperature_same_length_tensor[:, test_index, :].to(DEVICE)
    train_env = torch.nan_to_num(train_env, nan=0.0, posinf=0.0, neginf=0.0)
    train_y = torch.nan_to_num(train_y, nan=0.0, posinf=0.0, neginf=0.0)


    # average replicates
    train_y,_= average_based_on_group_df(train_y, copy.deepcopy(group_df).iloc[train_index,:])
    train_env,_ = average_based_on_group_df(train_env, copy.deepcopy(group_df).iloc[train_index, :])
    validation_y,_ = average_based_on_group_df(validation_y, copy.deepcopy(group_df).iloc[validation_index,:])
    validation_env,_ = average_based_on_group_df(validation_env, copy.deepcopy(group_df).iloc[validation_index, :])
    test_y,_ = average_based_on_group_df(test_y, copy.deepcopy(group_df).iloc[test_index,:])
    test_env,_ = average_based_on_group_df(test_env, copy.deepcopy(group_df).iloc[test_index, :])

    ts_train = torch.linspace(0.0, 284, 285).unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, train_y.shape[1],
                                                                                      1)[:-start_day, :,
               :].requires_grad_(True).to(DEVICE)  # time sequences steps
    ts_validation = torch.linspace(0.0, 284, 285).unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, validation_y.shape[1],
                                                                                           1)[:-start_day, :,
                    :].to(DEVICE)  # time sequences steps
    ts_test = torch.linspace(0.0, 284, 285).unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, test_y.shape[1],
                                                                                     1)[:-start_day, :, :].to(
        DEVICE)  # time sequences steps

    result_df = pd.read_csv('{}'.format(file_name),header=0,index_col=0).reset_index(drop=True)
    print(result_df)
    try:
        result_df=result_df.drop(columns='index')
    except:
        print('no column named index in result_df')
    result_df['genotype_train_RMSE']=None
    result_df['genotype_validation_RMSE'] = None
    result_df['genotype_test_RMSE'] = None
    result_df['genotype_train_dtw']=None
    result_df['genotype_validation_dtw'] = None
    result_df['genotype_test_dtw'] = None
    for i in result_df.index:
        match = re.search(r'PINN_mask_loss_(.*)\.csv', file_name)
        if match:
            mode_name = match.group(1)
            # print(mode_name)  # Output: example
        # print(result_df.iloc[i,:])
        hidden = result_df.loc[i,'hidden_size']
        num_layer = result_df.loc[i,'num_layer']
        # ts_layer = result_df.loc[i, 'ts_layer']
        lr = result_df.loc[i,'lr']
        weight_physic = result_df.loc[i, 'weight_physic']
        # drop_out = result_df.loc[i, 'dropout']
        ode_int_loss =result_df.loc[i, 'ode_int']
        # print(ode_int_loss)
        L2= result_df.loc[i, 'l2']
        j = result_df.loc[i,'random_sees']
        model_name= 'pinn_result/model/trained_model_{0}split_hidden{1}_env{2}_ts{3}_lr_{4}_w_ph{5}_ode_int_{6}_l2_{7}_{8}_rs_{9}.dill'.format(
                    n, hidden, num_layer, num_layer, lr, weight_physic, ode_int_loss, L2, mode_name, j)
        # print(model_name)
        with open(
                model_name, 'rb') as file:
            model1 = dill.load(file).to(DEVICE)

        # count_parameters(model1)
        with torch.no_grad():
            pred_train = model1(x=train_env, ts=ts_train)
            train_rmse = mask_rmse_loss(true_y=train_y, predict_y=pred_train).item()
            train_dtw = mask_dtw_loss(true_y=train_y, predict_y=pred_train).item()
            # print("train rmse:{}\ntrain dtw:{}".format(train_rmse,train_dtw))
            pred_validation = model1(x=validation_env, ts=ts_validation)
            val_rmse = mask_rmse_loss(true_y=validation_y, predict_y=pred_validation).item()
            val_dtw = mask_dtw_loss(true_y=validation_y, predict_y=pred_validation).item()
            # print("validation rmse:{}\nvalidation dtw:{}".format(val_rmse,val_dtw))
            pred_test = model1(x=test_env, ts=ts_test)
            test_rmse = mask_rmse_loss(true_y=test_y, predict_y=pred_test).item()
            test_dtw = mask_dtw_loss(true_y=test_y, predict_y=pred_test).item()
            # print("test rmse:{}\ntest dtw:{}".format(test_rmse,test_dtw))
            #update df
            result_df.loc[i,'train_rMSE'] = round(train_rmse,3)
            result_df.loc[i,'validation_rMSE'] = round(val_rmse,3)
            result_df.loc[i,'test_rMSE'] = round(test_rmse,3)
            result_df.loc[i,'train_shapeDTW'] = round(train_dtw,3)
            result_df.loc[i,'validation_shapeDTW'] = round(val_dtw,3)
            result_df.loc[i,'test_shapeDTW'] = round(test_dtw,3)
        # print(result_df)
        # save to file
    else:
        result_df.to_csv('{}'.format(file_name))

def read_run_cmd_cv_single_g():
    from sys import argv
    cmd_line = argv[1:]
    print('input command line: \n {}'.format(cmd_line))
    try:
        index_mode = cmd_line.index("-genotype") + 1
        genotype = int(cmd_line[index_mode])
    except:
        print('did not receive genotype input, Error')
        raise ValueError

    try:
        index_mode = cmd_line.index("-if_pinn") + 1
        if_pinn = cmd_line[index_mode]
        if str(if_pinn) =='True':
            if_pinn = True
        elif str(if_pinn) =='False':
            if_pinn = False
    except:
        print('did not receive if_pinn input, use default setting: True')
        if_pinn = True

    try:
        index_mode = cmd_line.index("-smooth_temp") + 1
        smooth_temp = cmd_line[index_mode]
        if smooth_temp == 'True':
            smooth_temp = True
        elif smooth_temp == 'False':
            smooth_temp = False
    except:
        print('did not receive -smooth_temp input: False')
        smooth_temp = False

    return genotype, if_pinn,smooth_temp


def main():
    plt.rcParams["font.family"] = "Times New Roman"
    """
    #This part is for logistic ODE model#
    #fit logistic ODE for replicates
    genotype_fit_logistic_ode_result = pd.DataFrame(columns=['genotype', 'rmse'], index=[0])
    # # genotype_list = pd.read_csv('../processed_data/fouryear_genotypes.csv',header=0,index_col=0)['genotype_id'].to_list()
    genotype_list = [106, 122, 133, 17, 218, 254, 282, 294, 2, 301, 302, 30, 335, 339, 33, 341, 362, 5, 6]
    for g in genotype_list:
        print(g)
        genotype_fit_logistic_ode_result = fit_ode_for_seq_seperatelly_plot(
            data_path='../processed_data/align_height_env_same_length.csv', genotype=g, start_day=91,
            years=(2018, 2019, 2021, 2022), result_df=genotype_fit_logistic_ode_result)
    # use logistic ODE as single genotype model
    ODE_fit_run()
    """

    """
    #this is for cross validation with different data split, only un comment this part and run via command line#
    genotype, if_pinn,smooth_temp = read_run_cmd_cv_single_g()
    load_best_hyperparameters_and_cross_validate(best_hyperparameter_file='best_model_result_summary/PINN_mask_loss_gpu_lstm_corr_same_length_best_hyperparameters_result.csv',
                                                 if_pinn=if_pinn,genotype=genotype,smooth_temp=smooth_temp,rescale=False)
    """


if __name__ == '__main__':
    main()
