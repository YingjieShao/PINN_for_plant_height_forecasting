import time

import numpy as np
import torch
import torch.nn as nn
import random
import pandas as pd
import dill
import torch.optim as optim

import wandb
import copy
from torch.utils.data import DataLoader, TensorDataset
from DataPrepare import reverse_min_max_scaling,train_test_split_based_on_group,train_test_split_based_on_two_groups, count_parameters,manually_data_split_based_on_one_group,manually_split_on_two_groups
import matplotlib.pyplot as plt
import seaborn as sns
from torchdiffeq import odeint
from DataPrepare import minmax_scaler
from NNmodel_training import average_based_on_group_df,mask_rmse_loss,smooth_tensor_ignore_nan,plot_multiple_sequences_colored_based_on_label_df

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  #device cup or gpu
class simple_genotype_code_temperature_input_height_prediction(nn.Module):
    def __init__(self, genetic_feature_size, input_size, hidden_size, num_layer, last_fc_hidden_size=5,genetics_embedding_size=5):
        super().__init__()
        # self.init_network()
        self.g_embedding = nn.Sequential(nn.Linear(in_features=genetic_feature_size, out_features=genetics_embedding_size),
                                          # nn.LeakyReLU(),
                                          # nn.Linear(in_features=5, out_features=194),
                                          nn.Tanh())
        # self.g_embedding = nn.Linear(in_features=genetic_feature_size, out_features=194)
        # self.g_embedding = nn.Sequential(nn.Conv1d(in_channels=1,out_channels=5,kernel_size=36,stride=36),
        #                                  nn.MaxPool1d(kernel_size=36,stride=36),
        #                                  nn.Flatten())

        self.leakyrelu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.lstm1 = nn.LSTM(input_size=(input_size + 1), hidden_size=hidden_size, num_layers=num_layer)
        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=1, num_layers=1)
        self.lstm3 = nn.LSTM(input_size=(1+genetics_embedding_size), hidden_size=last_fc_hidden_size, num_layers=1)
        self.lstm4 = nn.LSTM(input_size=last_fc_hidden_size, hidden_size=1, num_layers=1)

    def forward(self, inputx, input_genotype_code,ts_input):

        # input_genotype_code= input_genotype_code.unsqueeze(1)
        # print('g input shape:{}'.format(input_genotype_code.shape))
        genotype_effect_vector = self.g_embedding(input_genotype_code)

        # shape=(time_step,n_samples,feature_size)
        # genotype_effect_vector = 0.2*genotype_effect_vector.permute(1,0).unsqueeze(dim=-1)
        # print(genotype_effect_vector.shape)
        genotype_effect_vector = genotype_effect_vector.unsqueeze(dim=0).repeat(170, 1,
                                                                                      1)  # resahpe to (seq_length,sample size, 2)

        combine_vector = torch.cat([inputx,ts_input], dim=-1)
        out_put,c = self.lstm1(combine_vector)
        # out_put = self.leakyrelu(out_put)
        out_put,c= self.lstm2(out_put)
        out_put = self.leakyrelu(out_put)
        # plt.plot(torch.squeeze(copy.deepcopy(out_put.detach())[:, 0, :].cpu()), c='blue')
        add_genetics_vector = torch.cat([genotype_effect_vector, out_put], dim=-1)
        out_put, c = self.lstm3(add_genetics_vector)
        out_put, c = self.lstm4(out_put)
        out_put = self.leakyrelu(out_put)
        return out_put,combine_vector,None

    def init_network(self):
        # initialize weight and bias(use xavier and 0 for weight and bias separately)
        for name, param in self.named_parameters():
            if 'weight' in name:
                # nn.init.constant_(param, 0.0046)
                # random_number_init = np.random.uniform(0.001,0.0001)
                # nn.init.constant_(param, random_number_init)
                # nn.init.xavier_uniform_(param)
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
            # print(name, param)
class genotype_code_temperature_input_height_prediction_fc_CNN(nn.Module):
    def __init__(self, genetic_feature_size, input_size, hidden_size, num_layer, last_fc_hidden_size=5,genetics_embedding_size=19):
        super().__init__()
        # self.init_network()

        # self.g_embedding = nn.Linear(in_features=genetic_feature_size, out_features=194)
        self.g_embedding = nn.Sequential(nn.Conv1d(in_channels=1,out_channels=3,kernel_size=12,stride=7),
                                         nn.MaxPool1d(kernel_size=7,stride=5),
                                         nn.Flatten(),nn.Tanh())
        cnn_out=int(1+(309-3)/7)
        genetics_embedding_size = int(3*(1+(cnn_out-7)/5))
        print(genetics_embedding_size)
        self.leakyrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        # self.tanh = nn.Tanh()
        self.lstm1 = nn.LSTM(input_size=(input_size + 1), hidden_size=hidden_size, num_layers=num_layer)
        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=3, num_layers=1)
        self.fc= nn.Sequential(nn.Linear(in_features=(3+24),out_features=last_fc_hidden_size),
                               nn.LeakyReLU(),nn.Linear(last_fc_hidden_size,1))
        # self.weight_g = nn.Parameter(torch.randn(1,1))
        # self.weight_h = nn.Parameter(torch.randn(1,1))
    def forward(self, inputx, input_genotype_code,ts_input):

        input_genotype_code= input_genotype_code.unsqueeze(dim=1)
        #Conv1d input shape: [batch_size, channels, seq_len]
        # genotype_effect_vector = input_genotype_code
        # print('g input shape:{}'.format(genotype_effect_vector.shape))
        genotype_effect_vector = self.g_embedding(input_genotype_code)
        # genotype_effect_vector = self.leakyrelu(genotype_effect_vector)
        combine_vector = torch.cat([inputx,ts_input], dim=-1)
        out_put,c = self.lstm1(combine_vector)
        # out_put = self.leakyrelu(out_put)
        out_put,c= self.lstm2(out_put)
        out_put = self.leakyrelu(out_put)
        # plt.plot(torch.squeeze(copy.deepcopy(out_put.detach())[:,0,:].cpu()),c='blue')
        outputs=[]
        for t in range(out_put.shape[0]):
            #input for fc layer: samplesize feature size)
            outputs.append(self.fc(torch.cat([out_put[t,:,:],genotype_effect_vector],dim=-1)))
            # outputs.append(self.weight_h*out_put[t,:,:]+self.weight_g*(genotype_effect_vector[:,t].unsqueeze(dim=-1)))
        out_put = torch.stack(outputs,dim=0)
        # print('output shape')
        # print(out_put.shape)
        out_put = self.leakyrelu(out_put)
        return out_put,combine_vector,None

    def init_network(self):
        # initialize weight and bias(use xavier and 0 for weight and bias separately)
        for name, param in self.named_parameters():
            if 'weight' in name:
                # # nn.init.constant_(param, 0.0046)
                # random_number_init = np.random.uniform(0.001,0.0001)
                # nn.init.constant_(param, random_number_init)
                # nn.init.xavier_uniform_(param)
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
            # print(name, param)
class genotype_code_temperature_input_height_prediction_fc(nn.Module):
    def __init__(self, genetic_feature_size, input_size, hidden_size, num_layer, last_fc_hidden_size=5,genetics_embedding_size=19):
        super().__init__()
        # self.init_network()
        self.g_embedding = nn.Sequential(nn.Linear(in_features=genetic_feature_size, out_features=genetics_embedding_size),
                                          # nn.LeakyReLU(),
                                          # nn.Linear(in_features=5, out_features=194),
                                         nn.Tanh())
        # self.g_embedding = nn.Linear(in_features=genetic_feature_size, out_features=194)
        # self.g_embedding = nn.Sequential(nn.Conv1d(in_channels=1,out_channels=5,kernel_size=36,stride=36),
        #                                  nn.MaxPool1d(kernel_size=36,stride=36),
        #                                  nn.Flatten())

        self.leakyrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        # self.tanh = nn.Tanh()
        print(type(hidden_size))
        print(type(input_size))
        print(type(1))
        self.lstm1 = nn.LSTM(input_size=(input_size + 1), hidden_size=hidden_size, num_layers=num_layer)
        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=3, num_layers=1)
        self.fc= nn.Sequential(nn.Linear(in_features=(3+genetics_embedding_size),out_features=last_fc_hidden_size),
                               nn.LeakyReLU(),nn.Linear(last_fc_hidden_size,1))
        # self.weight_g = nn.Parameter(torch.randn(1,1))
        # self.weight_h = nn.Parameter(torch.randn(1,1))
    def forward(self, inputx, input_genotype_code,ts_input):
        # print(inputx)
        # input_genotype_code= input_genotype_code.unsqueeze(1)
        # genotype_effect_vector = input_genotype_code
        # print('g input shape:{}'.format(genotype_effect_vector.shape))
        genotype_effect_vector = self.g_embedding(input_genotype_code)
        # genotype_effect_vector = self.leakyrelu(genotype_effect_vector)
        combine_vector = torch.cat([inputx,ts_input], dim=-1)
        out_put,c = self.lstm1(combine_vector)
        # out_put = self.leakyrelu(out_put)
        out_put,c= self.lstm2(out_put)
        out_put = self.leakyrelu(out_put)
        # plt.plot(torch.squeeze(copy.deepcopy(out_put.detach())[:,0,:].cpu()),c='blue')
        outputs=[]
        for t in range(out_put.shape[0]):
            #input for fc layer: samplesize feature size)
            outputs.append(self.fc(torch.cat([out_put[t,:,:],genotype_effect_vector],dim=-1)))
            # outputs.append(self.weight_h*out_put[t,:,:]+self.weight_g*(genotype_effect_vector[:,t].unsqueeze(dim=-1)))
        out_put = torch.stack(outputs,dim=0)
        # print('output shape')
        # print(out_put.shape)
        out_put = self.leakyrelu(out_put)
        # print(out_put)
        return out_put,combine_vector,None

    def init_network(self):
        # initialize weight and bias(use xavier and 0 for weight and bias separately)
        for name, param in self.named_parameters():
            if 'weight' in name:
                # # nn.init.constant_(param, 0.0046)
                # random_number_init = np.random.uniform(0.001,0.0001)
                # nn.init.constant_(param, random_number_init)
                # nn.init.xavier_uniform_(param)
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
            # print(name, param)
class simple_genotype_code_temperature_input_height_prediction_CUSTOMIZE_RNN(nn.Module):
    def __init__(self, genetic_feature_size, input_size, hidden_size, num_layer, genotype_num=19):
        super().__init__()
        # self.init_network()
        # self.linear = nn.Linear(in_features=genetic_feature_size, out_features=seq_len)

        # self.leakyrelu = nn.LeakyReLU()
        self.rnn1 = RNNModel(input_size=(input_size+genetic_feature_size+1), hidden_size=hidden_size, num_layers=num_layer)
        self.rnn2 = RNNModel(input_size=hidden_size, hidden_size=1, num_layers=1)

    def forward(self, inputx, input_genotype_code,ts_input):
        genotype_effect_vector = input_genotype_code.unsqueeze(dim=0).repeat(194, 1,
                                                                                      1)  # resahpe to (seq_length,sample size, 2)
        # print(genotype_effect_vector.shape)
        combine_vector = torch.cat([genotype_effect_vector, inputx,ts_input], dim=-1)
        out_put1,smooth_1 = self.rnn1(combine_vector)
        out_put2,smooth_2= self.rnn2(out_put1)
        out_put = torch.abs(out_put2)
        # out_put = self.leakyrelu(out_put)
        #then we are predict the change of plant height from rnn then accumulated them
        # out_put = torch.cumsum(out_put, dim=0)

        # print(out_put)
        smooth_loss = torch.mean((out_put1-smooth_1)**2)+torch.mean((out_put2-smooth_2)**2)
        return out_put,combine_vector,smooth_loss

    def init_network(self):
        # initialize weight and bias(use xavier and 0 for weight and bias separately)
        for name, param in self.named_parameters():
            if 'weight' in name:
                # nn.init.constant_(param, 0.0046)
                # random_number_init = np.random.uniform(0.001,0.0001)
                # nn.init.constant_(param, random_number_init)
                # nn.init.xavier_uniform_(param)
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
            # print(name, param)

class pinn_genotype_embedding(nn.Module):
    def __init__(self, genetic_feature_size, input_size, hidden_size, num_layer,last_fc_hidden_size=5,genetics_embedding_size=5):
        super().__init__()
        # self.init_network()
        # self.g_embedding = nn.Sequential(nn.Linear(in_features=genetic_feature_size, out_features=5),
        #                                   nn.LeakyReLU(),
        #                                   nn.Linear(in_features=5, out_features=194))
        self.g_embedding1 = nn.Linear(in_features=genetic_feature_size, out_features=genetics_embedding_size)
        self.g_parameters= nn.Linear(in_features=genetics_embedding_size, out_features=2)
        # self.g_embedding2=nn.Linear(in_features=5, out_features=170)
        #need to use nn.linear to predict new genotype
        # self.linear = nn.Linear(in_features=2,out_features=2)
        # self.g_parameters = nn.Embedding(num_embeddings=genotype_num,embedding_dim=2) #related to  r and ymax
        self.leakyrelu = nn.LeakyReLU()
        self.lstm1 = nn.LSTM(input_size=(input_size + 1), hidden_size=hidden_size, num_layers=num_layer)
        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=1, num_layers=1)
        self.lstm3 = nn.LSTM(input_size=(1+genetics_embedding_size), hidden_size=last_fc_hidden_size, num_layers=1)
        self.lstm4 = nn.LSTM(input_size=last_fc_hidden_size, hidden_size=1, num_layers=1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        # self.relu = nn.ReLU()
    def forward(self, inputx, input_genotype_code,ts_input):
        # print(input_genotype_code)
        #first embedding layer
        genotype_effect_vector = self.g_embedding1(input_genotype_code)
        genotype_effect_vector = self.tanh(genotype_effect_vector)
        #convet to 2 value
        genotype_effect_embedding = self.g_parameters(genotype_effect_vector) #length2, r and ymax
        # genotype_effect_embedding = self.tanh(genotype_effect_embedding)
        # genotype_effect_vector_param = self.g_parameters(genotype_effect_embedding)

        self.r = self.sigmoid(genotype_effect_embedding[:,0])
        self.y_max= self.tanh(genotype_effect_embedding[:,1]) +1

        genotype_effect_vector = genotype_effect_vector.unsqueeze(dim=0).repeat(170, 1,
                                                                                1)  # resahpe to (seq_length,sample size, 2)

        combine_vector = torch.cat([inputx, ts_input], dim=-1)
        out_put, c = self.lstm1(combine_vector)
        # out_put = self.leakyrelu(out_put)
        out_put, c = self.lstm2(out_put)
        out_put = self.leakyrelu(out_put)
        # plt.plot(torch.squeeze(copy.deepcopy(out_put.detach())[:, 0, :].cpu()), c='blue')
        add_genetics_vector = torch.cat([genotype_effect_vector, out_put], dim=-1)
        out_put, c = self.lstm3(add_genetics_vector)
        out_put, c = self.lstm4(out_put)
        out_put = self.leakyrelu(out_put)
        return out_put,genotype_effect_embedding,None

    def init_network(self):
        # initialize weight and bias(use xavier and 0 for weight and bias separately)
        for name, param in self.named_parameters():
            if 'weight' in name:
                # random_number_init = np.random.uniform(0.001,0.0001)
                # nn.init.constant_(param, random_number_init)
                # nn.init.xavier_uniform_(param)
                nn.init.orthogonal_(param)
                # print(name)
                # print(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
class pinn_genotype_embedding_fc(nn.Module):
    def __init__(self, genetic_feature_size, input_size, hidden_size, num_layer, last_fc_hidden_size=5,genetics_embedding_size=19):
        super().__init__()
        # self.init_network()
        self.g_embedding = nn.Sequential(
            nn.Linear(in_features=genetic_feature_size, out_features=genetics_embedding_size),
            # nn.LeakyReLU(),
            # nn.Linear(in_features=5, out_features=194),
            # nn.Tanh()
            )
        # self.g_embedding1 = nn.Linear(in_features=genetic_feature_size, out_features=genetics_embedding_size)
        self.tanh = nn.Tanh()
        self.g_parameters= nn.Linear(in_features=genetics_embedding_size, out_features=2)

        # self.g_parameters = nn.Embedding(num_embeddings=genotype_num,embedding_dim=2) #related to  r and ymax

        self.lstm1 = nn.LSTM(input_size=(input_size + 1), hidden_size=hidden_size, num_layers=num_layer)
        self.leakyrelu = nn.LeakyReLU()
        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=3, num_layers=1)
        self.fc= nn.Sequential(nn.Linear(in_features=(3+genetics_embedding_size),out_features=last_fc_hidden_size),
                               nn.LeakyReLU(),nn.Linear(last_fc_hidden_size,1))

        self.sigmoid = nn.Sigmoid()

        # self.relu = nn.ReLU()
    def forward(self, inputx, input_genotype_code,ts_input):
        # print(input_genotype_code)
        #first embedding layer
        input_genotype_code = self.g_embedding(input_genotype_code)
        # genotype_effect_vector = self.leakyrelu(genotype_effect_vector)

        #convet to 2 value
        genotype_effect_embedding = self.g_parameters(input_genotype_code) #length2, r and ymax
        self.r = self.sigmoid(genotype_effect_embedding[:,0])
        self.y_max= self.tanh(genotype_effect_embedding[:,1]) +1

        input_genotype_code = self.tanh(input_genotype_code)
        # print(self.r)
        # print(self.y_max)
        combine_vector = torch.cat([inputx, ts_input], dim=-1)
        out_put, c = self.lstm1(combine_vector)
        # out_put = self.leakyrelu(out_put)
        out_put, c = self.lstm2(out_put)
        out_put = self.leakyrelu(out_put)
        # plt.plot(torch.squeeze(copy.deepcopy(out_put.detach())[:, 0, :].cpu()), c='blue')
        outputs = []
        for t in range(out_put.shape[0]):
            # input for fc layer: samplesize,feature size)

            outputs.append(
                self.fc(torch.cat([out_put[t, :, :], input_genotype_code], dim=-1)))
            # outputs.append(self.weight_h*out_put[t,:,:]+self.weight_g*(genotype_effect_vector[:,t].unsqueeze(dim=-1)))
        out_put = torch.stack(outputs, dim=0)
        # print('output shape')
        # print(out_put.shape)
        out_put = self.leakyrelu(out_put)

        return out_put,genotype_effect_embedding,None
    def init_network(self):
        # initialize weight and bias(use xavier and 0 for weight and bias separately)
        for name, param in self.named_parameters():
            if 'weight' in name:
                # random_number_init = np.random.uniform(0.001,0.0001)
                # nn.init.constant_(param, random_number_init)
                # nn.init.xavier_uniform_(param)
                nn.init.orthogonal_(param)
                # print(name)
                # print(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)


class pinn_genotype_embedding_trainable_parameter_fc(nn.Module):
    def __init__(self, genetic_feature_size, input_size, hidden_size, num_layer, last_fc_hidden_size=5,
                 genetics_embedding_size=19,unique_genotypes:tuple=('33', '106', '122', '133', '5', '30', '218', '2', '17', '254', '282', '294', '301', '302', '335', '339', '341', '6', '362')):
        super().__init__()
        # self.init_network()
        self.g_embedding = nn.Sequential(
            nn.Linear(in_features=genetic_feature_size, out_features=genetics_embedding_size),
            # nn.LeakyReLU(),
            # nn.Linear(in_features=5, out_features=194),
            # nn.Tanh()
        )
        # self.g_embedding1 = nn.Linear(in_features=genetic_feature_size, out_features=genetics_embedding_size)
        self.tanh = nn.Tanh()
        # Create trainable parameters for unique genotypes
        self.genotype_params = nn.ParameterDict({
            genotype: nn.Parameter(torch.tensor([0.1,0.8]), requires_grad=True)
            for genotype in unique_genotypes
        })
        self.trainable_param =True

        # self.g_parameters = nn.Embedding(num_embeddings=genotype_num,embedding_dim=2) #related to  r and ymax

        self.lstm1 = nn.LSTM(input_size=(input_size + 1), hidden_size=hidden_size, num_layers=num_layer)
        self.leakyrelu = nn.LeakyReLU()
        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=3, num_layers=1)
        self.fc = nn.Sequential(nn.Linear(in_features=(3 + genetics_embedding_size), out_features=last_fc_hidden_size),
                                nn.LeakyReLU(), nn.Linear(last_fc_hidden_size, 1))

        self.sigmoid = nn.Sigmoid()

        # self.relu = nn.ReLU()

    def forward(self, inputx, input_genotype_code, ts_input,genotype_indices):
        # print(input_genotype_code)
        # first embedding layer
        input_genotype_code = self.g_embedding(input_genotype_code)
        # genotype_effect_vector = self.leakyrelu(genotype_effect_vector)

        # convet to 2 value
        # genotype_effect_embedding = self.g_parameters(input_genotype_code)  # length2, r and ymax

        params = torch.stack([self.genotype_params[str(genotype.item())] for genotype in genotype_indices])
        self.r = params[:, 0].unsqueeze(1)       # r values for each sample
        self.y_max = params[:, 1].unsqueeze(1)   # y_max values for each sample

        input_genotype_code = self.tanh(input_genotype_code)
        # print(self.r)
        # print(self.y_max)
        combine_vector = torch.cat([inputx, ts_input], dim=-1)
        out_put, c = self.lstm1(combine_vector)
        # out_put = self.leakyrelu(out_put)
        out_put, c = self.lstm2(out_put)
        out_put = self.leakyrelu(out_put)
        # plt.plot(torch.squeeze(copy.deepcopy(out_put.detach())[:, 0, :].cpu()), c='blue')
        outputs = []
        for t in range(out_put.shape[0]):
            # input for fc layer: samplesize,feature size)

            outputs.append(
                self.fc(torch.cat([out_put[t, :, :], input_genotype_code], dim=-1)))
            # outputs.append(self.weight_h*out_put[t,:,:]+self.weight_g*(genotype_effect_vector[:,t].unsqueeze(dim=-1)))
        out_put = torch.stack(outputs, dim=0)
        # print('output shape')
        # print(out_put.shape)
        out_put = self.leakyrelu(out_put)

        return out_put, None, None

    def init_network(self):
        # initialize weight and bias(use xavier and 0 for weight and bias separately)
        for name, param in self.named_parameters():
            if 'weight' in name:
                # random_number_init = np.random.uniform(0.001,0.0001)
                # nn.init.constant_(param, random_number_init)
                # nn.init.xavier_uniform_(param)
                nn.init.orthogonal_(param)
                # print(name)
                # print(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)


class pinn_genotype_embedding_CUSTOMIZE_RNN(nn.Module):
    def __init__(self, genetic_feature_size, input_size, hidden_size, num_layer, genotype_num=19):
        super().__init__()
        # self.init_network()
        self.g_parameters = nn.Linear(in_features=genetic_feature_size, out_features=2) #need to use nn.linear to predict new genotype
        # self.linear = nn.Linear(in_features=2,out_features=2)
        # self.g_parameters = nn.Embedding(num_embeddings=genotype_num,embedding_dim=2) #related to  r and ymax
        self.leakyrelu = nn.LeakyReLU()
        self.rnn1 = RNNModel(input_size=(input_size+2+1), hidden_size=hidden_size, num_layers=num_layer)
        self.rnn2 = RNNModel(input_size=hidden_size, hidden_size=1, num_layers=1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
    def forward(self, inputx, input_genotype_code,ts_input):
        # print(input_genotype_code)
        genotype_effect_vector_param = self.g_parameters(input_genotype_code) #length2, r and ymax
        # genotype_effect_vector_param = self.linear(self.relu(genotype_effect_vector_param))
        self.r = self.sigmoid(genotype_effect_vector_param[:,0])
        self.y_max= self.tanh(genotype_effect_vector_param[:,1]) +1
        # print(self.r)
        # print(self.y_max)
        genotype_effect_vector=genotype_effect_vector_param.unsqueeze(dim=0).repeat(194,1,1) #resahpe to (seq_length,sample size, 2)
        combine_vector = torch.cat([genotype_effect_vector, inputx,ts_input], dim=-1)
        out_put1, smooth_1 = self.rnn1(combine_vector)
        out_put2, smooth_2 = self.rnn2(out_put1)
        # out_put = torch.abs(out_put2)
        out_put = self.leakyrelu(out_put2)
        # then we are predict the change of plant height from rnn then accumulated them
        # out_put = torch.cumsum(out_put, dim=0)

        # print(out_put)
        #difference between use 0.5 time step and 1 time step
        smooth_loss = torch.mean((out_put1 - smooth_1) ** 2) + torch.mean((out_put2 - smooth_2) ** 2)

        return out_put,genotype_effect_vector_param,smooth_loss

    def init_network(self):
        # initialize weight and bias(use xavier and 0 for weight and bias separately)
        for name, param in self.named_parameters():
            if 'weight' in name:
                # random_number_init = np.random.uniform(0.001,0.0001)
                # nn.init.constant_(param, random_number_init)
                # nn.init.xavier_uniform_(param)
                nn.init.orthogonal_(param)
                # print(name)
                # print(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
                # print(name, param)

def smoothing_spline(input_tensor,num_knots=10):
    """
    smooth to fill na for training
    """
    from scipy.interpolate import splrep, BSpline
    from scipy.interpolate import LSQUnivariateSpline

    input_tensor = copy.deepcopy(input_tensor)
    ori_type=input_tensor.dtype
    #drop where y = na
    input_tensor[input_tensor == 0.0] = np.nan
    # sns.scatterplot(input_tensor[:,:,0])
    # plt.show()
    # Convert tensor to NumPy for SciPy compatibility
    tensor_np = input_tensor.cpu().numpy()

    # Prepare an empty array for the result
    filled_np = np.copy(tensor_np)
    nan_mask = np.isnan(tensor_np)
    # Iterate over each sample and feature
    for sample in range(tensor_np.shape[1]):
        for feature in range(tensor_np.shape[2]):
            # Get the sequence for this feature across the sequence length
            y = tensor_np[:, sample, feature]
            t = np.arange(len(y))

            # Mask out NaNs for fitting
            valid_mask = ~np.isnan(y)
            t_valid = t[valid_mask]
            y_valid = y[valid_mask]

            if len(y_valid) > num_knots:
                # Define knot positions for the P-spline
                knots = np.linspace(t_valid[1], t_valid[-2], num_knots - 2)

                # Fit the P-spline to valid (non-NaN) data
                try:
                    spline = LSQUnivariateSpline(t_valid, y_valid, t=knots, k=3)
                except:
                    print(len(y_valid))
                    sns.scatterplot(y_valid)
                    sns.scatterplot(t_valid,color='g')
                    plt.show()
                    spline = LSQUnivariateSpline(t_valid, y_valid, t=knots, k=3)
                # Calculate smoothed values across the full range `t`
                smoothed_y = spline(t)

                # Replace internal NaN values (between the first and last valid indices)
                first_valid_index = t_valid[0]
                last_valid_index = t_valid[-1]

                # Only replace NaNs within the range of the first and last valid indices
                nan_indices = np.where(np.isnan(y[first_valid_index:last_valid_index + 1]))[0] + first_valid_index
                filled_np[nan_indices, sample, feature] = smoothed_y[nan_indices]

    # Convert back to PyTorch tensors
    filled_tensor = torch.tensor(filled_np, dtype=ori_type)
    nan_mask_tensor = torch.tensor(nan_mask, dtype=torch.bool)
    #plot after smooth filled value
    # plot_filled_values(input_tensor, filled_tensor, nan_mask_tensor)
    #change na back to 0.0
    filled_tensor = torch.nan_to_num(filled_tensor, nan=0.0, posinf=0.0, neginf=0.0).to(DEVICE)
    print('na in smoothed data')
    print(torch.sum(torch.isnan(filled_tensor)))
    return filled_tensor


def plot_filled_values(tensor, filled_tensor, nan_mask):
    seq_len, num_samples, feature_size = tensor.shape

    for sample in range(num_samples):
        for feature in range(feature_size):
            t = np.arange(seq_len)
            y_orig = tensor[:, sample, feature].numpy()
            y_filled = filled_tensor[:, sample, feature].numpy()

            # Determine colors: red for filled values, blue for original values
            color = np.where(nan_mask[:, sample, feature].numpy(), 'red', 'blue')

            # Plot original and filled values
            plt.figure(figsize=(8, 5))
            plt.scatter(t, y_filled, c=color, label='Filled (red) / Original (blue)')
            plt.plot(t, y_filled, linestyle='--', alpha=0.5, label='Smoothed curve')
            plt.xlabel("Time Step")
            plt.ylabel("Value")
            plt.title(f"Sample {sample + 1}, Feature {feature + 1}")
            plt.legend()
            plt.show()

def physics_loss(model,predict_y,ts_x):

    dy_dt_nn = torch.autograd.grad(
        predict_y, ts_x, grad_outputs=torch.ones_like(predict_y), create_graph=True, is_grads_batched=False
    )[0]
    # print('autogradient shape')
    # print(dy_dt_nn.shape)
    # print(model.r.shape)
    # print(predict_y.shape)
    reshpae_r = model.r.view(1, model.r.shape[0], 1)
    reshpae_ymax = model.y_max.view(1, model.y_max.shape[0], 1)
    dY_dt_ode = reshpae_r * predict_y * (1 - (predict_y / reshpae_ymax))
    # print(dY_dt_ode.shape) [194, 88, 1]
    physics_loss = torch.mean((dy_dt_nn - dY_dt_ode) ** 2)  # logistic ode
    return physics_loss,dY_dt_ode,dy_dt_nn

def l2_loss(l2_regulization,model):
    if l2_regulization != False:
        number_weights = 0
        for name, weights in model.named_parameters():
            if 'bias' not in name:
                number_weights = number_weights + weights.numel()
        # Calculate L2 term
        L2_term = torch.tensor(0., requires_grad=True)
        for name, weights in model.named_parameters():
            if 'bias' not in name:
                weights_sum = torch.sum(weights ** 2)
                L2_term = L2_term + weights_sum
        # print(number_weights)
        L2_term = L2_term / number_weights  # weight**2/number weights
        # print(L2_term * l2_regulization)
        # loss + L2 regularization
        l2_loss = L2_term * l2_regulization
    else:
        l2_loss = torch.tensor([0.0])
    return l2_loss

def mask_rmse_loss_weight(true_y:torch.tensor, predict_y:torch.tensor):
    '''
    calculate mask where (which time step) inout value is zero, weight based on true height
    https://discuss.pytorch.org/t/how-does-applying-a-mask-to-the-output-affect-the-gradients/126520/2
    '''
    device = true_y.device
    mask = (~torch.isin(true_y, torch.tensor(0.0).to(device))).float()

    mask_rmse_loss_value = (torch.sum((true_y+1.0)*(((true_y - predict_y) * mask) ** 2)) / torch.count_nonzero(mask))**0.5

    return mask_rmse_loss_value#,torch.count_nonzero(mask)
def mask_dtw_loss(true_y:torch.tensor, predict_y:torch.tensor, shapedtw=None):
    from shapedtw.shapedtw import shape_dtw
    from shapedtw.shapeDescriptors import SlopeDescriptor, PAADescriptor, CompoundDescriptor, DerivativeShapeDescriptor,DWTDescriptor,RawSubsequenceDescriptor
    from shapedtw.dtwPlot import dtwPlot

    device = true_y.device
    shape_dtw_distance = 0.0
    print(true_y.shape)
    for seq in range(true_y.shape[1]):
        seq1= torch.squeeze(copy.deepcopy(true_y.detach().cpu())[:,seq]).numpy()
        index_not_na = seq1 != 0.0
        seq1 = seq1[index_not_na]
        seq2= torch.squeeze(copy.deepcopy(predict_y.detach().cpu())[:,seq]).numpy()
        seq2 = seq2[index_not_na]
        # print(seq2.shape)
        # print(seq1)
        # print(seq2)
        #define descriptor
        slope_descriptor = SlopeDescriptor(slope_window=5)
        derivative_descriptor = DerivativeShapeDescriptor()
        paa_descriptor = PAADescriptor(piecewise_aggregation_window=5)
        dwt_descriptor = DWTDescriptor()
        # raw_descriptor = RawSubsequenceDescriptor()
        compound_descriptor = CompoundDescriptor([derivative_descriptor, slope_descriptor, paa_descriptor,dwt_descriptor],
                                                 descriptors_weights=[1., 1., 1.,1.])
        shape_dtw_results = shape_dtw(
            x=seq1,
            y=seq2,
            subsequence_width=5,  # need to figure out how to choose that
            shape_descriptor=compound_descriptor
        )
        # fig1,ax = plt.subplots()
        # ax = dtwPlot(shape_dtw_results, plot_type="twoway", yoffset=1,axis=ax)
        # wandb.log({'dtw_plot':wandb.Image(ax)})
        # plt.close()
        shape_dtw_distance += shape_dtw_results.shape_normalized_distance
        # normalized_distance attributes of classes representing shape dtw results.
        # print(round(shape_dtw_results.shape_normalized_distance, 2))
    else:
        return shape_dtw_distance/true_y.shape[1]

# def transfer_prediction_to_ranking(plant_height_tensor:torch.tensor)->torch.tensor:
#     """
#     This function is to rank the plant height curve, the simplest case, use average value of the plant height curve to rank
#     """
#     from differentiable_sorting.torch import bitonic_matrices, diff_sort
#     from torch.autograd import Variable
#     plant_height_tensor = copy.deepcopy(plant_height_tensor.detach()).squeeze()
#     # corresponding_genotype_list = copy.deepcopy(corresponding_genotype_list).squeeze()
#     # plant_height_tensor[plant_height_tensor==0.0] = torch.nan
#     # print('shape plant height after squeeze:{}'.format(plant_height_tensor.shape))
#     max_plant_height = torch.max(plant_height_tensor,dim=0,keepdim=True).values
#     # print('max plant height:{}'.format(max_plant_height))
#     # rank_tensor = torch.argsort(max_plant_height).argsort() # argsort is not differentiable
#     # ranked_genotype = torch.gather(corresponding_genotype_list,dim=0,index=rank_tensor)
#
#     rank_index = torchsort.soft_rank(max_plant_height, regularization_strength=0.005)
#     # print(rank_index)

    # return rank_index.squeeze()

def genotype_ranking_loss(true_rank,predict_rank):
    """
    This function is to calculated genotype height ranking loss by comparing the true rank and predict genotype rank
    """
    from torchmetrics.regression import SpearmanCorrCoef
    assert true_rank.shape == predict_rank.shape
    # print(true_rank.shape,predict_rank.shape)
    loss_metric = SpearmanCorrCoef()
    spear_man_loss = 1-loss_metric(true_rank.float(),predict_rank.float())
    # mse = nn.MSELoss()
    # spear_man_loss = mse(true_rank.float(),predict_rank.float())
    return spear_man_loss
    # mismatch_rank_count =0
    # for i in range(len(true_rank)):
    #     if true_rank[i] != predict_rank[i]:
    #         mismatch_rank_count +=1
    # else:
    #     # print("mismatched_count:{}".format(mismatch_rank_count))
    #     return mismatch_rank_count

def train_and_validate(train_set, val_set, test_set, model, optimizer, epochs, pinn_weight,l2=0.5,y_max_bound=False,
                       smooth_loss=False,rule=False,ode_intergration_loss=False):
    #

    def logistic_growth_ode(t, y):
        # data generate from logistic growth model
        # r, y_max = model.r, model.y_max
        r = model.r.view(1, model.r.shape[0], 1)
        y_max = model.y_max.view(1, model.y_max.shape[0], 1)
        dY_dT = r * y * (1 - (y / y_max))
        return dY_dT
    # plt.ion()
    # fig = plt.figure()
    wandb.watch(model, log="all",log_freq=10)
    # model.init_network()

    model_dict={}
    train_validation_sum_losses=[]
    inputs_e, input_g, y, ts_train,g_id_train = train_set[0], train_set[1], train_set[2], train_set[3], train_set[4]
    print(inputs_e.shape, input_g.shape, y.shape, ts_train.shape)
    inputs_e_val, input_g_val, y_val, ts_val,g_id_val = val_set[0], val_set[1], val_set[2], val_set[3], val_set[4]
    inputs_e_test, input_g_test, y_test,ts_test,g_id_test = test_set[0], test_set[1], test_set[2], test_set[3], test_set[4]
    batch_size=256
    # Create DataLoaders for batching
    print(inputs_e.permute(1,0,2).shape,input_g.shape,y.permute(1,0,2).shape,ts_train.permute(1,0,2).shape,g_id_train.shape
          )
    train_loader = DataLoader(TensorDataset(inputs_e.permute(1,0,2),input_g,y.permute(1,0,2),ts_train.permute(1,0,2),g_id_train),
                              batch_size=batch_size, shuffle=False)


    # #mask every two days input
    # masked_e = mask_data_every_n_day(inputs_e, 2)
    # masked_ts = mask_data_every_n_day(ts_train, 2)
    # masked_g = input_g.clone().requires_grad_(True).to(DEVICE)

    # sns.scatterplot(torch.squeeze(y).detach())
    # plt.show()
    # raise EOFError
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_data_loss_running = 0.0
        derivative_loss_running =0.0
        penalize_r_loss_running=0.0
        y_max_loss_running =0.0
        # trainig_rank_loss=0

        for inputs_e, input_g, y, ts_train,g_id_train in train_loader:
            inputs_e = inputs_e.permute(1, 0, 2).requires_grad_(True)
            y = y.permute(1, 0, 2)
            ts_train=ts_train.permute(1, 0, 2)

            optimizer.zero_grad()

            if hasattr(model, 'trainable_param'):
                outputs,g_embed,smooth_loss_item = model(inputs_e,input_g,ts_train,g_id_train) #torch.Size([194, 88, 1])
            else:
                # print(inputs_e,input_g,ts_train)
                outputs, g_embed, smooth_loss_item = model(inputs_e, input_g, ts_train)
            # outputs.retain_grad()#convert it to a leaf node, retains gradient use for check later
            # check genotype rank loss
            # rank_true = transfer_prediction_to_ranking(y)
            # rank_predict = transfer_prediction_to_ranking(outputs)
            # rank_loss=genotype_ranking_loss(rank_true, rank_predict)
            # trainig_rank_loss += rank_loss
            data_loss = mask_rmse_loss(predict_y=outputs, true_y=y)
            train_data_loss_running+=data_loss
            # data_loss_plot = mask_rmse_loss(predict_y=outputs, true_y=y)
            # shapedtw_loss_train = mask_dtw_loss(true_y=y, predict_y=outputs)
            # mask = (~torch.isin(y, torch.tensor(0.0).to(DEVICE))).float()
            l2_loss_term = l2_loss(l2, model)
            total_loss = data_loss + l2_loss_term #+ rank_loss #+distance_loss


            #loss will be negetive when it increase
            if rule:
                diff_increase = torch.mean((outputs[:-1, :, :] - outputs[1:, :, :]))
                total_loss = total_loss + diff_increase
            if smooth_loss and epoch>200:
                raise EOFError ('this function is not used currently')
                masked_pred_y,_,_ = model(masked_e,masked_g,masked_ts)
                smooth_loss_item = torch.mean((outputs - masked_pred_y)**2)  # shift one time step, calculate the difference l2 penalize
                # # smoothness_loss=diff
                print('smooth_loss:{}'.format(smooth_loss_item))
                # # smoothness_loss = 1/(2+torch.exp(0.5-2*diff))
                total_loss = total_loss+smooth_loss_item #- diverse_loss
            else:
                smooth_loss_item=None
            if pinn_weight:
                derivative_loss,dy_dt_ode,dy_dt_nn = physics_loss(model,predict_y=outputs,ts_x=ts_train)
                derivative_loss_running += derivative_loss
                total_loss = total_loss + derivative_loss * pinn_weight
                penalize_r = torch.mean(0.1 * (0.00001 ** model.r))
                penalize_r_loss_running +=penalize_r
                total_loss = total_loss + penalize_r

                # T_np = ts_train.detach().numpy()
                # Temp_np = inputs_e.detach().numpy()
                # Y_np = outputs.detach().numpy()
                # dy_dt_np = dy_dt_nn.detach().numpy()
                # dy_dtemp = torch.autograd.grad(
                #     outputs, inputs_e, grad_outputs=torch.ones_like(outputs), create_graph=True, is_grads_batched=False
                # )[0]
                # dy_dtemp_np = dy_dtemp.detach().numpy()
                #
                # # Create 3D figure
                # fig = plt.figure(figsize=(10, 7))
                # ax = fig.add_subplot(111, projection='3d')
                #
                # # Plot the surface of y
                # ax.plot_surface(T_np, Temp_np, Y_np, cmap='coolwarm', alpha=0.6)
                # # Quiver plot: gradient field (dy/dt, dy/dtemp)
                # ax.quiver(T_np, Temp_np, Y_np, dy_dt_np, dy_dtemp_np, np.zeros_like(dy_dt_np), color='black',
                #           length=0.5, normalize=True)
                # # Labels and title
                # ax.set_xlabel('t (Time)')
                # ax.set_ylabel('Temp (Temperature)')
                # ax.set_zlabel('y (Function Output)')
                # ax.set_title('Gradient Field of y = sin(t) + log(temp)')
                # plt.show()

                if y_max_bound:
                    # reshpae_ymax = model.y_max.view(1, model.y_max.shape[0], 1)
                    ymax_true,_=torch.max(y, dim=0)

                    reshpae_ymax = model.y_max.view(model.y_max.shape[0], 1)

                    y_max_loss = torch.mean((reshpae_ymax-ymax_true)**2)**0.5 #outputs[-1,:,:]

                    # print('ymax_bound_loss :{}'.format(y_max_loss))
                    total_loss = total_loss+y_max_loss
                else:
                    y_max_loss = None
            else:
                if y_max_bound:
                    # reshpae_ymax = model.y_max.view(1, model.y_max.shape[0], 1)
                    ymax_true,_=torch.max(y, dim=0)
                    ymax_predict,_= outputs.max(dim=0)
                    y_max_loss = torch.mean(((ymax_predict - ymax_true) ** 2) ** 0.5)  # outputs[-1,:,:]
                    y_max_loss_running += y_max_loss
                    # print('ymax_bound_loss :{}'.format(y_max_loss))
                    total_loss = total_loss + y_max_loss
                else:
                    y_max_loss=None

            # time1 = time.time()
            # print(total_loss.device)
            # print()
            total_loss.backward()
            # sns.lineplot(torch.squeeze(copy.deepcopy(output.grad.detach()))[:, :10])
            optimizer.step()
            # time2 = time.time()
            # print('update time:{}'.format(time2-time1))
            running_loss = total_loss.item()
            train_loss += running_loss

        else:
            # print(len(train_loader))
            train_loss /= len(train_loader)
            train_data_loss_running /= len(train_loader)
            # trainig_rank_loss /= len(train_loader)
            if derivative_loss_running !=0.0:
                derivative_loss_running /=len(train_loader)
            if penalize_r_loss_running!=0.0:
                penalize_r_loss_running /=len(train_loader)
            if y_max_loss_running!=0.0:
                y_max_loss_running /= len(train_loader)

            # Validation
            model.eval()
            if hasattr(model, 'trainable_param'):
                outputs_val,_,_= model(inputs_e_val,input_g_val,ts_val,g_id_val) #torch.Size([194, 88, 1])
            else:
                outputs_val,_,_= model(inputs_e_val,input_g_val,ts_val)
            loss_val = mask_rmse_loss(predict_y=outputs_val, true_y=y_val)
            val_loss_plot = mask_rmse_loss(predict_y=outputs_val, true_y=y_val)
            if pinn_weight:
                derivative_loss_val, dy_dt_val_ode,dy_dt_nn_val = physics_loss(model, predict_y=outputs_val, ts_x=ts_val)#it does not affect weight update
                val_loss_plot = copy.deepcopy(loss_val.item())  # +derivative_loss_val.item()
                val_loss = loss_val.item()+derivative_loss_val.item()

                if y_max_bound:
                    ymax_true_val, _ = torch.max(y_val, dim=0)
                    reshpae_ymax_val = model.y_max.view(model.y_max.shape[0], 1)
                    # print(ymax_true.shape) #shapw=[sample_size,1]
                    # print(reshpae_ymax.shape)
                    y_max_loss_val = torch.mean((reshpae_ymax_val - ymax_true_val) ** 2) ** 0.5  # outputs[-1,:,:]
                    print("validation set y bound loss: {}".format(y_max_loss_val))
                    # val_loss = val_loss+y_max_loss_val
            else:
                val_loss = loss_val.item()
                # val_loss_plot = val_loss_plot.item()
                if y_max_bound:
                    ymax_true_val, _ = torch.max(y_val, dim=0)
                    ymax_predict_val, _ = torch.max(outputs_val, dim=0)
                    y_max_loss_val = torch.mean((ymax_predict_val - ymax_true_val) ** 2) ** 0.5  # outputs[-1,:,:]
                    val_loss = val_loss+ y_max_loss_val
            model.eval()
            with torch.no_grad():
                if hasattr(model, 'trainable_param'):
                    outputs_test,_,_ = model(inputs_e_test,input_g_test,ts_test,g_id_test)
                else:
                    outputs_test, _, _ = model(inputs_e_test, input_g_test, ts_test)
                loss_test = mask_rmse_loss(predict_y=outputs_test, true_y=y_test)
                test_loss = loss_test.item()

            # Log validation metrics
            if pinn_weight:
                #'val_train_sum_loss':train_validation_sum_losses[-1],
                wandb.log({"train_loss": train_loss, "val_loss": val_loss_plot, "test_loss": test_loss,
                           "physics_loss":derivative_loss_running,'penalize_r':penalize_r_loss_running,'train_data_loss':train_data_loss_running,"smooth_loss":smooth_loss_item,
                           "y_bound_loss":y_max_loss_running,'mean_r':torch.mean(model.r),'mean_ymax':torch.mean(model.y_max),
                           'std_r':torch.std(model.r),'std_ymax':torch.std(model.y_max)})#,"trainig_rank_loss":trainig_rank_loss})#'distance_loss':distance_loss
            else:
                #'val_train_sum_loss':train_validation_sum_losses[-1],
                wandb.log({"train_loss":train_loss,'train_data_loss':train_data_loss_running,"val_loss": val_loss, "test_loss":test_loss,
                           "smooth_loss":smooth_loss_item})#,"trainig_rank_loss":trainig_rank_loss}) #,"smooth_loss":smoothness_loss,"diverse_loss":diverse_loss*0.012

            # # # # # print(model.r) plot
            # # data_loss_plot = ((((y - outputs) * (y+1.0)) ** 2)** 0.5).cpu()
            # # # sns.lineplot(torch.squeeze(copy.deepcopy(outputs.grad.detach()))[:, :10])
            # sns.lineplot(torch.squeeze(copy.deepcopy(outputs.detach().cpu()))[:, :10])
            # # # for i in range(10):
            # # # #     sns.lineplot(torch.squeeze(copy.deepcopy(outputs.detach().cpu()))[:, i],color='r')
            # # #     # sns.lineplot(torch.squeeze(copy.deepcopy(ode_intergrate_loss.detach()))[:,:6])
            # # #     sns.lineplot(torch.squeeze(copy.deepcopy(dy_dt_ode.detach()))[:, i],color='g')
            # #
            # # #     data loss
            # # #     data_loss_plot=5*((y - outputs)*mask)**2
            # # #     print(data_loss_plot[:,0,:])
            # # #     data_loss_plot=torch.squeeze(copy.deepcopy(data_loss_plot.detach()))[:, :8]
            # # #     sns.lineplot(data_loss_plot)
            # # #     for i in range(data_loss_plot.shape[1]):  # Iterate over each column
            # # #         sns.lineplot(x=range(data_loss_plot.shape[0]), y=data_loss_plot[:, i].numpy(), color='green')
            # # # Calculate standard deviation of auto-gradients at each time step
            # # # grad_std_per_time = torch.std(copy.deepcopy(dy_dt_nn.detach()), dim=0)  # Std along the sample axis (dim=0)
            # # # sns.lineplot(torch.squeeze(grad_std_per_time), color='blue')
            # # # sns.lineplot(torch.squeeze(copy.deepcopy(dy_dt_nn.detach()))[:, :10], color='r')
            # # sns.lineplot(torch.squeeze(data_loss_plot.detach().cpu()* mask.cpu())[:,:10])
            # sns.scatterplot(torch.squeeze(copy.deepcopy(y).detach().cpu())[:,:10])
            # plt.legend(loc='upper right')
            # fig.canvas.draw_idle()
            # fig.canvas.flush_events()
            # plt.clf()


            # Save model checkpoints periodically
            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, loss: {running_loss:.4f}")
                print('validation loss: {}'.format(val_loss))
                print('Test loss: {}'.format(test_loss))
                # print('rank loss:{}'.format(trainig_rank_loss))
                # hue_g = torch.squeeze(g_id_train).detach().numpy()
                # print(hue_g)
                # plot_y = copy.deepcopy(torch.squeeze(y.detach())).numpy()
                # plot_pred_y = copy.deepcopy(torch.squeeze(outputs.detach())).numpy()
                # print(plot_pred_y)
                # colors = sns.color_palette("viridis", plot_y.shape[1])
                #
                # for seq in range(plot_y.shape[1]):
                #     sns.scatterplot(x=torch.squeeze(ts_train[:,0,:]).detach().numpy(),y =plot_y[:,seq],color=colors[seq])
                #     sns.lineplot(x=torch.squeeze(ts_train[:,0,:]).detach().numpy(),y=plot_pred_y[:,seq],color=colors[seq])
                # plt.ylim(-0.2,1.5)
                # plt.xlim(0, 285)
                # fig.canvas.draw_idle()
                # fig.canvas.flush_events()
                # plt.clf()
                if epoch >= 1500:
                    model_dict[str(val_loss)] = copy.deepcopy(model)
                    train_validation_sum_losses.append(val_loss)
                    # print('val physics loss:{}'.format(derivative_loss_val.item()))
    else:
        model_return = model_dict[str(min(train_validation_sum_losses))]
        epoch_num = list(model_dict.keys()).index(str(min(train_validation_sum_losses)))

        # torch.save(model_return.state_dict(), 'model_checkpoint.pth')
        return model_return,epoch_num*10+1500

def distance_criterion(output: torch.Tensor) -> torch.Tensor: #, class_id: int
    cluster_num = output.shape[1]
    # print(cluster_num)

    # Ensure tensors are on the right device and initialized properly
    output = output.to(DEVICE)
    euclidean_distance = torch.zeros(1, device=DEVICE)
    pairs = torch.zeros(1, device=DEVICE)

    for i in range(cluster_num):
        for j in range(i + 1, cluster_num):  # Start from i+1 to avoid duplicate pairs and self-pairs
            vector_i = output[ :,i]
            vector_j = output[:,j ]
            distance = torch.nn.functional.pairwise_distance(vector_i.unsqueeze(0), vector_j.unsqueeze(0))
            euclidean_distance += distance
            pairs += 1
    # Avoid division by zero
    if pairs.item() == 0:
        loss = torch.tensor(float('inf'), device=DEVICE)  # or any large value, or handle it differently
    else:
        loss = 1 / (euclidean_distance / pairs)
    return loss

def mask_data_every_n_day(input_tensor,n):
    """
    This function to to mask every n days with 0.0 to produce lower resolution time series
    input_tensor:torch.tensor need to be masked
    n:int, each n step
    """
    print("check input shape:{}".format(input_tensor.shape))
    input_tensor_mask = input_tensor.clone().detach()
    input_tensor_mask[::n] = 0.0
    if input_tensor.requires_grad:
        input_tensor_mask = input_tensor_mask.requires_grad_(True).to(DEVICE)
    else:
        input_tensor_mask = input_tensor_mask.to(DEVICE)
    return input_tensor_mask

def train_simple_g_e_interaction_model(if_pinn=False, model_name="", smooth_input=False, smooth_loss=False,
                                           genotype_encoding='binary_encoding', split_group='year_site.harvest_year',customize=False,
                                       reduce_time_resolution=False,smooth_temp=True,add_more_train_g=True):

    genetics_input_tensor, genotype_list, group_df, model_name, num_genotypes, plant_height_tensor, \
        temperature_full_length_tensor, temperature_same_length_tensor = load_data_from_dill(
        genotype_encoding, model_name, smooth_temp,plant_height_file='../temporary/plant_height_tensor_all.dill',
        group_df_file='../temporary/group_list_df_all.dill',temperature_file='../temporary/temperature_tensor_same_length_all.dill',
        fill_in_na_at_start=True,rescale=True)

    if smooth_temp:
        model_name = model_name +'smooth_temp_True'
    else:
        model_name = model_name + 'smooth_temp_False'
    if reduce_time_resolution:
        #mask every two days
        print('Before reduce resolution, {} days plant height left'.format(
            plant_height_tensor.count_nonzero(dim=0)))
        plant_height_tensor = mask_data_every_n_day(plant_height_tensor,2)
        print(plant_height_tensor[:, 0, 0])
        temperature_same_length_tensor = mask_data_every_n_day(temperature_same_length_tensor,2)
        temperature_full_length_tensor = mask_data_every_n_day(temperature_full_length_tensor, 2)
        print('After reduce resolution, {} days plant height left'.format(plant_height_tensor.count_nonzero(dim=0)))
        model_name = model_name+'half_resolution'
        # raise EOFError
    if split_group == 'g_e':
        #split to predict new year, new genotype
        train_test_validation_dictionary = train_test_split_based_on_two_groups(copy.deepcopy(group_df), copy.deepcopy(group_df),
                                                                                n_split=5)
        model_name = model_name + split_group +'_split'
        # raise EOFError
    else:
        # split based on either year or genotype, if get two columns,
        # it can only make sure at least one of them is different for samples be splitted into different groups(train test etc.)
        train_test_validation_dictionary = train_test_split_based_on_group(copy.deepcopy(group_df), copy.deepcopy(group_df),
                                                                           group_name=[
                                                                               split_group
                                                                           ], n_split=5)
        # train_test_validation_dictionary,train_years = manually_data_split_based_on_one_group(copy.deepcopy(group_df),
        #                                                                           split_group=split_group, n_splits=None)

        if split_group == 'year_site.harvest_year':
            model_name = model_name + 'year_split'
        elif split_group == 'genotype.id':
            model_name = model_name + 'genotype_split'
        else:
            raise ValueError ('the input split_group is wrong!')
    # print(train_test_validation_dictionary)
    n_split =len(train_test_validation_dictionary.keys())
    # assert n_split==6
    # temperature_same_length_tensor,_= minmax_scaler(temperature_same_length_tensor)
    print(plant_height_tensor.shape)
    print(temperature_same_length_tensor.shape)
    print(genetics_input_tensor.shape)
    print(genotype_list)
    print(group_df)

    # raise EOFError
    for n in [0]:

        train_index, validation_index, test_index = train_test_validation_dictionary['splits_{}'.format(n)]
        print('before adding genotype train sequences number')
        print(len(train_index))

        train_y = plant_height_tensor[115:, train_index, :].to(DEVICE)
        train_env = temperature_same_length_tensor[115:, train_index, :].to(DEVICE)
        # sns.lineplot(train_env.squeeze())
        # plt.show()
        # train_env_full = temperature_full_length_tensor[115:, train_index, :].to(DEVICE)
        train_group_df = copy.deepcopy(group_df).iloc[train_index,:]

        keep_genotype_kinship = [33, 106, 122, 133, 5, 30, 218, 2, 17, 254, 282, 294, 301, 302, 335, 339, 341, 6, 362]
        genetics_input_tensor = read_genetics_kinship_matrix_with_training_genotype(
            genotype_tensor=genetics_input_tensor, g_order=genotype_list, traing_genotype=keep_genotype_kinship)
        # if split_group == 'g_e':
        #     keep_genotype_kinship=[33, 106, 122, 133, 5, 30, 218, 2, 17, 254, 282, 294, 301, 302, 335, 339, 341, 6, 362]
        #     genetics_input_tensor = read_genetics_kinship_matrix_with_training_genotype(
        #         genotype_tensor=genetics_input_tensor, g_order=genotype_list, traing_genotype=keep_genotype_kinship) #train_group_df['genotype.id'].unique()

        validation_y = plant_height_tensor[115:, validation_index, :].to(DEVICE)
        validation_env = temperature_same_length_tensor[115:, validation_index, :].to(DEVICE)
        # validation_env_full = temperature_full_length_tensor[115:, validation_index, :].to(DEVICE)
        validation_group_df = copy.deepcopy(group_df).iloc[validation_index,:]

        test_y = plant_height_tensor[115:, test_index, :].to(DEVICE)
        test_env = temperature_same_length_tensor[115:, test_index, :].to(DEVICE)
        # test_env_full = temperature_full_length_tensor[115:, test_index, :].to(DEVICE)
        test_group_df = copy.deepcopy(group_df).iloc[test_index, :]

        original_genotype_ids = torch.tensor(genotype_list)

        train_g = genetics_input_tensor[ train_index, :].to(DEVICE)
        validation_g = genetics_input_tensor[validation_index, :].to(DEVICE)
        test_g = genetics_input_tensor[test_index, :].to(DEVICE)

        # get genotype id list after averge replicates
        train_genotype_id = original_genotype_ids[train_index].unsqueeze(-1).unsqueeze(0)
        val_genotype_id = original_genotype_ids[validation_index].unsqueeze(-1).unsqueeze(0)
        test_genotype_id = original_genotype_ids[test_index].unsqueeze(-1).unsqueeze(0)

        print(test_genotype_id,test_genotype_id.shape)
        if add_more_train_g:
            #add_more_training_data
            overlap_g_list = tuple(copy.deepcopy(original_genotype_ids)[train_index].tolist() + copy.deepcopy(original_genotype_ids)[validation_index].tolist())
            overlap_year_list = tuple(copy.deepcopy(train_group_df)['year_site.harvest_year'].unique().tolist()) + tuple(copy.deepcopy(validation_group_df)['year_site.harvest_year'].unique().tolist())
            add_plant_height_tensor, add_temperature_same_length_tensor, add_group_df, add_genetics_input_tensor,add_genotype_id = load_more_training_data(
                data_plit=split_group,
                overlap_g=overlap_g_list, overlap_year=overlap_year_list,fill_in_na_at_start=True,smooth_temp=smooth_temp,genotype_encoding='kinship_matrix_encoding',rescale=True)

            train_y = torch.cat([train_y,add_plant_height_tensor],dim=1)
            train_env = torch.cat([train_env,add_temperature_same_length_tensor],dim=1)
            train_group_df = pd.concat([train_group_df,add_group_df],axis=0)
            print(train_group_df)
            train_g = torch.cat([train_g,add_genetics_input_tensor],dim=0)
            train_genotype_id=torch.cat([train_genotype_id,add_genotype_id],dim=1)
            train_genotype_id_before_average= copy.deepcopy(train_genotype_id).squeeze().to(DEVICE)
            print('train shape after add more train_g')
            print(train_genotype_id.shape)
            print(train_y.shape)
            print(train_env.shape)
            print(train_group_df.shape)
            print(train_g.shape)
            # raise EOFError
            # raise EOFError
        else:
            train_genotype_id_before_average = copy.deepcopy(train_genotype_id).squeeze().to(DEVICE)
        train_genotype_id, _ = average_based_on_group_df(copy.deepcopy(train_genotype_id),
                                                         train_group_df)
        train_genotype_id = torch.squeeze(train_genotype_id)
        val_genotype_id, _ = average_based_on_group_df(copy.deepcopy(val_genotype_id),
                                                         validation_group_df)
        val_genotype_id = torch.squeeze(val_genotype_id)
        test_genotype_id, _ = average_based_on_group_df(copy.deepcopy(test_genotype_id),
                                                       test_group_df)
        test_genotype_id = torch.squeeze(test_genotype_id) #
        # creat time sequence, same as input shape
        ts_train = torch.linspace(0.0, 284, 285).unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, train_y.shape[1],
                                                                                          1)[:-115, :,
                   :].requires_grad_(True).to(DEVICE)  # time sequences steps
        ts_validation = torch.linspace(0.0, 284, 285).unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, validation_y.shape[1],
                                                                                               1)[:-115, :,
                        :].requires_grad_(True).to(DEVICE)  # time sequences steps
        ts_test = torch.linspace(0.0, 284, 285).unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, test_y.shape[1],
                                                                                         1)[:-115, :, :].to(
            DEVICE)  # time sequences steps

        # average across replicates predict result
        print('train')
        train_env_avg, train_g_avg, train_y_avg, ts_train_avg, train_group_df_avg = calculate_average_across_replicates(
            train_env, train_g, train_group_df, train_y, ts_train)
        print('val')
        validation_env_avg, validation_g_avg, validation_y_avg, ts_validation_avg, validation_group_df_avg = calculate_average_across_replicates(
            validation_env, validation_g, validation_group_df, validation_y,
            ts_validation)
        print('test')
        test_env_avg, test_g_avg, test_y_avg, ts_test_avg, test_group_df_avg = \
            calculate_average_across_replicates(
                test_env, test_g, test_group_df, test_y,
                ts_test)

        different_genotype_curve_train = pd.DataFrame(
            torch.squeeze(copy.deepcopy(train_y_avg.detach())).cpu().numpy())
        different_genotype_curve_train.columns = train_genotype_id.tolist()
        different_genotype_curve_train.to_csv(
            "pinn_result/multipl_g/train_true_curves_{}.csv".format(
                 model_name))

        different_genotype_curve_val = pd.DataFrame(
            torch.squeeze(copy.deepcopy(validation_y_avg.detach())).cpu().numpy())
        different_genotype_curve_val.columns = val_genotype_id.tolist()
        different_genotype_curve_val.to_csv(
            "pinn_result/multipl_g/val_true_curves_{}.csv".format(
                 model_name))

        different_genotype_curve_test = pd.DataFrame(
            torch.squeeze(copy.deepcopy(test_y_avg.detach())).cpu().numpy())
        different_genotype_curve_test.columns = test_genotype_id.tolist()
        different_genotype_curve_test.to_csv(
            "pinn_result/multipl_g/test_true_curves_{}.csv".format(
                model_name))

        try:
            result_df = pd.read_csv('pinn_result/result_summary/multiple_g_result_pinn_{}_{}.csv'.format(if_pinn,model_name),header=0,index_col=0)
            print(result_df)
        except:
            result_df =pd.DataFrame()
            # raise EOFError
        for hidden in [5]: #3,5
            for genetics_embedding_size in [3,5]: #3,5
                for last_fc_hidden_size in [3,5]:
                    for num_layer in [1]:
                        for l2 in [1.0]: #0.1,0.5,1.0
                            for lr in [0.001]: #0.001,0.0005
                                if if_pinn:
                                    pinn_weight = [2] #2,5,7,9
                                    param_bounds = [True]
                                else:
                                    pinn_weight = [None]
                                    param_bounds = [False]
                                if smooth_input:
                                    smooth_loss_list = [False]
                                else:
                                    smooth_loss_list = [False]
                                for smooth_loss in smooth_loss_list:
                                    for y_max_bound in param_bounds:
                                        for pinn in pinn_weight:
                                            for j in [1,2,3,4,5]:
                                                random.seed(j)
                                                np.random.seed(j)
                                                torch.manual_seed(j)
                                                # Initialize W&B
                                                epoches=3000
                                                # wandb.init(project='simple_g_e_height_prediction_smooth')
                                                run = wandb.init(
                                                    # Set the project where this run will be logged
                                                    project='{}_multiple_g_pinn'.format(model_name),
                                                    # Track hyperparameters and run metadata
                                                    config={
                                                        "learning_rate": lr,
                                                        "epochs": epoches,
                                                        "n_split": n,
                                                        "random_sees": j,
                                                        "hidden_size": hidden,
                                                        "num_layer":num_layer,
                                                        "physics_weight":pinn,
                                                        "y_max_bound":y_max_bound,
                                                        "smooth_loss":smooth_loss,
                                                        'smooth_input': smooth_input,
                                                        'reduce_time_resolution': reduce_time_resolution,
                                                        "l2":l2,
                                                        "last_fc_hidden_size":last_fc_hidden_size,
                                                        "genetics_embedding_size":genetics_embedding_size

                                                    },
                                                )

                                                if pinn:
                                                    if customize:
                                                        model = pinn_genotype_embedding_CUSTOMIZE_RNN(genetic_feature_size=train_g.shape[-1],
                                                                                                                 input_size=temperature_same_length_tensor.shape[-1],
                                                                                                                 hidden_size=hidden,num_layer=num_layer,
                                                                                                                 genotype_num=num_genotypes).to(DEVICE)
                                                    else:
                                                        model = pinn_genotype_embedding_fc(genetic_feature_size=train_g.shape[-1],
                                                                                                                     input_size=temperature_same_length_tensor.shape[-1],
                                                                                                                     hidden_size=hidden,num_layer=num_layer,
                                                                                                                     last_fc_hidden_size=last_fc_hidden_size,genetics_embedding_size=genetics_embedding_size).to(DEVICE)
                                                        # model = pinn_genotype_embedding_trainable_parameter_fc(genetic_feature_size=train_g.shape[-1],
                                                        #                                                              input_size=temperature_same_length_tensor.shape[-1],
                                                        #                                                              hidden_size=hidden,num_layer=num_layer,
                                                        #                                                              last_fc_hidden_size=last_fc_hidden_size,genetics_embedding_size=genetics_embedding_size).to(DEVICE)

                                                else:
                                                    if customize:
                                                        model = simple_genotype_code_temperature_input_height_prediction_CUSTOMIZE_RNN(genetic_feature_size=genetics_input_tensor.shape[-1],
                                                                                                                     input_size=temperature_same_length_tensor.shape[-1],
                                                                                                                     hidden_size=hidden,num_layer=num_layer,
                                                                                                                     genotype_num=train_y.shape[0]).to(DEVICE)
                                                    else:
                                                        model = genotype_code_temperature_input_height_prediction_fc(genetic_feature_size=genetics_input_tensor.shape[-1],
                                                                                                                     input_size=temperature_same_length_tensor.shape[-1],
                                                                                                                     hidden_size=hidden,num_layer=num_layer,
                                                                                                                     last_fc_hidden_size=last_fc_hidden_size,genetics_embedding_size=genetics_embedding_size).to(DEVICE)
                                                model.init_network()

                                                total_params = count_parameters(model)

                                                # Initialize model, criterion, and optimizer
                                                optimizer = optim.Adam(model.parameters(), lr=lr)

                                                # Train and validate the model
                                                model,epoch_num=train_and_validate([train_env, train_g, train_y,ts_train,train_genotype_id_before_average],
                                                                                   [validation_env, validation_g, validation_y,ts_validation,original_genotype_ids[validation_index]],
                                                                   [test_env, test_g, test_y,ts_test,original_genotype_ids[test_index]], model, optimizer, epoches, pinn_weight=pinn,l2=l2,y_max_bound=y_max_bound,smooth_loss=smooth_loss)
                                                print('find minimize validation loss at epoch:{}'.format(epoch_num))
                                                wandb.log({"result_epochs": epoch_num})

                                                print(train_env_avg.shape)
                                                print(train_g_avg.shape)
                                                print(ts_train_avg.shape)
                                                if hasattr(model, 'trainable_param'):
                                                    train_pred_y, _, _ = model(train_env_avg, train_g_avg, ts_train_avg,train_genotype_id)
                                                else:
                                                    train_pred_y,_,_ = model(train_env_avg,train_g_avg,ts_train_avg)
                                                if if_pinn:
                                                    r_train = model.r.detach().cpu()
                                                    y_max_train = model.y_max.detach().cpu()
                                                else:
                                                    r_train = None
                                                    y_max_train = None
                                                train_rmse = mask_rmse_loss(true_y=train_y_avg,predict_y=train_pred_y)
                                                # shapedtw_loss_train = mask_dtw_loss(true_y=train_y_avg,predict_y=train_pred_y)
                                                # print('train shapedtw distance:{}'.format(shapedtw_loss_train))
                                                if hasattr(model, 'trainable_param'):
                                                    val_pred_y,_,_ = model(validation_env_avg, validation_g_avg, ts_validation_avg,val_genotype_id)
                                                else:
                                                    val_pred_y, _, _ = model(validation_env_avg, validation_g_avg,
                                                                             ts_validation_avg)
                                                if if_pinn:
                                                    r_val = model.r.detach().cpu()
                                                    y_max_val = model.y_max.detach().cpu()
                                                else:
                                                    r_val = None
                                                    y_max_val = None
                                                val_rmse = mask_rmse_loss(true_y=validation_y_avg, predict_y=val_pred_y)
                                                if hasattr(model, 'trainable_param'):
                                                    test_pred_y, _, _ = model(test_env_avg, test_g_avg, ts_test_avg,test_genotype_id)
                                                else:
                                                    test_pred_y, _,_ = model(test_env_avg, test_g_avg, ts_test_avg)
                                                test_rmse = mask_rmse_loss(true_y=test_y_avg, predict_y=test_pred_y)

                                                corre_train = mask_dtw_loss(true_y=train_y_avg,predict_y=train_pred_y)
                                                print('train shapeDTW')
                                                print(corre_train)
                                                corre_validation = mask_dtw_loss(true_y=validation_y_avg, predict_y=val_pred_y)
                                                print('validation shapeDTW')
                                                print(corre_validation)
                                                corre_test = mask_dtw_loss(true_y=test_y_avg, predict_y=test_pred_y)
                                                print('test shapeDTW')
                                                print(corre_test)

                                                if if_pinn:
                                                    r_test = model.r.detach().cpu()
                                                    y_max_test = model.y_max.detach().cpu()
                                                else:
                                                    r_test = None
                                                    y_max_test = None

                                                new_row = pd.DataFrame({
                                                        "learning_rate": lr,
                                                        "epochs": epoch_num, "n_split": n,
                                                        "random_sees": j,
                                                        "hidden_size": hidden,
                                                        'l2':l2,
                                                        "num_layer":num_layer,
                                                    "last_fc_hidden_size":last_fc_hidden_size,
                                                    "genetics_embedding_size":genetics_embedding_size,
                                                    'trainable_parameters':total_params,
                                                        "physics_weight":pinn,
                                                    "y_max_bound":y_max_bound,
                                                    "smooth_loss":smooth_loss,
                                                    "train_rMSE":round(train_rmse.item(),3),"validation_rMSE":round(val_rmse.item(),3),
                                                    "test_rMSE":round(test_rmse.item(),3),
                                                    'train_shapeDTW': round(corre_train, 3),
                                                    'validation_shapeDTW': round(corre_validation, 3),
                                                    "test_shapeDTW": round(corre_test, 3),
                                                    # 'train_r2': round(corre_train.item(), 3),
                                                    # 'validation_r2': round(corre_validation.item(), 3),
                                                    # "test_r2": round(corre_test.item(), 3),
                                                    },index=[0])
                                                result_df = pd.concat([result_df,new_row])
                                                result_df.to_csv('pinn_result/result_summary/multiple_g_result_pinn_{}_{}.csv'.format(if_pinn,model_name))
                                                name_str = "pinn_result/multipl_g/pinn_weight{}_{}_train_lr_{}_hidden_{}_fc_hidden_{}_g_embed{}_num_layers_{}_l2_{}_ymax_bound_{}_smooth_loss{}_seed_{}".format(pinn,model_name,lr,hidden,last_fc_hidden_size,genetics_embedding_size,num_layer,l2,y_max_bound,smooth_loss,j)

                                                #save predicted curves
                                                if if_pinn:
                                                    print(r_val.numpy(),val_genotype_id)
                                                    df_ode_parameters = pd.DataFrame(
                                                        {'genotype': val_genotype_id, 'predicted_r_val': r_val.squeeze().numpy(),
                                                         'predicted_y_max_val': y_max_val.squeeze().numpy()})
                                                    df_ode_parameters.to_csv("pinn_result/multipl_g/pinn_weight_{}_multiple_g_parameters_{}_train_lr_{}_hidden_{}_fc_hidden_{}_g_embed{}_num_layers_{}_l2_{}_ymax_bound_{}_smooth_loss{}_seed_{}.csv".format(pinn,model_name,lr,hidden,last_fc_hidden_size,genetics_embedding_size,num_layer,l2,y_max_bound,smooth_loss,j))
                                                # print(test_y.shape)

                                                different_genotype_predict_curve_train = pd.DataFrame(
                                                    torch.squeeze(copy.deepcopy(train_pred_y.detach())).cpu().numpy())
                                                different_genotype_predict_curve_train.columns = train_genotype_id.tolist()
                                                different_genotype_predict_curve_train.to_csv(
                                                    "pinn_result/multipl_g/pinn_weight_{}_train_curves_{}_train_lr_{}_hidden_{}_fc_hidden_{}_g_embed{}_num_layers_{}_l2_{}_ymax_bound_{}_smooth_loss{}_seed_{}.csv".format(
                                                        pinn, model_name, lr, hidden, last_fc_hidden_size,genetics_embedding_size,num_layer, l2,y_max_bound,smooth_loss,j))
                                                different_genotype_predict_curve_val = pd.DataFrame(
                                                    torch.squeeze(copy.deepcopy(val_pred_y.detach())).cpu().numpy())
                                                different_genotype_predict_curve_val.columns = val_genotype_id.tolist()
                                                different_genotype_predict_curve_val.to_csv(
                                                    "pinn_result/multipl_g/pinn_weight_{}_val_curves_{}_train_lr_{}_hidden_{}_fc_hidden_{}_g_embed{}_num_layers_{}_l2_{}_ymax_bound_{}_smooth_loss{}_seed_{}.csv".format(
                                                        pinn, model_name, lr, hidden, last_fc_hidden_size,genetics_embedding_size,num_layer, l2,y_max_bound,smooth_loss,j))
                                                #save predicted curves from train test validation, columns name is genotype id
                                                different_genotype_predict_curve_test = pd.DataFrame(torch.squeeze(copy.deepcopy(test_pred_y.detach())).cpu().numpy())
                                                different_genotype_predict_curve_test.columns = test_genotype_id.tolist()
                                                print('predict test curve')
                                                different_genotype_predict_curve_test.to_csv(
                                                    "pinn_result/multipl_g/pinn_weight_{}_test_curves_{}_train_lr_{}_hidden_{}_fc_hidden_{}_g_embed{}_num_layers_{}_l2_{}_ymax_bound_{}_smooth_loss{}_seed_{}.csv".format(
                                                        pinn, model_name, lr, hidden,last_fc_hidden_size,genetics_embedding_size,num_layer, l2, y_max_bound,smooth_loss,j))

                                                with open(
                                                        'pinn_result/multipl_g/model/pinn_weight_{}_model_{}_train_lr_{}_hidden_{}_fc_hidden_{}_g_embed{}_num_layers_{}_l2_{}_ymax_bound_{}_smooth_loss{}_seed_{}.dill'.format(
                                                        pinn, model_name, lr, hidden, last_fc_hidden_size,genetics_embedding_size,num_layer, l2, y_max_bound,smooth_loss,j), 'wb') as file:
                                                    dill.dump(model, file)
                                                    # torch.save(model, file)
                                                    # Load the entire model and use it directly
                                                file.close()

                                                fig_train = plot_multiple_genotype(train_pred_y, train_y_avg, color_label=train_genotype_id,
                                                                       marker_label=train_group_df_avg['year_site.harvest_year'],
                                                                                   name=name_str+'train',y_max_pred=y_max_train,
                                                                                   r_value_pred=r_train,num_epoch=str(epoch_num),env=train_env_avg)
                                                wandb.log({'train_prediction_plot':wandb.Image(fig_train)})
                                                plt.close()
                                                fig_val = plot_multiple_genotype(val_pred_y, validation_y_avg, color_label=val_genotype_id,
                                                                       marker_label=validation_group_df_avg['year_site.harvest_year'],
                                                                                 name=name_str+'val',y_max_pred=y_max_val,
                                                                                 r_value_pred=r_val,num_epoch=str(epoch_num),env=validation_env_avg)
                                                wandb.log({'val_prediction_plot': wandb.Image(fig_val)})
                                                plt.close()
                                                fig_test = plot_multiple_genotype(test_pred_y, test_y_avg, color_label=test_genotype_id,
                                                                       marker_label=test_group_df_avg['year_site.harvest_year'],
                                                                                  name=name_str+'test',y_max_pred=y_max_test,
                                                                                  r_value_pred=r_test,num_epoch=str(epoch_num),env=test_env_avg)
                                                wandb.log({'test_prediction_plot': wandb.Image(fig_test)})
                                                plt.close()
                                                run.finish()

def load_more_training_data(data_plit='genotype.id',overlap_g:tuple=(106),overlap_year:tuple=('2019'),plant_height_file='../temporary/plant_height_tensor_more.dill'
                        ,group_df_file='../temporary/group_list_df_more.dill',temperature_file ='../temporary/temperature_tensor_same_length_more.dill'
                        ,smooth_temp=True,genotype_encoding:str='kinship_matrix_encoding',fill_in_na_at_start=True,rescale=False):
    if data_plit=='genotype.id':
        overlap_year=()
    elif data_plit== 'year_site.harvest_year':
        overlap_g=()
    elif data_plit=='g_e':
        print('exclude overlap year and g')
    else:
        raise ValueError ('the input data_split parameter is wrong!')
    with open(plant_height_file, 'rb') as f:
        plant_height_tensor = dill.load(f)
    f.close()
    with open(group_df_file, 'rb') as f:
        group_df = dill.load(f)
    f.close()
    with open(temperature_file, 'rb') as f:
        temperature_same_length_tensor = dill.load(f)
        print(temperature_same_length_tensor[:,0,0])
        # SCALE TO BETWEEN(0, 1), first rescale then conver na to zero because there are 0.0 in real data for temperature
        if rescale:
            temperature_same_length_tensor, _ = minmax_scaler(temperature_same_length_tensor)
        if smooth_temp:
            temperature_same_length_tensor = smooth_tensor_ignore_nan(temperature_same_length_tensor, window_size=15)
            temperature_same_length_tensor[plant_height_tensor==0.0]=np.nan
        temperature_same_length_tensor = torch.nan_to_num(temperature_same_length_tensor, nan=0.0, posinf=0.0,
                                                          neginf=0.0)
    f.close()
    print(group_df)
    group_df = group_df[['year_site.harvest_year', 'genotype.id']]
    group_df['new_group_list'] = group_df.astype(str).apply(lambda row: '_'.join(row), axis=1)
    genotype_list = group_df['genotype.id'].to_list()
    tensor_list = []
    print('using {} genotype encoding method'.format(genotype_encoding))
    for genotype in genotype_list:
        # print(' open and load from ../temporary/{}_{}_all_present_genotype.dill'.format(genotype, genotype_encoding))
        with open('../temporary/{}_{}_all_present_genotype.dill'.format(genotype, genotype_encoding), 'rb') as file1:
            genotype_tensor = dill.load(file1)
            # print(genotype_tensor.shape)
            # conver to between 0 and 1
            # print(genotype_tensor)
            # if genotype_encoding == 'distance_encoding' or  ('kinship_matrix_encoding' in genotype_encoding):
            #     print(genotype_tensor.shape)
            tensor_list.append(genotype_tensor)
    genetics_input_tensor = torch.cat(tensor_list, dim=1).float()
    # print(genetics_input_tensor.shape)
    genetics_input_tensor = torch.permute(genetics_input_tensor, (1, 0))

    keep_genotype_kinship = [33, 106, 122, 133, 5, 30, 218, 2, 17, 254, 282, 294, 301, 302, 335, 339, 341, 6, 362]
    genetics_input_tensor = read_genetics_kinship_matrix_with_training_genotype(
        genotype_tensor=genetics_input_tensor, g_order=genotype_list, traing_genotype=keep_genotype_kinship)

    # check shape to perform split
    print('plant_height_tensor shape more g')
    print(plant_height_tensor.shape)
    print('genetics_input_tensor shape more g')
    print(genetics_input_tensor.shape)  # sample size,1(feature size)
    print('environment tensor shape more g:')
    print(temperature_same_length_tensor.shape)

    if fill_in_na_at_start:
        #find the minimum value position, set all value before as a very small number 0.0001
        plant_height_tensor[plant_height_tensor == 0.0] = 999.0 #set na to 999 to find minimum
        for seq in range(plant_height_tensor.shape[1]):
            plant_height_tensor = torch.nan_to_num(plant_height_tensor, nan=0.0, posinf=0.0, neginf=0.0)
            min_position = torch.argmin(plant_height_tensor[:,seq,:]).item()

            if torch.min(plant_height_tensor[:,seq,:]).item()>0.0:
                plant_height_tensor[:min_position+1, seq,:] = torch.min(plant_height_tensor[:,seq,:]).item()
            else:
                plant_height_tensor[:min_position+1, seq,:] = 0.0001
        else:
            plant_height_tensor[plant_height_tensor == 999.0] = 0.0 #set nan back to 0.0

    full_index = group_df.index.to_list()
    #get index for years and genotypes already in validation and test set.
    overlap_g_index = group_df[group_df['genotype.id'].isin(overlap_g)].index.tolist()
    print('the following genotypes has already used in validation ande test: \n {}'.format(group_df[group_df['genotype.id'].isin(overlap_g)]['genotype.id'].unique()))

    overlap_year_index = group_df[group_df['year_site.harvest_year'].isin(overlap_year)].index.tolist()
    #combine overlap year and overlap g index
    overlap_index_all = sorted(list(set(overlap_g_index+overlap_year_index)))


    assert len(set(full_index))==len(full_index) #index shouldn't have overlap
    # drop sequences with based on thoes index
    kept_index = sorted(set(full_index) - set(overlap_index_all))
    plant_height_tensor = plant_height_tensor[115:, kept_index, :].to(DEVICE)
    temperature_same_length_tensor = temperature_same_length_tensor[115:, kept_index, :].to(DEVICE)
    group_df = copy.deepcopy(group_df).iloc[kept_index, :]
    genetics_input_tensor = genetics_input_tensor[kept_index, :].to(DEVICE)
    genotype_id_list = copy.deepcopy(group_df['genotype.id']).to_list()

    genotype_id=torch.tensor(genotype_id_list).unsqueeze(-1).unsqueeze(0)

    return plant_height_tensor,temperature_same_length_tensor,group_df,genetics_input_tensor,genotype_id


def load_data_from_dill(genotype_encoding, model_name, smooth_temp,plant_height_file='../temporary/plant_height_tensor_all.dill'
                        ,group_df_file='../temporary/group_list_df_all.dill',temperature_file ='../temporary/temperature_tensor_same_length_all.dill',
                        fill_in_na_at_start=True,rescale=False):
    with open(plant_height_file, 'rb') as f:
        plant_height_tensor = dill.load(f)
    f.close()
    with open(group_df_file, 'rb') as f:
        group_df = dill.load(f)
    f.close()
    with open(temperature_file, 'rb') as f:
        temperature_same_length_tensor = dill.load(f)
        print(temperature_same_length_tensor[:,0,0])
        if rescale:
            # SCALE TO BETWEEN(0, 1), first rescale then conver na to zero because there are 0.0 in real data for temperature
            temperature_same_length_tensor, _ = minmax_scaler(temperature_same_length_tensor)
        if smooth_temp:
            temperature_same_length_tensor = smooth_tensor_ignore_nan(temperature_same_length_tensor, window_size=15)
            # temperature_same_length_tensor[plant_height_tensor==0.0]=np.nan
        temperature_same_length_tensor = torch.nan_to_num(temperature_same_length_tensor, nan=0.0, posinf=0.0,
                                                          neginf=0.0)
    f.close()
    with open('../temporary/temperature_tensor_all.dill', 'rb') as f:
        temperature_full_length_tensor = dill.load(f)
    f.close()
    model_name = model_name + genotype_encoding
    print(group_df)
    group_df = group_df[['year_site.harvest_year', 'genotype.id']]
    group_df['new_group_list'] = group_df.astype(str).apply(lambda row: '_'.join(row), axis=1)
    genotype_list = group_df['genotype.id'].to_list()
    num_genotypes = len(group_df['genotype.id'].unique())
    tensor_list = []
    print('using {} genotype encoding method'.format(genotype_encoding))
    for genotype in genotype_list:
        # print(' open and load from ../temporary/{}_{}.dill'.format(genotype, genotype_encoding))
        with open('../temporary/{}_{}.dill'.format(genotype, genotype_encoding), 'rb') as file1:
            genotype_tensor = dill.load(file1)
            # print(genotype_tensor.shape)
            # conver to between 0 and 1
            # print(genotype_tensor)
            # if genotype_encoding == 'distance_encoding' or  ('kinship_matrix_encoding' in genotype_encoding):
                # print(genotype_tensor.shape)
                # raise EOFError
                # genotype_tensor, _ = minmax_scaler(genotype_tensor, min=-1, max=1)
                # genotype_tensor = torch.tanh(genotype_tensor)
            # print('after scaling')
            # print(genotype_tensor)
            tensor_list.append(genotype_tensor)
    genetics_input_tensor = torch.cat(tensor_list, dim=1).float()
    # print(genetics_input_tensor.shape)
    genetics_input_tensor = torch.permute(genetics_input_tensor, (1, 0))
    # check shape to perform split
    print('plant_height_tensor shape and na')
    print(plant_height_tensor.shape)
    print(plant_height_tensor.isnan().sum())
    print('genetics_input_tensor shape')
    print(genetics_input_tensor.shape)  # sample size,1(feature size)
    print('environment tensor shape:')
    print(temperature_same_length_tensor.shape)

    # plant_height_tensor = torch.nan_to_num(plant_height_tensor, nan=0.0, posinf=0.0,
    #                                                   neginf=0.0)
    if fill_in_na_at_start:
        #find the minimum value position, set all value before as a very small number 0.0001
        plant_height_tensor[plant_height_tensor == 0.0] = 999.0 #set na to 999 to find minimum
        for seq in range(plant_height_tensor.shape[1]):
            plant_height_tensor = torch.nan_to_num(plant_height_tensor, nan=0.0, posinf=0.0, neginf=0.0)

            min_position = torch.argmin(plant_height_tensor[:,seq,:]).item()
            # print(tensor_dataset[:, seq, :])
            # raise EOFError
            if torch.min(plant_height_tensor[:,seq,:]).item()>0.0:
                plant_height_tensor[:min_position+1, seq,:] = torch.min(plant_height_tensor[:,seq,:]).item()
            else:
                plant_height_tensor[:min_position+1, seq,:] = 0.0001
        else:
            plant_height_tensor[plant_height_tensor == 999.0] = 0.0 #set nan back to 0.0
    return genetics_input_tensor, genotype_list, group_df, model_name, num_genotypes, plant_height_tensor, temperature_full_length_tensor, temperature_same_length_tensor


def calculate_average_across_replicates(train_env, train_g, train_group_df, train_y, ts_train):


    print('train_g shape:{}'.format(train_g.shape))
    train_g=train_g.unsqueeze(0) #change to shape[1,sample size,feature size] for average
    train_env_avg,average_group_df = average_based_on_group_df(train_env, df=train_group_df)

    train_g_avg,_ = average_based_on_group_df(train_g,
                                               df=train_group_df)
    train_g_avg = train_g_avg.squeeze(dim=0)
    print('train_g shape avg:{}'.format(train_g_avg.shape))
    ts_train_avg,_ = average_based_on_group_df(ts_train,
                                                df=train_group_df)
    train_y_avg,_ = average_based_on_group_df(train_y,
                                               df=train_group_df)
    print(
        'average y shape:{}'.format(train_y_avg.shape)
    )

    return train_env_avg, train_g_avg, train_y_avg, ts_train_avg,average_group_df


def plot_multiple_genotype(predict_y,y,color_label,marker_label,name,r_value_pred=None,y_max_pred=None,num_epoch=0,env=None):
    # Create a DataFrame where rows are time steps and columns are series
    if isinstance(color_label, torch.Tensor):
        color_label = pd.DataFrame({'COLOR_LABLE': copy.deepcopy(color_label.detach().cpu()).numpy()})
        color_label = color_label['COLOR_LABLE']
    if isinstance(marker_label, torch.Tensor):
        marker_label = pd.DataFrame({'marker_label': copy.deepcopy(marker_label.detach().cpu()).numpy()})
        marker_label = marker_label['marker_label']

    mask = (~torch.isin(y, torch.tensor(0.0).to(DEVICE))).float()
    # print(torch.argmax(mask, dim=0))
    for seq_index in range(mask.shape[1]):
        max_index= torch.argmax((mask[:,seq_index,:].flip(0)), dim=0)
        # print(max_index)
        # print(y.shape[0])
        max_index = y.shape[0] - 1 - max_index
        # print('maximum value')
        # print(max_index)
        mask[:max_index+1, seq_index, :]=1.0
        mask[max_index + 1:, seq_index, :] = 0.0

    # Convert the mask to float if needed
    mask = mask.to(DEVICE)
    # print(mask)

    # detach and change the shape for plotting, only
    plot_predicted_y = torch.squeeze(copy.deepcopy(mask*predict_y.detach())).cpu()
    plot_predicted_y[plot_predicted_y==0.0]=np.nan

    marker_list = ['x', '.', '1', '*', "$\u266B$"]  # year is no more tha 5 currently
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

    y_test_copy = copy.deepcopy(torch.squeeze(y[:, :, 0]).detach()).cpu().numpy()
    y_test_copy[y_test_copy == 0.0] = np.nan  # set 0.0 to nan, so it will not plot in the figure
    # Need Python>3.7, which dictionaries are ordered
    from matplotlib.lines import Line2D
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))
    # dict_map_plot_index= {v: k for k, V in unique_colors.items() for v in V}
    ax=axs[0]
    ax.set_ylim(-0.1, 1.2)
    ax.set_xlim(0, 285)
    # ax.set_xlim(0, 285)
    for seq, type, label in zip(range(y_test_copy.shape[1]), marker_label, color_label):
        sns.scatterplot(y_test_copy[:, seq], label=label, ax=ax, marker=markers_dictionary[type],
                        color=color_dictionary[label])
        # print("seq number:{}".format(seq))
        # print(label)
    print(plot_predicted_y.shape)
    for seq, type, label in zip(range(plot_predicted_y.shape[1]), marker_label, color_label):
        # ax1.plot(plot_predicted_y,color=color_dictionary[label])
        # print("seq number:{}".format(seq))
        sns.lineplot(plot_predicted_y[:, seq], ax=ax, color=color_dictionary[label])
        # ax.set_title('genotype:{}'.format(label))
    # sns.scatterplot(y_test_copy, ax1=ax1)
    # set legend
    f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
    labels = list(color_dictionary.keys()) + list(markers_dictionary.keys())
    # print(labels)
    handles = [f('s', x) for x in color_dictionary.values()] + [f(x, 'k') for x in list(markers_dictionary.values())]
    # ax1.legend(handles, labels, title="color:genotype, markers:year")

    ax_1 = ax.twinx()

    if (r_value_pred!=None) & (y_max_pred!=None) :
        ts = torch.linspace(0.0, 284, 285).unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, plot_predicted_y.shape[1],
                                                                                             1)[:-115, :, :]
        from LoadData import logistic_ode_model
        for seq in range(plot_predicted_y.shape[1]):
            yt0 = plot_predicted_y[0, seq]
            if yt0 <= 0.0 or np.isnan(yt0):
                yt0 = 0.0001

            # plot predicted y
            if torch.is_tensor(r_value_pred):
                # if genetics input create multiple r
                r_value_pred_seq = torch.squeeze(r_value_pred)[seq].detach().cpu()
                y_max_pred_seq = torch.squeeze(y_max_pred)[seq].detach().cpu()
                parameters = [r_value_pred_seq, y_max_pred_seq, yt0]
            else:
                # otherwise it will be float
                parameters = [r_value_pred, y_max_pred, yt0]
            # print(parameters)
            # calcualted derivate from logistic ODE
            dy_dt = parameters[0] * plot_predicted_y[:, seq] * (1 - (plot_predicted_y[:, seq] / parameters[1]))
            # print(dy_dt)
            # x_y = odeint(func=logistic_ode_model, y0=yt0, t=ts_test[:, 0, 0].detach().cpu(), args=(parameters,))[:, 0]
            # print(x_y)
            # x_y[x_y == 0.0] = np.nan
            # sns.lineplot(dy_dt,ax=ax1, color="g")
            ax_1.plot(ts[:, seq, 0].detach().cpu(), dy_dt.squeeze(), color="g", )  # label="Height (Model)"

    ax.set_title('Logistic ODE informed PINN (epoch:{})'.format(num_epoch), fontsize=14)
    ax_1.set_title('Logistic ODE informed PINN (epoch:{})'.format(num_epoch), fontsize=14)

    ax_env = axs[1]

    env_plot = torch.squeeze(copy.deepcopy(mask*env.detach())).cpu()
    for seq,mark_env in zip(range(env_plot.shape[1]),marker_label):
        env_plot[env_plot==0.0]=np.nan
        sns.lineplot(env_plot[:,seq], ax=ax_env, markers=mark_env)
    plt.savefig("{}_predict_curve.svg".format(name))
    # plt.show()
    return fig

def read_genetics_kinship_matrix_with_training_genotype(genotype_tensor:torch.tensor,g_order,traing_genotype):
    """
    This function is to select genotype kinship column with thoes genotype present in training set
    """
    print(" original genotype tensor shape:{}".format(genotype_tensor.shape))
    indices = [g_order.index(genotype) for genotype in traing_genotype]

    # Convert indices to a PyTorch tensor
    indices = torch.tensor(indices, dtype=torch.long)

    # Select the corresponding columns from the tensor
    selected_tensor = genotype_tensor[:, indices]
    print('only use kinship matrix from training data')
    print(selected_tensor.shape)
    return selected_tensor

def read_best_hyperparameter_retrain_with_different_data_split_seed(best_hyperparameter_file:str,
                                                                    model_name,genotype_encoding='kinship_matrix_encoding',split_group='g_e',
                                                                    smooth_temp=True,rescale=False,if_pinn=True,more_train_data=True):

    #read best hyperparameter
    best_df = pd.read_csv(best_hyperparameter_file,header=0,index_col=0)
    print(best_df.columns)
    hidden = best_df['hidden_size'].astype(int).unique().item() #.item() will result in type int, while [0] will give numpy.int32

    num_layer = best_df['num_layer'].astype(int).unique().item()
    lr = best_df['learning_rate'].astype(float).unique().item()
    weight_physic = best_df['physics_weight'].astype(int).unique().item()
    L2 = best_df['l2'].astype(float).unique().item()
    last_fc_hidden_size = best_df['last_fc_hidden_size'].astype(int).unique().item()
    genetics_embedding_size = best_df['genetics_embedding_size'].astype(int).unique().item()
    y_max_bound=best_df["y_max_bound"].astype(int).unique().item()
    smooth_loss=best_df["smooth_loss"].astype(int).unique().item()

    print('hidden size: {}'.format(hidden))
    print('num_layer:{}'.format(num_layer))
    print('weight_physic:{}'.format(weight_physic))
    print('L2:{}'.format(L2))
    print('lr:{}'.format(lr))
    print('last_fc_hidden_size:{}'.format(last_fc_hidden_size))
    print('genetics_embedding_size:{}'.format(genetics_embedding_size))
    print("y_max_bound:{}".format(y_max_bound))
    print("smooth_loss:{}".format(smooth_loss))


    #load data
    genetics_input_tensor, genotype_list, group_df, model_name, num_genotypes, plant_height_tensor, \
        temperature_full_length_tensor, temperature_same_length_tensor = load_data_from_dill(
        genotype_encoding, model_name, smooth_temp, plant_height_file='../temporary/plant_height_tensor_all.dill',
        group_df_file='../temporary/group_list_df_all.dill',
        temperature_file='../temporary/temperature_tensor_same_length_all.dill',
        fill_in_na_at_start=True,rescale=rescale)
    if split_group == 'g_e':
        #split to predict new year, new genotype
        train_test_validation_dictionary = manually_split_on_two_groups(group_df=copy.deepcopy(group_df),training_percentage=0.7)
        model_name = model_name + split_group +'_split'
        # raise EOFError
    else:
        # split based on either year or genotype, if get two columns,
        # it can only make sure at least one of them is different for samples be splitted into different groups(train test etc.)
        train_test_validation_dictionary,train_years = manually_data_split_based_on_one_group(copy.deepcopy(group_df),split_group=split_group,n_splits=6)
        # raise EOFError
        if split_group == 'year_site.harvest_year':
            model_name = model_name + 'year_split'
        elif split_group == 'genotype.id':
            model_name = model_name + 'genotype_split'
        else:
            raise ValueError ('the input split_group is wrong!')
    # print(train_test_validation_dictionary)
    n_split =len(train_test_validation_dictionary.keys())

    try:
        result_df = pd.read_csv(
            'pinn_result/result_summary/best_model_cv/multiple_g_result_pinn_{}_{}_cv.csv'.format(if_pinn, model_name), header=0,
            index_col=0)
        print(result_df)
    except:
        result_df = pd.DataFrame()
        # raise EOFError

    for n in range(n_split):
        train_index, validation_index, test_index = train_test_validation_dictionary['splits_{}'.format(n)]
        train_index, validation_index, test_index = train_test_validation_dictionary['splits_{}'.format(n)]
        print('before adding genotype train sequences number')
        print(len(train_index))

        train_y = plant_height_tensor[115:, train_index, :].to(DEVICE)
        train_env = temperature_same_length_tensor[115:, train_index, :].to(DEVICE)
        # sns.lineplot(train_env.squeeze())
        # plt.show()
        # train_env_full = temperature_full_length_tensor[115:, train_index, :].to(DEVICE)
        train_group_df = copy.deepcopy(group_df).iloc[train_index,:]

        keep_genotype_kinship = [33, 106, 122, 133, 5, 30, 218, 2, 17, 254, 282, 294, 301, 302, 335, 339, 341, 6, 362]
        genetics_input_tensor = read_genetics_kinship_matrix_with_training_genotype(
            genotype_tensor=genetics_input_tensor, g_order=genotype_list, traing_genotype=keep_genotype_kinship)
        # if split_group == 'g_e':
        #     keep_genotype_kinship=[33, 106, 122, 133, 5, 30, 218, 2, 17, 254, 282, 294, 301, 302, 335, 339, 341, 6, 362]
        #     genetics_input_tensor = read_genetics_kinship_matrix_with_training_genotype(
        #         genotype_tensor=genetics_input_tensor, g_order=genotype_list, traing_genotype=keep_genotype_kinship) #train_group_df['genotype.id'].unique()

        validation_y = plant_height_tensor[115:, validation_index, :].to(DEVICE)
        validation_env = temperature_same_length_tensor[115:, validation_index, :].to(DEVICE)
        # validation_env_full = temperature_full_length_tensor[115:, validation_index, :].to(DEVICE)
        validation_group_df = copy.deepcopy(group_df).iloc[validation_index,:]

        test_y = plant_height_tensor[115:, test_index, :].to(DEVICE)
        test_env = temperature_same_length_tensor[115:, test_index, :].to(DEVICE)
        # test_env_full = temperature_full_length_tensor[115:, test_index, :].to(DEVICE)
        test_group_df = copy.deepcopy(group_df).iloc[test_index, :]

        original_genotype_ids = torch.tensor(genotype_list)

        train_g = genetics_input_tensor[ train_index, :].to(DEVICE)
        validation_g = genetics_input_tensor[validation_index, :].to(DEVICE)
        test_g = genetics_input_tensor[test_index, :].to(DEVICE)

        # get genotype id list after averge replicates
        train_genotype_id = original_genotype_ids[train_index].unsqueeze(-1).unsqueeze(0)
        val_genotype_id = original_genotype_ids[validation_index].unsqueeze(-1).unsqueeze(0)
        test_genotype_id = original_genotype_ids[test_index].unsqueeze(-1).unsqueeze(0)

        print(test_genotype_id,test_genotype_id.shape)

        if more_train_data:
            #add_more_training_data
            overlap_g_list = tuple(copy.deepcopy(original_genotype_ids)[train_index].tolist() + copy.deepcopy(original_genotype_ids)[validation_index].tolist())
            overlap_year_list = tuple(copy.deepcopy(train_group_df)['year_site.harvest_year'].unique().tolist()) + tuple(copy.deepcopy(validation_group_df)['year_site.harvest_year'].unique().tolist())
            add_plant_height_tensor, add_temperature_same_length_tensor, add_group_df, add_genetics_input_tensor,add_genotype_id = load_more_training_data(
                data_plit=split_group,
                overlap_g=overlap_g_list, overlap_year=overlap_year_list,fill_in_na_at_start=True,smooth_temp=smooth_temp,
                genotype_encoding='kinship_matrix_encoding')

            train_y = torch.cat([train_y,add_plant_height_tensor],dim=1)
            train_env = torch.cat([train_env,add_temperature_same_length_tensor],dim=1)
            train_group_df = pd.concat([train_group_df,add_group_df],axis=0)
            print(train_group_df)
            train_g = torch.cat([train_g,add_genetics_input_tensor],dim=0)
            train_genotype_id=torch.cat([train_genotype_id,add_genotype_id],dim=1)
            train_genotype_id_before_average= copy.deepcopy(train_genotype_id).squeeze().to(DEVICE)
            # print(train_genotype_id_before_average.shape)
            # raise EOFError
            print('train shape after add more train_g')
            print(train_genotype_id.shape)
            print(train_y.shape)
            print(train_env.shape)
            print(train_group_df.shape)
            print(train_g.shape)
        else:
            train_genotype_id_before_average = copy.deepcopy(train_genotype_id).squeeze().to(DEVICE)

        train_genotype_id, _ = average_based_on_group_df(copy.deepcopy(train_genotype_id),
                                                         train_group_df)
        train_genotype_id = torch.squeeze(train_genotype_id)
        val_genotype_id, _ = average_based_on_group_df(copy.deepcopy(val_genotype_id),
                                                       validation_group_df)
        val_genotype_id = torch.squeeze(val_genotype_id)
        test_genotype_id, _ = average_based_on_group_df(copy.deepcopy(test_genotype_id),
                                                        test_group_df)
        test_genotype_id = torch.squeeze(test_genotype_id)  #

        # creat time sequence, same as input shape
        ts_train = torch.linspace(0.0, 284, 285).unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, train_y.shape[1],
                                                                                          1)[:-115, :,
                   :].requires_grad_(True).to(DEVICE)  # time sequences steps
        ts_validation = torch.linspace(0.0, 284, 285).unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, validation_y.shape[1],
                                                                                               1)[:-115, :,
                        :].requires_grad_(True).to(DEVICE)  # time sequences steps
        ts_test = torch.linspace(0.0, 284, 285).unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, test_y.shape[1],
                                                                                         1)[:-115, :, :].to(
            DEVICE)  # time sequences steps

        # average across replicates predict result
        print('train')
        train_env_avg, train_g_avg, train_y_avg, ts_train_avg, train_group_df_avg = calculate_average_across_replicates(
            train_env, train_g, train_group_df, train_y, ts_train)

        print('val')
        validation_env_avg, validation_g_avg, validation_y_avg, ts_validation_avg, validation_group_df_avg = calculate_average_across_replicates(
            validation_env, validation_g, validation_group_df, validation_y,
            ts_validation)

        print('test')
        test_env_avg, test_g_avg, test_y_avg, ts_test_avg, test_group_df_avg = \
            calculate_average_across_replicates(
                test_env, test_g, test_group_df, test_y,
                ts_test)

        for j in [1, 2, 3, 4, 5]:
            random.seed(j)
            np.random.seed(j)
            torch.manual_seed(j)
            epoches=3000
            run = wandb.init(
                # Set the project where this run will be logged
                project='{}_multiple_g_pinn'.format(model_name),
                # Track hyperparameters and run metadata
                config={
                    "learning_rate": lr,
                    "epochs": epoches,
                    "n_split": n,
                    "random_sees": j,
                    "hidden_size": hidden,
                    "num_layer": num_layer,
                    "physics_weight": weight_physic,
                    "y_max_bound": y_max_bound,
                    "l2": L2,
                    "last_fc_hidden_size": last_fc_hidden_size,
                    "genetics_embedding_size": genetics_embedding_size

                },
            )
            if weight_physic!=0:
                model = pinn_genotype_embedding_fc(genetic_feature_size=train_g.shape[-1],
                                                 input_size=temperature_same_length_tensor.shape[-1],
                                                 hidden_size=hidden,num_layer=num_layer,
                                                 last_fc_hidden_size=last_fc_hidden_size,genetics_embedding_size=genetics_embedding_size).to(DEVICE)

            else:
                model = genotype_code_temperature_input_height_prediction_fc(genetic_feature_size=train_g.shape[-1],
                                                                             input_size=temperature_same_length_tensor.shape[-1],
                                                                             num_layer=num_layer,hidden_size=hidden,
                                                                             last_fc_hidden_size=last_fc_hidden_size,genetics_embedding_size=genetics_embedding_size).to(DEVICE)

            model.init_network()
            total_params = count_parameters(model)

            # Initialize model, criterion, and optimizer
            optimizer = optim.Adam(model.parameters(), lr=lr)
            # plt.plot(train_y.squeeze())
            # plt.show()
            # Train and validate the model
            model, epoch_num = train_and_validate(
                [train_env, train_g, train_y, ts_train, train_genotype_id_before_average],
                [validation_env, validation_g, validation_y, ts_validation, original_genotype_ids[validation_index]],
                [test_env, test_g, test_y, ts_test, original_genotype_ids[test_index]], model, optimizer, epoches,
                pinn_weight=weight_physic, l2=L2, y_max_bound=y_max_bound, smooth_loss=False)
            print('find minimize validation loss at epoch:{}'.format(epoch_num))
            wandb.log({"result_epochs": epoch_num})

            print(train_env_avg.shape)
            print(train_g_avg.shape)
            print(ts_train_avg.shape)
            if hasattr(model, 'trainable_param'):
                train_pred_y, _, _ = model(train_env_avg, train_g_avg, ts_train_avg, train_genotype_id)
            else:
                train_pred_y, _, _ = model(train_env_avg, train_g_avg, ts_train_avg)
            if if_pinn:
                r_train = model.r.detach().cpu()
                y_max_train = model.y_max.detach().cpu()
            else:
                r_train = None
                y_max_train = None
            train_rmse = mask_rmse_loss(true_y=train_y_avg, predict_y=train_pred_y)
            # shapedtw_loss_train = mask_dtw_loss(true_y=train_y_avg,predict_y=train_pred_y)
            # print('train shapedtw distance:{}'.format(shapedtw_loss_train))
            if hasattr(model, 'trainable_param'):
                val_pred_y, _, _ = model(validation_env_avg, validation_g_avg, ts_validation_avg, val_genotype_id)
            else:
                val_pred_y, _, _ = model(validation_env_avg, validation_g_avg,
                                         ts_validation_avg)
            if if_pinn:
                r_val = model.r.detach().cpu()
                y_max_val = model.y_max.detach().cpu()
            else:
                r_val = None
                y_max_val = None
            val_rmse = mask_rmse_loss(true_y=validation_y_avg, predict_y=val_pred_y)
            if hasattr(model, 'trainable_param'):
                test_pred_y, _, _ = model(test_env_avg, test_g_avg, ts_test_avg, test_genotype_id)
            else:
                test_pred_y, _, _ = model(test_env_avg, test_g_avg, ts_test_avg)
            test_rmse = mask_rmse_loss(true_y=test_y_avg, predict_y=test_pred_y)

            corre_train = mask_dtw_loss(true_y=train_y_avg, predict_y=train_pred_y)
            print('train shapeDTW')
            print(corre_train)
            corre_validation = mask_dtw_loss(true_y=validation_y_avg, predict_y=val_pred_y)
            print('validation shapeDTW')
            print(corre_validation)
            corre_test = mask_dtw_loss(true_y=test_y_avg, predict_y=test_pred_y)
            print('test shapeDTW')
            print(corre_test)

            if if_pinn:
                r_test = model.r.detach().cpu()
                y_max_test = model.y_max.detach().cpu()
            else:
                r_test = None
                y_max_test = None

            new_row = pd.DataFrame({
                "learning_rate": lr,
                "epochs": epoch_num, "n_split": n,
                "random_sees": j,
                "hidden_size": hidden,
                'l2': L2,
                "num_layer": num_layer,
                "last_fc_hidden_size": last_fc_hidden_size,
                "genetics_embedding_size": genetics_embedding_size,
                'trainable_parameters': total_params,
                "physics_weight": weight_physic,
                "y_max_bound": y_max_bound,
                "smooth_loss": smooth_loss,
                "train_rMSE": round(train_rmse.item(), 3), "validation_rMSE": round(val_rmse.item(), 3),
                "test_rMSE": round(test_rmse.item(), 3),
                'train_shapeDTW': round(corre_train, 3),
                'validation_shapeDTW': round(corre_validation, 3),
                "test_shapeDTW": round(corre_test, 3),
            }, index=[0])
            result_df = pd.concat([result_df, new_row])
            result_df.to_csv('pinn_result/result_summary/best_model_cv/multiple_g_result_pinn_{}_{}.csv'.format(if_pinn, model_name))
            name_str = "pinn_result/multipl_g/pinn_weight{}_{}_train_lr_{}_hidden_{}_fc_hidden_{}_g_embed{}_num_layers_{}_l2_{}_ymax_bound_{}_smooth_loss{}_seed_{}".format(
                if_pinn, model_name, lr, hidden, last_fc_hidden_size, genetics_embedding_size, num_layer, L2, y_max_bound,
                smooth_loss, j)

            # save model
            with open(
                    'pinn_result/multipl_g/model/pinn_weight_{}_model_{}_train_lr_{}_hidden_{}_fc_hidden_{}_g_embed{}_num_layers_{}_l2_{}_ymax_bound_{}_smooth_loss{}_seed_{}.dill'.format(
                        if_pinn, model_name, lr, hidden, last_fc_hidden_size, genetics_embedding_size, num_layer, L2,
                        y_max_bound, smooth_loss, j), 'wb') as file:
                dill.dump(model, file)
                # torch.save(model, file)
                # Load the entire model and use it directly
            file.close()

            fig_train = plot_multiple_genotype(train_pred_y, train_y_avg, color_label=train_genotype_id,
                                               marker_label=train_group_df_avg['year_site.harvest_year'],
                                               name=name_str + 'train', y_max_pred=y_max_train, r_value_pred=r_train,
                                               num_epoch=str(epoch_num),env=train_env_avg)
            wandb.log({'train_prediction_plot': wandb.Image(fig_train)})
            plt.close()
            fig_val = plot_multiple_genotype(val_pred_y, validation_y_avg, color_label=val_genotype_id,
                                             marker_label=validation_group_df_avg['year_site.harvest_year'],
                                             name=name_str + 'val', y_max_pred=y_max_val, r_value_pred=r_val,
                                             num_epoch=str(epoch_num),env=validation_env_avg)
            wandb.log({'val_prediction_plot': wandb.Image(fig_val)})
            plt.close()
            fig_test = plot_multiple_genotype(test_pred_y, test_y_avg, color_label=test_genotype_id,
                                              marker_label=test_group_df_avg['year_site.harvest_year'],
                                              name=name_str + 'test', y_max_pred=y_max_test, r_value_pred=r_test,
                                              num_epoch=str(epoch_num),env=test_env_avg)
            wandb.log({'test_prediction_plot': wandb.Image(fig_test)})
            plt.close()
            run.finish()


def read_cmd():
    from sys import argv
    cmd_line = argv[1:]
    print('input command line: \n {}'.format(cmd_line))
    try:
        index_mode = cmd_line.index("-mode") + 1
        mode = cmd_line[index_mode]
    except:
        print('did not receive mode input, use default setting: \'\'')
        mode = ''

    try:
        index_genotype_encoding = cmd_line.index("-genotype_encoding") + 1
        genotype_encoding = cmd_line[index_genotype_encoding]
    except:
        print('did not receive mode input, use default setting: \'\'')
        genotype_encoding = 'one_hot_similarity_encoding'

    try:
        index_split_group = cmd_line.index("-split_group") + 1
        split_group = cmd_line[index_split_group]
    except:
        print('did not receive mode input, use default setting: \'\'')
        split_group = 'year_site.harvest_year'

    try:
        index_mode = cmd_line.index("-if_pinn") + 1
        if_pinn = cmd_line[index_mode]
        if str(if_pinn) == 'True':
            if_pinn = True
        elif str(if_pinn) == 'False':
            if_pinn = False
    except:
        print('did not receive if_pinn input, use default setting: True')
        if_pinn = True
    try:
        index_mode = cmd_line.index("-smooth_loss") + 1
        smooth_loss = cmd_line[index_mode]
        if smooth_loss == 'True':
            smooth_loss = True
        elif smooth_loss == 'False':
            smooth_loss = False
    except:
        print('did not receive -smooth_loss input,False')
        smooth_loss = False

    try:
        index_mode = cmd_line.index("-reduce_time_resolution") + 1
        reduce_time_resolution = cmd_line[index_mode]
        if reduce_time_resolution == 'True':
            reduce_time_resolution = True
        elif reduce_time_resolution == 'False':
            reduce_time_resolution = False
    except:
        print('did not receive -smooth_loss input,False')
        reduce_time_resolution = False

    try:
        index_mode = cmd_line.index("-smooth_input") + 1
        smooth_input = cmd_line[index_mode]
        if smooth_input == 'True':
            smooth_input = True
        elif smooth_input == 'False':
            smooth_input = False
    except:
        print('did not receive -smooth_input input,False')
        smooth_input = False

    try:
        index_mode = cmd_line.index("-smooth_temp") + 1
        smooth_temp = cmd_line[index_mode]
        if smooth_temp == 'True':
            smooth_temp = True
        elif smooth_temp == 'False':
            smooth_temp = False
    except:
        print('did not receive -smooth_temp input,False')
        smooth_temp = False

    return if_pinn,mode,genotype_encoding,smooth_loss,split_group,reduce_time_resolution,smooth_input,smooth_temp
def main():
    # from sktime.classification.distance_based import ShapeDTW
    if_pinn,mode,genotype_encoding,smooth_loss,split_group,reduce_time_resolution,smooth_input,smooth_temp = read_cmd()
    train_simple_g_e_interaction_model(if_pinn=if_pinn, model_name=mode,smooth_loss=smooth_loss,
                                       genotype_encoding=genotype_encoding,split_group=split_group,
                                       reduce_time_resolution=reduce_time_resolution,smooth_input=smooth_input,
                                       smooth_temp=smooth_temp,add_more_train_g=False)
    # if_pinn,mode,genotype_encoding,smooth_loss,split_group,reduce_time_resolution,smooth_input,smooth_temp = read_cmd()
    # train_simple_g_e_interaction_model(if_pinn=True, model_name='test_rank',smooth_loss=smooth_loss,
    #                                    genotype_encoding='kinship_matrix_encoding',split_group='genotype.id',
    #                                    reduce_time_resolution=reduce_time_resolution,smooth_input=smooth_input,
    #                                    smooth_temp=False,add_more_train_g=False)
    # hyperparameter_csv = 'multiple_g_result_pinn_True_pinn_result_lstm_sameval_g_more_train_gkinship_matrix_encoding_all_present_genotypeg_e_split_best_hyperparameters_result.csv'
    # if_pinn = bool(hyperparameter_csv.find('pinn_True'))
    #
    # read_best_hyperparameter_retrain_with_different_data_split_seed(best_hyperparameter_file='best_model_result_summary/{}'.format(hyperparameter_csv),
    #                                                                 model_name='pinn_False_NN_result_lstm_ameval_g_more_train_g',
    #                                                                 genotype_encoding='kinship_matrix_encoding_all_present_genotype',
    #                                                                 smooth_temp=False,split_group='g_e',rescale=True,if_pinn=if_pinn)
    #
    # hyperparameter_csv = 'multiple_g_result_pinn_True_pinn_result_lstm_sameval_g_more_train_gkinship_matrix_encoding_all_present_genotypegenotype_split_best_hyperparameters_result.csv'
    # if_pinn = bool(hyperparameter_csv.find('pinn_True'))
    # #
    # read_best_hyperparameter_retrain_with_different_data_split_seed(best_hyperparameter_file='best_model_result_summary/{}'.format(hyperparameter_csv),
    #                                                                 model_name='pinn_False_NN_result_lstm_ameval_g_more_train_g',
    #                                                                 genotype_encoding='kinship_matrix_encoding_all_present_genotype',
    #                                                                 smooth_temp=False,split_group='genotype.id',rescale=True,if_pinn=if_pinn)

if __name__ == '__main__':
    main()
