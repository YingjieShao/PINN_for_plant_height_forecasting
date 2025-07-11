# correct spatial trend with NN model compared base on MSE by using time series for yield prediction, genotype specific growth curve(NA is removd when calculated MAPE)
import copy
import time
from operator import itemgetter
import shap
import dill
import numpy as np  # for maths
import pandas as pd  # for data manipulation
import matplotlib.pyplot as plt  # for visualization
import torch
from torch import nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader
from prettytable import PrettyTable
import torch.optim as optim
import random
from model_selection import data_load
from sklearn.metrics import accuracy_score
import model_selection
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import GroupShuffleSplit
from model_selection import load_multiple_year_data, average_based_on_days

from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
import matplotlib.patches as mpatches
import colorsys
from scipy.stats import pearsonr, spearmanr
# from convlstm import ConvLSTM
import unittest
# from torch.masked import masked_tensor, as_masked_tensor
import warnings


class LSTM_fit_curve_genotype_predict(nn.Module):
    # without position input, only use sequence input for genotype predict and curve fit
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1, batch_first=True, cnn_kernel=5,
                 maxpooling_kernel=3):
        super().__init__()
        self.input_size = input_size  # number of features
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = 1
        self.cnn_kernel = cnn_kernel
        self.maxpooling_kernel = maxpooling_kernel

        self.lstm1 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=2,
                             dropout=dropout, batch_first=batch_first)
        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=input_size, num_layers=1
                             , batch_first=batch_first)
        self.relu = nn.ReLU()  # convert output value to >= 0

        self.cnn1 = nn.Conv1d(in_channels=input_size, out_channels=3, kernel_size=cnn_kernel)
        self.leakyrelu = nn.LeakyReLU()
        self.out_size1 = int(
            (46 - 1 * cnn_kernel // 1) + 1)  # output size for cnn layer, 46 is the length of time
        self.maxpooling = nn.MaxPool1d(kernel_size=maxpooling_kernel, stride=5)  # reduce parameter number
        pooling_out_size = int(((self.out_size1 - 1 * maxpooling_kernel) // 5) + 1)  # outputsize after maximum pooling
        self.cnn2 = nn.Conv1d(in_channels=5, out_channels=5, kernel_size=cnn_kernel)
        self.out_size2 = int(
            (pooling_out_size - 1 * cnn_kernel // 1) + 1)  # output size for cnn layer
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.out_size2 * 5, output_size)  # pooling out * channel

        self.softmax = nn.Softmax(
            dim=-1)  # https://stackoverflow.com/questions/42081257/why-binary-crossentropy-and-categorical-crossentropy-give-different-performances/46038271#46038271

    def forward(self, x, location):
        # Initialize

        # h0 shape：(num_layers * num_directions, batch, hidden_size)
        # c0 shape：(num_layers * num_directions, batch, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(
            x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(
            x.device)

        # Forward propagate LSTM
        output_1, hidden = self.lstm1(x, (h0, c0))
        # take the output of the last time step, feed it in to a fully connected layer
        output_2, hidden2 = self.lstm2(output_1)
        output_curve = self.relu(output_2)  # covert to positive
        # print(output_1.shape)
        output_permute = torch.permute(output_curve, (0, 2, 1))
        out_cnn1 = self.cnn1(output_permute)
        out_cnn1 = self.leakyrelu(out_cnn1)
        out_cnn1 = self.maxpooling(out_cnn1)
        out_cnn2 = self.cnn2(out_cnn1)
        out_cnn2 = self.leakyrelu(out_cnn2)

        out_faltten = self.flatten(out_cnn2)
        output_fc = self.fc(out_faltten)

        output_label = self.softmax(output_fc)
        return output_curve, output_label

    def init_network(self):
        # initialize weight and bias(use xavier and 0 for weight and bias separately)
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def custom_loss(self, fit_curve=None, predict_Y=None, true_label=None, input_curve=None):

        criterion_genotype = nn.CrossEntropyLoss()
        genotype_loss = criterion_genotype(true_label, predict_Y)
        # print(genotype_loss)

        # tried MAPE loss for curve fit, while when data include zero it doesn't make sence
        # MAPE_loss = torch.mean(torch.abs((input_curve - fit_curve) / input_curve))
        criterion_curve_fit = nn.MSELoss()
        fit_curve_loss = criterion_curve_fit(input_curve, fit_curve)

        total_loss = genotype_loss + fit_curve_loss

        return total_loss, fit_curve_loss, genotype_loss


class LSTM_yield_predict(nn.Module):
    # without position input, only use sequence input for genotype predict and curve fit
    def __init__(self, input_size, hidden_size, output_size, seq_len, dropout=0.1, batch_first=True, cnn_kernel=5,
                 maxpooling_kernel=3, position=False):
        super().__init__()
        self.input_size = input_size  # number of features
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.seq_len = seq_len
        self.num_layers = 1
        self.cnn_kernel = cnn_kernel
        self.maxpooling_kernel = maxpooling_kernel
        # https: // pytorch.org / docs / stable / generated / torch.nn.LSTM.html  # torch.nn.LSTM
        self.lstm1 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=1,
                             dropout=dropout, batch_first=batch_first)
        # self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=input_size, num_layers=1,
        #                      dropout=dropout,batch_first=batch_first)
        self.relu = nn.ReLU()  # convert to >= 0

        self.cnn1 = nn.Conv1d(in_channels=hidden_size, out_channels=6, kernel_size=cnn_kernel)
        self.leakyrelu = nn.LeakyReLU()
        self.out_size1 = int(
            (self.seq_len - 1 * (cnn_kernel - 1) // 1) + 1)  # output size for cnn layer
        self.maxpooling = nn.MaxPool1d(kernel_size=maxpooling_kernel, stride=5)
        pooling_out_size = int(
            ((self.out_size1 - 1 * (maxpooling_kernel - 1)) // 5) + 1)  # outputsize after maximum pooling
        # (inputsize+2*padding-dilation*(kernerl_size-1)-1)/stride +1 https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html

        if position:
            print('with pos')
            self.cnn1 = nn.Conv1d(in_channels=hidden_size + 2, out_channels=6,
                                  kernel_size=cnn_kernel)  # inputsize plus position 3+2 overwrite self.cnn1
            out_size2 = int(
                (self.seq_len - 1 * (cnn_kernel - 1) // 1) + 1)
            print(out_size2)  # output size for cnn layer
            # out_size2 = int(
            #     (out_size2 - 1*cnn_kernel // 1) + 1)
            # print(out_size2)
            pooling_out_size = int(
                ((out_size2 - 1 * (maxpooling_kernel - 1)) // 5) + 1)  # outputsize after maximum pooling
            print(pooling_out_size)
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(pooling_out_size * 6, output_size)  # pooling out * cnn2 out channel
        else:
            # #without location information
            # self.cnn2 = nn.Conv1d(in_channels=3,out_channels=5,kernel_size=cnn_kernel)
            # self.out_size2 = int(
            #     (pooling_out_size - 1*cnn_kernel // 1) + 1)  # output size for cnn layer
            self.flatten = nn.Flatten()
            print(pooling_out_size * 6)
            self.fc = nn.Linear(pooling_out_size * 6, output_size)  # pooling out size* out channel

    def forward(self, x, position=None, use_position=False):
        # Initialize

        # h0 shape：(num_layers * num_directions, batch, hidden_size)
        # c0 shape：(num_layers * num_directions, batch, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(
            x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(
            x.device)

        # Forward propagate LSTM
        output_2, hidden = self.lstm1(x, (h0, c0))

        output_2 = self.relu(output_2)  # covert all value to positive
        # print('out 2 shape')
        # print(output_2.shape)
        output_permute = torch.permute(output_2,
                                       (0, 2, 1))  # convert to (batch,feature size, time steps) as input for CNN
        # out_cnn1 = self.cnn1(output_permute) #https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html#torch.nn.Conv1d

        if use_position:
            print('with position')
            position = torch.unsqueeze(position, -1)
            # print(position.shape)
            position = position.repeat(
                [1, 1, output_permute.shape[-1]])  # duplicate location tensor to mathc the dimension after first cnn
            # print(position.shape)
            # print(position.dtype)
            # print(out_cnn1.dtype)
            output_permute = torch.cat([output_permute, position], dim=1)  # concat cnn1 output with position
            # covert to float as the calculation will transfer it to double
            out_cnn1 = self.cnn1(output_permute.float())  # concat cnn1 output with position 3+2=5
            out_cnn1 = self.leakyrelu(out_cnn1)
            out_cnn1 = self.maxpooling(out_cnn1)
            out_faltten = self.flatten(out_cnn1)
            # print(out_faltten.shape)
            output_fc = self.fc(out_faltten)
            return output_fc
        else:
            # print('out_p')
            # print(output_permute.shape)
            out_cnn1 = self.cnn1(output_permute)
            out_cnn1 = self.leakyrelu(out_cnn1)
            out_cnn1 = self.maxpooling(out_cnn1)
            # print(out_cnn1.shape)
            out_faltten = self.flatten(out_cnn1)
            # print(out_faltten.shape)
            output_fc = self.fc(out_faltten)
            # print('shape before output layer')
            # print(output_fc.shape)
            # add spatial information
            return output_fc

    def init_network(self):
        # initialize weight and bias(use xavier initialization for weight and bias )
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.00)
            # print(name, param)

    def custom_loss(self, fit_curve=None, predict_Y=None, true_label=None, input_curve=None):
        # check shape

        criterion_yield = nn.MSELoss()
        yield_loss = criterion_yield(true_label, predict_Y)
        # loss = spearman(true_label,predict_Y, regularization_strength=1e-3)
        # print(yield_loss)
        return yield_loss

    def spearman_rank_loss(self, std=None, mean=None, predict_Y=None, true_label=None):

        # reshape and use for rank #! For reshape,do not pack it in another torch tensor, otherwise will lose gradient
        predict_Y = predict_Y.reshape(1, predict_Y.shape[0])
        true_label = true_label.reshape(1, true_label.shape[0])

        import torchsort  # use softrank from torchsort to use correlationas loss function

        def corrcoef(target, pred):
            pred_n = pred - pred.mean()
            target_n = target - target.mean()

            pred_n = pred_n / pred_n.norm()

            target_n = target_n / target_n.norm()
            return (pred_n * target_n).sum()

        def spearman(target, pred, regularization="l2", regularization_strength=1.0, ):
            pred = torchsort.soft_rank(
                pred,
                regularization=regularization,
                regularization_strength=regularization_strength,
            )
            target = torchsort.soft_rank(
                target,
                regularization=regularization,
                regularization_strength=regularization_strength,
            )
            # print('target rank')
            # print(target)
            # print("pre rank")
            # print(pred)
            coef = corrcoef(target, pred / pred.shape[-1])
            # print(coef)
            return coef

        loss = (1 - spearman(true_label, predict_Y))  # the orginal spearman coef is between -1 and 1
        criterion_yield = nn.MSELoss()
        yield_loss = criterion_yield(true_label, predict_Y)
        loss = (yield_loss + loss) / 2
        return loss

    def spearman_MSE_loss(self, std=None, mean=None, predict_Y=None, true_label=None):

        # reshape and use for rank #! For reshape,do not pack it in another torch tensor, otherwise will lose gradient
        predict_Y = predict_Y.reshape(1, predict_Y.shape[0])
        true_label = true_label.reshape(1, true_label.shape[0])

        import torchsort  # use softrank from torchsort to use correlationas loss function

        def corrcoef(target, pred):
            """
            calculate coefficient
            """
            pred_n = pred - pred.mean()
            target_n = target - target.mean()
            pred_n = pred_n / pred_n.norm()
            target_n = target_n / target_n.norm()
            return (pred_n * target_n).sum()

        def spearman(target, pred, regularization="l2", regularization_strength=1.0, ):
            pred = torchsort.soft_rank(
                pred,
                regularization=regularization,
                regularization_strength=regularization_strength,
            )
            target = torchsort.soft_rank(
                target,
                regularization=regularization,
                regularization_strength=regularization_strength,
            )
            coef = corrcoef(target, pred / pred.shape[-1])
            # print(coef)
            return coef

        loss = (1 - spearman(true_label, predict_Y))  # the orginal spearman coef is between -1 and 1
        criterion_yield = nn.MSELoss()
        yield_loss = criterion_yield(true_label, predict_Y)
        combine_loss = yield_loss + loss
        return combine_loss


class ConvLSTM_yield_prediction(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        from convlstm import ConvLSTM
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.convlstm = ConvLSTM(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size,
                                 num_layers=num_layers, batch_first=False)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, inpux):
        # inpux shape = (t, b, c, h, w)
        layer_output_list, output1_list = self.convlstm(inpux)
        hidden, cell = output1_list.pop()
        print(hidden.shape, cell.shape)

        output = self.fc(hidden)
        return output

    def init_network(self):
        # initialize weight and bias(use xavier and 0 for weight and bias separately)
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
            # print(name, param)


class onelayer_LSTM_yield_prediction(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1, batch_first=True):
        super().__init__()
        self.input_size = input_size  # number of features
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = 1

        self.lstm1 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=1,
                             dropout=dropout, batch_first=batch_first)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, position_batch, with_position):
        # Initialize

        # h0 shape：(num_layers * num_directions, batch, hidden_size)
        # c0 shape：(num_layers * num_directions, batch, hidden_size)
        # batch first, the x.size(0) is batch
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(
            x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(
            x.device)

        # Forward propagate LSTM
        output_1, hidden = self.lstm1(x, (h0, c0))
        output = self.fc(output_1[:, -1, :])  # last time step output to fully connected layer
        return output

    def init_network(self):
        # initialize weight and bias(use xavier initialization for weight and bias )
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
            # print(name, param)

    def spearman_rank_loss(self, std=None, mean=None, predict_Y=None, true_label=None):

        # reshape and use for rank #! For reshape,do not pack it in another torch tensor, otherwise will lose gradient
        predict_Y = predict_Y.reshape(1, predict_Y.shape[0])
        true_label = true_label.reshape(1, true_label.shape[0])

        import torchsort  # use softrank from torchsort to use correlationas loss function

        def corrcoef(target, pred):
            pred_n = pred - pred.mean()
            target_n = target - target.mean()

            pred_n = pred_n / pred_n.norm()

            target_n = target_n / target_n.norm()
            return (pred_n * target_n).sum()

        def spearman(target, pred, regularization="l2", regularization_strength=1.0, ):
            pred = torchsort.soft_rank(
                pred,
                regularization=regularization,
                regularization_strength=regularization_strength,
            )
            target = torchsort.soft_rank(
                target,
                regularization=regularization,
                regularization_strength=regularization_strength,
            )

            coef = corrcoef(target, pred / pred.shape[-1])
            # print(coef)
            return coef

        loss = (1 - spearman(true_label, predict_Y))  # the orginal spearman coef is between -1 and 1
        criterion_yield = nn.MSELoss()
        yield_loss = criterion_yield(true_label, predict_Y)
        loss = yield_loss + loss
        return loss

    def spearman_MSE_loss(self, std=None, mean=None, predict_Y=None, true_label=None):

        # reshape and use for rank #! For reshape,do not pack it in another torch tensor, otherwise will lose gradient
        predict_Y = predict_Y.reshape(1, predict_Y.shape[0])
        true_label = true_label.reshape(1, true_label.shape[0])

        import torchsort  # use softrank from torchsort to use correlationas loss function

        def corrcoef(target, pred):
            """
            calculate coefficient
            """
            pred_n = pred - pred.mean()
            target_n = target - target.mean()
            pred_n = pred_n / pred_n.norm()
            target_n = target_n / target_n.norm()
            return (pred_n * target_n).sum()

        def spearman(target, pred, regularization="l2", regularization_strength=1.0, ):
            pred = torchsort.soft_rank(
                pred,
                regularization=regularization,
                regularization_strength=regularization_strength,
            )
            target = torchsort.soft_rank(
                target,
                regularization=regularization,
                regularization_strength=regularization_strength,
            )
            coef = corrcoef(target, pred / pred.shape[-1])
            # print(coef)
            return coef

        loss = (1 - spearman(true_label, predict_Y))  # the orginal spearman coef is between -1 and 1
        criterion_yield = nn.MSELoss()
        yield_loss = criterion_yield(true_label, predict_Y)
        combine_loss = yield_loss + loss
        return combine_loss

    def custom_loss(self, fit_curve=None, predict_Y=None, true_label=None, input_curve=None):
        # check shape

        criterion_yield = nn.MSELoss()
        yield_loss = criterion_yield(true_label, predict_Y)
        # loss = spearman(true_label,predict_Y, regularization_strength=1e-3)
        # print(yield_loss)
        return yield_loss


# def masked_tensor(input_tensor)->masked_tensor:
#     'mask na'
#     raise ValueError ('masked tensor can not be operate in most of the layers such as nn.Linear')
#     mask_input = masked_tensor(input_tensor, ~torch.isnan(input_tensor))
#
#     return mask_input
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

    # activations_used = []
    # # Define a hook function to track activation functions
    # def activation_hook(module, input, output):
    #     if isinstance(module, (
    #     torch.nn.ReLU, torch.nn.Sigmoid, torch.nn.Tanh, torch.nn.LeakyReLU, torch.nn.ELU, torch.nn.Softmax)):
    #         activations_used.append(type(module).__name__)
    # # Register the hook to each module
    # for name, module in model.named_modules():
    #     module.register_forward_hook(activation_hook)
    # # Dummy input to call forward pass
    # input_data = torch.randn(10,3,15902)
    # # Forward pass
    # output = model(input_data)
    # # Print the tracked activation functions
    # print("Activation functions used in order:")
    # for i, activation in enumerate(activations_used):
    #     print(f"{i + 1}: {activation}")
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
        yield_df = model_selection.data_load(yield_file, year=year)
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


class Prepare_Rf_model:
    def __init__(self, window_size, year, n_split, seed, load_data_boolean=True, data=None):

        self.window_size = window_size
        if load_data_boolean:
            self.load_data(year, n_split, seed)
        else:
            print('prepare data...')
            self.prepare_data(year, n_split, seed, data)

    def load_data(self, year, n_split, seed):
        '''
                create input tensor and yield tensor with corresponding train test validation index
                '''
        print('data for year {}'.format(year))
        canopy_df = data_load("C:/data_from_paper/ETH/olivia_WUR/process_data/Canopy coverage.csv", year)

        data_canopy = canopy_df[
            ['timestamp', 'plot.UID', 'value', 'genotype.id', 'plot.row_global', 'plot.range_global', 'lot',
             'year_site.harvest_year']]
        # print(data_canopy)
        height_df = data_load("C:/data_from_paper/ETH/olivia_WUR/process_data/Plant height.csv", year)
        data_height = height_df[
            ['timestamp', 'plot.UID', 'value', 'genotype.id', 'plot.row_global', 'plot.range_global', 'lot',
             'year_site.harvest_year']]

        creat_tensor_class = create_tensor_dataset([data_canopy, data_height], year=year, n_split=n_split,
                                                   average_endpoint_value=True, random_seed=seed)

        remove_na_dfs, genotype_df, dictionary_of_n_splits_index, yield_tensor, env_dfs = creat_tensor_class.create_train_test_validation_index(
            year=year)
        # creat_tensor_class.train_test_validation_tensor()

        tensor_dataset = convert_inputx_to_tesor_list(remove_na_dfs)
        # print(tensor_dataset.shape)

        self.x = tensor_dataset
        self.label = yield_tensor
        assert (dictionary_of_n_splits_index['splits_0'][-1] == dictionary_of_n_splits_index['splits_1'][
            -1]).all()  # validation set in n split should be the same
        self.index_dictionary = dictionary_of_n_splits_index
        # print(genotype_df)
        self.genotype_df = genotype_df[['genotype.id']]

    def prepare_data(self, year, n_split, seed, data):
        data_canopy = data[0]
        data_height = data[1]
        print('year {}'.format(year))
        print('time stmps:{}'.format(
            len(data_height[data_height['year_site.harvest_year'] == year]['timestamp'].unique())))
        creat_tensor_class = create_tensor_dataset([data_canopy, data_height], year=year, n_split=n_split,
                                                   average_endpoint_value=True, random_seed=seed)

        remove_na_dfs, genotype_df, dictionary_of_n_splits_index, yield_tensor, env_dfs = creat_tensor_class.create_train_test_validation_index(
            year=year)
        # creat_tensor_class.train_test_validation_tensor()

        tensor_dataset = convert_inputx_to_tesor_list(remove_na_dfs)
        # print(tensor_dataset.shape)
        self.genotype_df = genotype_df[['genotype.id']]
        self.x = tensor_dataset
        self.label = yield_tensor
        assert (dictionary_of_n_splits_index['splits_0'][-1] == dictionary_of_n_splits_index['splits_1'][
            -1]).all()  # validation set in n split should be the same
        self.index_dictionary = dictionary_of_n_splits_index

    def cv_train_test_data_iterater(self):

        index_dixtionary = copy.deepcopy(self.index_dictionary)
        print(index_dixtionary)
        for key in index_dixtionary.keys():
            train_index = index_dixtionary[key][0]
            test_index = index_dixtionary[key][1]
            validation_index = index_dixtionary[key][2]
            yield train_index, test_index

    def create_input_data(self):
        window_size = self.window_size
        x = copy.deepcopy(self.x)
        y, y_scaler = minmax_scaler(copy.deepcopy(self.label))
        if not window_size:
            print(x.shape)
            # exclude nan when calculate mean
            inputx = torch.nanmean(x,
                                   dim=1)  # average based on the time step demension, convert to Xarray-like of shape (n_samples, n_features)
            print(inputx.shape)
            y = torch.squeeze(y)
            print(y)
            # the shape of y need to be an 1d array like for random forest
            validation_indexes = self.index_dictionary['splits_0'][-1]
            validation_x = torch.squeeze(inputx)[validation_indexes, :]
            validation_y = y[validation_indexes]

            # feature name
            feature_length = int(inputx.shape[1] / 2)
            print(inputx.shape[1] / 2)
            feature_name = ['canopy' + str(x) for x in range(feature_length)] + ['height' + str(x) for x in
                                                                                 range(feature_length)]
            inputx = pd.DataFrame(inputx, columns=feature_name)

            return inputx, y, validation_x, validation_y, y_scaler, self.genotype_df, validation_indexes
        elif str(window_size).isnumeric():
            window_number = int(x.shape[1] / window_size)
            reformat_x_list = []
            for step in range(window_number):
                if (window_number + 1) * window_size <= x.shape[1]:
                    x_window_mean = torch.nanmean(
                        x[:, window_number * window_size:(window_number + 1) * window_size, :], dim=1)
                    reformat_x_list.append(x_window_mean)
                else:
                    x_window_mean = torch.nanmean(
                        x[:, window_number * window_size:, :], dim=1)
                    reformat_x_list.append(x_window_mean)
            else:
                reformat_x = torch.stack(reformat_x_list, dim=1)
                print(reformat_x.shape)
                reformat_x = torch.cat([reformat_x[:, :, 0], reformat_x[:, :, 1]], dim=1)
                reformat_x = torch.nan_to_num(reformat_x)
                print(reformat_x.shape)
            y = torch.squeeze(y)
            # the shape of y need to be an 1d array like for random forest
            validation_indexes = self.index_dictionary['splits_0'][-1]
            validation_x = torch.squeeze(reformat_x)[validation_indexes, :]
            validation_y = y[validation_indexes]

            feature_length = int(reformat_x.shape[1] / 2)
            feature_name = ['canopy' + str(x) for x in range(feature_length)] + ['height' + str(x) for x in
                                                                                 range(feature_length)]
            reformat_x = pd.DataFrame(reformat_x, columns=feature_name)
            print(reformat_x)
            return reformat_x, y, validation_x, validation_y, y_scaler, self.genotype_df, validation_indexes

            # slide window and take mean
        elif window_size == 'last_time_step':
            print(x.shape)
            inputx = x[:, -1,
                     :]  # take the last time step value, convert to Xarray-like of shape (n_samples, n_features)
            print(inputx.shape)
            y = torch.squeeze(y)  # the shape need to be an 1d array like for random forest
            validation_indexes = self.index_dictionary['splits_0'][-1]
            print('validation index {}'.format(validation_indexes))
            validation_x = torch.squeeze(inputx)[validation_indexes, :]
            print(validation_x)
            validation_y = y[validation_indexes]
            feature_length = int(inputx.shape[1] / 2)
            print(inputx.shape[1] / 2)
            feature_name = ['canopy' + str(x) for x in range(feature_length)] + ['height' + str(x) for x in
                                                                                 range(feature_length)]
            inputx = pd.DataFrame(inputx, columns=feature_name)
            return inputx, y, validation_x, validation_y, y_scaler, self.genotype_df, validation_indexes
        elif window_size == 'all_time_step':
            inputx = torch.cat([x[:, :, 0], x[:, :, 1]], dim=1)
            inputx = torch.nan_to_num(inputx, -999)
            print(inputx.shape)

            y = torch.squeeze(y)  # the shape need to be an 1d array like for random forest
            print(y)
            validation_indexes = self.index_dictionary['splits_0'][-1]
            print('validation index {}'.format(validation_indexes))
            validation_x = torch.squeeze(inputx)[validation_indexes, :]
            validation_y = y[validation_indexes]
            print('validation yyyy')
            print(validation_y)
            feature_length = int(inputx.shape[1] / 2)
            print(inputx.shape[1] / 2)
            feature_name = ['canopy' + str(x) for x in range(feature_length)] + ['height' + str(x) for x in
                                                                                 range(feature_length)]
            inputx = pd.DataFrame(inputx, columns=feature_name)
            print(inputx)
            return inputx, y, validation_x, validation_y, y_scaler, self.genotype_df, validation_indexes


class Random_forest_yield_prediction:

    def RF_model(self, n_split=5, n_validation=10, window_size: str | int = 2, year=2019):

        # inscease
        hyperparameters = {'n_estimators': [50, 100, 300, 500, 1000],  # 1
                           'max_features': [0.3, 0.5, 0.7],
                           'max_depth': [5, 10, 20, None],  # deeper, or use number of node, or leave it open
                           'criterion': ['squared_error']}
        print('{} times validation '.format(n_validation))

        try:
            rf_mean_input_validation_result = pd.read_csv('rf_mean_input_validation_result.csv', header=0, index_col=0)
            rf_mean_input_test_result = pd.read_csv('rf_mean_input_test_result_test.csv',
                                                    header=0, index_col=0)

        except:
            rf_mean_input_validation_result = pd.DataFrame()
            rf_mean_input_test_result = pd.DataFrame()

        for n in range(n_validation):
            prepare_rf = Prepare_Rf_model(year=year, n_split=n_split, seed=n, window_size=window_size)
            X, y, X_validation, y_validation, y_scaler, group_df, validation_indexes = prepare_rf.create_input_data()
            # About the scoring: https://github.com/scikit-learn/scikit-learn/issues/2439 In summary it's a negative value to make it the larger the better
            # refit set to false to make sure the whole dataset won't be used for refit the model, will do this manually with trainning data
            print('___________________________________')
            # print(X)
            # print(y)
            # unscale_y = y_scaler.inverse_transform(torch.unsqueeze(torch.tensor(y),dim=1))
            # print(unscale_y)
            print(group_df)
            # print(validation_indexes)

            # #use split function in GridSearchCV
            # best_rf, best_parameters,cv_result= self.cross_validation_rf(X,  group_df, hyperparameters, n,
            #                                     validation_indexes, y)
            # use self generate train test index in GridSearchCV
            best_rf, best_parameters, cv_result = self.cross_validation_rf_self_define_cv(X, hyperparameters,
                                                                                          prepare_rf, y)

            plt.clf()
            sns.lineplot(y=cv_result["mean_test_score"],
                         x=cv_result['param_max_depth'].data,
                         hue=cv_result['param_n_estimators'])
            plt.xlabel("max_depth")
            plt.ylabel("mean_test_error (mean of 5 times random split)")
            plt.title(
                "mean_squared_error with different max_depth and features for RF model")
            plt.tight_layout()
            plt.savefig(
                '../samples_visualization/rf_plots/mean input_yield prediction_training with different hyperparameters_validation{}_window_size__{}_new.png'.format(
                    n, window_size))
            plt.clf()

            # predict_y = best_rf.predict(X_validation)
            coef, predict_y, validation_MSE, validation_percentage_MSE = self.validation_result(X_validation, best_rf,
                                                                                                y_scaler,
                                                                                                y_validation, n,
                                                                                                window_size)
            # inverse scale
            y_validation = torch.unsqueeze(torch.tensor(y_validation), dim=1)
            print(y_validation.shape)
            print(predict_y.shape)
            y_validation = y_scaler.inverse_transform(y_validation)

            test_result_df = pd.DataFrame(cv_result)
            test_result_df['validation_rep'] = n
            test_result_df['window_size'] = window_size
            rf_mean_input_test_result = pd.concat([rf_mean_input_test_result, test_result_df])
            rf_mean_input_test_result.to_csv('rf_mean_input_test_result_test.csv')

            plt.clf()
            plt.scatter(x=np.squeeze(y_validation), y=np.squeeze(predict_y), c='r')
            # plt.scatter(x=np.squeeze(y_validation), y=np.squeeze(predict_y1), c='b')
            plt.xlim(0, 20)
            plt.ylim(0, 20)
            plt.ylabel('predicted yield')
            plt.xlabel('true yield')
            plt.tight_layout()
            plt.savefig(
                '../samples_visualization/rf_plots/yield_predict_scatter_validation{}_window_size_{}.svg'.format(n,
                                                                                                                 window_size))
            # plt.clf()
            # plt.show()

            self.plot_feature_importance(model=best_rf, X=X, name=str(n) + '_window_size__' + str(window_size))
            result_dictionary = {'validation_rep': n, 'validation_MSE': validation_MSE,
                                 'validation_MAPE': validation_percentage_MSE, 'pearson_coef': coef,
                                 'window_size': window_size}
            criterion_yield = nn.MSELoss()
            print(y_validation.shape)
            print(predict_y.shape)
            MSE_loss = criterion_yield(torch.tensor(y_validation), torch.tensor(predict_y))
            print('MSE calculate by nn model {}'.format(MSE_loss))
            print(result_dictionary)
            result_dictionary.update(best_parameters)
            new_row = pd.DataFrame(result_dictionary, index=[0])
            rf_mean_input_validation_result = pd.concat([rf_mean_input_validation_result, new_row])
            rf_mean_input_validation_result.to_csv('rf_mean_input_validation_result.csv')

    def cross_validation_rf_self_define_cv(self, X, hyperparameters, prepare_rf, y):
        rf_cv_self = GridSearchCV(
            RandomForestRegressor(random_state=0),
            hyperparameters,
            cv=prepare_rf.cv_train_test_data_iterater(),
            verbose=5, return_train_score=True, refit=False,
            scoring='r2')  # https://scikit-learn.org/stable/modules/model_evaluation.html score
        # use r2 score for regression if scoring is not specified
        rf_cv_self.fit(X,
                       y)  # the whole dataset is passed here, but for cv validation only perform train and test index, so validation set is still unsean
        # use the best rf model hyperparameters to train the model for validation(X_validation)
        best_parameters = rf_cv_self.best_params_
        print(best_parameters)
        print(rf_cv_self.cv_results_)
        # get train and test index from one split refit model to get estimator
        train_index, test_index = list(prepare_rf.cv_train_test_data_iterater())[0]
        # get train and test from dataset, fit a rf model with the best parameters
        X_train = X.loc[train_index, :]
        y_train = y[train_index]
        best_rf = RandomForestRegressor(random_state=0, n_estimators=best_parameters['n_estimators'],
                                        max_features=best_parameters['max_features'],
                                        max_depth=best_parameters['max_depth'])
        best_rf.fit(X_train, y_train)
        return best_rf, best_parameters, rf_cv_self.cv_results_

    def cross_validation_rf(self, X, group_df, hyperparameters, n, validation_indexes, y):
        splitter = GroupShuffleSplit(test_size=.2, n_splits=5, random_state=n)
        cross_validatiton_X = X.drop(index=validation_indexes)
        print(cross_validatiton_X)
        keep_index = pd.DataFrame({'index': X.index}).drop(index=validation_indexes)
        cross_validatiton_y = y[keep_index.index]
        cross_validatiton_group = group_df.drop(index=validation_indexes)
        split = splitter.split(cross_validatiton_X, cross_validatiton_y, groups=cross_validatiton_group)
        rf_cv = GridSearchCV(
            RandomForestRegressor(random_state=0),
            hyperparameters,
            cv=split,
            verbose=3, return_train_score=True, refit=True, scoring='r2')
        rf_cv.fit(cross_validatiton_X, cross_validatiton_y)
        cv_result = rf_cv.cv_results_
        print()
        best_parameters = rf_cv.best_params_
        best_rf1 = rf_cv.best_estimator_
        return best_rf1, best_parameters, cv_result

    def multiple_year_rf_model(self, n_split=5, n_validation=10, window_size: str | int = 2, years=[2018, 2019, 2021],
                               data=None):
        splitter = GroupShuffleSplit(test_size=.2, n_splits=5, random_state=0)
        hyperparameters = {'n_estimators': [50, 100, 300, 500, 1000],  # increase
                           'max_features': [0.3, 0.5, 0.7],
                           'max_depth': [5, 10, 20, None],
                           'criterion': ['squared_error']}
        print('{} times validation '.format(n_validation))
        try:
            rf_mean_input_validation_result = pd.read_csv('rf_mean_input_validation_result_align_test.csv', header=0,
                                                          index_col=0)
            rf_mean_input_test_result = pd.read_csv('rf_mean_input_test_result_align_test.csv',
                                                    header=0, index_col=0)

        except:
            rf_mean_input_validation_result = pd.DataFrame()
            rf_mean_input_test_result = pd.DataFrame()

            # use single year for training and test in other years
        year = years[1]  # use 2019 as training
        for n in range(n_validation):
            if str(window_size).isnumeric():
                print('aligned and averaged data input')
                prepare_rf = Prepare_Rf_model(year=year, n_split=n_split, seed=n, window_size='all_time_step',
                                              load_data_boolean=False, data=data)
            else:
                print('use window size : {}'.format(window_size))
                prepare_rf = Prepare_Rf_model(year=year, n_split=n_split, seed=n, window_size=window_size,
                                              load_data_boolean=False, data=data)

            X, y, X_test, y_test, y_scaler, group_df, validation_indexes = prepare_rf.create_input_data()

            best_rf, best_parameters, cv_result = self.cross_validation_rf_self_define_cv(X, hyperparameters,
                                                                                          prepare_rf, y)

            # save test result in df
            test_result_df = pd.DataFrame(cv_result)
            test_result_df['validation_rep'] = n
            test_result_df['window_size'] = window_size
            rf_mean_input_test_result = pd.concat([rf_mean_input_test_result, test_result_df])
            rf_mean_input_test_result.to_csv('rf_mean_input_test_result_align_test.csv')

            # save best model from each validation

            self.plot_tree_from_rf(list(X.columns), best_rf)
            with open('model/rf_model/best_validation{}_rf_after_fit.dill', 'wb') as file:
                dill.dump(best_rf, file)
            file.close()

            # use the best rf model hyperparameters for validation(X_test)
            coef, predict_y, validation_MSE, validation_percentage_MSE = self.validation_result(X_test, best_rf,
                                                                                                y_scaler,
                                                                                                y_test, n, window_size)
            plt.clf()
            # plt.show()
            self.plot_feature_importance(model=best_rf, X=X,
                                         name=str(n) + '_window_size__' + str(window_size) + 'align')
            result_dictionary = {'validation_rep': n, 'validation_MSE': validation_MSE,
                                 'validation_MAPE': validation_percentage_MSE, 'pearson_coef': coef,
                                 'window_size': 'align_' + str(window_size), 'different_year_validation': False}

            print(predict_y.shape)
            print(result_dictionary)
            result_dictionary.update(best_parameters)
            new_row = pd.DataFrame(result_dictionary, index=[0])
            rf_mean_input_validation_result = pd.concat([rf_mean_input_validation_result, new_row])

            # #use other year as validation 2021 not the same length
            # prepare_rf = Prepare_Rf_model(year=2021, n_split=n_split, seed=n, window_size='all_time_step',load_data_boolean=False,data=data)
            # X,y,X_test,y_test,y_scaler = prepare_rf.create_input_data()
            # coef, predict_y, validation_MSE, validation_percentage_MSE= self.validation_result(X, best_rf,
            #                                                                                             y_scaler,
            #                                                                                             y)
            # result_dictionary = {'validation_rep': n, 'validation_MSE': validation_MSE,
            #                      'validation_MAPE': validation_percentage_MSE, 'spearman_coef': coef,
            #                      'window_size': window_size,'same_year_validation':2021}
            # result_dictionary.update(best_parameters)
            # new_row = pd.DataFrame(result_dictionary, index=[0])
            # rf_mean_input_validation_result = pd.concat([rf_mean_input_validation_result, new_row])

            rf_mean_input_validation_result.to_csv('rf_mean_input_validation_result_align_test.csv')

    def plot_tree_from_rf(self, feature_names, best_rf):
        from sklearn import tree
        fig, ax = plt.subplots()
        tree.plot_tree(best_rf.estimators_[0],
                       feature_names=feature_names,
                       filled=True,
                       ax=ax)
        fig.savefig('tree_1.svg', dpi=20000)
        # plt.show()

    def validation_result(self, X_test, best_rf, y_scaler, y_test, n, window_size):
        predict_y = best_rf.predict(X_test)
        # inverse scale
        y_test = torch.unsqueeze(torch.tensor(y_test), dim=1)
        print(y_test.shape)

        y_test = y_scaler.inverse_transform(y_test)
        predict_y = torch.unsqueeze(torch.tensor(predict_y), dim=1)
        predict_y = y_scaler.inverse_transform(predict_y)
        validation_MSE = mean_squared_error(y_true=y_test, y_pred=predict_y)
        validation_percentage_MSE = mean_absolute_percentage_error(y_true=y_test, y_pred=predict_y)
        coef, p = pearsonr(np.squeeze(y_test), np.squeeze(predict_y))
        plt.clf()
        plt.scatter(x=np.squeeze(y_test), y=np.squeeze(predict_y))
        plt.xlim(0, 20)
        plt.ylim(0, 20)
        plt.ylabel('predicted yield')
        plt.xlabel('true yield')
        plt.tight_layout()
        plt.savefig(
            '../samples_visualization/rf_plots/yield_predict_scatter_validation{}_window_size__{}_same_year_validation.svg'.format(
                n,
                window_size))
        return coef, predict_y, validation_MSE, validation_percentage_MSE

    def linear_regression(self, n_split=5, n_validation=10, window_size: str | int = 2, year=2019, data=None):

        print('{} times validation '.format(n_validation))
        try:
            rf_mean_input_validation_result = pd.read_csv('lm_mean_input_validation_result.csv', header=0, index_col=0)
        except:
            rf_mean_input_validation_result = pd.DataFrame()
        for n in range(n_validation):
            if data != None:
                if str(window_size).isnumeric():
                    print('use aligned and averaged data input')
                    prepare_data = Prepare_Rf_model(year=year, n_split=n_split, seed=n, window_size='all_time_step',
                                                    load_data_boolean=False, data=data)
                else:
                    print('use window size : {} to calculate average'.format(window_size))
                    prepare_data = Prepare_Rf_model(year=year, n_split=n_split, seed=n, window_size=window_size,
                                                    load_data_boolean=False, data=data)
            else:
                print('use window size : {} to calculate average'.format(window_size))
                prepare_data = Prepare_Rf_model(year=year, n_split=n_split, seed=n, window_size=window_size
                                                )
            X, y, X_validation, y_validation, y_scaler, group_df, validation_indexes = prepare_data.create_input_data()  # X train is the whole dataset, X test is the validation set
            # About the scoring: https://github.com/scikit-learn/scikit-learn/issues/2439 In summary it's a negative value to make it the larger the better
            # hyperparemters: number of features used in lm

            lm = LinearRegression()
            rfe = RFE(lm)
            hyper_params = {'n_features_to_select': list(range(1, X.shape[1]))}
            linear_regression_cv = GridSearchCV(estimator=rfe,
                                                param_grid=hyper_params,
                                                scoring='r2',
                                                cv=prepare_data.cv_train_test_data_iterater(),
                                                verbose=3,
                                                return_train_score=True, refit=False)

            linear_regression_cv.fit(X,
                                     y)  # the whole dataset is passed here, but for cv validation only perform train and test index, so validation set is still unsean
            best_parameters = linear_regression_cv.best_params_

            best_lm = RFE(lm, n_features_to_select=best_parameters['n_features_to_select'])
            # get train and test index from one split refit model to get coefficient
            train_index, test_index = list(prepare_data.cv_train_test_data_iterater())[0]
            # get train and test from dataset
            X_train = X.loc[train_index, :]
            y_train = y[train_index]
            print(X_train)
            best_lm.fit(X_train, y_train)
            best_lm.estimator.fit(X_train, y_train)
            print('coefficient {}'.format(best_lm.estimator.coef_))

            # use the best rf model hyperparameters for validation
            predict_y = best_lm.predict(X_validation)

            # inverse scale
            y_validation = torch.unsqueeze(torch.tensor(y_validation), dim=1)
            print(y_validation.shape)
            y_validation = y_scaler.inverse_transform(y_validation)
            predict_y = torch.unsqueeze(torch.tensor(predict_y), dim=1)
            predict_y = y_scaler.inverse_transform(predict_y)

            validation_MSE = mean_squared_error(y_true=y_validation, y_pred=predict_y)
            validation_percentage_MSE = mean_absolute_percentage_error(y_true=y_validation, y_pred=predict_y)
            coef, p = pearsonr(np.squeeze(y_validation), np.squeeze(predict_y))
            spearmancoef, p = spearmanr(np.squeeze(y_validation), np.squeeze(predict_y))
            plt.clf()
            plt.scatter(x=np.squeeze(y_validation), y=np.squeeze(predict_y))
            plt.xlim(0, 20)
            plt.ylim(0, 20)
            plt.ylabel('predicted yield')
            plt.xlabel('true yield')
            plt.tight_layout()
            # plt.show()
            plt.savefig(
                '../samples_visualization/lm_plots/yield_predict_scatter_validation{}_window_size_{}.svg'.format(n,
                                                                                                                 window_size))
            plt.clf()
            # plt.show()
            # self.plot_feature_importance(model=linear_regression_cv.best_estimator_, X=X,
            #                              name=str(n) + '_window_size__' + str(window_size),model_name='lm')
            result_dictionary = {'validation_rep': n, 'validation_MSE': validation_MSE,
                                 'validation_MAPE': validation_percentage_MSE, 'pearson_coef': coef,
                                 'spearman_coef': spearmancoef,
                                 'window_size': window_size}
            print(y_validation.shape)
            print(predict_y.shape)
            print(result_dictionary)
            result_dictionary.update(best_parameters)
            new_row = pd.DataFrame(result_dictionary, index=[0])
            rf_mean_input_validation_result = pd.concat([rf_mean_input_validation_result, new_row])
            rf_mean_input_validation_result.to_csv('lm_mean_input_validation_result.csv')

    def plot_feature_importance(self, model, X, name, model_name='rf'):
        # Plot feature importance
        # print(len(X[0]))
        plt.clf()
        fi = pd.DataFrame(data=model.feature_importances_,
                          index=X.columns,
                          columns=['Importance']).sort_values(by=['Importance'], ascending=False)

        print(fi.index)
        ax1 = sns.barplot(data=fi.head(20), x="Importance",
                          y=(fi.head(20)).index)
        ax1.set_title(
            "feature importance for {} model".format(model_name))
        plt.ylabel('Features')
        plt.tight_layout()
        plt.savefig(
            '../samples_visualization/{}_plots/yield_predict_feature_importance_validation{}.jpg'.format(model_name,
                                                                                                         name))
        # plt.clf()
        # plt.show()


def convert_input_tensor_for_Convlstm(input_x: torch.tensor, position_tensor: torch.tensor, h: int = None,
                                      w: int = None):
    """Combine location with input tensor into 5 dimension tensor"""
    # h,w maximum row, col number
    # inpux shape need to be convert into (t, b, c, h, w)

    import warnings
    shape = input_x.shape  # batch, time step,feature
    position_shape = position_tensor.shape
    print(shape, position_shape)
    b, t, c = shape
    if (h is None) or (w is None):
        warnings.warn(
            "Warning: Missing row and column number for input_x, will take the maximum value from position_tensor")
        h = torch.max(position_tensor[:, 0])  # maximum row number
        w = torch.max(position_tensor[:, 1])  # maximum col number

    # create a new tensor shape (b,t, c, h, w)
    reformat_tensor = torch.zeros([b, t, c, h, w])
    print('reformat tensor shape {}'.format(reformat_tensor.shape))
    for sample_index in range(0, b):
        sample_index_tensor = torch.tensor([sample_index])
        sample_position = position_tensor[sample_index, :]
        sample = torch.index_select(input_x, 0, sample_index_tensor)
        reformat_tensor[sample_index, :, :, sample_position[0] - 1,
        sample_position[1] - 1] = sample  # row and col number-1, use as index
    else:

        reformat_tensor = torch.permute(reformat_tensor, (1, 0, 2, 3, 4))
        print('reformat tensor shape {}'.format(reformat_tensor.shape))  # shape (t, b, c, h, w)

        return reformat_tensor


class PlotX:
    def __init__(self, color_label):
        raise RuntimeError('still does not work for this formate of input')
        self.color_label = color_label

    def _get_colors(self):
        num_colors = len(self.color_label.unique())

        colors = []
        random.seed(0)
        for i in np.arange(0.0, 360.0, 360.0 / num_colors):
            hue = i / 360.0
            lightness = (50 + np.random.rand() * 10) / 100.0
            saturation = (90 + np.random.rand() * 10) / 100.0
            colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
        return colors

    def label_with_label(self):
        map_dictionary = {}
        label_df = copy.deepcopy(self.color_label)
        colours = self._get_colors()
        for type in label_df.unique():
            map_dictionary[type] = colours.pop(0)
        colour_label = label_df.map(map_dictionary)

        return colour_label, map_dictionary


def plot_X(genotype, tensor_dataset, n, name):
    '''
    plot and save the samples figure
    '''
    # print(keep_index)
    print('{}_split {}set'.format(n, name))
    # genotype = genotype.iloc[keep_index,:]

    plot_df = pd.DataFrame(torch.squeeze(tensor_dataset[:, :, 1].T)).astype(float)
    plot_df.columns = genotype['genotype.id'].astype('str').to_list()
    # print(plot_df)
    hue_list = genotype['genotype.id'].astype(int)
    # print(hue_list)
    plt.clf()
    ax = sns.lineplot(data=plot_df)
    plt.legend(fontsize="6")
    sns.move_legend(ax, 'upper center', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.xlabel("days")
    plt.ylabel("plant height)")
    plt.title(" Samples' Plant height in {} set".format(name))
    plt.tight_layout()
    plt.show()
    # plt.savefig("../samples_visualization/Plant_height_in_{}_set_split_{}.svg".format(name,n))
    #
    # plt.close()

    plot_df = pd.DataFrame(torch.squeeze(tensor_dataset[:, :, 0].T)).astype(float)
    plot_df.columns = genotype['genotype.id'].astype('str').to_list()

    plt.clf()
    ax = sns.lineplot(data=plot_df)
    plt.legend(fontsize="6")
    sns.move_legend(ax, 'upper center', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.xlabel("days")
    plt.ylabel("Canopy coverage)")
    plt.title(" Samples' Plant height in {} set".format(name))
    plt.tight_layout()
    plt.show()
    # plt.savefig("../samples_visualization/Canopy_coverage_in_{}_set_split_{}.svg".format(name, n))
    #
    # plt.close()


def training(model, x, y, position, genotype, batch_size, test_x, test_y, test_pos, validation_x, validation_y,
             validation_pos, epoch: int, lr=0.001, optimize="SGD", mode='predict_yield', with_position=True,
             file_name=''):
    """

    :param model:
    :param x: shape: number of sequences,time step,  inputsize(3) (the first dimension should be the same as y
    :param y: label, shape: (num_sequences, output_size) output size is 1 for binary classification
    :param batch_size:
    :param input_size:
    :param output_size:
    :return:
    """
    # # set random seed
    # np.random.seed(123)
    # random.seed(123)
    # torch.manual_seed(123)

    num_sequences = x.shape[0]
    test_n_seq = test_x.shape[0]
    validation_n_seq = validation_x.shape[0]
    print(num_sequences)
    seq_length = x.shape[1]
    # Define training parameters
    learning_rate = lr  #
    num_epochs = epoch

    # Convert input and target data to PyTorch datasets
    print(x.shape)
    print(y.shape)
    print(position.shape)
    print(genotype.shape)
    dataset = TensorDataset(x, y, position, genotype)

    # Create data loader for iterating over dataset in batches during training
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # initialize model
    model.init_network()

    if optimize == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimize == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    loss_list = []
    loss_test_list = []
    curve_loss_list = []
    gene_loss_list = []
    x_axis = []
    loss_validation_list = []
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    previous_loss = 0
    previous_curve_loss = 0
    previous_genoype_loss = 0
    for epoch in range(num_epochs):
        running_loss = 0.0  # running loss for every epoch
        running_loss_curve = 0.0
        running_loss_gene = 0.0
        running_loss_test = 0.0
        running_loss_validation = 0.0
        for index1, (inputs, targets, position_batch, genotype_batch) in enumerate(dataloader):
            # print(index1)
            # print(inputs.shape[0])
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            if mode == 'predict_yield':
                # print(inputs.shape)
                yield_pre = model(inputs.float(), position_batch,
                                  with_position)  # batch first, the input is (110,46,2) for 2019
                yield_pre_test = model(test_x.float(), test_pos, with_position)

                yield_pre_validation = model(validation_x.float(), validation_pos, with_position)
                # print("target:{}".format(predict_genotype.shape))
                # print("output_fc{}".format(output_fc.shape))
                targets = targets.float()
                test_target = test_y.float()
                validation_target = validation_y.float()
                # print(targets)
                # print(yield_pre.shape)
                loss = model.spearman_rank_loss(true_label=targets, predict_Y=yield_pre)
                loss_test = model.spearman_rank_loss(true_label=test_target, predict_Y=yield_pre_test)
                loss_validation = model.spearman_rank_loss(true_label=validation_target, predict_Y=yield_pre_validation)
                running_loss_test += loss_test.item() * test_x.shape[0]
                running_loss_validation += loss_validation.item() * validation_x.shape[0]
                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.shape[0]  # loss.item() is the mean loss of the total batch
            elif mode == 'predict_genotype':
                output_curve, predict_genotype = model(inputs.float())
                # print("output_fc{}".format(output_fc.shape))
                targets = targets.float()
                # print(targets)

                loss, fit_curve_loss, genotype_loss = model.spearman_rank_loss(fit_curve=output_curve,
                                                                               predict_genotype=predict_genotype,
                                                                               true_label=targets, input_curve=inputs)
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                # sum the running loss for every batch, the last batch maybe smaller
                running_loss_curve += fit_curve_loss.item() * inputs.shape[
                    0]  # loss.item() is the mean loss of the total batch
                running_loss_gene += genotype_loss.item() * inputs.shape[0]
        if (epoch + 1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs,
                                                       running_loss / (num_sequences)))
            print('Epoch [{}/{}], Test Loss: {:.4f}'.format(epoch + 1, num_epochs,
                                                            (running_loss_test / (test_n_seq)) / (index1 + 1)))
            print('Epoch [{}/{}], validation Loss: {:.4f}'.format(epoch + 1, num_epochs,
                                                                  (running_loss_validation / (validation_n_seq)) / (
                                                                              index1 + 1)))
            # print('Epoch [{}/{}], fit curve Loss: {:.4f}'.format(epoch + 1, num_epochs,
            #                                            running_loss_curve/(num_sequences)))
            # print('Epoch [{}/{}], genotype predict Loss: {:.4f}'.format(epoch + 1, num_epochs,
            #                                            running_loss_gene/(num_sequences)))
            x_axis.append(epoch + 1)
            # plot loss
            if (epoch + 1) == 10:
                loss_list.append(0)
                loss_test_list.append(0)
                loss_validation_list.append(0)
                # curve_loss_list.append(0)
                # gene_loss_list.append(0)

            else:
                loss_list.append(running_loss / (num_sequences))
                loss_test_list.append((running_loss_test / (test_n_seq)) / (index1 + 1))
                loss_validation_list.append((running_loss_validation / (validation_n_seq)) / (index1 + 1))
                # curve_loss_list.append(running_loss_curve/(num_sequences)-previous_curve_loss)
                # gene_loss_list.append(running_loss_gene/(num_sequences)-previous_genoype_loss)
                #
                #

                sns.lineplot(x=x_axis, y=loss_list, c='blue', label='train_loss')
                sns.lineplot(x=x_axis, y=loss_test_list, c='red', label='test_loss')
                sns.lineplot(x=x_axis, y=loss_validation_list, c='orange', label='validation_loss')
                # line3, = ax.plot(x_axis, gene_loss_list, c='orange')
                plt.ylabel("loss")
                plt.legend()
                # plt.show()
                # plt.savefig('../samples_visualization/nn_plot/train_validationloss_during_training_{}.png'.format(file_name))
                # plt.clf()
                # line1.set_ydata(loss_list)
                # line1.set_xdata(x_axis)
                # line2.set_ydata(curve_loss_list)
                # line2.set_xdata(x_axis)
                # line3.set_ydata(gene_loss_list)
                # line3.set_xdata(x_axis)
                # plt.axhline(y=0, color='g', linestyle='-')
                fig.canvas.draw_idle()
                fig.canvas.flush_events()

            # previous_loss = running_loss / (num_sequences)
            # previous_curve_loss = running_loss_curve / (num_sequences)
            # previous_genoype_loss = running_loss_gene / (num_sequences)
    else:
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            # predict_curve,predict_genotypes,_ = model(x.float())
            if mode == 'predict_yield':
                predict_yield = model(x.float(), position, with_position)
                loss = model.spearman_rank_loss(true_label=y, predict_Y=predict_yield)  # ,std=std,mean=mean
                return predict_yield, model, loss
            elif mode == 'predict_genotype':
                raise EOFError('have not finished yet')
                predict_curve, predict_genotype = model(inputs.float())
                loss, fit_curve_loss, genotype_loss = model.custom_loss(fit_curve=predict_curve,
                                                                        predict_genotype=predict_genotype,
                                                                        true_label=targets, input_curve=inputs)

                return model, predict_curve, predict_genotype, loss, fit_curve_loss, genotype_loss


def test_result(model, xtest, ytest, scaler, position=None, with_position=True, file_name=''):
    from scipy.stats import spearmanr, pearsonr
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        if position != None:
            predict_yield = model(xtest.float(), position, with_position)
        else:
            forecast, predict_yield = model(xtest.float())
        unscale_predict_yield = torch.tensor(scaler.inverse_transform(predict_yield))
        unscaled_ytest = torch.tensor(scaler.inverse_transform(ytest))
        criterion_yield = nn.MSELoss()
        MSE_loss = criterion_yield(unscaled_ytest, unscale_predict_yield)
        print(MSE_loss)
        print(unscaled_ytest.shape, unscale_predict_yield.shape)
        coef, p = spearmanr(unscaled_ytest, unscale_predict_yield)
        # coef, p = pearsonr(unscaled_ytest,unscale_predict_yield)
        # print(unscaled_ytest)
        # print(unscale_predict_yield)
        print(coef, p)
        plt.clf()
        plt.scatter(x=unscaled_ytest, y=unscale_predict_yield)
        plt.xlim(0, 20)
        plt.ylim(0, 20)
        plt.xlabel('true')
        plt.ylabel('predict')
        if file_name != '':
            plt.savefig('../samples_visualization/nn_plot/{}_yield_prediction.png'.format(file_name))
        # plt.show()

        # _, _, _ = mean_yield_benchmark(unscaled_ytest)

    return MSE_loss.item(), coef, p


def mean_yield_benchmark(unscaled_ytest):
    criterion_yield = nn.MSELoss()
    # if use mean as output, the result will be as following
    mean_predict_yield = torch.full(unscaled_ytest.shape, unscaled_ytest.mean())  # use mean to fill the tensor
    mean_predict_yield = mean_predict_yield + 0.001 * torch.randn(unscaled_ytest.shape)  # add a small gussian noise
    # print(mean_predict_yield)
    print(unscaled_ytest.shape, mean_predict_yield.shape)
    coef, p = spearmanr(unscaled_ytest, mean_predict_yield)
    pearson_correlation = pearsonr(torch.squeeze(unscaled_ytest), torch.squeeze(mean_predict_yield))
    print('peason correlation : {}'.format(pearson_correlation))
    MSE_loss = criterion_yield(unscaled_ytest, mean_predict_yield)
    print('use mean as predict:MSE{},coef{},p value{}'.format(MSE_loss, coef, p))
    return MSE_loss, coef, p


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
    # print('std')
    # print(std) # one value
    # print(std.shape)
    # print(mean.shape)
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


def plot_growth_curve_colored_based_on_yield(X, y):
    sns.color_palette("viridis", as_cmap=True)
    sns.lineplot(data=torch.squeeze(X), )
    plt.show()


def multiple_years_yield(average_genotype=True, n_split=5, year=[2019], random_seed_for_validation_split=0,
                         slide_window_average: int | bool = False):
    list_tensor_x = []
    list_tensor_y = []
    list_tensor_position = []
    list_tensor_genotype = []

    # print('data for year {}'.format(year))
    # canopy_df = data_load("C:/data_from_paper/ETH/olivia_WUR/process_data/Canopy coverage.csv", year)
    #
    # data_canopy = canopy_df[
    #     ['timestamp', 'plot.UID', 'value', 'genotype.id', 'plot.row_global', 'plot.range_global', 'lot']]
    # #print(data_canopy)
    # height_df = data_load("C:/data_from_paper/ETH/olivia_WUR/process_data/Plant height.csv", year)
    # data_height = height_df[
    #     ['timestamp', 'plot.UID', 'value', 'genotype.id', 'plot.row_global', 'plot.range_global', 'lot']]
    # canopy, height = load_multiple_year_data(years=year)
    canopy = pd.read_csv('../processed_data/align_canopy_drop_na.csv', header=0, index_col=0)
    height = pd.read_csv('../processed_data/align_height_drop_na.csv', header=0, index_col=0)
    dfs = [canopy, height]
    if slide_window_average:
        dfs = average_based_on_days([canopy, height], 5)
    creat_tensor_class = create_tensor_dataset(dfs, year=year, n_split=n_split, average_endpoint_value=average_genotype,
                                               random_seed=random_seed_for_validation_split)

    yield from creat_tensor_class.train_test_validation_tensor_iterator()
    # for tensor_dataset_list,yield_tensor_list,position_tensor_list,genotype_tensor_list in creat_tensor_class.train_test_validation_tensor_iterator():
    #
    #     yield tensor_dataset_list,yield_tensor_list,position_tensor_list,genotype_tensor_list


def feature_importance(model, datax, datay, batch):
    dataset = TensorDataset(datax, datay)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=True)
    batch = next(iter(dataloader))
    seqs, _ = batch

    background = seqs[:100]
    test_seq = seqs[100:103]
    e = shap.DeepExplainer(model, background)
    shap_values = e.shap_values(test_seq)


def yield_prediction_one_year(X, Y, position, genotype, with_position_str='with_position'):
    # for tensor_dataset,Y_tensor,position_tensor,genotype_tensor in zip(X,Y,position,genotype):
    # print(X)
    print(len(X))
    X_train, X_test, X_validation = X
    y_train, y_test, y_validation = Y
    position_tensor_train, position_tensor_test, position_tensor_validation = position
    genotype_tensor_train, genotype_tensor_test, genotype_tensor_validation = genotype

    X_train = torch.nan_to_num(X_train)
    X_test = torch.nan_to_num(X_test)
    X_validation = torch.nan_to_num(X_validation)
    '''
    #masked na
    X_train = masked_tensor(X_train)
    X_test = masked_tensor(X_test)
    X_validation = masked_tensor(X_validation)
    '''
    assert torch.isnan(X_train).any() == False

    scaled_Y_train_tensor, scaler_Y_train = minmax_scaler(y_train)
    scaled_position_train_tensor, _ = minmax_scaler(position_tensor_train)

    scaled_Y_test_tensor, scaler_y_test = minmax_scaler(y_test)
    scaled_position_test_tensor, _ = minmax_scaler(position_tensor_test)

    scaled_Y_validation_tensor, scaler_y_test = minmax_scaler(y_validation)
    scaled_position_validation_tensor, _ = minmax_scaler(position_tensor_validation)
    # print(scaled_Y_train_tensor)
    try:
        yield_result = pd.read_csv("yield{}_mse_loss.csv".format(with_position_str), header=0, index_col=0)
    except:
        yield_result = pd.DataFrame()
    print('with position:{}'.format(with_position_str != ''))
    # define model lr 0.001,0.005,0.0005
    best_test = -1.0
    best_parameters = {}
    best_model = None
    for hidden in [5]:  # [3,5,7,9]
        print('X train shape{}'.format(X_train.shape))  # [110, 46, 2] or seq len 59( for align feature seq)
        print('yield train shape{}'.format(y_train.shape))
        print('position shape {}'.format(scaled_position_train_tensor.shape))
        print('genotype shape {}'.format(genotype_tensor_train.shape))
        if with_position_str != 'one_layer_lstm':
            lstm_model = LSTM_yield_predict(input_size=2, hidden_size=hidden, batch_first=True,
                                            output_size=scaled_Y_train_tensor.shape[1], seq_len=X_train.shape[1],
                                            position=(with_position_str != ''))
        else:
            print('###########################{}'.format(scaled_Y_train_tensor.shape[1]))
            lstm_model = onelayer_LSTM_yield_prediction(input_size=2, hidden_size=hidden,
                                                        output_size=scaled_Y_train_tensor.shape[1], batch_first=True)
        count_parameters(lstm_model)
        for lr in [0.005]:  # [0.005,0.001, 0.01, 0.0001]
            for batch_size in [100]:  # [50,30]
                for epoch in [1000]:  # 300
                    print("hidden size {}".format(hidden))
                    print('learning rate:{}'.format(lr))
                    print('batch_size:{}'.format(batch_size))
                    print('epoch:{}'.format(epoch))
                    predict_yield, model, train_loss = training(model=lstm_model,
                                                                x=X_train,
                                                                y=scaled_Y_train_tensor,
                                                                position=scaled_position_train_tensor,
                                                                genotype=genotype_tensor_train,
                                                                batch_size=batch_size, epoch=epoch,
                                                                lr=lr,
                                                                optimize='Adam',
                                                                with_position=(with_position_str != ''), test_x=X_test,
                                                                test_y=scaled_Y_test_tensor,
                                                                test_pos=scaled_position_test_tensor,
                                                                validation_x=X_validation,
                                                                validation_y=scaled_Y_validation_tensor,
                                                                validation_pos=scaled_position_validation_tensor,
                                                                file_name='lr{}_hd{}_batch{}_epoch{}_{}'.format(lr,
                                                                                                                hidden,
                                                                                                                batch_size,
                                                                                                                epoch,
                                                                                                                with_position_str))
                    # test MSE
                    MSE_loss, spearmanr, p = test_result(model, X_test, scaled_Y_test_tensor, scaler=scaler_y_test,
                                                         position=scaled_position_test_tensor,
                                                         with_position=(with_position_str != ''),
                                                         file_name='lr{}_hd{}_batch{}_epoch{}_{}'.format(lr, hidden,
                                                                                                         batch_size,
                                                                                                         epoch,
                                                                                                         with_position_str) + 'test')
                    if spearmanr > best_test:
                        best_test = spearmanr
                        best_parameters = {"lr": lr,
                                           "batch_size": batch_size,
                                           "hidden_size": hidden,
                                           'epoch': epoch,
                                           'train_loss': train_loss,
                                           "test_MSE_loss": MSE_loss,
                                           'test_spearman_rank': spearmanr,
                                           'test_pvalue': p}
                        best_model = model

                    print('test MSE loss:{}'.format(MSE_loss))
                    print('test spearman loss:{}'.format(spearmanr))
                    new_row = pd.DataFrame(
                        data={"lr": lr,
                              "batch_size": batch_size,
                              "hidden_size": hidden,
                              'epoch': epoch,
                              'train_loss': train_loss,
                              "test_MSE_loss": MSE_loss,
                              'test_spearman_rank': spearmanr,
                              'test_pvalue': p
                              },
                        index=[0])
                    yield_result = pd.concat(
                        [yield_result, new_row])
                    yield_result.to_csv('yield{}_spearmanr_loss.csv'.format(with_position_str))
    else:
        return best_test, best_parameters, best_model


def spatial_correct_genotype_prediction():
    # load data
    canopy_df = data_load("C:/data_from_paper/ETH/olivia_WUR/process_data/Canopy coverage.csv", 2021)

    data_canopy = canopy_df[
        ['timestamp', 'plot.UID', 'value', 'genotype.id', 'plot.row_global', 'plot.range_global', 'lot']]

    height_df = data_load("C:/data_from_paper/ETH/olivia_WUR/process_data/Plant height.csv", 2021)
    data_height = height_df[
        ['timestamp', 'plot.UID', 'value', 'genotype.id', 'plot.row_global', 'plot.range_global', 'lot']]
    # tensor_dataset,n_features,Y_tensor,label_number,yield_tensor = create_tensor_dataset([data_canopy,data_height])
    tensor_dataset, n_features, Y_tensor, label_number = create_tensor_dataset([data_canopy, data_height],
                                                                               'genotype_predict')
    # save tensor X and Y
    tensor_dataset = torch.nan_to_num(tensor_dataset)
    '''
    tensor_dataset = masked_tensor(tensor_dataset)
    '''
    print(torch.isnan(tensor_dataset).any())

    # save height and canopy 2019 after processing
    curve_canopy = pd.DataFrame(tensor_dataset[:, :, 0].numpy())
    # print(curve_canopy)
    # curve_canopy.to_csv("canopy_curve_fill_na_2019.csv")
    curve_height = pd.DataFrame(tensor_dataset[:, :, 1].numpy())
    # print(curve_height)
    # curve_height.to_csv("height_curve_fill_na_2019.csv")

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(tensor_dataset, Y_tensor, test_size=0.3)
    try:
        cross_validation_result = pd.read_csv("loss.csv", header=0, index_col=0)
    except:
        cross_validation_result = pd.DataFrame()
    # define model lr 0.001,0.005,0.0005
    for hidden in [3, 5, 7, 9]:
        lstm_model = LSTM_fit_curve_genotype_predict(input_size=2, hidden_size=hidden, output_size=y_train.shape[1])
        count_parameters(lstm_model)
        for lr in [0.005, 0.001, 0.01, 0.0001]:
            for batch_size in [100, 50]:
                print("hidden size {}".format(hidden))
                print('learning rate:{}'.format(lr))
                print('batch_size:{}'.format(batch_size))
                # training
                model, output_curve, predict_genotype, train_loss, curve_loss, genotype_loss = training(lstm_model,
                                                                                                        X_train,
                                                                                                        y_train,
                                                                                                        batch_size,
                                                                                                        y_train, 300,
                                                                                                        lr=lr,
                                                                                                        optimize='Adam')
                fit_curve_canopy = pd.DataFrame(output_curve[:, :, 0].numpy())
                print(fit_curve_canopy)
                # fit_curve_canopy.to_csv("canopy_curve{}_lr{}_batch_size{}_fit_curve.csv".format(hidden,lr,batch_size))
                fit_curve_height = pd.DataFrame(output_curve[:, :, 1].numpy())
                print(fit_curve_height)
                # fit_curve_height.to_csv("height_curve{}_lr{}_batch_size{}_fit_curve.csv".format(hidden,lr,batch_size))
                predict_genotype = torch.round(predict_genotype)
                # print(predict_genotype.shape)
                predict_label = convert_one_hot_endoding_to_label(predict_genotype)
                true_label = convert_one_hot_endoding_to_label(Y_tensor)

                accuracy = accuracy_score(true_label, predict_label)
                print('training genotype accuracy{}'.format(accuracy))

                # test procedure
                print('----------test result---------')
                output_curve, output_label = model(X_test, y_test)

                fit_curve_canopy = pd.DataFrame(output_curve[:, :, 0].numpy())
                print(fit_curve_canopy)
                fit_curve_height = pd.DataFrame(output_curve[:, :, 1].numpy())
                print(fit_curve_height)
                # fit_curve_canopy.to_csv("canopy_curve{}_lr{}_batch_size{}_fit_curve.csv".format(hidden,lr,batch_size))
                # fit_curve_height.to_csv("height_curve{}_lr{}_batch_size{}_fit_curve.csv".format(hidden,lr,batch_size))
                predict_genotype = torch.round(output_label)
                # print(predict_genotype.shape)
                predict_label = convert_one_hot_endoding_to_label(predict_genotype)
                true_label = convert_one_hot_endoding_to_label(Y_tensor)
                test_accuracy = accuracy_score(true_label, predict_label)
                print('test loss:{}'.format(test_accuracy))
                new_row = pd.DataFrame(
                    data={"lr": lr,
                          "batch_size": batch_size,
                          "hidden_size": hidden,
                          'train_total': train_loss,
                          "test_total_loss": test_accuracy
                          },
                    index=[0])
                cross_validation_result = pd.concat(
                    [cross_validation_result, new_row])
                cross_validation_result.to_csv("loss.csv")


def run_rf_model():
    # run random forest model, with different window size average
    rf_train = Random_forest_yield_prediction()
    rf_train.RF_model(n_split=5, n_validation=5, window_size=5)

    rf_train2 = Random_forest_yield_prediction()
    rf_train2.RF_model(n_split=5, n_validation=5, window_size=10)

    rf_train3 = Random_forest_yield_prediction()
    rf_train3.RF_model(n_split=5, n_validation=5, window_size='last_time_step')

    rf_train1 = Random_forest_yield_prediction()
    rf_train1.RF_model(n_split=5, n_validation=5, window_size='all_time_step')


def run_lm_model():
    # run linear regression, with different window size average
    lm_train0 = Random_forest_yield_prediction()
    lm_train0.linear_regression(n_split=5, n_validation=5, window_size=5)

    lm_train = Random_forest_yield_prediction()
    lm_train.linear_regression(n_split=5, n_validation=5, window_size=10)

    lm_train2 = Random_forest_yield_prediction()
    lm_train2.linear_regression(n_split=5, n_validation=5, window_size='last_time_step')

    rf_train1 = Random_forest_yield_prediction()
    rf_train1.linear_regression(n_split=5, n_validation=5, window_size='all_time_step')

    '''
    canopy = pd.read_csv('../processed_data/align_canopy_drop_na.csv', header=0, index_col=0) #data after align canopy coverage and plant height
    height = pd.read_csv('../processed_data/align_height_drop_na.csv', header=0, index_col=0)
    dfs=average_based_on_days([canopy,height],5)

    rf_train1 = Random_forest_yield_prediction()
    rf_train1.linear_regression(n_split=5,n_validation=5,window_size=5,data=dfs,year=2019)

    dfs=average_based_on_days([canopy,height],10)
    rf_train1 = Random_forest_yield_prediction()
    rf_train1.linear_regression(n_split=5,n_validation=5,window_size=10,data=dfs,year=2019)

    dfs=[canopy,height]
    rf_train1 = Random_forest_yield_prediction()
    rf_train1.linear_regression(n_split=5,n_validation=5,window_size='last_time_step',data=dfs,year=2019)

    dfs=[canopy,height]
    rf_train1 = Random_forest_yield_prediction()
    rf_train1.linear_regression(n_split=5,n_validation=5,window_size='all_time_step',data=dfs,year=2019)
    '''


def align_dfs():
    #
    #
    # canopy,height = load_multiple_year_data()
    # dfs=average_based_on_days([canopy,height],5)
    # # import dill
    # # with open('average_dfs.dill', 'rb') as file:
    # #     dfs=dill.load(file)
    # # file.close()
    #
    # rf_train1 = Random_forest_yield_prediction()
    # rf_train1.multiple_year_rf_model(n_split=5,n_validation=5,window_size=5,data=dfs,years=[2018,2019,2021])
    #
    # canopy,height = load_multiple_year_data()
    # dfs=average_based_on_days([canopy,height],10)
    # # import dill
    # # with open('average_dfs.dill', 'rb') as file:
    # #     dfs=dill.load(file)
    # # file.close()
    #
    # rf_train1 = Random_forest_yield_prediction()
    # rf_train1.multiple_year_rf_model(n_split=5,n_validation=5,window_size=10,data=dfs,years=[2018,2019,2021])
    #
    # canopy = pd.read_csv('align_canopy.csv', header=0, index_col=0)
    # height = pd.read_csv('align_height.csv', header=0, index_col=0)
    # dfs=average_based_on_days([canopy,height],20)
    #
    # rf_train1 = Random_forest_yield_prediction()
    # rf_train1.multiple_year_rf_model(n_split=5,n_validation=5,window_size=20,data=dfs,years=[2018,2019,2021])

    canopy = pd.read_csv('../processed_data/align_canopy_drop_na.csv', header=0, index_col=0)
    height = pd.read_csv('../processed_data/align_height_drop_na.csv', header=0, index_col=0)
    dfs = average_based_on_days([canopy, height], 5)
    rf_train1 = Random_forest_yield_prediction()
    rf_train1.multiple_year_rf_model(n_split=5, n_validation=5, window_size='last_time_step', data=dfs,
                                     years=[2018, 2019, 2021])

    canopy = pd.read_csv('../processed_data/align_canopy_drop_na.csv', header=0, index_col=0)
    height = pd.read_csv('../processed_data/align_height_drop_na.csv', header=0, index_col=0)
    dfs = [canopy, height]
    rf_train1 = Random_forest_yield_prediction()
    rf_train1.multiple_year_rf_model(n_split=5, n_validation=5, window_size='last_time_step', data=dfs,
                                     years=[2018, 2019, 2021])
    canopy = pd.read_csv('../processed_data/align_canopy_drop_na.csv', header=0, index_col=0)
    height = pd.read_csv('../processed_data/align_height_drop_na.csv', header=0, index_col=0)
    dfs = [canopy, height]
    rf_train1 = Random_forest_yield_prediction()
    rf_train1.multiple_year_rf_model(n_split=5, n_validation=5, window_size='all_time_step', data=dfs,
                                     years=[2018, 2019, 2021])


def run_lstm_models(save_result_name='', ):
    n = 5  # n_split for train and test, which the hold out(validation set) is the same
    cross_validation = pd.DataFrame()
    cross_validation2 = pd.DataFrame()
    cross_validation_one_layer = pd.DataFrame()
    validation_num = 5
    for validation_split in range(validation_num):
        start_time = time.time()
        print(multiple_years_yield(n_split=n, year=[2019], random_seed_for_validation_split=validation_split))
        for i, (X, Y, position, genotype) in enumerate(
                multiple_years_yield(n_split=n, year=[2019], random_seed_for_validation_split=validation_split)):
            print('________training fold {}/{}________'.format(i, n))
            # fill na
            X_validation = torch.nan_to_num(X[-1])
            y_validation = torch.nan_to_num(Y[-1])
            position_tensor_validation = torch.nan_to_num(position[-1])
            genotype_tensor_validation = torch.nan_to_num(genotype[-1])
            plot_growth_curve_colored_based_on_yield(X_validation[:, :, 0], y_validation)

            '''
            #mask na
            X_validation = masked_tensor(X[-1])
            y_validation = masked_tensor(Y[-1])
            position_tensor_validation = masked_tensor(position[-1])
            genotype_tensor_validation = masked_tensor(genotype[-1])
            '''

            # standard scalar
            scaled_Y_validation_tensor, scaler_validation = minmax_scaler(y_validation)
            scaled_position_validation_tensor, _ = minmax_scaler(position_tensor_validation)

            print(X_validation.shape)
            best_test_one_layer, best_parameters_one_layer, best_model_one_layer = yield_prediction_one_year(X, Y,
                                                                                                             position,
                                                                                                             genotype,
                                                                                                             with_position_str='one_layer_lstm')
            # validate result

            MSE_loss, spearmanr, p = test_result(best_model_one_layer, X_validation, scaled_Y_validation_tensor,
                                                 scaler_validation,
                                                 scaled_position_validation_tensor, with_position=False,
                                                 file_name='one_layer_no_pos_validation{}_train_fold_{}'.format(
                                                     validation_split, i, save_result_name))

            print('validation result:{}'.format(MSE_loss, spearmanr))
            best_parameters_one_layer['validation_MSE'] = MSE_loss
            best_parameters_one_layer['spearmanr_validation'] = spearmanr
            best_parameters_one_layer['validation_rep'] = validation_split

        #     best_test, best_parameters, best_model = yield_prediction_one_year(X, Y, position, genotype,
        #                                                                        with_position_str='')
        #     # validate result
        #
        #     MSE_loss, spearmanr, p = test_result(best_model, X_validation, scaled_Y_validation_tensor,
        #                                          scaler_validation,
        #                                          scaled_position_validation_tensor, with_position=False,file_name='no_pos_validation{}_train_fold{}_{}'.format(validation_split,i,save_result_name))
        #
        #     best_parameters['validation_MSE'] = MSE_loss
        #     best_parameters['spearmanr_validation'] = spearmanr
        #     best_parameters['validation_rep'] = validation_split
        #     best_test2, best_parameters2, best_model2 = yield_prediction_one_year(X, Y, position, genotype,
        #                                                                           with_position_str='with_scaled_position')
        #     # validate result
        #
        #     MSE_loss, spearmanr, p = test_result(best_model2, X_validation, scaled_Y_validation_tensor,
        #                                          scaler_validation,
        #                                          scaled_position_validation_tensor, with_position=True,file_name='with_pos_validation{}_train_fold{}_{}'.format(validation_split,i,save_result_name))
        #
        #     best_parameters2['validation_MSE'] = MSE_loss
        #     best_parameters2['spearmanr_validation'] = spearmanr
        #     best_parameters2['validation_rep'] = validation_split
        #
        #     # save in dataframe
        #     new_row_one_layer = pd.DataFrame(data=best_parameters_one_layer, index=[i])
        #     cross_validation_one_layer = pd.concat([cross_validation_one_layer, new_row_one_layer])
        #
        #     new_row = pd.DataFrame(data=best_parameters, index=[i])
        #     cross_validation = pd.concat([cross_validation, new_row])
        #
        #     new_row2 = pd.DataFrame(data=best_parameters2, index=[i])
        #     cross_validation2 = pd.concat([cross_validation2, new_row2])
        #     cross_validation_one_layer.to_csv('best_test_one_layer_spearmanr_align_{}.csv'.format(save_result_name))
        #     cross_validation.to_csv('best_test_1_spearmanr_align_{}.csv'.format(save_result_name))
        #     cross_validation2.to_csv('best_test_2_spearmanr_align_{}.csv'.format(save_result_name))
        #
        #     with open('model/model_one_layer{}_spearmanr_align_{}.dill'.format(i,save_result_name), 'wb') as file:
        #         dill.dump(best_model_one_layer, file)
        #     file.close()
        #     with open('model/model_{}_spearmanr_align_{}.dill'.format(i,save_result_name), 'wb') as file:
        #         dill.dump(best_model, file)
        #     file.close()
        #     with open('model/model_{}_with_position_spearmanr_align_{}.dill'.format(i,save_result_name), 'wb') as file:
        #         dill.dump(best_model2, file)
        #     file.close()
        #
        #     with open('model/validation_{}_align_{}.dill'.format(validation_split,save_result_name), 'wb') as file:
        #         dill.dump([X_validation, y_validation, position_tensor_validation, genotype_tensor_validation], file)
        #     file.close()
        # else:
        #     print('##########{}/{} validation finished########'.format(validation_split, validation_num))
        #     print('training time: {}'.format(time.time() - start_time))
        # calculate mean error inside this validation


def main():
    run_lstm_models()
    # run_lm_model()
    # run_rf_model()
    # align_dfs()


if __name__ == '__main__':
    main()