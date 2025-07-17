import pandas as pd
import dill
import matplotlib.pyplot as plt
import statsmodels.api as sm
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from prettytable import PrettyTable
import torch.optim as optim


from DataPrepare import multiple_years_yield, minmax_scaler, test_result,count_parameters,mean_yield_benchmark,onelayer_LSTM_yield_prediction
import math
class Autocorrelation_analysis():
    """
    analysis autocorrelation in innput x
    """
    def __init__(self, input_x):

        # self.plot_autocorrelation(input_x)
        self.AR_model(input_x,23,46)

    def plot_autocorrelation(self,data):
        """
        average ACF and PACF for plant height and canopy coverage, respectively
        """
        acf_plant_height_df = pd.DataFrame()
        pacf_plant_height_df = pd.DataFrame()
        ci_plant_height_df_acf = pd.DataFrame() #confident interval
        ci_plant_height_df_pacf = pd.DataFrame()
        for sample in range(data.shape[0]):
            #plant height
            #https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acf.html
            acf, acf_ci = sm.tsa.acf(data[sample,:,-1].squeeze(),nlags=(data.shape[1]-1),missing="conservative",alpha=0.05)
            pacf, pacf_ci = sm.tsa.pacf(data[sample,:,-1].squeeze(),nlags=(int(data.shape[1]/2)-1),alpha=0.05)

            try:
                acf_col_names != None
                pacf_col_names != None
            except:
                acf_col_names = ['lag'+str(x) for x in range(len(acf))]
                pacf_col_names = ['lag' + str(x) for x in range(len(pacf))]

            acf_new_row = pd.DataFrame(data=acf)
            acf_plant_height_df = pd.concat([acf_plant_height_df, acf_new_row],axis=1)

            pacf_new_row = pd.DataFrame(data=pacf)
            pacf_plant_height_df = pd.concat([pacf_plant_height_df, pacf_new_row],axis=1)

            acf_new_row_ci = pd.DataFrame(data=acf-acf_ci[:,1].squeeze())
            ci_plant_height_df_acf = pd.concat([ci_plant_height_df_acf, acf_new_row_ci],axis=1)
            pacf_new_row_ci = pd.DataFrame(data=pacf-pacf_ci[:,1].squeeze())

            ci_plant_height_df_pacf = pd.concat([ci_plant_height_df_pacf, pacf_new_row_ci],axis=1)
        else:
            acf_plant_height_df = acf_plant_height_df.T.mean(axis=0)
            acf_plant_height_df.columns = acf_col_names
            pacf_plant_height_df = pacf_plant_height_df.T.mean(axis=0)
            pacf_plant_height_df.columns = pacf_col_names

            ci_plant_height_df_acf = ci_plant_height_df_acf.T.mean(axis=0)
            ci_plant_height_df_pacf = ci_plant_height_df_pacf.T.mean(axis=0)


        acf_canopy_coverage_df = pd.DataFrame()
        pacf_canopy_coverage_df = pd.DataFrame()
        ci_canopy_coverage_df_acf = pd.DataFrame() #confident interval
        ci_canopy_coverage_df_pacf = pd.DataFrame()
        for sample in range(data.shape[0]):
            #canopy coverage
            #https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acf.html
            acf,acf_ci = sm.tsa.acf(data[sample,:,0].squeeze(),nlags=(data.shape[1]-1),missing="conservative",alpha=0.05)

            pacf,pacf_ci = sm.tsa.pacf(data[sample,:,0].squeeze(),nlags=(int(data.shape[1]/2)-1),alpha=0.05)
            try:
                acf_col_names != None
                pacf_col_names != None
            except:
                acf_col_names = ['lag'+str(x) for x in range(len(acf))]
                pacf_col_names = ['lag' + str(x) for x in range(len(pacf))]

            acf_new_row = pd.DataFrame(data=acf)
            acf_canopy_coverage_df = pd.concat([acf_canopy_coverage_df, acf_new_row],axis=1)
            pacf_new_row = pd.DataFrame(data=pacf)
            pacf_canopy_coverage_df = pd.concat([pacf_canopy_coverage_df, pacf_new_row],axis=1)
            #CI
            acf_new_row_ci = pd.DataFrame(data=acf-acf_ci[:,1].squeeze())
            ci_canopy_coverage_df_acf = pd.concat([ci_canopy_coverage_df_acf, acf_new_row_ci],axis=1)
            pacf_new_row_ci = pd.DataFrame(data=pacf-pacf_ci[:,1].squeeze())
            ci_canopy_coverage_df_pacf = pd.concat([ci_canopy_coverage_df_pacf, pacf_new_row_ci],axis=1)
        else:

            acf_canopy_coverage_df = acf_canopy_coverage_df.T.mean(axis=0)
            acf_canopy_coverage_df.columns = acf_col_names
            pacf_canopy_coverage_df = pacf_canopy_coverage_df.T.mean(axis=0)
            pacf_canopy_coverage_df.columns = pacf_col_names
            #confident interval tranpose
            ci_canopy_coverage_df_acf = ci_canopy_coverage_df_acf.T.mean(axis=0)
            ci_canopy_coverage_df_pacf = ci_canopy_coverage_df_pacf.T.mean(axis=0)


        print(acf_canopy_coverage_df)
        print(ci_canopy_coverage_df_acf)
        print((acf_canopy_coverage_df - ci_canopy_coverage_df_acf))

        # https://www.statsmodels.org/stable/_modules/statsmodels/graphics/tsaplots.html#plot_acf
        fig, axs = plt.subplots(2)
        fig.suptitle('ACF')
        axs[0].stem(acf_plant_height_df)
        axs[0].set_title('plant height')
        axs[0].fill_between(acf_plant_height_df.index, (-ci_plant_height_df_acf),
                            ci_plant_height_df_acf, color='b', alpha=.1)
        axs[0].set_xlabel('lag')
        axs[1].stem(acf_canopy_coverage_df)
        axs[1].set_title('canopy coverage')

        axs[1].fill_between(acf_canopy_coverage_df.index, (- ci_canopy_coverage_df_acf),
                            ci_canopy_coverage_df_acf, color='b', alpha=.1)
        axs[1].set_xlabel('lag')
        plt.tight_layout()
        plt.savefig('ACF plot.jpg',dpi=2400)
        # plt.show()

        fig, axs = plt.subplots(2)
        fig.suptitle('PACF')
        axs[0].stem(pacf_plant_height_df)
        axs[0].set_title('plant height')
        axs[0].fill_between(pacf_plant_height_df.index, (- ci_plant_height_df_pacf),
                            ci_plant_height_df_pacf, color='b', alpha=.1)
        axs[0].set_xlabel('lag')
        axs[1].stem(pacf_canopy_coverage_df)
        axs[1].set_title('canopy coverage')
        axs[1].fill_between(pacf_canopy_coverage_df.index, (- ci_canopy_coverage_df_pacf),
                            ci_canopy_coverage_df_pacf, color='b', alpha=.1) #alpha control the transparent
        axs[1].set_xlabel('lag')
        plt.tight_layout()
        plt.savefig('PACF plot.jpg',dpi=2400)
        # plt.show()

    def AR_model(self, sample, train_len, predict_len):
        # raise ValueError ('exponential prediction result')
        #inspired by https://www.kaggle.com/code/iamleonie/time-series-interpreting-acf-and-pacf
        from statsmodels.tsa.ar_model import AutoReg
        import seaborn as sns
        sample = torch.squeeze(sample[0,:,1]).numpy()
        train = sample[:train_len]

        print(train.shape)

        ar_model = AutoReg(train, lags=[1,2,5]).fit()#-15 out of index if use lags=[1,15]

        print(ar_model.summary())

        pred = ar_model.predict(start=train_len, end=predict_len, dynamic=False)
        print(pred)
        print(sample[train_len:predict_len+1])
        f, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4))
        sns.lineplot(x=range(train_len,predict_len), y=sample[train_len:predict_len+1], marker='o',
                     label='test', color='grey')
        sns.lineplot(x=range(train_len), y=sample[:train_len], marker='o', label='train')
        sns.lineplot(x=range(train_len,predict_len+1), y=pred, marker='o', label='pred')

        ax.set_title('Sample 0 Time Series')
        plt.tight_layout()
        plt.show()

class Autoregression(nn.Module):
    def __init__(self,input_feature):
        super().__init__()
        self.linear = nn.Linear(input_feature,input_feature)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(input_feature,1)

    def forecasting(self,input_x):
        output = self.linear(input_x)
        output = self.sigmoid(output)
        return output
    def yield_predict(self,input_x):
        output = self.fc(input_x)
        output = self.sigmoid(output)
        return output
    def forward(self,input_x):
        forecasting_out = input_x[:,0,:]

        list_predict = []
        for t in range(input_x.shape[1]-1):
            forecasting_out = self.forecasting(forecasting_out)
            list_predict.append(forecasting_out)
        else:
            out_yield = self.yield_predict(forecasting_out)
            forecast_out = torch.stack(list_predict).float()
            # print(forecasting_out.shape)
            forecast_out = forecast_out.permute((1, 0, 2)).float()
            return forecast_out,out_yield.float()
    def init_network(self):
        # initialize weight and bias(use xavier initialization for weight and bias )
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param,0.00)

def training_autoregression(model,lr,x:torch.tensor,y:torch.tensor,epoch:int,batch_size:int,optimize:str):
    num_sequences = x.shape[0]
    print('number of sequence{}'.format(num_sequences))
    seq_length = x.shape[1]
    # Define training parameters
    learning_rate = lr  #
    num_epochs = epoch

    # Convert input and target data to PyTorch datasets
    x = torch.nan_to_num(x)
    print(x.shape)
    print(y.shape)

    dataset = TensorDataset(x, y)

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
    x_axis = []
    # plt.ion()
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    criterian = nn.MSELoss()
    for epoch in range(num_epochs):
        running_loss = 0.0  # running loss for every epoch

        for i, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.float()
            targets = targets.float()
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass

            forecast, yield_predict = model(inputs.float())  # batch first, the input is (110,46,2) for 2019
            loss1 = criterian(targets, yield_predict)
            loss2 = criterian(forecast,inputs[:,1:,:])
            loss = (loss1+loss2).float()
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.shape[1]  # loss.item() is the mean loss of the total batch
        if (epoch + 1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs,
                                                       running_loss/(num_sequences)))
            x_axis.append(epoch + 1)
            loss_list.append(running_loss / (num_sequences))
            # line1, = ax.plot(x_axis, loss_list, c='blue')
            # fig.canvas.draw_idle()
            # fig.canvas.flush_events()
            # plt.show(block=False)
    else:
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            forecast,predict_yield = model(x.float())
            loss = criterian(y,predict_yield)  #
            return predict_yield, model, loss

def hyperparameters_selection(X,Y):
    model = Autoregression(2)

    # for tensor_dataset,Y_tensor,position_tensor,genotype_tensor in zip(X,Y,position,genotype):
    # print(X)
    print(len(X))
    X_train, X_test,X_validation=X
    y_train, y_test,y_validation=Y
    X_train = torch.nan_to_num(X_train)
    X_test = torch.nan_to_num(X_test)
    y_train_scaled, train_scaler = minmax_scaler(y_train)
    scaled_Y_test_tensor,scaler_y_test = minmax_scaler(y_test)


    assert torch.isnan(X_train).any() ==False

    try:
        yield_result = pd.read_csv("yield{}_MSE_loss.csv".format('autoregression'), header=0, index_col=0)
    except:
        yield_result = pd.DataFrame()

    # define model lr 0.001,0.005,0.0005
    best_test = -1.0
    best_parameters={}
    best_model = None
    model = Autoregression(input_feature=2)
    count_parameters(model)
    print('X train shape{}'.format(X_train.shape)) #[110, 46, 2]
    print('yield train shape{}'.format(y_train.shape))
    print('###########################{}'.format(y_train_scaled.shape[1]))
    # count_parameters(model)
    for lr in [0.005,0.001, 0.01, 0.0001]: #
        for batch_size in [100,50,30]:#
            for epoch in [100,300,500]: #
                print('learning rate:{}'.format(lr))
                print('batch_size:{}'.format(batch_size))
                print('epoch:{}'.format(epoch))
                train_predict_yield, model, train_MSE_loss = training_autoregression(model=model, lr=lr,
                                                                                     x=X_train,
                                                                                     y=y_train_scaled, epoch=epoch,
                                                                                     batch_size=batch_size, optimize='Adam')
                # test MSE
                MSE_loss,spearmanr,p = test_result(model, X_test, scaled_Y_test_tensor, scaler=scaler_y_test,with_position=False)
                if spearmanr>best_test:
                    best_test = spearmanr
                    best_parameters={"lr": lr,
                          "batch_size": batch_size,
                          'epoch':epoch,
                          'train_loss': train_MSE_loss,
                          "test_MSE_loss": MSE_loss,
                          'test_spearman_rank':spearmanr,
                          'test_pvalue':p}
                    best_model = model

                print('test loss:{}'.format(MSE_loss))
                new_row = pd.DataFrame(
                    data=best_parameters,
                    index=[0])
                yield_result = pd.concat(
                    [yield_result, new_row])
                yield_result.to_csv('yield{}_MSE_loss.csv'.format('autoregression'))
    else:
        return best_test,best_parameters,best_model

def main():

    import dill
    n=5 # n_split for train and test, which the hold out(validation set) is the same
    cross_validation = pd.DataFrame()

    validation_num=5
    for validation_split in range(validation_num):
        for i,(X,Y,position,genotype) in enumerate(multiple_years_yield(n_split=n,year=2019,random_seed_for_validation_split=validation_split)):

            # plot autocorrelation
            all_X = torch.cat([X[0],X[1],X[2]],dim=0)
            all_Y = torch.cat([Y[0],Y[1],Y[2]],dim=0)
            mean_yield_benchmark(all_Y)

            autorregression = Autocorrelation_analysis(all_X)

            print('________training fold {}/{}________'.format(i,n))
            # fill na
            X_validation = torch.nan_to_num(X[-1])
            y_validation = torch.nan_to_num(Y[-1])
            lstm_model = onelayer_LSTM_yield_prediction(input_size=2, hidden_size=5,
                                                        output_size=1, batch_first=True)
            from Plot_analysis_result import plot_loss

            plot_loss(lstm_model, torch.nan_to_num(X[0]), torch.nan_to_num(Y[0]), position[0], genotype[0], with_position='')
            '''
            #standard scalar for validation
            scaled_Y_validation_tensor, scaler_validation= minmax_scaler(y_validation)

            best_test, best_parameters, best_model = hyperparameters_selection(X,Y)

            MSE_loss, spearmanr, p = test_result(best_model, X_validation, scaled_Y_validation_tensor,
                                                 scaler_validation, with_position=False)
            #add validation result to dictionary
            best_parameters['validation_MSE'] = MSE_loss
            best_parameters['spearmanr_validation'] = spearmanr
            best_parameters['validation_rep'] = validation_split
            new_row2 = pd.DataFrame(data=best_parameters, index=[i])
            cross_validation = pd.concat([cross_validation, new_row2])
            cross_validation.to_csv('best_test_2_mse_spearman.csv')

            with open('model/model_{}_with_position_mse_spearman.dill'.format(i), 'wb') as file:
                dill.dump(best_model, file)
            file.close()

            with open('model/validation_{}_autoregression.dill'.format(validation_split), 'wb') as file:
                dill.dump([X_validation,y_validation], file)
            file.close()
            '''
        else:
            print('##########{}/{} validation finished########'.format(validation_split,validation_num))
            #calculate mean error inside this validation





if __name__ == '__main__':
    main()