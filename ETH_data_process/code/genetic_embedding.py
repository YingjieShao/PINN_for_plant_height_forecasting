import math
import dill
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast
from visualize_data import find_genotype_present_at_multiple_years,genotype_present_in_specific_years,genotype_present_in_at_least_in_one_of_selected_years
import blosum as bl
from pard.grantham import grantham
import re
import torch
import torch.nn as nn
from DataPrepare import minmax_scaler, count_parameters
from NNmodel_training import plot_loss_change
import torch.optim as optim
import copy
import random
import subprocess
import sys
import wandb
DEVICE = torch.device("cpu")
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  #device cup or gpu

def transform_AA_mutation(value):
    """
    convert 'A->G' to 'AG' otherwise set to na
    """

    pattern =  r'^([A-Z])->([A-Z])$'
    match = re.fullmatch(pattern, value)
    if match:
        return match.group(1) + match.group(2)  # change 'A->G' to 'AG'
    else:
        return np.nan  # Set non-matching values to NaN
def encode_aminoacid_changes_blosum_62_matrix(snp_info_df:pd.DataFrame):
    """
    calculate and save blosum 62 score for AA mutation: https://www.nlm.nih.gov/ncbi/workshops/2023-08_BLAST_evol/blast_score.html
    The numbers in BLOSUM62 are log odds ratios of the observed substitution frequency to the background frequency.
    score of self match: common AA has lower score -3 to 9
    """
    bl_matrix = bl.BLOSUM(62) #import blosum 62 scoring matrix as dictionary
    # print(bl_matrix)
    def get_blosum_score(aa_str):
        if isinstance(aa_str,str): #otherwise will be np.nan
            # print(aa_str)
            assert len(aa_str)==2 #two amino acid
            a1= aa_str[0]
            a2 = aa_str[1]
            value = bl_matrix[a1][a2]

        else:
            value = np.nan
        # print(value)
        return value
    snp_info_df['blosum_score'] = snp_info_df['Amino_Acid_Change'].apply(get_blosum_score)
    return snp_info_df

def encode_aminoacid_changes_grantham_score(snp_info_df:pd.DataFrame):
    """
    The higher the distance, the more deleterious the substitution is expected to be. 0-205
    https://aacrjournals.org/cebp/article/14/11/2598/257044/The-Predicted-Impact-of-Coding-Single-Nucleotide
    """
    bl_matrix = bl.BLOSUM(62) #import blosum 62 scoring matrix as dictionary
    # print(bl_matrix)
    def get_grantham_score(aa_str):
        if isinstance(aa_str,str): #otherwise will be np.nan
            # print(aa_str)
            assert len(aa_str)==2 #two amino acid
            a1= aa_str[0]
            a2 = aa_str[1]
            value = grantham(a1, a2)
        else:
            value = np.nan
        # print(value)
        return value
    snp_info_df['grantham_score'] = snp_info_df['Amino_Acid_Change'].apply(get_grantham_score)
    return snp_info_df

def read_SNPs_file_based_on_genotype_list(genotype_list, ordinal_encoded=True):
    """
    read 90k chips snps information and save selected genotype in ../processed data'
    """
    #match genotype id with genotype name
    genotype_name_df = pd.read_csv('C:\data_from_paper/ETH/olivia_WUR/genotypes.csv',header=0)[['genotype.id','genotype.name','genotype.keys']]
    genotype_name_df = genotype_name_df[genotype_name_df['genotype.id'].isin(genotype_list)].reset_index(drop=True)
    genotype_name_df['genotype.keys'] = genotype_name_df['genotype.keys'].apply(ast.literal_eval)
    genotype_name_df['GABI_code'] = genotype_name_df['genotype.keys'].apply(lambda x: x.get('GABI'))
    genotype_id_list = genotype_name_df['genotype.id'].astype(int).to_list()
    gabi_id_list = genotype_name_df['GABI_code'].to_list()
    dict_gabi_id= dict(zip(genotype_name_df['GABI_code'], genotype_name_df['genotype.id']))
    print(genotype_name_df)
    #maybe use to merge with files from other source

    print(genotype_id_list)
    # raise EOFError
    genotype_name_df.to_csv('../processed_data/genotypes_filtered.csv')
    if ordinal_encoded:
        #this is from ETH genetics imputed file which already convert to 0,1,2..
        ETH_data_snps ='C:\data_from_paper/ETH/olivia_WUR/GABI_marker_data_imputed.csv'
        snp_genotype_df = pd.read_csv(ETH_data_snps,header=0,index_col=0)
        snp_genotype_df = snp_genotype_df[snp_genotype_df['genotype_id'].isin(genotype_list)]
        snp_genotype_df.index = snp_genotype_df['genotype_id'].astype(int)
        genotype_id_list = snp_genotype_df['genotype_id'].astype(int).to_list()
        print(genotype_id_list)
        snp_genotype_df.drop(columns=['genotype_id','genotype_GABI'],inplace=True)#last two columns are GABI key and genotype id
        print(snp_genotype_df)
        print('find overlap: genotype_ids with GABI key')
        print(len(snp_genotype_df.index))
        print('locus number:')
        locus_names = snp_genotype_df.columns.to_list()
        print(len(locus_names))
        locus_names = ["_".join(i.split('_')[:-2]) for i in locus_names]
        snp_genotype_df = snp_genotype_df.T
        snp_genotype_df['Locus_Name'] = locus_names
        snp_genotype_df.reset_index(inplace=True,drop=True)
        snp_genotype_df = snp_genotype_df.drop_duplicates(subset=genotype_id_list) #if a snps is identical in all genotyep plants, drop it
        print(snp_genotype_df) #4535 snps left after drop duplicate
        snp_genotype_df.to_csv('../processed_data/GABI_marker_data_imputed_filtered.csv')
    else:
        #otherwise use snps infromation for single alle(did not encode to 0,1,2)
        #read from GABI_WHEAT_90k.txt
        snp_genotype_df = pd.read_csv('C:\data_from_paper/ETH/olivia_WUR/genotype_markers/GABI_WHEAT_90k.txt',sep=' ',header=0,index_col=0).T
        # snp_genotype_df['GABI_code'] = snp_genotype_df.index #index is different cultivars columns are snps name
        print(snp_genotype_df)
        snp_genotype_df = snp_genotype_df[snp_genotype_df.index.isin(gabi_id_list)]#should be 19
        #change gabi key to genotype id
        snp_genotype_df.index = snp_genotype_df.index.map(dict_gabi_id)
        print(snp_genotype_df)
        print('locus number:')
        locus_names = snp_genotype_df.columns.to_list()
        print(len(locus_names))
        snp_genotype_df = snp_genotype_df.T
        snp_genotype_df['Locus_Name'] = locus_names
        snp_genotype_df.reset_index(inplace=True, drop=True)


        # raise EOFError
        genotype_id_list = snp_genotype_df.columns.to_list().remove('Locus_Name')
        print(genotype_id_list)
        # print(locus_names)
        snp_genotype_df = snp_genotype_df.drop_duplicates(subset=genotype_id_list) #if a snps is identical in all genotyep plants, drop it
        # print(snp_genotype_df)
        #convert 'failed' to na
        snp_genotype_df = snp_genotype_df.replace('failed', np.nan)
        #then drop if the whole row is na
        snp_genotype_df = snp_genotype_df.dropna(subset=genotype_id_list, how='all')
        print('snps number:{}'.format(len(snp_genotype_df.index)))
        raise EOFError
        snp_genotype_df.to_csv('../processed_data/GABI_WHEAT_90k_filtered.csv')

    #read snps information location, value
    snps_info_file = 'C:\data_from_paper/ETH/olivia_WUR/genotype_markers/Annotation of SNP loci._90K.xlsx'
    snp_info_df = pd.read_excel(snps_info_file,header=1)
    # print(snp_info_df.columns)
    # snp_info_df = snp_info_df[snp_info_df['Locus_Name'].isin(locus_names)].reset_index(drop =True)
    # print(snp_info_df)
    #covert amino acide change into standard type in blosum matrix
    snp_info_df['Amino_Acid_Change'] = snp_info_df['Amino_Acid_Change'].apply(transform_AA_mutation)
    # print(snp_info_df['Amino_Acid_Change'])
    snp_info_df = encode_aminoacid_changes_blosum_62_matrix(snp_info_df)
    snp_info_df = encode_aminoacid_changes_grantham_score(snp_info_df)
    # print(snp_info_df)

    #read snp name and order based on consensus SNPs genetics link map
    snp_consensus_map = pd.read_csv('C:\data_from_paper/ETH/olivia_WUR/genotype_markers/Consensus SNP genetic linkage map for hexaploid wheat_90K.csv',header=1)
    snp_consensus_map['Locus_Name'] = snp_consensus_map['SNP Name']
    snp_consensus_map.drop(columns=['SNP Name'])
    # 19111 after merge, as the same snps are mapped to different position on consensus map
    snp_info_df = pd.merge(snp_info_df,snp_consensus_map,how='left',on=['Locus_Name'])
    snp_info_df=snp_info_df.sort_values(by='Order').reset_index(drop=True)
    snp_info_df = snp_info_df.dropna(subset=['Order'])
    snp_info_df.to_csv('../processed_data/90k_locus_info_filtered.csv')
    return genotype_id_list,snp_genotype_df,snp_info_df

def snps_encoding(snp_genotype_df,snp_info_df,genoype_list,scoring_method:tuple=('blosum62','grantham_score'),name=''):
    """
    encoding methods for snps input:
    use 0,1,2 and consider adding amino acide change (not hit as 0, synonymous as 1, nonsynonymous using blousm score),
    save to csv file with snps code and locus information, 18850 locus intotal
    :return: genotype snps information tensor saved in dictionary, which key is genotype id.
    """
    #one hot encoding dictionary
    #https://www.hgmd.cf.ac.uk/docs/nuc_lett.html
    one_hot_dictionary = {'A':[1,0,0,0],'G':[0,1,0,0],'C':[0,0,1,0],'T':[0,0,0,1],'R':[0.5,0.5,0,0],'Y':[0,0,0.5,0.5],
    'K':[0,0.5,0,0.5],'M':[0.5,0,0.5,0],'S':[0,0.5,0.5,0],'W':[0.5,0,0,0.5],'B':[0,0.3,0.3,0.3],'D':[0.3,0.3,0,0.3],
                          'H':[0.3,0,0.3,0.3],'V':[0.3,0.3,0.3,0],'N':[0.25,0.25,0.25,0.25],np.nan:[0,0,0,0]}
    print(snp_genotype_df)
    print(snp_info_df)
    # #drop drop snps whose value is the same for all genotype
    # drop_subset = snp_info_df.columns.to_list()
    # drop_subset.remove('Locus_Name')
    # print(drop_subset)
    # snp_info_df.drop_duplicates(inplace=True,subset=drop_subset)
    # print('after drop duplicate')
    # print(snp_info_df)
    #merge snp_info_df with snp_genotype_df based on Locus_name
    snp_merge_df = pd.merge(snp_genotype_df,snp_info_df,on='Locus_Name',how='inner')
    # print(snp_merge_df)

    # snp_merge_df.to_csv('../processed_data/merge_genotype_snps_input.csv')
    genotype_tensor_dictionary={}
    #convert pd dataframe to torch tensor for each genotype
    for genotype in genoype_list:
        # print(genotype)
        if name == '':
            #default use 0,1,2
            snp_value = snp_merge_df[genotype].to_numpy()
            # print(snp_value)
            snp_value_tensor = torch.tensor(snp_value).unsqueeze(dim=-1)
        elif name == 'one_hot':
            # print(genotype)
            np_snp =[]
            for nucleotide in snp_merge_df[genotype]:
                values = np.array(one_hot_dictionary[nucleotide])#length four
                # print(values)
                # raise EOFError
                # print(values.shape)
                np_snp.append(values)
            else:
                snp_value = np.vstack(np_snp) #(7637, 4)
            snp_value_tensor = torch.tensor(snp_value)#.unsqueeze(dim=-1)
            # print(snp_value_tensor)
        # print(snp_value_tensor.shape)
        if 'blosum62' in scoring_method:
            blosum_score = snp_merge_df[['blosum_score']].to_numpy()
            blosum_score = torch.tensor(blosum_score)
            # print('blosum')
            blosum_score,scaler =minmax_scaler(blosum_score)
            # print(blosum_score.shape)
            snp_value_tensor = torch.cat([snp_value_tensor,blosum_score],dim=-1)
            #min_max_scaling to between 0 to 1
        if 'grantham_score' in scoring_method:
            grantham_score = snp_merge_df[['grantham_score']].to_numpy()
            grantham_score = torch.tensor(grantham_score)
            # print('grantham_score')
            grantham_score, scaler = minmax_scaler(grantham_score)
            snp_value_tensor = torch.cat([snp_value_tensor, grantham_score], dim=-1)
        # print(snp_value_tensor.shape) #shape=[snps_num,feature_size]

        #save temporary files to csv and dill
        # if name == '':
        #     df = pd.DataFrame(data=snp_value_tensor,columns=['snp_value','blosum62_scaled','grantham_scaled'],index=range(snp_value_tensor.shape[0]))
        #     df.to_csv('../temporary/{}_{}.csv'.format(int(genotype),name))
        # elif name == 'one_hot':
        #     df = pd.DataFrame(data=snp_value_tensor,columns=['snp_value_A','snp_value_G','snp_value_C','snp_value_T','blosum62_scaled','grantham_scaled'],index=range(snp_value_tensor.shape[0]))
        #     df.to_csv('../temporary/{}_{}.csv'.format(int(genotype),name))
        if scoring_method!=():
            with open('../temporary/{}_{}_{}.dill'.format(genotype,name,"_".join(scoring_method)),'wb') as file1:
                dill.dump(snp_value_tensor,file1)
            file1.close()
        else:
            with open('../temporary/{}_{}.dill'.format(genotype,name),'wb') as file1:
                dill.dump(snp_value_tensor,file1)
            file1.close()
        genotype_tensor_dictionary[genotype] = snp_value_tensor #[snps_num,feature_size]
    else:
        return genotype_tensor_dictionary

def genetics_embedding_NN(method:int,):
    """
    This is the function to embed SNPs information into ODE parameters.
    """
    #METHOD 1: train together with PINN
        #1dCNN, all locus in one dimension,from chromosome 1A to 7D. [bathc_size,snps_number,feature_size]
        #2dCNN, [bathc_size,chr_num,snps_number,feature_size]

    #Method2: pretrain network for classification (use training set), then freeze parameter except last layer to predict r and ymax in PINN

    #Methods3: pretrain network for classification, take last layer output as input together with ts and env input for PINN.

class embedding_by_1dCNN(nn.Module):
    def __init__(self,snps_number,feature_size,out_channel_1,out_channel_2,kernel_size_1,kernel_size_2,max_pooling_stride,output_size=2,snp_enbed=''):
        super().__init__()
        self.vector_length = 6 #embedded size, embedded snps into a vector, and pretrained loss is to maximize distance among vectors
        self.seq_len = snps_number
        #use stride ==kernel size for CNN layer input shape: (N,C,L)
        self.cnn1 = nn.Conv1d(in_channels=feature_size,out_channels=out_channel_1,kernel_size=kernel_size_1,stride=kernel_size_1)
        self.leakyrelu = nn.LeakyReLU()
        out_size1 = int(
            ((self.seq_len - (kernel_size_1-1)-1) / kernel_size_1) + 1)  # output size for cnn layer
        print(out_size1)
        self.max_pooling1 = nn.MaxPool1d(kernel_size=kernel_size_1,stride=max_pooling_stride)#same kenerl size as cnn1
        pooling_out_size1 = int(
            ((out_size1 - (kernel_size_1-1)-1) / max_pooling_stride) + 1)  # output size after maximum pooling
        print(pooling_out_size1)
        self.cnn2 = nn.Conv1d(in_channels=out_channel_1, out_channels=out_channel_2, kernel_size=kernel_size_2,stride=kernel_size_2)
        out_size2 = int(
            ((pooling_out_size1 - (kernel_size_2-1)-1) / kernel_size_2) + 1)  # output size for cnn layer
        print('output size2:{}'.format(out_size2))
        self.maxpooling2 = nn.MaxPool1d(kernel_size=kernel_size_2, stride=max_pooling_stride)
        pooling_out_size2 = int(
            ((out_size2 - (kernel_size_2-1)-1)/ max_pooling_stride) + 1)  # outputsize after maximum pooling
        # (inputsize+2*padding-dilation*(kernerl_size-1)-1)/stride +1 https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html
        print("maxpooling2 output:{}".format(pooling_out_size2))
        self.flatten = nn.Flatten()
        # pooling out * cnn1 out channel ouputsize is 2, which link to parameter r and ymax
        self.cluster = nn.Linear(pooling_out_size2 * out_channel_2,self.vector_length)
        if snp_enbed =='one_hot':
            self.fc = nn.Linear(self.vector_length+4, output_size) #concat year code here
        else:
            #if not one hot, year is embed into a number
            self.fc = nn.Linear(self.vector_length + 1, output_size)  # concat year code here
        self.relu=nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
    def forward(self, inputx):

        year_encode = inputx[:, :, -1]
        input_x = inputx[:,:,:-1]
        cnn1_out = self.cnn1(input_x)
        cnn1_out = self.leakyrelu(cnn1_out)
        max_pooling1_out = self.max_pooling1(cnn1_out)
        cnn2_out = self.cnn2(max_pooling1_out)
        cnn2_out = self.leakyrelu(cnn2_out)
        max_pooling2_out = self.maxpooling2(cnn2_out)
        # print('max pooling out:{}'.format(max_pooling2_out))
        # print(max_pooling2_out.shape)
        flatten_out = self.flatten(max_pooling2_out)
        # print(flatten_out.shape)
        cluster_out = self.cluster(flatten_out)
        #print(cluster_out.shape,year_encode.shape)
        parameter_input = torch.cat([cluster_out,year_encode],dim=-1) #[sample_num,10]
        # print(parameter_input[0,:])
        # print(torch.unique(parameter_input,dim=1).shape)
        out_fc = self.fc(parameter_input)
        # out_fc = self.out_fc)+1 #make sure lager than 0 less than 2
        self.r = self.sigmoid(out_fc[:,0]) #shape should be [batchsize,1]
        # print('self r shape: {}'.format(self.r.shape))
        self.y_max=self.tanh(out_fc[:,1])+1
        # print(self.r,self.y_max)
        # print(cluster_out.shape)
        return cluster_out
    def init_network(self):
        # initialize weight and bias(use xavier and 0 for weight and bias separately)
        for name, param in self.named_parameters():
            if 'weight' in name:
                #https://pytorch.org/cppdocs/api/function_namespacetorch_1_1nn_1_1init_1ace282f75916a862c9678343dfd4d5ffe.html
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
class MLP_embedding(nn.Module):
    def __init__(self,input_size=20,out_1_size=6):
        super().__init__()
        self.init_network()
        self.fc = nn.Linear(in_features=input_size,out_features=2)
        self.Relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(in_features=out_1_size,out_features=2)
        self.sigmoid = nn.Sigmoid()
    def forward(self,inputx):
        out1 = self.fc(inputx)

        self.r = self.sigmoid(out1[:, 0])  # shape should be [batchsize,1]
        # print('self r shape: {}'.format(self.r.shape))
        self.y_max = self.tanh(out1[:, 1]) + 1
        return out1
    def init_network(self):
        # initialize weight and bias(use xavier and 0 for weight and bias separately)
        for name, param in self.named_parameters():
            if 'weight' in name:
                #https://pytorch.org/cppdocs/api/function_namespacetorch_1_1nn_1_1init_1ace282f75916a862c9678343dfd4d5ffe.html
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

def calculate_distance_matrix(genotype_list,snp_encode_name=''):
    """
    I want to calculate a distance matrix ans use that as input for simple genotype embedding
    """
    similarity_matrix_df = pd.DataFrame({'genotype':genotype_list},index=range(len(genotype_list)))
    for i in range(len(genotype_list)):
        print(i)
        gene_i_distance_list = []
        g_tensor_i = genotype_list[i]
        file_path_i = '../temporary/{}_{}.dill'.format(g_tensor_i, snp_encode_name)
        with open(file_path_i,'rb') as file1:
            i_tensor = dill.load(file1)
            i_tensor = i_tensor.float()
        for j in range(len(genotype_list)):
            g_tensor_j = genotype_list[j]
            file_path_j = '../temporary/{}_{}.dill'.format(g_tensor_j, snp_encode_name)
            with open(file_path_j,'rb') as file2:
                j_tensor = dill.load(file2)
                j_tensor = j_tensor.float()
            print(i_tensor.shape)
            distance = torch.mean(torch.nn.functional.pairwise_distance(i_tensor,j_tensor))
            print(distance)
            gene_i_distance_list.append(distance.reshape(1))
        else:
            i_tensor_new = torch.cat(gene_i_distance_list).unsqueeze(dim=-1)
            similarity_matrix_df[genotype_list[i]]=i_tensor_new

    else:
        print(similarity_matrix_df)
        similarity_matrix_df.to_csv('../processed_data/distance_encoding_matrix_full.csv')
def genotype_similarity_encoding(genotype_list,snp_encode_name=''):
    all_genotype_tensor=[]
    for i in range(len(genotype_list)):
        gene_i_distance_list = []
        g_tensor_i = genotype_list[i]
        file_path_i = '../temporary/{}_{}.dill'.format(g_tensor_i, snp_encode_name)
        with open(file_path_i,'rb') as file1:
            i_tensor = dill.load(file1)
            # print(i_tensor)
        all_genotype_tensor.append(i_tensor)
    else:
        #calculate average genotype tensor (average snp value, encode to 0,1,2 or including blosume score for all genotype)
        tensor_with_nan=torch.stack(all_genotype_tensor)
        print(tensor_with_nan.shape)
        print(torch.isnan(tensor_with_nan).sum())
        mask = ~torch.isnan(tensor_with_nan)

        # Replace NaNs with zero for summing purposes
        tensor_without_nan = torch.where(mask, tensor_with_nan, torch.zeros_like(tensor_with_nan))
        # Sum across the dimension of interest (e.g., dim=0) while ignoring NaNs
        sum_values = torch.sum(tensor_without_nan, dim=0)
        # Count non-NaN values along the same dimension
        count_non_nan = torch.sum(mask, dim=0)
        # Calculate the mean by dividing the sum by the count of non-NaN values
        average_genotype_tensor = sum_values / count_non_nan
        # print(average_genotype_tensor.shape)
        # print(average_genotype_tensor)
    for j in range(len(genotype_list)):
        g_tensor_j = genotype_list[j]
        file_path_j = '../temporary/{}_{}.dill'.format(g_tensor_j, snp_encode_name)
        with open(file_path_j,'rb') as file2:
            j_tensor = dill.load(file2)
        # print(i_tensor.shape)
        distance = mask_Euclidean_distance(average_genotype_tensor, j_tensor).unsqueeze(dim=0).unsqueeze(dim=0)
        print(distance) #reshape to shape [1,1]
        with open('../temporary/{}_{}_similarity_encoding.dill'.format(g_tensor_j,snp_encode_name),'wb') as file_save:
            #save distance encoding tensor
            dill.dump(distance,file_save)

def mask_Euclidean_distance(tensor1, tensor2):
    mask1 = ~torch.isnan(tensor2)
    mask2 = ~torch.isnan(tensor1)
    # Combined mask: only include positions where both tensors are not NaN
    combined_mask = mask1 & mask2
    # Replace NaN values with zeros temporarily for calculation
    x1_nonan = torch.where(mask1, tensor2, torch.zeros_like(tensor2))
    x2_nonan = torch.where(mask2, tensor1, torch.zeros_like(tensor1))
    # Calculate squared differences, ignoring NaNs
    squared_diff = (x1_nonan - x2_nonan) ** 2 * combined_mask
    # Sum all valid squared differences and count non-NaN positions across entire tensor
    sum_squared_diff = squared_diff.sum()
    num_valid_elements = combined_mask.sum()
    # Compute the overall Euclidean distance
    distance = torch.sqrt(sum_squared_diff / num_valid_elements)
    # print("Pairwise distance ignoring NaNs:", distance)
    return distance


def genotype_binary_encoding(genotype_list, snp_encode_name='binary_encoding'):

    """
    encode genotype is list to unique 5-bit vector and save in dill
    """

    def int_to_binary_vector(n, num_bits=5):
        #convert an integer to a 5-bit binary vector
        return [int(x) for x in format(n, f'0{num_bits}b')]

    def int_to_onehot(n, num_classes=19):
        # Convert an integer to a one-hot encoded vector of length num_classes
        if n >= num_classes or n < 0:
            raise ValueError("Input value out of range for the specified number of classes.")
        return [1 if i == n else 0 for i in range(num_classes)]
    # Create a dictionary to map genotype IDs to unique binary vectors
    genotype_to_binary = {}
    if snp_encode_name=='binary_encoding':

        for idx, genotype_id in enumerate(genotype_list):
            # Encode the index as a 5-bit vector
            binary_vector = int_to_binary_vector(idx)
            genotype_to_binary[genotype_id] = torch.tensor(binary_vector).unsqueeze(dim=-1)

        for genotype_id, binary_vector in genotype_to_binary.items():
            print(f"Genotype ID {genotype_id}: {binary_vector}")
            print(binary_vector.shape)
            with open('../temporary/{}_{}.dill'.format(genotype_id,snp_encode_name), 'wb') as file_save:
                # save distance encoding tensor
                dill.dump(binary_vector, file_save)
    elif snp_encode_name=='genotype_one_hot_encoding':
        for idx, genotype_id in enumerate(genotype_list):
            # Encode the index as a 5-bit vector
            one_hot_vector = int_to_onehot(idx)
            genotype_to_binary[genotype_id] = torch.tensor(one_hot_vector).unsqueeze(dim=-1)

        for genotype_id, one_hot_vector in genotype_to_binary.items():
            print(f"Genotype ID {genotype_id}: {one_hot_vector}")
            print(one_hot_vector.shape)
            with open('../temporary/{}_{}.dill'.format(genotype_id, snp_encode_name), 'wb') as file_save:
                # save distance encoding tensor
                dill.dump(one_hot_vector, file_save)


def encode_snp_to_tensor(genotype_list, genotype_encode_name:str='',snp_encode:str='one_hot',scoring_method:tuple=()):

    """check if tempoary file with encoded snps information for chosen genotype exist, otherwise create
    genotype_encode_name: 'binary_encoding', 'distance_encoding','similarity_encoding','genotype_one_hot_encoding'
    snp_encode: 'one_hot' or ''
    scoring_method: choose from ('blosum62','grantham_score') or both or ()"""

    import os
    for genotype in genotype_list:
        # Specify the file path
        file_path = '../temporary/{}_{}_blosum62.dill'.format(genotype, genotype_encode_name)
        # Check if the file exists
        if os.path.exists(file_path):
            print("The {}_{}.dill exists.".format(genotype, genotype_encode_name))
        else:
            print("The file {}_{}.dill  does not exist. create genetics_snps_tensor.....".format(genotype, genotype_encode_name))
            if genotype_encode_name == 'one_hot':

                ordinal_encoded = bool(snp_encode != 'one_hot')
                genotype_name_list,snp_genotype_df,snp_info_df =read_SNPs_file_based_on_genotype_list(genotype_list,
                                                                                                      ordinal_encoded=ordinal_encoded) #returned genotype list with GABI key
                snps_encoding(snp_genotype_df, snp_info_df, genotype_name_list, name=snp_encode, scoring_method=scoring_method)
            elif genotype_encode_name == 'distance_encoding':
                #encoding fo genotype based on addtive effect
                calculate_distance_matrix(genotype_list=genotype_list, snp_encode_name=snp_encode)
            elif (genotype_encode_name == 'binary_encoding') or (genotype_encode_name == 'genotype_one_hot_encoding'):
                genotype_binary_encoding(genotype_list, genotype_encode_name)
            elif genotype_encode_name == 'similarity_encoding':

                ordinal_encoded = bool(snp_encode!='one_hot')
                print(ordinal_encoded)
                genotype_name_list,snp_genotype_df,snp_info_df =read_SNPs_file_based_on_genotype_list(genotype_list,
                                                                                                      ordinal_encoded=ordinal_encoded) #returned genotype list with GABI key
                #snp_encode_name read snp encoding and then calculate similarity, '' is default 0,1,2 code(if name==one_hot then use one hot encoding to calculate
                # if name=='', ordinal_encoded need to be true
                snps_encoding(snp_genotype_df, snp_info_df, genotype_name_list, name='{}'.format(snp_encode), scoring_method=scoring_method)
                if scoring_method!=():
                    genotype_similarity_encoding(genotype_list,snp_encode_name='{}_{}'.format(snp_encode,"_".join(scoring_method)))
                else:genotype_similarity_encoding(genotype_list,snp_encode_name='{}'.format(snp_encode))
            elif genotype_encode_name == 'kinship_matrix_encoding':
                #https: // biometris.github.io / statgenGWAS / articles / GWAS.html  # kinship-matrices
                # import subprocess
                # from pathlib import Path
                #
                # # Call the R script
                # file_path = Path('../temporary/kinship_matrix_astle.csv')
                # if not file_path.exists():
                #     print(f"File does not exist. Running the R script to create: {file_path}")
                #     # Run the R script using rpy2
                #     try:
                #         os.system('kinship_calculation.R')  # Make sure the kinship_calculation.R is in the same directory or provide full path
                #         print("R script executed successfully.")
                #     except Exception as e:
                #         print(f"Error running R script: {e}")
                #
                # else:
                #     print(f"File exists: {file_path}")
                kinship_df = pd.read_csv('../temporary/kinship_matrix_astle_all_present_genotype.csv',header=0,index_col=0)
                kinship_df.columns = kinship_df.columns.astype(int)
                genotype_kinship_similarity = torch.tensor(kinship_df[genotype].values).reshape(len(kinship_df[genotype]), 1)
                print(genotype_kinship_similarity.shape)
                with open('../temporary/{}_{}_all_present_genotype.dill'.format(genotype, genotype_encode_name),
                          'wb') as file_save:
                    # save distance encoding tensor
                    dill.dump(genotype_kinship_similarity, file_save)
            else:
                genotype_name_list,snp_genotype_df,snp_info_df =read_SNPs_file_based_on_genotype_list(genotype_list,
                                                                                                      ordinal_encoded=True)
                snps_encoding(snp_genotype_df, snp_info_df, genotype_name_list, name=genotype_encode_name, scoring_method=scoring_method)


def distance_criterion(output: torch.Tensor) -> torch.Tensor: #, class_id: int
    cluster_num = output.shape[0]
    # print(cluster_num)

    # Ensure tensors are on the right device and initialized properly
    output = output.to(DEVICE)
    euclidean_distance = torch.zeros(1, device=DEVICE)
    pairs = torch.zeros(1, device=DEVICE)

    for i in range(cluster_num):
        for j in range(i + 1, cluster_num):  # Start from i+1 to avoid duplicate pairs and self-pairs
            vector_i = output[i, :]
            vector_j = output[j, :]
            distance = torch.nn.functional.pairwise_distance(vector_i.unsqueeze(0), vector_j.unsqueeze(0))
            euclidean_distance += distance
            pairs += 1
    # Avoid division by zero
    if pairs.item() == 0:
        loss = torch.tensor(float('inf'), device=DEVICE)  # or any large value, or handle it differently
    else:
        loss = 1 / (euclidean_distance / pairs)
    return loss
    # euclidean_distance_same_cluster=0.0
    # pairs_same_cluster =0
    # # this is to maximize distance between vects not in the same group and minimmize ingroup distance
    # # try:
    # #     class_id = class_id.to_list()
    # # except:
    # #     class_id = class_id.tolist()
    # # print(class_id)
    # # for id in class_id.unique():
    # #     #loop through all genotype id
    # #     id_index = [index for index, value in enumerate(class_id) if str(value) == str(id)]
    # #     if len(id_index)>=2: #if one genotype has a replicates, then calculate distance for the same genotype
    # #         for i in id_index:
    # #             id_index.remove(i) #remove to avoid calculated distance for the vector itself
    # #             for j in id_index:
    # #                 euclidean_distance_same_cluster += torch.nn.functional.pairwise_distance(output[i, :], output[j, :])
    # #                 pairs_same_cluster = pairs_same_cluster + 1
    # # else:
    # #     try:
    # #         euclidean_distance_same_cluster_average =euclidean_distance_same_cluster/pairs_same_cluster
    # #     except:
    # #         euclidean_distance_same_cluster_average = 0.000001
    # euclidean_distance = 0.0
    # pairs=0
    # id_no_replicate_list = copy.deepcopy(class_id.unique())
    # for index_no_rep,id_i in enumerate(id_no_replicate_list):
    #     id_index = [index for index, value in enumerate(class_id) if str(value) == str(id_i)]
    #     print(id_index)
    #     euclidean_distance_same_cluster += torch.nn.functional.pairwise_distance(output[id_index[0], :],output[id_index[1], :])
    #     pairs_same_cluster = pairs_same_cluster + 1
    #     average_output_i = torch.mean(output[id_index, :],dim=0)
    #     for index_j_no_rep in range(index_no_rep,len(id_no_replicate_list)):#loop through all other genotype id except id_i
    #         id_j = id_no_replicate_list[index_j_no_rep]
    #         id_index_j = [index for index, value in enumerate(class_id) if str(value) == str(id_j)]
    #         average_output_j = torch.mean(output[id_index_j, :], dim=0)
    #         # sum distance between i and each j
    #         euclidean_distance += torch.nn.functional.pairwise_distance(average_output_i, average_output_j)
    #         pairs = pairs + 1
    # else:
    #     try:
    #         euclidean_distance_same_cluster_average =euclidean_distance_same_cluster/pairs_same_cluster
    #     except:
    #         euclidean_distance_same_cluster_average = 0.000001
    # # loss = 1 /((euclidean_distance / pairs)-0) #euclidean_distance_same_cluster_average
    # # print('out group distance :{}.'.format((euclidean_distance / pairs)))
    # # print('in group distance :{}.'.format(euclidean_distance_same_cluster_average))
    # loss = 1/((euclidean_distance / pairs) -euclidean_distance_same_cluster_average)#in group distance/ out group length
    # return loss

def train_clustering_model(train, validation, test, model, epochs, lr,genotype_train,genotype_validation,genotype_test):

    model.train()
    losses = []
    validation_losses_plot=[]
    test_losses_plot = []
    validation_losses = []
    model_dict = {}
    optimiser = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):  # loop over the dataset multiple times

        # input to device
        inputs = train.to(DEVICE)

        # zero the parameter gradients
        optimiser.zero_grad()
        # forward
        outputs = model(inputs)
        loss = distance_criterion(outputs,genotype_train)
        losses.append(loss.item())
        loss.backward() #backward
        optimiser.step() #update parameters

        # Validation loss
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            #calculate test and validation loss for plot and stop criteria
            validation_outputs = model(validation)
            validation_loss = distance_criterion(validation_outputs,genotype_validation)
            validation_losses_plot.append(validation_loss.item())
            test_outputs = model(test)
            test_loss = distance_criterion(test_outputs,genotype_test)
            test_losses_plot.append(test_loss.item())
        if (epoch + 1) % 10 == 0:
            validation_losses.append(validation_loss)
            model_dict[str(validation_loss)] = copy.deepcopy(model)

            #print loss every 10 epochs
            print(f"Epoch {epoch+1}/{epochs}, loss: {losses[-1]:.4f}")
            # print_parameters_from_ode(model)
            print('validation loss:{}'.format(validation_loss))
    else:
        model_return = model_dict[str(min(validation_losses))]
        figure =plot_loss_change(losses, validation_losses_plot,test_losses_plot, name='loss_change')
        epoch_num = list(model_dict.keys()).index(str(min(validation_losses)))

    return epoch_num,model_return

def pretrain_genetics_embedding(genotype_split:tuple,year_list:tuple,file_name='',snp_encode_name=''):
    """
    call train_clustering_model for pretrain and find optimal hyperparameters set to maximize distance among different genotype
    """
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)  # the random seeds
    train_validation_test_list = []
    genotype_id_list =[]
    for genotype_list in genotype_split:
        # genotype_list = [33, 106, 122, 133, 5, 30, 218, 2, 17, 254, 282, 294, 301, 302, 335, 339, 341, 6, 362]
        # genotype_list = list(set(genotype_list))
        # check if genetics input tensor exist, otherwise create
        genotype_id_list.append(genotype_list) #genotype id for train validation and test
        encode_snp_to_tensor(genotype_list, genotype_encode_name=snp_encode_name)
        tensor_list = []
        print(genotype_list)
        for genotype in genotype_list:
            with open('../temporary/{}_{}.dill'.format(genotype,snp_encode_name), 'rb') as file1:
                genotype_tensor = dill.load(file1)
                tensor_list.append(genotype_tensor.unsqueeze(dim=0))
        genetics_input_tensor = torch.cat(tensor_list, dim=0).float()
        print(genetics_input_tensor.shape)
        genetics_input_tensor = torch.permute(genetics_input_tensor, (0, 2, 1))
        # print(genetics_input_tensor.shape)
        train_validation_test_list.append(genetics_input_tensor)
    train,validation,test= train_validation_test_list[0],train_validation_test_list[1],train_validation_test_list[2]
    train_id, validation_id, test_id = genotype_id_list[0].to(DEVICE),genotype_id_list[1].to(DEVICE),genotype_id_list[2].to(DEVICE)
    year_train,year_validation,year_test =year_list[0].to(DEVICE),year_list[1].to(DEVICE),year_list[2].to(DEVICE)

    # print(year_tensor_train)
    print('year tensor shape:{}'.format(year_train.shape))

    #convert nan to0.0
    train = torch.nan_to_num(train, nan=0.0, posinf=0.0, neginf=0.0).to(DEVICE)
    validation = torch.nan_to_num(validation, nan=0.0, posinf=0.0, neginf=0.0).to(DEVICE)
    test = torch.nan_to_num(test, nan=0.0, posinf=0.0, neginf=0.0).to(DEVICE)
    #concat year code to the end of input tensor
    print(train.shape,year_train.shape)
    # # add year code
    # if snp_encode_name !='one_hot':
    #     train = torch.cat([train, year_train.unsqueeze(-1).unsqueeze(-1)], dim=-1)
    #     validation = torch.cat([validation, year_validation.unsqueeze(-1).unsqueeze(-1)], dim=-1)
    #     test = torch.cat([test, year_test.unsqueeze(-1).unsqueeze(-1)], dim=-1)
    # else:
    #     train = torch.cat([train,year_train],dim=-1)
    #     validation = torch.cat([validation, year_validation], dim=-1)
    #     test = torch.cat([test, year_test], dim=-1)

    print('training shape')
    print(train.shape)
    print(validation.shape)
    print(test.shape)
    snps_embedding_pre_train_result = pd.DataFrame()
    if snp_encode_name == '' or snp_encode_name == 'one_hot':
        for out_channel_1 in [2]:
            for out_channel_2 in [1]:
                for kernel1 in [9]:
                    for kernel2 in [5]:
                        model_snp = embedding_by_1dCNN(snps_number=train.shape[-1],feature_size=train.shape[1],out_channel_1=out_channel_1,out_channel_2=out_channel_2,
                                                       kernel_size_1=kernel1,kernel_size_2=kernel2,max_pooling_stride=9,snp_enbed=snp_encode_name).to(DEVICE)
                        count_parameters(model_snp)
                        model_snp.init_network()
                        stop_epoch,model_trained = train_clustering_model(train=train,validation=validation,test=test,model=model_snp,epochs=30,lr=0.001,
                                                                          genotype_train=train_id,genotype_validation=validation_id,genotype_test=test_id)

                        with torch.no_grad():
                            model_trained.eval()
                            train_loss = distance_criterion(model_trained(train),train_id).item()
                            validation_loss = distance_criterion(model_trained(validation),validation_id).item()
                            test_loss = distance_criterion(model_trained(test),test_id).item()
                        new_row = pd.DataFrame({"out_channel_1":out_channel_1,'out_channel_2':out_channel_2,"kernel1":kernel1,
                                                    "kernel2":kernel2,'train_loss':train_loss,"validation_loss":validation_loss,
                                                "test_loss":test_loss},index=[0]).astype(int)
                        snps_embedding_pre_train_result=pd.concat([snps_embedding_pre_train_result,new_row])
                        with open('snps_embedding_model/{}_{}_{}_{}_{}_{}.dill'.format(file_name,out_channel_1,out_channel_2,kernel1,kernel2,snp_encode_name),'wb') as file_2:
                            dill.dump(model_trained,file_2)
                        file_2.close()
        else:
            snps_embedding_pre_train_result = snps_embedding_pre_train_result.sort_values(by="validation_loss").reset_index(drop=True)
            print(snps_embedding_pre_train_result)
            hyperparameter = snps_embedding_pre_train_result.loc[0,['out_channel_1','out_channel_2',
            'kernel1','kernel2']].values.tolist()
            print(hyperparameter)
            return hyperparameter,(train,validation,test)
    else:
        for size in [3]:
            model_snp = MLP_embedding(input_size=train.shape[-1],out_1_size=size).to(DEVICE)
            count_parameters(model_snp)
            model_snp.init_network()
            train = train.squeeze().to(DEVICE)
            validation = validation.squeeze().to(DEVICE)
            test = test.squeeze().to(DEVICE)
            stop_epoch, model_trained = train_clustering_model(train=train, validation=validation, test=test,
                                                               model=model_snp, epochs=300, lr=0.005,
                                                               genotype_train=train_id, genotype_validation=validation_id,
                                                               genotype_test=test_id)

            with torch.no_grad():
                model_trained.eval()
                train_loss = distance_criterion(model_trained(train), train_id).item()
                validation_loss = distance_criterion(model_trained(validation), validation_id).item()
                test_loss = distance_criterion(model_trained(test), test_id).item()
            new_row = pd.DataFrame({"out_1_size": size, 'train_loss': train_loss, "validation_loss": validation_loss,
                                    "test_loss": test_loss}, index=[0]).astype(int)
            snps_embedding_pre_train_result = pd.concat([snps_embedding_pre_train_result, new_row])
            with open('snps_embedding_model/{}_{}_{}.dill'.format(file_name, size, snp_encode_name), 'wb') as file_2:
                dill.dump(model_trained, file_2)
            file_2.close()
        else:
            snps_embedding_pre_train_result = snps_embedding_pre_train_result.sort_values(by="validation_loss").reset_index(drop=True)
            print(snps_embedding_pre_train_result)
            hyperparameter = snps_embedding_pre_train_result.loc[0,['out_1_size']].values.tolist()
            print(hyperparameter)
            return hyperparameter,(train,validation,test)
def save_marker_information_to_dill_seperately(genotype_list):

    genotype_name_list, snp_genotype_df, snp_info_df = read_SNPs_file_based_on_genotype_list(genotype_list,
                                                                                             ordinal_encoded=True)
    print('genotype without gabi key will be dropped, {} genotypes left'.format(len(genotype_name_list)))
    # genotype_name_list_df = pd.DataFrame({'genotype_id':genotype_name_list},index=range(len(genotype_name_list)))
    # genotype_name_list_df.to_csv('../processed_data/fouryear_genotypes.csv')
    snps_encoding(snp_genotype_df, snp_info_df, genotype_name_list, name='one_hot',
                  scoring_method=())
    # encode_snp_to_tensor(genotype_name_list, 'kinship_matrix_encoding', snp_encode='', scoring_method=())
    # calculate_distance_matrix(genotype_name_list,snp_encode_name='')
def main():
    # import rpy2.robjects as ro
    # ro.r('R.version')
    from torch.utils.data import random_split
    # genotype_list = find_genotype_present_at_multiple_years()
    genotype_list=genotype_present_in_at_least_in_one_of_selected_years(year_list=[2018,2019,2021,2022])
    # save_marker_information_to_dill_seperately(genotype_list)
    # genotype_list=[33, 106, 122, 133, 5, 30, 218, 2, 17, 254, 282, 294, 301, 302, 335, 339, 341, 6, 362]
    encode_snp_to_tensor(genotype_list, 'similarity_encoding', snp_encode='one_hot', scoring_method=())
    # encode_snp_to_tensor(genotype_list,'similarity_encoding',snp_encode='',scoring_method=('blosum62',))
    # encode_snp_to_tensor(genotype_list, 'kinship_matrix_encoding', snp_encode='', scoring_method=())

    # encode_snp_to_tensor(genotype_list, '', snp_encode='', scoring_method=())
    # genotype_similarity_encoding(genotype_list)
    # read_SNPs_file_based_on_genotype_list(genotype_list,snps_encoded=False)

    # pretrain_genetics_embedding(genotype_list,snp_encode_name='one_hot')

    # df =pd.read_table("C:\data_from_paper/ETH/olivia_WUR/genotype_markers/GABI_WHEAT_90k.txt", header=0, sep=' ')
    # print(df)
    # df.to_csv('90k.csv')


if __name__ == '__main__':
    main()
