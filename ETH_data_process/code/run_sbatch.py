#!/usr/bin/env python
import time
import os


def run_slurm_with_gpu(job_file='gpu',genetics_embedding=False):
    mode='gpu_lstm_NN_save_model'
    if_pinn =False
    if genetics_embedding:
        genotype_list = [[33, 106, 122, 133, 5, 30, 218, 2, 17, 254, 282, 294, 301, 302, 335, 339, 341, 6, 362]]

    else:
        genotype_list = [33, 106, 122, 133, 5, 30, 218, 2, 17, 254, 282, 294, 301, 302, 335, 339, 341, 6, 362]

    for genotype in genotype_list:
        for parameter_boundary in ['None']:#hard constrain of r in CNN of genetics embedding
            for same_length in ['_same_length']: #'_same_length'
                for rescale in ['False']:
                    for weight in [0]:
                        if genetics_embedding:
                            job_file = str(mode) + 'multi_g' + str(parameter_boundary) + 'smooth_False' + str(same_length) \
                                       + str(rescale) +str(genetics_embedding)+ '.sh'
                        else:
                            job_file = str(mode) + str(genotype) + str(parameter_boundary) + 'smooth_False' + str(same_length) \
                                       + str(rescale) +str(genetics_embedding)+ '.sh'
                        with open(job_file, 'w') as file1:
                            file1.writelines("#!/bin/bash\n")
                            file1.writelines("#SBATCH --job-name=%s\n" % job_file)
                            file1.writelines("#SBATCH --output=/lustre/backup/WUR/AIN/shao015/pinn/outputs/output_%j.txt\n")
                            file1.writelines("#SBATCH --error=/lustre/backup/WUR/AIN/shao015/pinn/errors/error_output_%j.txt\n")
                            file1.writelines("#SBATCH --mail-user=yingjie.shao@wur.nl\n")
                            file1.writelines("#SBATCH --partition=gpu\n")
                            file1.writelines('#SBATCH --gres=gpu:1\n')
                            file1.writelines("#SBATCH --mem=4G\n")
                            file1.writelines("#SBATCH --mail-type=ALL\n")
                            file1.writelines("#SBATCH --time=0-02:30:00\n")
                            file1.writelines('#SBATCH --ntasks=1\n')
                            file1.writelines('#SBATCH --cpus-per-task=4\n')

                            file1.writelines('source activate /lustre/backup/WUR/AIN/shao015/conda/envs/test\n')
                            file1.writelines('conda run -n test\n')
                            if genetics_embedding:
                                genotype = [str(i) for i in genotype]
                                file1.writelines(
                                    "python3 run_model_cmd.py -mode {0} -if_pinn {1} -start_day 115 -smooth False -fill_in_na_at_start True {2} -rescale {3} -weight {4} -genetics_embed {5} -genotype {6} -end\n"
                                    .format(mode, if_pinn, same_length, rescale, weight,genetics_embedding,  " ".join(genotype)))
                            else:
                                file1.writelines(
                                    "python3 run_model_cmd.py -mode {0} -if_pinn {1} -start_day 115 -smooth False -fill_in_na_at_start True {2} -rescale {3} -weight {4} -genetics_embed {5} -genotype {6} -end\n"
                                    .format(mode, if_pinn, same_length, rescale, weight,genetics_embedding, genotype))
                            file1.close()

                        # print(
                        #         "python3 run_model_cmd.py -mode {0} -if_pinn {1} -start_day 115 -smooth False -fill_in_na_at_start True {2} -rescale {3} -weight {4} -genetics_embed {5} -genotype {6} -end &\n"
                        #         .format(mode, if_pinn, same_length, rescale, weight,genetics_embedding, genotype))

                        # os.system(
                        #         "python3 run_model_cmd.py -mode {0} -if_pinn {1} -start_day 115 -smooth False -fill_in_na_at_start True {2} -rescale {3} -weight {4} -genetics_embed {5} -genotype {6} -end &\n"
                        #         .format(mode, if_pinn, same_length, rescale, weight,genetics_embedding, genotype))
                        # time.sleep(5000)
                        os.system("sbatch %s" % job_file)
                        os.system('rm {}'.format(job_file))

def run_sbatch_cpu(genetics_embedding=False):

        for mode in ['cpu_logi_pinn_save_model_']:
            for if_pinn in [True]:
                if genetics_embedding:
                    genotype_list = [[33, 106, 122, 133, 5, 30, 218, 2, 17, 254, 282, 294, 301, 302, 335, 339, 341, 6, 362]]
                    parameter_boundary_list = ['None']
                    encode_type =  ['ordinal','one_hot']
                else:
                    genotype_list = [33, 106, 122, 133, 5, 30, 218, 2, 17, 254, 282, 294, 301, 302, 335, 339, 341, 6, 362]
                    parameter_boundary_list = ['penalize_r']#'None'
                    encode_type = [None]
                for genotype in genotype_list:
                    for parameter_boundary in parameter_boundary_list: #'penalize_r',
                        for same_length in ['_same_length']:
                            for rescale in ['False']:
                                for weight in [None]:
                                    for snp_encode_type in encode_type:
                                        # for penalize_y in ['']:
                                        if genetics_embedding:
                                            job_file = str(mode)+'multi_g'+str(parameter_boundary)+'smooth_False'+str(same_length)\
                                            +str(rescale) +str(weight)+ snp_encode_type+'.sh'
                                        else:
                                            job_file = str(mode)+str(genotype)+str(parameter_boundary)+'smooth_False'+str(same_length)\
                                            +str(rescale) +str(weight)+'.sh'
                                        with open(job_file,'w') as file1:
                                            file1.writelines("#!/bin/bash\n")
                                            file1.writelines("#SBATCH --job-name=%s\n" % job_file)
                                            file1.writelines("#SBATCH --output=/lustre/backup/WUR/AIN/shao015/pinn/outputs/output_%j.txt\n")
                                            file1.writelines("#SBATCH --error=/lustre/backup/WUR/AIN/shao015/pinn/errors/error_output_%j.txt\n")
                                            file1.writelines("#SBATCH --time=0-32:00:00\n")
                                            file1.writelines("#SBATCH --cpus-per-task=1\n")
                                            file1.writelines("#SBATCH --mem-per-cpu=4GB\n")
                                            file1.writelines("#SBATCH --mail-type=ALL\n")
                                            file1.writelines("#SBATCH --mail-user=yingjie.shao@wur.nl\n")
                                            file1.writelines('source activate /lustre/backup/WUR/AIN/shao015/conda/envs/test\n')
                                            file1.writelines('conda run -n test\n')
                                            if genetics_embedding:
                                                genotype = [str(i) for i in genotype]
                                                file1.writelines("python3 run_model_cmd.py -mode {0} -if_pinn {1} -start_day 115 -parameter_boundary {2} -smooth False -fill_in_na_at_start True {3} -rescale {4} -weight {5} -genetics_embed {6} -genetics_encode {7} -genotype {8} -end\n"
                                                                 .format(mode,if_pinn,parameter_boundary,same_length,rescale,weight,genetics_embedding,snp_encode_type, " ".join(genotype)))
                                                # os.system("python3 run_model_cmd.py -mode {0} -if_pinn {1} -start_day None -parameter_boundary {2} -smooth False -fill_in_na_at_start True {3} -rescale {4} -weight {5} -genetics_embed {6} -genetics_encode {7} -genotype {8} -end &\n"
                                                #                  .format(mode,if_pinn,parameter_boundary,same_length,rescale,weight,genetics_embedding,snp_encode_type, " ".join(genotype)))
                                            else:
                                                file1.writelines(
                                                    "python3 run_model_cmd.py -mode {0} -if_pinn {1} -start_day 115 -parameter_boundary {2} -smooth False -fill_in_na_at_start True {3} -rescale {4} -weight {5} -genetics_embed {6} -genotype {7} -end\n"
                                                    .format(mode, if_pinn, parameter_boundary, same_length, rescale, weight,genetics_embedding,genotype
                                                           ))
                                                # os.system("python3 run_model_cmd.py -mode {0} -if_pinn {1} -start_day 115 -parameter_boundary {2} -smooth False -fill_in_na_at_start True {3} -rescale {4} -weight {5} -genetics_embed {6} -genotype {7} -end &\n"
                                                #     .format(mode, if_pinn, parameter_boundary, same_length, rescale, weight,genetics_embedding,genotype
                                                #            ))
                                            file1.close()
                                        os.system("sbatch %s" % job_file)
                                        # time.sleep(3600)#wait 3600 second
                                            #remove created temperoal file
                                        os.system('rm {}'.format(job_file))

def run_sbatch_pinn_cross_validation(smooth_temp=False):

    genotype_list = [33,106,122, 133, 5, 30, 218, 2, 17, 254, 282, 294, 301, 302, 335, 339, 341, 6,362]  # [33, 106, 122, 133, 5, 30, 218, 2, 17, 254, 282, 294, 301, 302, 335, 339, 341, 6,362]
    for genotype in genotype_list:
        job_file =  str(genotype)  + 'smooth_temp_' + str(smooth_temp) +'.sh'
        with open(job_file, 'w') as file1:
            file1.writelines("#!/bin/bash\n")
            file1.writelines("#SBATCH --job-name=%s\n" % job_file)
            file1.writelines(
                "#SBATCH --output=/lustre/backup/WUR/AIN/shao015/pinn/outputs/output_%j.txt\n")
            file1.writelines(
                "#SBATCH --error=/lustre/backup/WUR/AIN/shao015/pinn/errors/error_output_%j.txt\n")
            file1.writelines("#SBATCH --time=0-12:00:00\n")
            file1.writelines("#SBATCH --cpus-per-task=1\n")
            file1.writelines("#SBATCH --mem-per-cpu=8GB\n")
            file1.writelines("#SBATCH --mail-type=ALL\n")
            file1.writelines("#SBATCH --mail-user=yingjie.shao@wur.nl\n")
            file1.writelines('source activate /lustre/backup/WUR/AIN/shao015/conda/envs/test\n')
            file1.writelines('conda run -n test\n')
            file1.writelines(
                "python3 NNmodel_training.py -if_pinn False -genotype {0} -smooth_temp {1}\n"
                .format( genotype,smooth_temp))
            file1.close()
        # os.system("sbatch %s" % job_file)
        os.system("python3 NNmodel_training.py -if_pinn False -genotype {0} -smooth_temp {1} &\n"
                .format( genotype,smooth_temp))
        os.system('rm {}'.format(job_file))
        time.sleep(2000)

def run_multiple_genotype_model(mode_name,if_pinn=True,smooth_loss=False,reduce_time_resolution=False,smooth_input=False
                                ,smooth_temp=False):
    job_file=mode_name
    if if_pinn ==True:
        for split_group in ['year_site.harvest_year']:#,,'year_site.harvest_year','genotype.id'
            if split_group == 'year_site.harvest_year':
                encoding_list = ['genotype_one_hot_encoding','kinship_matrix_encoding'] #'distance_encoding',
            else:
                encoding_list = ['kinship_matrix_encoding']
            for genotype_encoding in encoding_list:#
                for smooth_input in [smooth_input]:
                    # job_file = job_file +str(split_group) +str(genotype_encoding) +'.sh'
                    with open(job_file, 'w') as file1:
                        file1.writelines("#!/bin/bash\n")
                        file1.writelines("#SBATCH --job-name=%s\n" % job_file)
                        file1.writelines("#SBATCH --output=/lustre/backup/WUR/AIN/shao015/pinn/outputs/output_%j.txt\n")
                        file1.writelines("#SBATCH --error=/lustre/backup/WUR/AIN/shao015/pinn/errors/error_output_%j.txt\n")
                        file1.writelines("#SBATCH --time=0-96:00:00\n")
                        file1.writelines("#SBATCH --cpus-per-task=1\n")
                        file1.writelines("#SBATCH --mem-per-cpu=16GB\n")
                        file1.writelines("#SBATCH --mail-type=ALL\n")
                        file1.writelines("#SBATCH --mail-user=yingjie.shao@wur.nl\n")
                        file1.writelines('source activate /lustre/backup/WUR/AIN/shao015/conda/envs/test\n')
                        file1.writelines('conda run -n test\n')

                        file1.writelines(
                            "python3 MultipleGenotypeModel.py -mode {0} -if_pinn {1} -smooth_loss {2} -genotype_encoding {3} -split_group {4} -reduce_time_resolution {5} -smooth_input {6} -smooth_temp {7}\n"
                            .format(job_file, if_pinn,smooth_loss,genotype_encoding,split_group,reduce_time_resolution,smooth_input,smooth_temp))
                    file1.close()
                    print("python3 MultipleGenotypeModel.py -mode {0} -if_pinn {1} -smooth_loss {2} -genotype_encoding {3} -split_group {4} -reduce_time_resolution {5} -smooth_input {6} -smooth_temp {7}\n"
                            .format(job_file, if_pinn,smooth_loss,genotype_encoding,split_group,reduce_time_resolution,smooth_input,smooth_temp))
                    os.system("sbatch %s" % job_file)
                    #os.system("python3 MultipleGenotypeModel.py -mode {0} -if_pinn {1} -smooth_loss {2} -genotype_encoding {3} -split_group {4} -reduce_time_resolution {5} -smooth_input {6} -smooth_temp {7}\n"
                    #       .format(job_file, if_pinn,smooth_loss,genotype_encoding,split_group,reduce_time_resolution,smooth_input,smooth_temp))
                    # time.sleep(7200)
    else:
        for split_group in [ 'year_site.harvest_year']: #'year_site.harvest_year', 'genotype.id','g_e'
            if split_group == 'year_site.harvest_year':
                encoding_list = ['genotype_one_hot_encoding','kinship_matrix_encoding'] #'genotype_one_hot_encoding',
            else:
                encoding_list = ['kinship_matrix_encoding']
            for genotype_encoding in encoding_list:
                for smooth_input in [smooth_input]:
                    # job_file = job_file + str(split_group) + str(genotype_encoding)+'.sh'
                    with open(job_file, 'w') as file1:
                        file1.writelines("#!/bin/bash\n")
                        file1.writelines("#SBATCH --job-name=%s\n" % job_file)
                        file1.writelines("#SBATCH --output=/lustre/backup/WUR/AIN/shao015/pinn/outputs/output_%j.txt\n")
                        file1.writelines("#SBATCH --error=/lustre/backup/WUR/AIN/shao015/pinn/errors/error_output_%j.txt\n")
                        file1.writelines("#SBATCH --mail-user=yingjie.shao@wur.nl\n")
                        file1.writelines("#SBATCH --partition=gpu\n")
                        file1.writelines('#SBATCH --gres=gpu:1\n')
                        file1.writelines("#SBATCH --mem=16G\n")
                        file1.writelines("#SBATCH --mail-type=ALL\n")
                        file1.writelines("#SBATCH --time=0-08:00:00\n")
                        file1.writelines('#SBATCH --ntasks=1\n')
                        file1.writelines('#SBATCH --cpus-per-task=4\n')
                        file1.writelines(
                            "python3 MultipleGenotypeModel.py -mode {0} -if_pinn {1} -smooth_loss {2} -genotype_encoding {3} -split_group {4} -reduce_time_resolution {5} -smooth_input {6} -smooth_temp {7}\n"
                            .format(job_file, if_pinn, smooth_loss, genotype_encoding, split_group,reduce_time_resolution,smooth_input,smooth_temp))
                    file1.close()
                    # os.system("sbatch %s" % job_file)
                    print('Attention!! (I am suppose to run this in WSL). The following command will run in current window that does not submit by job:')
                    print("python3 MultipleGenotypeModel.py -mode {0} -if_pinn {1} -smooth_loss {2} -genotype_encoding {3} -split_group {4} -reduce_time_resolution {5} -smooth_input {6} -smooth_temp {7}&\n"
                            .format(job_file, if_pinn, smooth_loss, genotype_encoding, split_group,reduce_time_resolution,smooth_input,smooth_temp))
                    os.system("python3 MultipleGenotypeModel.py -mode {0} -if_pinn {1} -smooth_loss {2} -genotype_encoding {3} -split_group {4} -reduce_time_resolution {5} -smooth_input {6} -smooth_temp {7}&\n"
                            .format(job_file, if_pinn, smooth_loss, genotype_encoding, split_group,reduce_time_resolution,smooth_input,smooth_temp))
                    # time.sleep(3600)


def main():
    # run_sbatch_pinn_cross_validation(smooth_temp=False)
    # run_multiple_genotype_model(mode_name='pinn_cv_final',if_pinn=True,smooth_loss=False,
    #                             reduce_time_resolution=False,smooth_temp=False)
    # run_multiple_genotype_model(mode_name='pinn_result_NEW_more_g',if_pinn=True,smooth_loss=False,
    #                             reduce_time_resolution=False,smooth_temp=False)
    # run_multiple_genotype_model(mode_name='pinn_result_reduce_time_resolution_lstm', if_pinn=True, smooth_loss=False,
    #                             reduce_time_resolution=True)
    # run_multiple_genotype_model(mode_name='pinn_result_lstm', if_pinn=True, smooth_loss=False,smooth_input=True)
    # run_multiple_genotype_model(mode_name='PINN_retun_final',if_pinn=True,smooth_loss=False,reduce_time_resolution=False,
    #                              smooth_temp=False)
    # run_multiple_genotype_model(mode_name='NN_rerun_final',if_pinn=False,smooth_loss=False,reduce_time_resolution=False,
    #                             smooth_temp=False)
    # run_multiple_genotype_model(mode_name='NN_result_NEW_more_g',if_pinn=False,smooth_loss=False,reduce_time_resolution=False,
    #                             smooth_temp=False)
    # run_multiple_genotype_model(mode_name='NN_result_reduce_time_resolution_final', if_pinn=False, smooth_loss=False,
    #                             reduce_time_resolution=True)
    # run_multiple_genotype_model(mode_name='NN_result_final', if_pinn=False, smooth_loss=False,smooth_input=True)

    # run_multiple_genotype_model(mode_name='NN_xvair_init_result', if_pinn=False, smooth_loss=False,reduce_time_resolution=False)
    # run_multiple_genotype_model(mode_name='NN_result_two_genetics_layer_weight_data_loss', if_pinn=False, smooth_loss=False,reduce_time_resolution=False)
    # run_multiple_genotype_model(mode_name='fix_error_gpu_orthogonal_init_yboundsmooth', if_pinn=False,smooth_loss=True)
    # run_sbatch_cpu(genetics_embedding=True)
    run_sbatch_cpu(genetics_embedding=False)
    run_slurm_with_gpu(genetics_embedding=False)
    # if not os.path.exists('submit_bash/'):
    #     os.mkdir('submit_bash/')

if __name__ == '__main__':
    main()
