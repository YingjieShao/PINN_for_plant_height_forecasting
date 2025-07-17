"""
This script is to run PINN
usage: python3 run_model_cmd.py -mode <str, corresponding to mode parameters in function run_logistic_ode_pinn>
        -if_pinn <boolean> -start_day <int> -penalize_r <str or None, if None will not penalize r>
       -smooth <boolean> -fill_in_na_at_start <boolean> -penalize_y <optional, otherwise will not penalize y>
       -same_length <str, if empty will use full environment input, if 'same_length' will use the same resolution environment input as phenotype>
       -genotype <int of genotype code or ints split based on space>
Description: when using command line to run the code, genotype input need to be the last
"""


def read_cmd_input():
    # read command line input, use as paramters for running PINN
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
        index_mode = cmd_line.index("-start_day") + 1
        start_day = int(cmd_line[index_mode])
        if str(start_day) == 'None':
            start_day =None
    except:
        print(
            'did not receive -start_day input, use default setting: None. The start date will be the date with minimum value')
        start_day = None

    try:
        index_mode = cmd_line.index("-smooth") + 1
        smooth = cmd_line[index_mode]
        if smooth =='True':
            smooth = True
        elif smooth =='False':
            smooth = False
    except:
        print('did not receive -smooth input, use default setting: False. Do not add smoothing layer')
        smooth = False
    try:
        index_mode = cmd_line.index("-fill_in_na_at_start") + 1
        fill_in_na_at_start = cmd_line[index_mode]
        if str(fill_in_na_at_start) == 'True':
            fill_in_na_at_start = True
        elif str(fill_in_na_at_start) == 'False':
            fill_in_na_at_start = False
    except:
        print('did not receive -fill_in_na_at_start input, use default setting: True.')
        fill_in_na_at_start = True

    try:
        if "_same_length" in cmd_line:
            same_length = "_same_length"
        else:
            same_length = ''
    except:
        print('did not receive -same_length input, will use the full length enivronment input.')
        same_length = ''
        assert 'same_length' not in mode
    try:
        index_mode = cmd_line.index("-rescale") + 1
        rescale = cmd_line[index_mode]
        if rescale =='True':
            rescale = True
        elif rescale =='False':
            rescale = False
    except:
        print('did not receive -same_length input, will use the full length enivronment input.')
        rescale = False

    try:
        index_mode = cmd_line.index("-genotype") + 1
        end_index = cmd_line.index("-end")
        genotype = cmd_line[index_mode:end_index]
        genotype = [int(x) for x in genotype]
    except:
        print('did not receive -genotype input, set as None, will loop through all genotypes later.')
        genotype = None
    try:
        index_mode = cmd_line.index("-seed") + 1
        seed = int(cmd_line[index_mode])
    except:
        print('did not receive -seed input, set as None, will loop through all random seed coded in NNmodel_training.py later.')
        seed = None
    try:
        index_mode = cmd_line.index("-weight") + 1
        weight = int(cmd_line[index_mode])
        print('weight: {}'.format(weight))

    except:
        print('did not receive -weight input, set as None, will loop through all weights coded in NNmodel_training.py later.')
        weight = None

    try:
        index_mode = cmd_line.index("-genetics_embed") + 1
        genetics_embed = cmd_line[index_mode]
        if str(genetics_embed) =='True':
            genetics_embed = True
        elif str(genetics_embed) =='False':
            genetics_embed = False
    except:
        print('did not receive genetics_embed input, use default setting: False')
        genetics_embed = False

    try:
        index_mode = cmd_line.index("-genetics_encode") + 1
        genetics_encode = cmd_line[index_mode]
        if genetics_encode != 'one_hot':
            genetics_encode = ''
    except:
        print('did not receive genetics_embed input, use default setting: False')
        genetics_encode = ''

    if if_pinn == False:
        parameter_boundary = ''
        penalize_y = ''
        return mode, if_pinn, start_day, parameter_boundary, penalize_y, smooth, fill_in_na_at_start, same_length, rescale, genotype,seed,weight,genetics_embed,genetics_encode
    try:
        index_mode = cmd_line.index("-parameter_boundary") + 1
        parameter_boundary = cmd_line[index_mode]
        if str(parameter_boundary) == 'None':
            parameter_boundary = ''
    except:
        print('did not receive -parameter_boundary input, use default setting: \'\', will not penalize negetive r')
        parameter_boundary = ''

    try:
        if "-penalize_y" in cmd_line:
            penalize_y = "penalize_y"
        else:
            penalize_y = ''
    except:
        print('did not receive -penalize_y input, use default setting: \'\', will not penalize negetive y')
        penalize_y = ''

    return mode, if_pinn, start_day, parameter_boundary, penalize_y, smooth, fill_in_na_at_start, same_length,rescale, genotype,seed,weight,genetics_embed,genetics_encode

def main():
    from NNmodel_training import run_logistic_ode_pinn
    from raw_data_merge_visualize import find_genotype_present_at_multiple_years
    print("""
        This script is to run PINN
        usage: python3 run_model_cmd.py -mode <str, corresponding to mode parameters in function run_logistic_ode_pinn> 
                -if_pinn <boolean> -start_day <int> -parameter_boundary <str, if not '' penalize r>
               -smooth <boolean> -fill_in_na_at_start <boolean> -penalize_y <str, if not '' penalize y> 
               -same_length <str, if empty will use full environment input, if 'same_length' will use the same resolution environment input as phenotype>
               -rescale <boolean>
               -genotype <int of genotype code or ints split based on space>
        Description: when using command line to run the code, genotype input need to be the last
        """)

    mode, if_pinn, start_day, parameter_boundary, penalize_y, smooth, fill_in_na_at_start, same_length,rescale, genotype,seed,weight,genetics_embed,genetics_encode = read_cmd_input()
    print(genotype)
    if genotype == None:
        geneotype_list = find_genotype_present_at_multiple_years()  # after remove, [133, 294, 122, 218, 362, 459, 106, 301, 302, 335, 17, 466, 339, 341, 472, 282, 254]
    else:
        geneotype_list = genotype
    # print(geneotype_list)
    path = '../processed_data/align_height_env{}.csv'.format(same_length)
    if same_length!='':
        mode = mode +'{}'.format(same_length)
    # print(path)
    if genetics_embed:
        g=geneotype_list
        if seed:
            print('input randomseed:{}'.format(seed))
            run_logistic_ode_pinn(data_path=path,
                                  mode=mode,
                                  genotype=g, if_pinn=if_pinn, years=(2018, 2019, 2021, 2022),
                                  start_day=start_day, parameter_boundary=parameter_boundary, smooth=smooth,
                                  fill_in_na_at_start=fill_in_na_at_start, penalize_y=penalize_y, rescale=rescale,
                                  randomseed=seed, weight=weight, environment_input=['Air_temperature_2_m'],
                                  genetics_input=genetics_embed,snp_encoding_type=genetics_encode)
        elif seed is None:
            print('do not receive randomseed input')
            run_logistic_ode_pinn(data_path=path,
                                  mode=mode,
                                  genotype=g, if_pinn=if_pinn, years=(2018, 2019, 2021, 2022),
                                  start_day=start_day, parameter_boundary=parameter_boundary, smooth=smooth,
                                  fill_in_na_at_start=fill_in_na_at_start, penalize_y=penalize_y, rescale=rescale,
                                  weight=weight,
                                  environment_input=['Air_temperature_2_m'], genetics_input=genetics_embed,snp_encoding_type=genetics_encode)
    else:
        for g in geneotype_list:
            if seed:
                print('input randomseed:{}'.format(seed))
                run_logistic_ode_pinn(data_path=path,
                                      mode=mode,
                                      genotype=[g], if_pinn=if_pinn, years=(2018, 2019, 2021, 2022),
                                      start_day=start_day, parameter_boundary=parameter_boundary, smooth=smooth,
                                      fill_in_na_at_start=fill_in_na_at_start, penalize_y=penalize_y,rescale=rescale,
                                      randomseed=seed,weight=weight,environment_input=['Air_temperature_2_m'],genetics_input=genetics_embed)
            elif seed is None:
                print('do not receive randomseed input')
                run_logistic_ode_pinn(data_path=path,
                                      mode=mode,
                                      genotype=[g], if_pinn=if_pinn, years=(2018, 2019, 2021, 2022),
                                      start_day=start_day, parameter_boundary=parameter_boundary, smooth=smooth,
                                      fill_in_na_at_start=fill_in_na_at_start, penalize_y=penalize_y, rescale=rescale,weight=weight,
                                      environment_input=['Air_temperature_2_m'],genetics_input=genetics_embed)


if __name__ == '__main__':
    main()
