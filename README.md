# Repository for the paper 'Physics-Informed Neural Network Methods for Forecasting Plant Height Development'.
## Floder structure
- `ETH_data_process/`: This folder contains the code for data processing, model training, and result analysis.
  - `code/`: This folder include scripts and result folders contains model prediction results
  - `figures/`: This folder stores all the figures visulise model prediction result and also figures create for manuscript.
  - `temporary/`: This folder is used for temporary files that it not part of the result but needed for plot or result analysis.

The data we used is from ETH Z\"{u}rich, which is not included in this repository.

## Script structure 
For data process and model training:
- `raw_data_merge_visualize` need to be run first to merge data download from ETH paper.(Note: This requires data which is not in this repository)
  - `LoadData` can run after to align plant height based on sowing date to create the dataframe
    - `genetic_embedding` can be run to create genotype encoding that is necessary for multiple-genotype model. 
      The `kinship_calculation.R` needed to get the kinship_matrix_encoding
        - `MultipleGenotypeModel` can be run directly or with `run_sbatch.py` from command line, which train multiple-genotype models.
    - `NNmodeltraining` can be run directly or via `run_sbatch.py` from command line, which train single-genotype models.
    - `rf_plant_height_prediction` can be run for rf and temperature ODE single-genotype models.
For result visualisation:
-`Plot_analysis_result` can be run directly with saved result in this repository (or after model training).