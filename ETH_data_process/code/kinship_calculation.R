# author: yingjie
# Usage: Just run the script, this script should be put under 'ETH_data_process/code/' directory
#        The script requires marker data for each genotype saved in '../temporary/<genotype.id>_.dill',
#        which can be create from genetic_embedding.py
# output: save kinship matrix to '../temporary/kinship_matrix_astle.csv'
# This script is to calculated kinship matrix using marker data, 
# based on classical way of statisticians.
# 
# The kinship matrix will used as genetics code for now, used as neural network model input


#install.packages("statgenGWAS")
#install.packages("reticulate")
library(statgenGWAS)
library(reticulate)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()

# Load reticulate library to interface with Python

use_python("C:/MyPrograms/anaconda/envs/test", required = TRUE)

# Import the dill library
dill <- import("dill")
builtins <- import_builtins()

file_directory <- "../temporary/"
# Initialize an empty list to store the data from each dill file
data_list <- list()
#genotype_list <-c(33, 106, 122, 133, 5, 30, 218, 2, 17, 254, 282, 294, 301, 302, 335, 339, 341, 6, 362)
genotype_list <- read.csv("../processed_data/fouryear_genotypes.csv")$genotype_id

# Loop through all 19 dill files and load them
for (i in genotype_list) {
  # Construct file name
  file_name <- paste0(file_directory, i, "_", ".dill")
  
  # Load the file using dill's load method
  python_data <- dill$load(builtins$open(file_name, "rb"))$numpy() 
  # Convert Python object to an R-friendly format (e.g., data.frame or list)
  data_list[[i]] <- t(as.data.frame(python_data)) # Assumes Python object is convertible
}

# Combine all 19 rows into a single dataframe
final_df <- do.call(rbind, data_list)
rownames(final_df) <- genotype_list

# Print the combined dataframe
print(final_df)
kinship_matrix <- kinship(
  final_df,
  method = c("astle"),
  MAF = NULL,
  denominator = NULL
)
#colnames(kinship_matrix)<-genotype_list
# Save as a CSV (optional)
write.csv(kinship_matrix, "../temporary/kinship_matrix_astle_all_present_genotype.csv", row.names = TRUE)

