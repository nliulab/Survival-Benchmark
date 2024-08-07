remove(list = ls())
library(AutoScore)
library(dplyr)
library(survival)
library(pec)
library(randomForestSRC)
library(ggplot2)

##----- Load data -----------------------------------------------

working_dir <- "C:/Users/zoewa/Desktop/Deep_survival/code/Survival_Benchmark"
train_set <- read.csv(file.path(working_dir, paste("data/train_set")), stringsAsFactors = TRUE)
val_set <- read.csv(file.path(working_dir, paste("data/val_set")), stringsAsFactors = TRUE)
test_set <- read.csv(file.path(working_dir, paste("data/test_set")), stringsAsFactors = TRUE)

# ## Another method: Load data and split data
# data("sample_data_survival_small")
# set.seed(4)
# out_split <- AutoScore::split_data(data = sample_data_survival_small, ratio = c(0.7, 0, 0.3),
#                         cross_validation = TRUE)
# train_set <- out_split$train_set
# validation_set <- out_split$validation_set
# test_set <- out_split$test_set

##=================================================================
## Step 1 : generate variable ranking list
##=================================================================
set.seed(30)

ranking <- AutoScore_rank_Survival(train_set = train_set, ntree = 500) 

### ---- fast generate the variable ranking list and can tune more para----------------------
source('scripts/R/functions.R')
ranking2 <- AutoScore_rank_Survival_fast(train_set, ntree= 500, mytre=2, nodesize=3, nsplit=3)
plot_importance(ranking2)
### ------------------------------------------------------------------------------------------

##=================================================================
## Step 2 : Select variables with parsimony plot
##=================================================================
#quant <- c(0, 0.2, 0.4, 0.6,0.8, 1)
quant <- c(0, 0.1, 0.3, 0.7,0.9, 1)
iAUC <- AutoScore_parsimony_Survival(
  train_set,
  val_set,
  rank = ranking,
  max_score = 100,
  n_min = 1,
  n_max = 20,
  categorize = "quantile",
  quantiles = quant,
  auc_lim_min = 0.6,
  auc_lim_max = "adaptive"
)

#write.csv(data.frame(iAUC), file = "iAUC.csv")

##=================================================================
## Step 3: Generate initial scores with final variables
##=================================================================

final_variables <- names(ranking[c(1:8)])
var_times <- c(5,10,15,30)

cut_vec <- AutoScore_weighting_Survival( 
  train_set,
  val_set,
  final_variables,
  max_score = 100,
  categorize = "quantile",
  quantiles = quant,
  time_point = var_times
)


##=================================================================
## Step 4: fine-tune initial score from step 3 (based on validation dataset)
##=================================================================

scoring_table <- 
  AutoScore_fine_tuning_Survival(train_set,
                                 val_set,
                                 final_variables,
                                 cut_vec,
                                 max_score = 100,
                                 time_point = var_times)

##=================================================================
## Step 5: evaluate final risk scores on test dataset
##=================================================================

pred_score <-
  AutoScore_testing_Survival(
    test_set,
    final_variables,
    cut_vec,
    scoring_table,
    threshold = "best",
    with_label = TRUE,
    time_point = var_times
  )

head(pred_score)



## ================================================================
## Bootstrap for CI of IBS
## ================================================================
source("scripts/R/functions.R")

risk_score <- pred_score$pred_score
IBS_autoscore <- get_brier_score_ci(test_set, scoring_table, 
                                    final_variables, cut_vec, n_boot = 100)
print(IBS_autoscore)
 

