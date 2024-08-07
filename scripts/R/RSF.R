rm(list = ls())
library(randomForestSRC)
library(survival)
library(pec)
library(tidyverse)

##----- Load data -----------------------------------------------
setwd(dirname(rstudioapi::getSourceEditorContext()$path))
working_dir <- "../../../data/"
train_set <- read.csv(file.path(working_dir, paste("train_set")), stringsAsFactors = TRUE)
val_set <- read.csv(file.path(working_dir, paste("val_set")), stringsAsFactors = TRUE)
test_set <- read.csv(file.path(working_dir, paste("test_set")), stringsAsFactors = TRUE)

des_vars <- c("Vital_G")
train_set[des_vars] <- lapply(train_set[des_vars], factor)
val_set[des_vars] <- lapply(val_set[des_vars], factor)
test_set[des_vars] <- lapply(test_set[des_vars], factor)
###===== param grid ========================================

param_grid <- expand.grid(
  ntree = c(500, 1000, 1500),
  mytre = c(2, 3),
  nodesize = c(3, 5, 10, 15),
  nsplit = c(3, 5, 10)
)

###===== CV search ========================================

cv_rfsrc <- function(data, param){
  set.seed(30)
  n<- nrow(data)
  folds <- sample(1:5, n, replace = TRUE)
  results <- sapply(1:5, function(fold){
    train <- data[folds != fold, ]
    test <- data[folds == fold, ]
    model <- rfsrc(Surv(label_time, label_status)~., data = train, 
                   save.memory = TRUE,
                   ntree = param$ntree, mtry = param$mtry, 
                   nodesize = param$nodesize, nsplit = param$nsplit,
                   seed = 30)
    pred <- predict(model, newdata = test)$predicted
    cindex <- 1 - get.cindex(test$label_time, test$label_status, pred)
    return(cindex)
  })
  return(mean(results))
}

results <- apply(param_grid, 1, function(row){
  param <- as.list(row)
  cv_rfsrc(train_set, param)
})

best_params <- param_grid[which.max(results), ]
print(best_params)

###===== rsf training ======================================

fit_rsf <- randomForestSRC::rfsrc(Surv(label_time, label_status)~., data= train_set, 
                                  save.memory = TRUE, ntree=best_params$ntree,
                                  mtry = best_params$mtry,
                                  nodesize = best_params$nodesize,
                                  nsplit = best_params$nsplit, 
                                  seed = 20 )

## ===== predict ===========================================
pre_rsf <- predict(fit_rsf, newdata = test_set)

## ========================================================
# Evaluation Metrics:
# (1) C-index (Concordance index)
# (2) BS(t) -----> Intergal Brier Score (IBS)
## ========================================================


## ----- C-index ------------------------------------------
rsf_cindex <- randomForestSRC::get.cindex(time = test_set$label_time, 
                                          censoring = test_set$label_status, 
                                          1-pre_rsf$predicted)
print(rsf_cindex)

# ## another method for C-index
# Surv_test <- Surv(test_set$label_time, test_set$label_status)
# AUC_c_test <- concordancefit(Surv_test, -pre_rsf$predicted)
# rsf_cindex2 <- AUC_c_test$concordance
# print(rsf_cindex2)

## ----- BS & IBS ------------------------------------------
surv_formula <- Surv(label_time, label_status) ~ 1 
brier_rsf <- pec::pec( object = fit_rsf, data = test_set, 
                       formula = surv_formula, 
                       splitMethod = "none")
rsf_ibs <- crps(brier_rsf, start = min(test_set$label_time), times = max(test_set$label_time))
print(rsf_ibs[2])


## ================================================================
## Bootstrap for CI
## ================================================================
source("scripts/R/functions.R")
set.seed(20)
print_result_rsf <- result_performance_ci(pre_rsf$predicted, test_set, fit_rsf, surv_formula, n_boot = 100)
print_result_rsf

write.csv(print_result_rsf, file.path(working_dir, paste("results/result_rsf.csv")), row.names = TRUE)

