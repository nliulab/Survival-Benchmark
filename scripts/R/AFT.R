rm(list = ls())
library(survival)
library(tidyverse)
library(pec)
library(rms)

##----- Load data -----------------------------------------------
setwd(dirname(rstudioapi::getSourceEditorContext()$path))
working_dir <- "../../../data/"
train_set <- read.csv(file.path(working_dir, paste("train_set")), stringsAsFactors = TRUE)
val_set <- read.csv(file.path(working_dir, paste("val_set")), stringsAsFactors = TRUE)
test_set <- read.csv(file.path(working_dir, paste("test_set")), stringsAsFactors = TRUE)

# ## Another method: Load data and split data
# data("sample_data_survival_small")
# set.seed(4)
# out_split <- AutoScore::split_data(data = sample_data_survival_small, ratio = c(0.7, 0, 0.3),
#                         cross_validation = TRUE)
# train_set <- out_split$train_set
# validation_set <- out_split$validation_set
# test_set <- out_split$test_set


## ===== AFT training =====================================

variables <- colnames(test_set)[-c(length(names(test_set))-1, length(names(test_set)) )]
surv_formula <- as.formula(paste0("Surv(label_time,label_status)~",
                                  paste0(variables,collapse="+")))  
fit_aft_train <- psm(surv_formula, data = train_set, dist= "weibull")

## ===== predict ==========================================
pre_aft <- predict(fit_aft_train, newdata=test_set)

## ========================================================
# Evaluation Metrics:
# (1) C-index (Concordance index)
# (2) BS(t) -----> Intergal Brier Score (IBS)
## ========================================================

## ----- C-index -------------------------------------
Surv_test <- Surv(test_set$label_time, test_set$label_status)
AUC_c_test <- concordancefit(Surv_test, pre_aft)
aft_cindex <- AUC_c_test$concordance
print(aft_cindex)

### ----- BS and IBS ----------------------------------
variables <- colnames(test_set)[-c(length(names(test_set))-1, length(names(test_set)) )]
surv_formula <- as.formula(paste0("Surv(label_time,label_status)~",
                                  paste0(variables,collapse="+")))                                  
                        
brier_aft <- pec::pec( object = fit_aft_train,  data = test_set, 
                       formula = surv_formula, 
                       splitMethod = "none")

aft_ibs <- crps(brier_aft, start = min(test_set$label_time), times = max(test_set$label_time) )
print(aft_ibs[2])


## ================================================================
## Bootstrap for CI
## ================================================================
source("scripts/R/functions.R")
set.seed(15)
print_result_aft <- result_performance_ci(-pre_aft, test_set, fit_aft_train, surv_formula, n_boot = 100)
print_result_aft

write.csv(print_result_aft, file.path(working_dir, paste("results/result_aft.csv")), row.names = TRUE)


##== plot effects ============================
aft_model <- survreg(Surv(label_time, label_status) ~ ., data = train_set, dist = "weibull")
coefficients <- summary(aft_model)$table
coefficients <- coefficients[!rownames(coefficients) %in% c("(Intercept)", "Log(scale)"), ]
coefficients_df <- as.data.frame(coefficients)
coefficients_df$Variable <- rownames(coefficients_df)
colnames(coefficients_df)[1] <- "Estimate"
colnames(coefficients_df)[2] <- "Std.Error"

vimp_aft <- ggplot(coefficients_df, aes(x = Estimate, y = Variable)) + 
  geom_point() +
  geom_errorbarh(aes(xmin = Estimate - `Std.Error`, xmax = Estimate + `Std.Error`), height = 0.2) +
  labs(x = "Coefficient", y = "") +
  theme_minimal()

vimp_aft
