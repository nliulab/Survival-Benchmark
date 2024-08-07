rm(list = ls())
library(survival)
library(pec)
# source("scripts/functions.R")

##----- Load data -----------------------------------------------

working_dir <- "C:/Users/zoewa/Desktop/Deep_survival/code/Survival_Benchmark"
train_set <- read.csv(file.path(working_dir, paste("data/train_set")), stringsAsFactors = TRUE)
val_set <- read.csv(file.path(working_dir, paste("data/val_set")), stringsAsFactors = TRUE)
test_set <- read.csv(file.path(working_dir, paste("data/test_set")), stringsAsFactors = TRUE)



## ===== Cox training =====================================
fit_cox_train <- coxph(Surv(label_time, label_status)~ ., 
                       x=TRUE, y=TRUE,
                       data = train_set)
#summary(fit_cox_train)

## ===== predict ==========================================
pre_cox <- predict(fit_cox_train, newdata=test_set, type='lp')

## ========================================================
# Evaluation Metrics:
# (1) C-index (Concordance index)
# (2) BS(t) -----> Intergal Brier Score (IBS)
## ========================================================

## ----- C-index ------------------------------------------
Surv_test <- Surv(test_set$label_time, test_set$label_status)
AUC_c_test <- concordancefit(Surv_test, -pre_cox)
cox_cindex <- AUC_c_test$concordance
print(cox_cindex)

### ---- BS and IBS ---------------------------------------
variables <- colnames(test_set)[-c(length(names(test_set))-1, length(names(test_set)) )]
surv_formula <- as.formula(paste0("Surv(label_time,label_status)~",
                                  paste0(variables,collapse="+")))                                  
                        
brier_cox <- pec::pec( object = fit_cox_train,  data = test_set, 
                       formula = surv_formula, 
                       splitMethod = "none")

survfit_test <- survfit(fit_cox_train, data = test_set)
event_times <- summary(survfit_test)$time #times of all events
cox_ibs <- crps(brier_cox, start = min(event_times), times = max(event_times))
print(cox_ibs[2])

## ================================================================
## Bootstrap for CI
## ================================================================
source("scripts/R/functions.R")
set.seed(15)
print_result_coxph <- result_performance_ci(pre_cox, test_set, fit_cox_train, surv_formula, n_boot = 100)
print_result_coxph

#write.csv(print_result_coxph, file.path(working_dir, paste("results/result_coxph.csv")), row.names = TRUE)
