
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

import sksurv
from sksurv.linear_model import CoxnetSurvivalAnalysis 
from sklearn.preprocessing import StandardScaler
from sksurv.column import encode_categorical

import warnings
from sklearn.exceptions import FitFailedWarning
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, KFold
from sksurv.metrics import integrated_brier_score

# Note that on gpu, there is still some randomness.
import helpers
SEED = 20 
helpers.seed_everything(SEED)

#--------------------------------------------------------
#  load data
#--------------------------------------------------------
working_dir = "~/Survival_Benchmark/data"
df_train = pd.read_csv(os.path.join(working_dir, "train_set"),  index_col = False)
df_val = pd.read_csv(os.path.join(working_dir, "val_set"), index_col = False)
df_test = pd.read_csv(os.path.join(working_dir, "test_set"), index_col = False)

#--------------------------------------------------------
#  variables-- categories vs continuous
#--------------------------------------------------------
cols_cat = ['Vital_G']  # Categories variables
y_name = ['label_status', 'label_time'] # Censor status & observed time
cols_standardize = [x for x in df_train.columns if x not in cols_cat + y_name]


#--------------------------------------------------------
#  continuous -- standardize
#--------------------------------------------------------
sta = StandardScaler()
df_train[cols_standardize] = sta.fit_transform(df_train[cols_standardize])
df_val[cols_standardize] = sta.fit_transform(df_val[cols_standardize])
df_test[cols_standardize] = sta.fit_transform(df_test[cols_standardize])

#--------------------------------------------------------
#  categories -- onehot encoding
#--------------------------------------------------------
df_train_ohe = encode_categorical(df_train, columns = cols_cat)
df_val_ohe = encode_categorical(df_val, columns = cols_cat)
df_test_ohe = encode_categorical(df_test, columns = cols_cat)

#--------------------------------------------------------
#  label transform
#--------------------------------------------------------
get_target = lambda df: (df['label_time'].values, df['label_status'].values)
y_train = durations_train, events_train = get_target(df_train_ohe)
y_test = durations_test, events_test = get_target(df_test_ohe)

train_target_new = sksurv.util.Surv.from_arrays(event= events_train, time=durations_train)
test_target_new = sksurv.util.Surv.from_arrays(event= events_test, time=durations_test)

train_feature_new = df_train_ohe.drop(["label_status", "label_time"], axis=1)
test_feature_new = df_test_ohe.drop(["label_status", "label_time"], axis=1)


##==========================================================
# CoxPH EN penalty model
##==========================================================
## Pipe
coxnet_pipe = make_pipeline(
    StandardScaler(),
    CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=0.01, max_iter=10)
)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FitFailedWarning)
coxnet_pipe.fit(train_feature_new, train_target_new) 
    
##-----------------------------------------------------------
# Find the best alpha
##----------------------------------------------------------
estimated_alphas = coxnet_pipe.named_steps["coxnetsurvivalanalysis"].alphas_
cv = KFold(n_splits=5, shuffle=True, random_state=0)
gcv = GridSearchCV(
    make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=0.9)),
    param_grid={"coxnetsurvivalanalysis__alphas": [[v] for v in estimated_alphas]},
    cv=cv,
    error_score=0.5,
    n_jobs=1).fit(train_feature_new, train_target_new)

cv_results = pd.DataFrame(gcv.cv_results_)
cv_results


##-----------------------------------------------------------
# plot
##----------------------------------------------------------
alphas = cv_results.param_coxnetsurvivalanalysis__alphas.map(lambda x: x[0])
mean = cv_results.mean_test_score
std = cv_results.std_test_score

plt.style.use('default')
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(alphas, mean)
ax.fill_between(alphas, mean - std, mean + std, alpha=.15)
ax.set_xscale("log")
ax.set_ylabel("concordance index")
ax.set_xlabel("alpha")
ax.axvline(gcv.best_params_["coxnetsurvivalanalysis__alphas"][0], c="C1")
ax.axhline(0.5, color="grey", linestyle="--")
ax.grid(True)

##-----------------------------------------------------------
# Fit model
##----------------------------------------------------------
best_model = gcv.best_estimator_.named_steps["coxnetsurvivalanalysis"]
best_coefs = pd.DataFrame(
    best_model.coef_,
    index=train_feature_new.columns,
    columns=["coefficient"]
)

non_zero = np.sum(best_coefs.iloc[:, 0] != 0)
print("Number of non-zero coefficients: {}".format(non_zero))

non_zero_coefs = best_coefs.query("coefficient != 0")
coef_order = non_zero_coefs.abs().sort_values("coefficient").index

#--------------------------------------------------------
#  Variable importance
#--------------------------------------------------------
plt.style.use('default')
_, ax = plt.subplots(figsize=(6, 8))
non_zero_coefs.loc[coef_order].plot.barh(ax=ax, legend=False)
ax.set_xlabel("coefficient")
#plt.yticks(range(28), xnames)
ax.grid(True)
plt.savefig('vimp_CoxEN.png')

##-----------------------------------------------------------
# Prediction  
##----------------------------------------------------------
coxnet_pred = make_pipeline(
    StandardScaler(),
    CoxnetSurvivalAnalysis(l1_ratio=0.9, fit_baseline_model=True)
)
coxnet_pred.set_params(**gcv.best_params_)
coxnet_pred.fit(train_feature_new, train_target_new)

##-----------------------------------------------------------
# C-index  
##----------------------------------------------------------
cindex_coxnet_best = coxnet_pred.score(test_feature_new, test_target_new)
print(cindex_coxnet_best)

##-----------------------------------------------------------
# Brier score
##----------------------------------------------------------
surv_fns = coxnet_pred.predict_survival_function(test_feature_new)
grad_times = np.arange(df_test.label_time.min(), df_test.label_time.max(), 1)
#coxnet_surv_prob = np.asarray([[fn(t) for t in grad_times] for fn in surv_fns])
coxnet_surv_prob = np.row_stack([fn(grad_times) for fn in surv_fns])
coxnet_ibs = integrated_brier_score(test_target_new, test_target_new, coxnet_surv_prob, grad_times)
print(coxnet_ibs)


#--------------------------------------------------------
#  Calculate Confidence interval
#--------------------------------------------------------

if __name__ == '__main__':
    result = helpers.bootstrap_ML(coxnet_pred, test_target_new, test_feature_new, len(test_feature_new), 100, 0.95)
    print(result)

result.to_csv('results/result_coxen.csv')

