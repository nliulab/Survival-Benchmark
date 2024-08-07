
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

import sksurv
from sklearn.preprocessing import StandardScaler
from sksurv.column import encode_categorical
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sklearn.model_selection import GridSearchCV, KFold
from sksurv.metrics import concordance_index_censored, integrated_brier_score


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


#-----------------------------------------------------------------------
# param selection
#-----------------------------------------------------------------------
param_grid = {
    'n_estimators': [500, 1000],
    'max_depth': [2, 3, 5],
    'min_samples_split': [3, 5, 7]
}

def c_index_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    return concordance_index_censored(y['event'], y['time'], y_pred)[0]

gbm = GradientBoostingSurvivalAnalysis()
cv = KFold(n_splits=5, shuffle=True, random_state=20)
grid_search = GridSearchCV(estimator = gbm, param_grid = param_grid, 
                           scoring = c_index_scorer, cv=cv, n_jobs=-1)
grid_search.fit(train_feature_new, train_target_new)
cv_results = pd.DataFrame(grid_search.cv_results_)


best_estimator = grid_search.best_estimator_
print("Best estimator: ", best_estimator)

best_params = grid_search.best_params_
print("Best parameters found: ", best_params)

best_c_index = grid_search.best_score_
print("Best C-index score: ", best_c_index)

###------------------------------------------------
### Model estimation
###------------------------------------------------
best_estimator.fit(train_feature_new, train_target_new)

##-----------------------------------------------------------
# C-index  
##----------------------------------------------------------
cindex_GBM = best_estimator.score(test_feature_new, test_target_new)
print(cindex_GBM)

##-----------------------------------------------------------
# Brier score
##----------------------------------------------------------
GBoost_pred = best_estimator.predict_survival_function(test_feature_new) 
max_x_values = [sf.x.max() for sf in GBoost_pred]
grad_times = np.arange(df_test.label_time.min()+0.1, max(max_x_values)-0.1, 1) 
GBoost_surv_prob = np.row_stack([fn(grad_times) for fn in GBoost_pred])
ibs_GBM = integrated_brier_score(train_target_new, test_target_new, GBoost_surv_prob, grad_times)
print(ibs_GBM)

#--------------------------------------------------------
#  Calculate Confidence interval
#--------------------------------------------------------
if __name__ == '__main__':
    result = helpers.bootstrap_ML(best_estimator, test_target_new, test_feature_new, len(test_feature_new), 100, 0.95)
    print(result)

result.to_csv('results/result_GBM.csv')
#--------------------------------------------------------
#  Variable importance
#--------------------------------------------------------
VIMP = best_estimator.feature_importances_
print("Feature Importance Scores:", VIMP)
df_vimp = {'name': train_feature_new.columns, 'vimp': VIMP}
sort_names = [name for _, name in sorted(zip(df_vimp['vimp'], df_vimp['name']), reverse=False)]
sort_df = sorted(df_vimp['vimp'], reverse=False)

# Visualize feature importance scores
plt.style.use('default')
_, ax = plt.subplots(figsize=(6, 8))
df_vimp_temp = pd.DataFrame(sort_df)
df_vimp_temp.plot.barh(ax=ax, legend=False)
ax.set_xlabel("Variable importance")
plt.yticks(range(len(VIMP)), sort_names)
ax.grid(True)
plt.savefig('vimp_GBM.png')

##---------------------------------------------------
## selection
##---------------------------------------------------
variables = sort_names[-8:]

train_feature_select = train_feature_new[variables]
test_feature_select = test_feature_new[variables]

est = best_estimator.fit(train_feature_select, train_target_new) 
# cindex = est.score(test_feature_select, test_target_new)
# print(cindex)


if __name__ == '__main__':
    result_select = helpers.bootstrap_ML(est, test_target_new, test_feature_select, len(test_feature_select), 100, 0.95)
    print(result_select)

result_select.to_csv('results/result_GBM_vimp.csv')
