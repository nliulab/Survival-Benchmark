
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

import sksurv
from lifelines import CoxPHFitter   
from sklearn.preprocessing import StandardScaler
from sksurv.column import encode_categorical
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

###------------------------------------------------
### Stepwise CoxpH 
###------------------------------------------------
cph_est = helpers.forward_select(df_train_ohe, 'label_status', 'label_time' )

## variable selection
selected_vars = cph_est.columns
cph_train = df_train_ohe[selected_vars]
cph_test = df_test_ohe[selected_vars]

## training model
cph = CoxPHFitter(penalizer = 0.01)
cph.fit(cph_train,  duration_col='label_time', event_col='label_status')
cph.print_summary()

#-------------------------------------------------------
# Target and Feature transform 
#--------------------------------------------------------
get_target = lambda df: (df['label_time'].values, df['label_status'].values)
y_train = durations_train, events_train = get_target(cph_train)
y_test = durations_test, events_test = get_target(cph_test)

train_target_new = sksurv.util.Surv.from_arrays(event= events_train, time=durations_train)
test_target_new = sksurv.util.Surv.from_arrays(event= events_test, time=durations_test)

##------------------------------------------------------
## C-index  
##------------------------------------------------------
cindex_cph = cph.score(cph_test, scoring_method="concordance_index")
print(cindex_cph) 

##------------------------------------------------------
## IBS  
##------------------------------------------------------
pre_surv = cph.predict_survival_function(cph_test).transpose().to_numpy()
grad_times = np.arange(cph_test.label_time.min()+0.1, cph_test.label_time.max()-0.1, 1)
surv_prob = pre_surv[:, :len(grad_times)]
ibs_cph = integrated_brier_score(train_target_new, test_target_new, surv_prob, grad_times)
print(ibs_cph)
  
#--------------------------------------------------------
#  Calculate Confidence interval
#--------------------------------------------------------
if __name__ == '__main__':
    result = helpers.bootstrap_stepwise(cph, pre_surv, test_target_new, cph_test, len(cph_test), 100, 0.95)
    print(result)

result.to_csv('results/result_stepwiseCox.csv')
#--------------------------------------------------------
#  Variable importance
#--------------------------------------------------------
xnames = cph_train.drop(columns=['label_time', 'label_status']).columns

plt.style.use('default')
_, ax = plt.subplots(figsize=(6, 8))
cph.plot(ax=ax)
ax.set_xlabel("Coefficient")
plt.yticks(range(len(xnames)), xnames)
ax.grid(True)
plt.savefig('vimp_stepwiseCox.png')
    
