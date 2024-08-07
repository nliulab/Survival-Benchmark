
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper 
import torchtuples as tt # Some useful functions

from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
from sksurv.column import encode_categorical
import os
import pandas as pd
import shap

# Note that on gpu, there is still some randomness.
import helpers
SEED = 30 
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
#cols_cat = []  # Categories variables
y_name = ['label_status', 'label_time'] # Censor status & observed time
cols_standardize = [x for x in df_train.columns if x not in cols_cat + y_name]

#--------------------------------------------------------
#  continuous variables -- standardize
#--------------------------------------------------------
sta = StandardScaler()
df_train[cols_standardize] = sta.fit_transform(df_train[cols_standardize])
df_val[cols_standardize] = sta.fit_transform(df_val[cols_standardize])
df_test[cols_standardize] = sta.fit_transform(df_test[cols_standardize])

#--------------------------------------------------------
#  categories variables -- onehot encoding
#--------------------------------------------------------
df_train_ohe = encode_categorical(df_train, columns = cols_cat)
df_val_ohe = encode_categorical(df_val, columns = cols_cat)
df_test_ohe = encode_categorical(df_test, columns = cols_cat)

#--------------------------------------------------------
#  label transform
#--------------------------------------------------------
cols_leave = [x for x in df_train_ohe.columns if x not in y_name]
leave = [(col, None) for col in cols_leave]

x_mapper = DataFrameMapper(leave)
x_train = x_mapper.fit_transform(df_train_ohe).astype('float32')
x_val = x_mapper.fit_transform(df_val_ohe).astype('float32')
x_test = x_mapper.transform(df_test_ohe).astype('float32')

get_target = lambda df: (df['label_time'].values, df['label_status'].values)
y_train = durations_train, events_train = get_target(df_train_ohe)
y_val  = get_target(df_val_ohe)
y_test = durations_test, events_test = get_target(df_test_ohe)

val = x_val, y_val  # for tuning parametric

#--------------------------------------------------------
#  Deepsurv NN
#--------------------------------------------------------
in_features = x_train.shape[1] # number of input features
out_features = 1
batch_norm = True
output_bias = False
num_nodes =[128, 64, 32]
dropout = 0.4

net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout, output_bias= output_bias)
model = CoxPH(net, tt.optim.Adam)

batch_size = 512
lrfinder = model.lr_finder(x_train, y_train, batch_size, tolerance=50)
model.optimizer.set_lr(0.001)
#_=lrfinder.plot()
#lrfinder.get_best_lr()

epochs = 128
callbacks = [tt.callbacks.EarlyStopping()]
verbose = True
log = model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose, val_data=val, val_batch_size=batch_size)

# Validation 
plt.style.use('default')
log.plot()
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Prediction
_ = model.compute_baseline_hazards()
surv = model.predict_surv_df(x_test) 


#--------------------------------------------------------
#  Evaluation -- Cindex
#--------------------------------------------------------
ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')  
cindex = ev.concordance_td() 
print(round(cindex, 5))

#--------------------------------------------------------
#  Evaluation -- Brier score & IBS
#--------------------------------------------------------
time_grid = np.linspace(durations_test.min()+0.001, durations_test.max()-0.001, 100)
ibs = ev.integrated_brier_score(time_grid) 
print(ibs)

#--------------------------------------------------------
#  Evaluation -- Negative binomial log-likelihood
#--------------------------------------------------------
nbll = ev.integrated_nbll(time_grid) 
print(nbll)

#--------------------------------------------------------
#  Calculate Confidence interval
#--------------------------------------------------------
if __name__ == '__main__':
    result = helpers.bootstrap_NN(model, durations_test, events_test, x_test,
                                    n=len(x_test), B=100, c=0.95)
    print(result)

result.to_csv('results/result_deepsurv.csv')
#--------------------------------------------------------
#  SHAP analysis
#--------------------------------------------------------
def wrapped_predict_risk(x):
    return helpers.predict_risk(x, model)

col =  df_train_ohe.drop(["label_status", "label_time"], axis=1).columns
explainer = shap.Explainer(wrapped_predict_risk, x_test, feature_names=col)
shap_values = explainer(x_test) 

# Plotting the SHAP values
#shap.summary_plot(shap_values, x_test)
shap.summary_plot(shap_values, x_test, show=False)
plt.savefig('vimp_Deepsurv')

