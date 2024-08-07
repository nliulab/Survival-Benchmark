import numpy as np
import pandas as pd
import torch
import random
import os
from lifelines import CoxPHFitter 
from pycox.evaluation import EvalSurv
from sksurv.metrics import integrated_brier_score

def seed_everything(seed):
    """"
    Seed everything.
    """   
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

##=== The summary results for nerual network (NN)
def bootstrap_NN(model, durations_test, events_test, x_test, n, B, c):
 
    cindex_sample_result_arr = [] 
    ibs_sample_result_arr = []
    nbll_sample_result_arr = []
    for i in range(B):
       index_n = np.random.randint(0, n, size= n)
       durations_test_sample = durations_test[index_n]
       events_test_sample = events_test[index_n]
       x_test_sample = x_test[index_n]
       surv_sample = model.predict_surv_df(x_test_sample) 
       ev_sample = EvalSurv(surv_sample, durations_test_sample, events_test_sample, censor_surv='km')  
       
       #C-index
       cindex_sample_result = ev_sample.concordance_td()
       cindex_sample_result_arr.append(cindex_sample_result)

       #IBS
       times = np.linspace(durations_test_sample.min(), durations_test_sample.max(), 100)
       ibs_sample_result = ev_sample.integrated_brier_score(times) 
       ibs_sample_result_arr.append(ibs_sample_result)
       
       #NBLL
       nbll_sample_result = ev_sample.integrated_nbll(times) 
       nbll_sample_result_arr.append(nbll_sample_result)
    
    aa = 1 - c
    k1 = int (B *  aa/2)  
    k2 = int (B * (1 - aa/2))
    
    cindex_sample_arr_sorted = sorted(cindex_sample_result_arr)
    cindex_lower_quantile = cindex_sample_arr_sorted[k1]
    cindex_upper_quantile = cindex_sample_arr_sorted[k2]
    cindex_mean = np.mean(cindex_sample_result_arr)
    cindex_se = np.std(cindex_sample_result_arr, ddof =1 )
   
    
    ibs_sample_arr_sorted = sorted(ibs_sample_result_arr)
    ibs_lower_quantile = ibs_sample_arr_sorted[k1]
    ibs_upper_quantile = ibs_sample_arr_sorted[k2]  
    ibs_mean = np.mean(ibs_sample_result_arr)
    ibs_se = np.std(ibs_sample_result_arr, ddof =1 )

    
    nbll_sample_arr_sorted = sorted(nbll_sample_result_arr)
    nbll_lower_quantile = nbll_sample_arr_sorted[k1]
    nbll_upper_quantile = nbll_sample_arr_sorted[k2] 
    nbll_mean = np.mean(nbll_sample_result_arr)
    nbll_se = np.std(nbll_sample_result_arr, ddof =1 )

    
    cindex_result = [cindex_mean, cindex_se, cindex_lower_quantile, cindex_upper_quantile]
    ibs_result = [ibs_mean, ibs_se, ibs_lower_quantile, ibs_upper_quantile]
    nbll_result = [nbll_mean, nbll_se, nbll_lower_quantile, nbll_upper_quantile]
    sum_result = pd.DataFrame([cindex_result, ibs_result, nbll_result], 
                              index= ['Cindex','IBS','NBLL'],
                              columns=['Mean','SE','lower_q','upper_q'] )
    
    return sum_result

##=== The summary results for machine learning (ML)
def bootstrap_ML(model, test_target_new, test_feature_new,  n, B, c):
 
    cindex_sample_result_arr = [] 
    ibs_sample_result_arr = []
    for i in range(B):
       index_n = np.random.randint(0, n, size= n)
       test_target_sample = test_target_new[index_n]
       test_feature_sample = test_feature_new.loc[index_n]
       
       cindex_sample = model.score(test_feature_sample, test_target_sample)
       cindex_sample_result_arr.append( cindex_sample )
       
       #times = np.linspace(test_target_sample['time'].min()+0.1, test_target_sample['time'].max()-0.1, 100)
       times = np.arange(test_target_sample['time'].min(), test_target_sample['time'].max(), 1)
       surv_sample = model.predict_survival_function(test_feature_sample) 
       preds_sample = np.row_stack([fn(times) for fn in surv_sample])
       ibs_sample = integrated_brier_score(test_target_sample, test_target_sample, preds_sample, times)
       ibs_sample_result_arr.append( ibs_sample )
       
    aa = 1 - c
    k1 = int (B *  aa/2)  
    k2 = int (B * (1 - aa/2))
    
    cindex_sample_arr_sorted = sorted(cindex_sample_result_arr)
    cindex_lower_quantile = cindex_sample_arr_sorted[k1]
    cindex_upper_quantile = cindex_sample_arr_sorted[k2]
    cindex_mean = np.mean(cindex_sample_result_arr)
    cindex_se = np.std(cindex_sample_result_arr, ddof =1 )
    
    ibs_sample_arr_sorted = sorted(ibs_sample_result_arr)
    ibs_lower_quantile = ibs_sample_arr_sorted[k1]
    ibs_upper_quantile = ibs_sample_arr_sorted[k2]  
    ibs_mean = np.mean(ibs_sample_result_arr)
    ibs_se = np.std(ibs_sample_result_arr, ddof =1 )
     
    cindex_result = [cindex_mean, cindex_se, cindex_lower_quantile, cindex_upper_quantile]
    ibs_result = [ibs_mean, ibs_se, ibs_lower_quantile, ibs_upper_quantile]
   
    sum_result = pd.DataFrame([cindex_result, ibs_result], 
                              index= ['Cindex','IBS'],
                              columns=['Mean','SE','lower_q','upper_q'] )
    return sum_result

##=== The summary results for stepwise CoxPH
def bootstrap_stepwise(model, pre_surv, test_target_new, testset, n, B, c):
 
    cindex_sample_result_arr = [] 
    ibs_sample_result_arr = []
    for i in range(B):
       index_n = np.random.randint(0, n, size= n)
       #durations_test_sample = durations_test[index_n]
       test_target_sample = test_target_new[index_n]
       test_sample = testset.iloc[index_n,:]

       cindex_sample_result = model.score(test_sample, scoring_method="concordance_index")
       cindex_sample_result_arr.append(cindex_sample_result)
       
       times = np.arange(test_sample['label_time'].min()+0.1, test_sample['label_time'].max()-0.1, 1)
       pre_surv_sample = pre_surv[index_n]
       surv_prob_sample = pre_surv_sample[:, :len(times)]
       ibs_sample_result = integrated_brier_score(test_target_sample, test_target_sample, surv_prob_sample, times)
       ibs_sample_result_arr.append(ibs_sample_result)
       
    aa = 1 - c
    k1 = int (B *  aa/2)  
    k2 = int (B * (1 - aa/2))
    
    cindex_sample_arr_sorted = sorted(cindex_sample_result_arr)
    cindex_lower_quantile = cindex_sample_arr_sorted[k1]
    cindex_upper_quantile = cindex_sample_arr_sorted[k2]
    cindex_mean = np.mean(cindex_sample_result_arr)
    cindex_se = np.std(cindex_sample_result_arr, ddof =1 )

    ibs_sample_arr_sorted = sorted(ibs_sample_result_arr)
    ibs_lower_quantile = ibs_sample_arr_sorted[k1]
    ibs_upper_quantile = ibs_sample_arr_sorted[k2]  
    ibs_mean = np.mean(ibs_sample_result_arr)
    ibs_se = np.std(ibs_sample_result_arr, ddof =1 )
       
    cindex_result = [cindex_mean, cindex_se, cindex_lower_quantile, cindex_upper_quantile]
    ibs_result = [ibs_mean, ibs_se, ibs_lower_quantile, ibs_upper_quantile]
    sum_result = pd.DataFrame([cindex_result, ibs_result], 
                              index= ['Cindex','IBS'],
                              columns=['Mean','SE','lower_q','upper_q'] )
    return sum_result

   
def forward_select(df, E, T):

    variate=list(df.columns)  #Initial variables 
    variate.remove(E)  
    variate.remove(T)  
    selected = []  
    best_score = float('inf')  # Initial value, the smaller the AIC, the better, so the initial value is set to infinity
 
    n = 1
    while True:
        scores_with_candidates = []
        for candidate in variate:
            data = df[selected + [candidate, E, T]].copy()
            cph = CoxPHFitter(penalizer=0.01)  
            score = cph.fit(data, duration_col=T, event_col=E).AIC_partial_
            scores_with_candidates.append((abs(score), candidate))
            
        scores_with_candidates.sort()  # Sort from smallest to largest
        current_best_score, current_best_candidate = scores_with_candidates[0]  # Select the minimum value
        print('round {}\nbest score is {:.4f}, selected candidantes '\
              'are {} \n '.format(n, current_best_score, selected + [current_best_candidate]))
        n += 1
        if current_best_score < best_score :
            variate.remove(current_best_candidate)
            selected.append(current_best_candidate)
            best_score = current_best_score
        else:
            break    
      
    data = df[selected + [E, T]].copy()
    return data


def predict_risk(x, model):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        return model.predict_surv_df(x).iloc[-1].to_numpy().astype(np.float32)


