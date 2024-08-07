## Survival modeling using deep learning, machine learning and statistical methods: A comparative analysis for predicting mortality after hospital admission

### Introduction 
Survival analysis is essential for studying time-to-event outcomes and providing a dynamic understanding of the probability of an event occurring over time. Various survival analysis techniques, from traditional statistical models to state-of-the-art machine learning algorithms, support healthcare intervention and policy decisions. However, there remains ongoing discussion about their comparative performance.
This study conducted benchmark comparisons of statistical modelling (the accelerated failure time model [AFT][^1], Cox proportional hazards model [CoxPH][^2], Stepwise CoxPH[^3], Elastic net penalized CoxPH [CoxEN][^4]), ensemble machine learning (randon survival forest [RSF][^5], Gradient boosting [GBM][^6]), interpretable machine learning (AutoScore-Survival[^7]), and deep learning (DeepSurv[^8], CoxTime[^9], and DeepHit[^10]).

### System requirements

+ **R packages**: 'dplyr', 'ggplot2', 'survival', 'rms',   'AutoScore', 'randomForestSRC', 'pROC', 'survAUC', 'tsutils'.
+ **Python**: version 3.9 (Windows)
To install the required Python packages, run: ``pip install -r requirements.txt`` .

### A demo for analyzing simulated data
#### 1. Load survival data
+ Read data from CSV or Excel files.
+ For this demo, use the generated survival data samples in the directory `data/sample_data_survival` or `data/sample_data_survival_small`.  
+   `sample_data_survival`  has 20000 simulated samples with survival outcomes, which has the same data distribution as the [MIMIC-III ICU database]([https://mimic.mit.edu/](https://mimic.mit.edu/)).  The corresponding training, validation, test data set are generated and saved in the `data/train_set`, `data/val_set` and `data/test_set`.
+   `sample_data_survival_small`  has 1000 simulated samples with survival outcomes, which are also from MIMIC-III ICU database. The corresponding training, validation, test data set are generated and saved in the `data/train_set_small`, `data/val_set_small` and `data/test_set_small`.

#### 2. Training model
 + The application of different statistical and machine learning frameworks is available in `script`. In addition, the required software and packages are summarized as follows.
 +   Description of various survival techniques
     | |Models|Software (Package)|
     |---|---|---|
     |Traditional statistical model|AFT| R (rms)|
     |      |CoxPH| R (survival)|
     |      |Stepwise CoxPH| Python (lifelines)|
     |      |CoxEN| Python (scikit-surv)|
     |Ensemble machine learning|RSF| R (randomForestRC)|
      |      |GBM| Python (scikit-surv)|
      |Interpretability machine learning|AutoScore-Survival| R (AutoScore)|
      |Feedforward deep neural network|DeepSurv| Python (Pycox)|
      | |CoxTime| Python (Pycox)|
      | |DeepHit| Python (Pycox)|
 
#### 3. Interpretability-performance trade-off

 + Our study incorporates survival metrics such as the concordance index (C-index)[^11] and integrated Brier score (IBS)[^12], which are essential for measuring the model's goodness-of-fit, providing a comprehensive assessment of discrimination, and assisting in the evaluation of calibration and stability of the models.
 + Select the required model in `script` to run,  with the  performance output files stored in the folder `results` and importance plot stored in the folder `figures`.  For instance, run `scripts/Python/DeepSurv.py` the performance results are saved in `results/result_deepsurv.csv` and the importance generated by SHAP analysis is saved in `figures/vimp_deepsurv.png`.
 + The importance of the neural network using SHAP analysis in this demo is represented as follows: 
![Variable importance](https://github.com/nliulab/Survival-Benchmark/blob/main/figures/combined_figures.pdf)


#### 4. Performance test and visualization
We further conducted the Multiple Comparisons with the Best (MCB) test for C-index and IBS measures to assess the statistical significance of different models' performance.

Run script `scripts/R/MCB/mcb_plot.R` to draw MCB plots for C-index and IBS, with results stored in `figures`. The MCB plot in this demo is represented as follows: 
![MCB Plot](https://github.com/nliulab/Survival-Benchmark/blob/main/figures/sum_mcb.pdf)

### Citing Our Work
Z. Wang, et al. "Survival modeling using deep learning, machine learning and statistical methods: A comparative analysis for predicting mortality after hospital admission". arXiv preprint [arXiv:2403.06999](https://arxiv.org/abs/2403.06999) (2024)

### Contact
+ Nan Liu, E-mail: liu.nan@duke-nus.edu.sg
+ Ziwen Wang, E-mail: ziwen.wang@duke-nus.edu.sg
+ Tanujit Chakraborty, E-mail: tanujitisi@gmail.com


### Reference
[^1]: Buckley J, James I: Linear regression with censored data. _Biometrika_ 1979, 66(3):429-436.
[^2]: Cox DR: Regression models and life‐tables. _Journal of the Royal Statistical Society: Series B (Methodological)_ 1972, 34(2):187-202.
[^3]: Liang H, Zou G: Improved AIC selection strategy for survival analysis. _Computational statistics & data analysis_ 2008, 52(5):2538-2548.
[^4]: Zou H, Hastie T: Regularization and variable selection via the elastic net. _Journal of the royal statistical society: series B (statistical methodology)_ 2005, 67(2):301-320.
[^5]: Ishwaran H, Kogalur UB, Blackstone EH, Lauer MS: Random survival forests. _The annals of applied statistics_ 2008, 2(3):841-860.
[^6]: Hothorn T, Bühlmann P, Dudoit S, Molinaro A, Van Der Laan MJ: Survival ensembles. _Biostatistics_ 2006, 7(3):355-373.
[^7]: Xie F, Ning Y, Yuan H, Goldstein BA, Ong MEH, Liu N, Chakraborty B: AutoScore-Survival: Developing interpretable machine learning-based time-to-event scores with right-censored survival data. _Journal of Biomedical Informatics_ 2022, 125:103959.
[^8]: Katzman JL, Shaham U, Cloninger A, Bates J, Jiang T, Kluger Y: DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network. _BMC medical research methodology_ 2018, 18(1):1-12.
[^9]: Kvamme H, Borgan Ø, Scheel I: Time-to-event prediction with neural networks and Cox regression. _arXiv preprint arXiv:190700825_ 2019.
[^10]: Lee C, Zame W, Yoon J, Van Der Schaar M: Deephit: A deep learning approach to survival analysis with competing risks. In: _Proceedings of the AAAI conference on artificial intelligence: 2018_; 2018.
[^11]: Harrell FE, Califf RM, Pryor DB, Lee KL, Rosati RA: Evaluating the yield of medical tests. _Jama_ 1982, 247(18):2543-2546.
[^12]: Harrell Jr FE, Lee KL, Mark DB: Multivariable prognostic models: issues in developing models, evaluating assumptions and adequacy, and measuring and reducing errors. _Statistics in medicine_ 1996, 15(4):361-387.
