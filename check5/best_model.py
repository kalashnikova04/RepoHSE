import numpy as np
import random
import optuna
from catboost import CatBoostClassifier
from objective_catboost import *

np.random.seed(42)
random.seed(42)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=15, timeout=6000)

catboost_params ={
    'iterations': study.best_trial.params['iterations'],
    'depth': 8,
    'min_data_in_leaf': study.best_trial.params['min_data_in_leaf'],
    'learning_rate': study.best_trial.params['learning_rate'],
    'eval_metric': 'TotalF1:average=Macro',
    "loss_function": 'MultiClass',
    'task_type': 'GPU',
    'verbose': 300,
    'early_stopping_rounds': study.best_trial.params['early_stopping_rounds']}


model = CatBoostClassifier(**catboost_params) 
model.fit(train_pool, eval_set=valid_pool)

preds = model.predict(X_test)

probs = model.predict_proba(X_test)