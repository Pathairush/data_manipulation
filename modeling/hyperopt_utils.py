# import 
import hyperopt as hp
import lightgbm as lgb

# helper function
def prepare_eval_df():
    '''
    Preparing the datafrme for scoring with Micro / Macro LRAP
    '''
    pass
    return eval_df

def macro_lrap(eval_df):
    '''
    scoring marcro lrap from the evaluation dataframe
    '''
    pass
    return score

# objective function
def objective(params,random_state, data_model):
    
    # extracting data
    X, y, X_val, y_val = data_model

    # initial model
    model = lgb.LGBMClassifier(**params,
    random_state=random_state)
    
    # fitting model
    model.fit(X, y,
    eval_set = [X_val, y_val],
    early_stopping_rounds = 100,
    verbose = 200)

    # prediction
    pred = model.predict_proba(y_val)
    eval_df = prepare_eval_df()

    # scoring model
    score = macro_lrap(eval_df)

    return {'score' : -score, 'status' : hp.STATUS_OK}

# initial hyper parameters space
space = dict()
space['n_estimator'] = hp.quniform('n_estimators', 100, 2000, 1)
space['max_depth'] = hp.uniform('max_depth', 2, 20, 1)
space['learning_rate'] = hp.loguniform('learning_rate', -5, 0)

# trials for logging information
trials = hp.Trials()

# max evaluation round
max_eval = 50

# running optimisation
best = hp.fmin(
    fn = objective,
    space = space,
    algo = tpe.suggest,
    max_evals = max_evals,
    trials = trials
)

print(f'best params is : {best}')
