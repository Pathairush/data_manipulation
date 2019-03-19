#!/usr/bin/env python
# coding: utf-8

# In[2]:


# utility
import os
print(os.getcwd())
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc("font", size=14)
import seaborn as sns
import gc # garbage collection
import time
from contextlib import contextmanager

# model
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold,train_test_split

# debugging
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[16]:


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}min".format(title, (time.time() - t0)/60))


# In[17]:


def one_hot_encoder(df, nan_as_category = True):
    original_columns = df.columns
    catagorical_columns = df.select_dtypes(include='object').columns
    df = pd.get_dummies(df, columns= catagorical_columns, dummy_na= nan_as_category)
    new_columns = df.columns.difference(original_columns)
    return df, new_columns


# In[18]:


def application_train_test(num_readrows=None, nan_as_category = False):
    # read data
    df = pd.read_csv("home-credit-default-risk/application_train.csv", nrows = num_readrows)
    test_df = pd.read_csv("home-credit-default-risk/application_test.csv", nrows = num_readrows)
    print("Train shape: {}, test shape: {}".format(df.shape, test_df.shape))
    df = df.append(test_df).reset_index(drop=True)
    
    # preprocessing
    df = df[df['CODE_GENDER'] != 'XNA'] # Remove 'XNA' type for code gender
    df["DAYS_EMPLOYED"].replace(365243,np.nan,inplace=True) # Replace some weird values
    df['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True) 
    
    # get dummies
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    
    # generate new simple features (percentages)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    
    del test_df
    gc.collect()
    
    return df


# In[19]:


def bureau_and_balance(num_readrows=None, nan_as_category = False):
    bureau = pd.read_csv("home-credit-default-risk/bureau.csv", nrows=num_readrows)
    balanced = pd.read_csv("home-credit-default-risk/bureau_balance.csv", nrows=num_readrows)
    
    # create more feature from HOME CREDIT - BUREAU DATA - FEATURE ENGINEERING
    bureau_loan_count = bureau[['SK_ID_CURR', 'DAYS_CREDIT']].groupby(by = ['SK_ID_CURR'])['DAYS_CREDIT'].count().reset_index().rename(index=str, columns={'DAYS_CREDIT': 'BUREAU_LOAN_COUNT'})
    bureau = bureau.merge(bureau_loan_count, on = ['SK_ID_CURR'], how = 'left')
    bureau_loan_type = bureau[['SK_ID_CURR', 'CREDIT_TYPE']].groupby(by = ['SK_ID_CURR'])['CREDIT_TYPE'].nunique().reset_index().rename(index=str, columns={'CREDIT_TYPE': 'BUREAU_LOAN_TYPES'})
    bureau = bureau.merge(bureau_loan_type, on = ['SK_ID_CURR'], how = 'left')
    bureau['AVERAGE_LOAN_TYPE'] = bureau['BUREAU_LOAN_COUNT']/bureau['BUREAU_LOAN_TYPES']
    del bureau_loan_count,bureau_loan_type
    gc.collect()
    
    # OHE for cat cols
    bureau, bureau_cat_cols = one_hot_encoder(bureau,nan_as_category=nan_as_category)
    balanced, balanced_cat_cols = one_hot_encoder(balanced,nan_as_category=nan_as_category)
    print("bureau shape : {}, balanced shape :{}".format(bureau.shape,balanced.shape))
    
    # impute some values
    bureau.loc[bureau['DAYS_CREDIT_ENDDATE'] < -40000,'DAYS_CREDIT_ENDDATE'] = np.nan
    bureau.loc[bureau['DAYS_CREDIT_UPDATE'] < -40000,'DAYS_CREDIT_UPDATE'] = np.nan
    bureau.loc[bureau['DAYS_ENDDATE_FACT'] < -40000,'DAYS_ENDDATE_FACT'] = np.nan
    
    # Bureau balance: Perform aggregations and merge with bureau.csv
    # the kinds of aggregation
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in balanced_cat_cols:
        bb_aggregations[col] = ['mean']
    # groupby and rename of cols
    bb_agg = balanced.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    # merge the balanced with bureau
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    del balanced, bb_agg
    gc.collect()
    
    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat_cols: cat_aggregations[cat] = ['mean']
    for cat in balanced_cat_cols: cat_aggregations[cat + "_MEAN"] = ['mean']
        
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg


# In[20]:


# Preprocess previous_applications.csv
def previous_applications(num_rows = None, nan_as_category = True):
    prev = pd.read_csv('../02 Home_credit_default_risk/home-credit-default-risk/previous_application.csv', nrows = num_rows)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category= True)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg


# In[21]:


# Preprocess POS_CASH_balance.csv
def pos_cash(num_rows = None, nan_as_category = True):
    pos = pd.read_csv('../02 Home_credit_default_risk/home-credit-default-risk/POS_CASH_balance.csv', nrows = num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg


# In[22]:


def installments_payments(num_rows = None, nan_as_category = True):
    ins = pd.read_csv('../02 Home_credit_default_risk/home-credit-default-risk/installments_payments.csv', nrows = num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category= True)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg


# In[23]:


def credit_card_balance(num_rows = None, nan_as_category = True):
    cc = pd.read_csv('../02 Home_credit_default_risk/home-credit-default-risk/credit_card_balance.csv', nrows = num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)
    
    # impute value
    cc.loc[cc['AMT_DRAWINGS_ATM_CURRENT'] < 0,'AMT_DRAWINGS_ATM_CURRENT'] = np.nan
    cc.loc[cc['AMT_DRAWINGS_CURRENT'] < 0,'AMT_DRAWINGS_CURRENT'] = np.nan
    
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg


# In[24]:


def kfold_lightgbm(df, num_folds, stratified = False, debug= False):
    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()
    
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=1001)
        
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(
            nthread=4,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            silent=-1,
            verbose=-1, )
        

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
            eval_metric= 'auc', verbose= 100, early_stopping_rounds= 200)

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    # Write submission file and plot feature importance
    if not debug:
        test_df['TARGET'] = sub_preds
        test_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_file_name, index= False)
    display_importances(feature_importance_df)
    return feature_importance_df


# In[25]:


# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')


# In[26]:


def drop_multicolinearity(df,threshold=0.9):
    print("the shape before drop_multicolinearity = {}".format(df.shape))
    # check colinearity
    threshold = threshold;
    corr_matrix = df.corr().abs()
    corr_matrix.head()
    # get upper matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    upper.head()
    to_drop = [cols for cols in upper.columns if any(upper[cols]>threshold)]
    print("There are {} cols that have r more than 0.9".format(len(to_drop)))
    df.drop(columns=to_drop,inplace=True)
    print("the shape after drop_multicolinearity = {}".format(df.shape))
    del corr_matrix, upper
    gc.collect()
    return df


# In[27]:


def drop_unimportance_feature(df):
    # preprocessing
    df = df[~df['TARGET'].isna()];
    train_labels = df.pop('TARGET');
    train = df;
    
    # Initialize an empty array to hold feature importances
    feature_importances = np.zeros(train.shape[1])

    # Create the model with several hyperparameters
    model = LGBMClassifier(objective='binary', boosting_type = 'goss', n_estimators = 10000, class_weight = 'balanced')

    # Fit the model twice to avoid overfitting
    for i in range(2):

        # Split into training and validation set
        train_features, valid_features, train_y, valid_y = train_test_split(train, train_labels, test_size = 0.25, 
                                                                            random_state = i)

        # Train using early stopping
        model.fit(train_features, train_y, early_stopping_rounds=100, eval_set = [(valid_features, valid_y)], 
                  eval_metric = 'auc', verbose = 200)

        # Record the feature importances
        feature_importances += model.feature_importances_

    # Make sure to average feature importances! 
    feature_importances = feature_importances / 2
    feature_importances = pd.DataFrame({'feature': list(train.columns),
                                        'importance': feature_importances}).sort_values('importance', ascending = False)
    display(feature_importances.head())

    # Find the features with zero importance
    zero_features = list(feature_importances[feature_importances['importance'] == 0.0]['feature'])
    print('There are %d features with 0.0 importance' % len(zero_features))
    display(feature_importances.tail())

    return zero_features


# In[4]:


def main(debug = False):
    num_rows = 10000 if debug else None
    df = application_train_test(num_rows)
    with timer("Process bureau and bureau_balance"):
        bureau = bureau_and_balance(num_rows)
        print("Bureau df shape:", bureau.shape)
        df = df.join(bureau, how='left', on='SK_ID_CURR')
        del bureau
        gc.collect()
    with timer("Process previous_applications"):
        prev = previous_applications(num_rows)
        print("Previous applications df shape:", prev.shape)
        df = df.join(prev, how='left', on='SK_ID_CURR')
        del prev
        gc.collect()
    with timer("Process POS-CASH balance"):
        pos = pos_cash(num_rows)
        print("Pos-cash balance df shape:", pos.shape)
        df = df.join(pos, how='left', on='SK_ID_CURR')
        del pos
        gc.collect()
    with timer("Process installments payments"):
        ins = installments_payments(num_rows)
        print("Installments payments df shape:", ins.shape)
        df = df.join(ins, how='left', on='SK_ID_CURR')
        del ins
        gc.collect()
    with timer("Process credit card balance"):
        cc = credit_card_balance(num_rows)
        print("Credit card balance df shape:", cc.shape)
        df = df.join(cc, how='left', on='SK_ID_CURR')
        print(df.shape)
        del cc
        gc.collect()
    with timer("drop_unimportance_feature"):
        drop_colinear = pd.read_csv('drop_colinearity_cols.csv')
        drop_colinear = pd.Index(list(drop_colinear.iloc[:,0].values))
        df = df.loc[:,drop_colinear];
        unimportance_feature_ind = drop_unimportance_feature(df)
        df = df.drop(columns=unimportance_feature_ind);
    return df
#     with timer("Run LightGBM with kfold"):
#         feat_importance = kfold_lightgbm(df, num_folds= 5, stratified= False, debug= debug)


# In[5]:


if __name__ == "__main__":
    submission_file_name = "submission_kernel02.csv"
    with timer("Full model run"):
        df = main(debug=False)


# In[ ]:


def plot_feature_importances(df, threshold = 0.9):
    """
    Plots 15 most important features and the cumulative importance of features.
    Prints the number of features needed to reach threshold cumulative importance.
    
    Parameters
    --------
    df : dataframe
        Dataframe of feature importances. Columns must be feature and importance
    threshold : float, default = 0.9
        Threshold for prining information about cumulative importances
        
    Return
    --------
    df : dataframe
        Dataframe ordered by feature importances with a normalized column (sums to 1)
        and a cumulative importance column
    
    """
    
    plt.rcParams['font.size'] = 18
    
    # Sort features according to importance
    df = df.sort_values('importance', ascending = False).reset_index()
    
    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    df['cumulative_importance'] = np.cumsum(df['importance_normalized'])

    # Make a horizontal bar chart of feature importances
    plt.figure(figsize = (10, 6))
    ax = plt.subplot()
    
    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.index[:15]))), 
            df['importance_normalized'].head(15), 
            align = 'center', edgecolor = 'k')
    
    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:15]))))
    ax.set_yticklabels(df['feature'].head(15))
    
    # Plot labeling
    plt.xlabel('Normalized Importance'); plt.title('Feature Importances')
    plt.show()
    
    # Cumulative importance plot
    plt.figure(figsize = (8, 6))
    plt.plot(list(range(len(df))), df['cumulative_importance'], 'r-')
    plt.xlabel('Number of Features'); plt.ylabel('Cumulative Importance'); 
    plt.title('Cumulative Feature Importance');
    plt.show();
    
    importance_index = np.min(np.where(df['cumulative_importance'] > threshold))
    print('%d features required for %0.2f of cumulative importance' % (importance_index + 1, threshold))
    
    return df


# In[38]:


df = application_train_test(num_readrows=None, nan_as_category = False);


# In[39]:


df.dropna(inplace=True)


# In[1]:


# random search for light GBM
import lightgbm as lgb
n_folds = 5;
max_evals = 5;


# In[2]:


def prepare_lgb_df(df,test_size=0.3):
    features = df.select_dtypes('number')
    labels = np.array(features['TARGET'].astype(np.int32)).reshape((-1,))
    features = features.drop(columns=['TARGET','SK_ID_CURR'])
    
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = test_size, random_state = 42)
    print("training shape : {} \n testing shape : {}".format(train_features.shape,test_features.shape))
    
    train_set = lgb.Dataset(data = train_features, label = train_labels)
    test_set = lgb.Dataset(data = test_features, label = test_labels)
    return train_set, test_set, train_features, test_features, train_labels, test_labels


# In[3]:


train_set, test_set, train_features, test_features, train_labels, test_labels = prepare_lgb_df(df)


# In[51]:


model = lgb.LGBMClassifier()
default_params = model.get_params()
del default_params['n_estimators']
print(default_params)

cv_results = lgb.cv(default_params,
                    train_set,
                    num_boost_round=10000,
                    early_stopping_rounds=100,
                   metrics='auc',
                   nfold=n_folds,
                   seed=50)


# In[84]:


print('The maximum validation ROC AUC was: {:.5f} with a standard deviation of {:.5f}.'.format(cv_results['auc-mean'][-1], cv_results['auc-stdv'][-1]))
print('The optimal number of boosting rounds (estimators) was {}.'.format(len(cv_results['auc-mean'])))


# In[87]:


# beseline model
from sklearn.metrics import roc_auc_score
model.n_estimators = len(cv_results['auc-mean']);

model.fit(train_features,train_labels);
pred = model.predict_proba(test_features)[:,1]
baseline = roc_auc_score(test_labels,pred)

print("baseline auc score = {:.5f}".format(baseline))


# In[88]:


def objective(train_set, hyperparameters, iteration, n_folds = 5):
    if 'n_estimators' in hyperparameters.keys():
        del hyperparameters['n_estimators']
    cv_results = lgb.cv(hyperparameters,
                        train_set,
                        num_boost_round=10000,
                        early_stopping_rounds=100,
                        metrics='auc',
                        nfold=n_folds,
                        seed=50)
    score = np.max(cv_results['auc-mean'])
    estimator = len(cv_results['auc-mean'])
    hyperparameters['n_estimators'] = estimator
    print("{} CV score auc = {:.5f}".format(n_folds,score))
    return [score, estimator, hyperparameters]


# In[63]:


# Hyperparameter grid
param_grid = {
    'boosting_type': ['gbdt', 'goss', 'dart'],
    'num_leaves': list(range(20, 150)),
    'learning_rate': list(np.logspace(np.log10(0.005), np.log10(0.5), base = 10, num = 1000)),
    'subsample_for_bin': list(range(20000, 300000, 20000)),
    'min_child_samples': list(range(20, 500, 5)),
    'reg_alpha': list(np.linspace(0, 1)),
    'reg_lambda': list(np.linspace(0, 1)),
    'colsample_bytree': list(np.linspace(0.6, 1, 10)),
    'subsample': list(np.linspace(0.5, 1, 100)),
    'is_unbalance': [True, False]
}


# In[89]:


import random
random.seed(50);
def random_search(train_set,param_grid, max_evals, n_folds = 5):
    # initial df
    results = pd.DataFrame(columns = ['score','iteration','params'],
                            index = list(range(max_evals)))
    for i in range(max_evals):
        hyperparameters = {k: random.sample(v,1)[0] for k,v in param_grid.items()}
        hyperparameters['subsample'] = 1.0 if boosting_type == 'goss' else random.sample(param_grid['subsample'],1)[0]
        
        eval_result = objective(train_set, hyperparameters, iteration=i, n_folds = n_folds)
        results.loc[i,:] = eval_result
        
    results.sort_values('score', ascending = False, inplace = True)
    results.reset_index(inplace = True)
    return results


# In[90]:


random_results = random_search(train_set,param_grid,max_evals);


# In[92]:


print("the best CV score is {:.5f}".format(random_results.loc[0,'score']))
print("\n the best params is :")
import pprint
pprint.pprint(random_results.loc[0, 'params'])


# In[93]:


random_search_bestparams = random_results.loc[0,"params"]
model = lgb.LGBMClassifier(**random_search_bestparams,random_state=36)
model.fit(train_features,train_labels)

preds = model.predict_proba(test_features)[:,1] #select for the pred = 1

print('The best model from random search scores {:.5f} ROC AUC on the test set.'.format(roc_auc_score(test_labels, preds)))


# In[ ]:


sns.pairplot(iris, hue='species', size=2.5);


# In[9]:


import numpy as np
import pandas as pd
temp = pd.cut(np.array([1, 7, 5, 4, 6, 3]), 3)


# In[7]:


temp.


# In[ ]:




