import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score


def split_train_test(data,train_feature,target_feature,test_size=0.3,random_state=0):
	X_train, X_test, y_train, y_test = train_test_split(
	    data[train_feature],
	    data[target_feature],
	    test_size=test_size,
	    random_state=random_state)

	print('train feature {}, test feature {}'.format(X_train.shape, X_test.shape))
	return (X_train,X_test,y_train,y_test)

def remove_constant(data_model):
	X_train, X_test, y_train, y_test = data_model
	# remove constant features
	constant_features = [
	    feat for feat in X_train.columns if X_train[feat].std() == 0
	]

	print('number of constant features {}'.format(len(constant_features)))
	X_train.drop(labels=constant_features, axis=1, inplace=True)
	X_test.drop(labels=constant_features, axis=1, inplace=True)

	X_train.shape, X_test.shape
	return (X_train,X_test,y_train,y_test)

def remove_quasi_constant(data_model):
	X_train, X_test, y_train, y_test = data_model
	# remove quasi-constant features
	sel = VarianceThreshold(
	    threshold=0.01)  # 0.1 indicates 99% of observations approximately

	sel.fit(X_train)  # fit finds the features with low variance

	print('number of quasi-constant features {} '.format(sum(sel.get_support()))) # how many not quasi-constant?

	features_to_keep = X_train.columns[sel.get_support()]


	# we can then remove the features like this
	X_train = sel.transform(X_train)
	X_test = sel.transform(X_test)

	X_train.shape, X_test.shape
	return (X_train,X_test,y_train,y_test)

def remove_duplicated_feature(data_model):
	X_train, X_test, y_train, y_test = data_model
	# check for duplicated features in the training set
	duplicated_feat = []
	for i in range(0, len(X_train.columns)):
	    if i % 50 == 0:  # this helps me understand how the loop is going
	        print(i)

	    col_1 = X_train.columns[i]

	    for col_2 in X_train.columns[i + 1:]:
	        if X_train[col_1].equals(X_train[col_2]):
	            duplicated_feat.append(col_2)
	            
	print('number of duplicated features {} '.format(len(duplicated_feat)))
	# remove duplicated features
	X_train.drop(labels=duplicated_feat, axis=1, inplace=True)
	X_test.drop(labels=duplicated_feat, axis=1, inplace=True)

	X_train.shape, X_test.shape
	return (X_train,X_test,y_train,y_test)

def remove_colinearity_feature(data_model):
	X_train, X_test, y_train, y_test = data_model

	def correlation(dataset, threshold=0.8):
	    col_corr = set()  # Set of all the names of correlated columns
	    corr_matrix = dataset.corr()
	    for i in range(len(corr_matrix.columns)):
	        for j in range(i):
	            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
	                colname = corr_matrix.columns[i]  # getting the name of column
	                col_corr.add(colname)
	    return col_corr

	corr_features = correlation(X_train, threshold)
	print('correlated features: ', len(set(corr_features)))

	# removed correlated  features
	X_train.drop(labels=corr_features, axis=1, inplace=True)
	X_test.drop(labels=corr_features, axis=1, inplace=True)

	X_train.shape, X_test.shape
	return (X_train,X_test,y_train,y_test)

def remove_feature_roc_auc(data_model, roc_auc_threshold=0.5):

	X_train, X_test, y_train, y_test = data_model

	roc_values = []
	for feature in X_train.columns:
	    clf = DecisionTreeClassifier()
	    clf.fit(X_train[feature].fillna(0).to_frame(), y_train)
	    y_scored = clf.predict_proba(X_test[feature].fillna(0).to_frame())
	    roc_values.append(roc_auc_score(y_test, y_scored[:, 1]))

    # let's add the variable names and order it for clearer visualisation
	roc_values = pd.Series(roc_values)
	roc_values.index = X_train.columns
	roc_values.sort_values(ascending=False).plot.bar(figsize=(20, 8))

	selected_feat = roc_values[roc_values>roc_auc_threshold]
	print('number of roc_auc > {} features: {}'.format(roc_auc_threshold, len(set(corr_features))))
	len(selected_feat), X_train.shape[1]
	X_train = X_train[selected_feat].copy()
	X_test = X_test[selected_feat].copy()

	return (X_train,X_test,y_train,y_test)

def compare_baseline_model(data_model,model='rf'):

		X_train,X_test,y_train,y_test = data_model


		if model == 'rf':
		    rf = RandomForestClassifier(n_estimators=200, random_state=39, max_depth=4)
		    rf.fit(X_train, y_train)
		    print('Train set')
		    pred = rf.predict_proba(X_train)
		    print('Random Forests roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
		    print('Test set')
		    pred = rf.predict_proba(X_test)
		    print('Random Forests roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))
	    elif model == 'logistic':
    	    # function to train and test the performance of logistic regression
		    logit = LogisticRegression(random_state=44)
		    logit.fit(X_train, y_train)
		    print('Train set')
		    pred = logit.predict_proba(X_train)
		    print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
		    print('Test set')
		    pred = logit.predict_proba(X_test)
		    print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))
