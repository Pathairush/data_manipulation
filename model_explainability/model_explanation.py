import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import eli5
from eli5.sklearn import PermutationImportance

def print_permutation_importance(fitted_model, data_model : tuple, random_state = 1):

	if len(data_model) == 4:
		X_train, X_test, y_train, y_test = data_model
	if len(data_model) == 2:
		X_test, y_test = data_model
	else:
		raise "the len of data model is neither 4 nor 2"

	feature_names = X_test.columns.tolist()
	perm = PermutationImportance(fitted_model, random_state = random_state).fit(X_test, y_test)
	display(eli5.show_weights(perm, feature_names = feature_names))

from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots

def plot_1D_partial_dependency(fitted_model, X_test : pd.DataFrame, model_features : list, feature : str ):
	# Create the data that we will plot
	pdp_obj = pdp.pdp_isolate(model = fitted_model,
								dataset = X_test,
								model_features = model_features,
								feature = feature)

	# plot it
	pdp.pdp_plot(pdp_obj, feature)
	plt.show()

def plot_2D_partial_dependency(fitted_model, X_test : pd.DataFrame, model_features : list, features : list , plot_type = 'contour'):
	""" have an error with the matplolib version 3.0.0 """

	# Create the data that we will plot
	pdp_obj = pdp.pdp_interact(model = fitted_model,
								dataset = X_test,
								model_features = model_features,
								features = features)

	# plot it
	pdp.pdp_interact_plot(pdp_interact_out = pdp_obj,
		feature_names = feature, plot_type = plot_type)
	plt.show()

import shap

def create_shap_object(my_model, model_type, **kwargs):
	if model_type == 'tree':
		explainer = shap.TreeExplainer(my_model)
	if model_type == 'deep':
		explainer = shap.DeepExplainer(my_model)
	if model_type == 'other':
		explainer = shap.KernelExplainer(my_model.predict_proba, train_X)
	return explainer

def calculate_shap_value(explainer, data_for_prediction : pd.DataFrame):
	shap_values = explainer.shap_values(data_for_prediction)
	return shap_values

def plot_shap_forceplot(explainer, shap_values, data_for_prediction  : pd.DataFrame):
	shap.initjs()
	# use [1] for binary classification to get the positive outcome
	display(shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction))

def plot_shap_summaryplot(shap_values, X_test : pd.DataFrame):

	"""When plotting, we call shap_values[1]. 
	For classification problems, there is a separate array of SHAP values for each possible outcome. 
	In this case, we index in to get the SHAP values for the prediction of "True".
	Calculating SHAP values can be slow. It isn't a problem here, because this dataset is small. 
	But you'll want to be careful when running these to plot with reasonably sized datasets. 
	The exception is when using an xgboost model, which SHAP has some optimizations 
	for and which is thus much faster."""

	shap.initjs()
	# use [1] for binary classification to get the positive outcome
	display(shap.summary_plot(shap_values[1], X_test))

def plot_shap_dependenceplot(shap_values, X, feature_names : str, interaction_feature : str):
	""" X = data[feature_names] before splitting train test """
	shap.initjs()
	display(shap.dependence_plot(feature_names, shap_values[1], X, interaction_index=interaction_feature))