import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def make_clustering_report(df_with_clustering_membership, col_clustering_membership, described_features = None):
	
	if described_features is None:
		described_features = df_with_clustering_membership.columns.tolist()
	
	is_feature_in_dataframe = any(feature not in df_with_clustering_membership.columns for feature in described_features)
	feature_notin_dataframe = [feature not in df_with_clustering_membership.columns for feature in described_features]
	
	if is_feature_in_dataframe:
		raise f"the {feature_notin_dataframe} doesn't contain in the dataframe"
	
	df_groupby = df_with_clustering_membership.groupby(col_clustering_membership)
	df_cluster_med = df_groupby.median()
	
