import pandas as pd
import numpy as np
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import pdb    
import os    

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm 

from copy import deepcopy
from collections import Counter
from random import gauss
from scipy.spatial import distance_matrix

from dataset import *


if os.path.isfile('data/ga_qualitative.npy'):
	try:
		os.remove('data/ga_qualitative.npy')
	except:
		pass

	try:
		os.remove('data/dice_qualitative.npy')
	except:
		pass

	try:
		os.remove('data/piece_qualitative.npy')
	except:
		pass

	try:
		os.remove('data/ga_qualitative_diverse.npy')
	except:
		pass
		
	try:
		os.remove('data/dice_qualitative_diverse.npy')
	except:
		pass


def analysis(DIVERSITY_SIZE, seed):

	dice_df = pd.read_csv('data/dice_diverse.csv')
	neme_df = pd.read_csv('data/neme_diverse.csv')
	ga_df = pd.read_csv('data/GA_Xps_diverse.csv')
	piece_df = pd.read_csv('data/piece_sfs.csv')

	print(" ")
	print('=====================')
	print(dice_df.shape, ga_df.shape, piece_df.shape, neme_df.shape)
	print(" ")

	CONT_PERTURB_STD = 0.05
	POSITIVE_CLASS = 1
	MAX_MC = 100

	continuous_features = continuous_feature_names
	categorical_features = categorical_feature_names

	# ## Robustness Function
	def get_reachability(solution):
		"""
		OOD Check using NN-dist metric
		"""
		
		l2s, _ = REACH_KNN.kneighbors(X=solution, n_neighbors=1, return_distance=True)    
		return l2s


	def get_gain(x, solution):
		"""
		Return mean distance between query and semifactuals
		"""
		
		l2s = np.sqrt(((x - solution)**2).sum(axis=1))    

		if len(l2s) == 1: l2s = l2s.reshape(1,-1)
		return l2s


	def get_robustness(x, solution, clf, cat_idxs, actionable_idxs, action_meta, continuous_features, categorical_features):
		"""
		Monte Carlo Approximation of e-ball robustness
		"""
		

		perturbation_preds = list()
		for x_prime in solution:
			instance_perturbations = list()
			for _ in range(MAX_MC):
				x_prime_clone = deepcopy(x_prime)        
				perturbed_instance = perturb_one_random_feature(x, 
																x_prime_clone,
																continuous_features,
																categorical_features,
																action_meta,
																cat_idxs,
																actionable_idxs)

				instance_perturbations.append(perturbed_instance.tolist())
			predictions = clf.predict(instance_perturbations) == POSITIVE_CLASS
			perturbation_preds.append(predictions.mean())

		return np.array(perturbation_preds)


	def get_diversity(solution):
		"""
		Return L2 distance between all vectors (the mean)
		"""
		
		if DIVERSITY_SIZE == 1:
			return 0
		
		# Take average distance
		scores = distance_matrix(solution, solution).flatten()

		num = sum(scores>0)

		if num > 0:
			return sum(scores) / len(scores)
		else:
			return np.nan


	def perturb_one_random_feature(x, x_prime, continuous_features, categorical_features, action_meta, cat_idxs, actionable_idxs):
		"""
		perturb one actionable feature for MC robustness optimization
		Really just for Monte Carlo Approximation Robustness
		"""
		
		feature_names = continuous_feature_names + categorical_feature_names
		change_idx    = get_rand_actionable_feature_idx(x, actionable_idxs, cat_idxs)[0]
		feature_num   = len(feature_names)
				
		# if categorical feature
		if feature_names[change_idx] in categorical_feature_names:
			perturbed_feature = generate_category(x,
												  x_prime,
												  change_idx-len(continuous_feature_names),  # index of category for function
												  cat_idxs,
												  action_meta,
												  replace=False)
			
			x_prime[cat_idxs[change_idx-len(continuous_feature_names)][0]: cat_idxs[change_idx-len(continuous_feature_names)][1]] = perturbed_feature

		# if continuous feature
		else:
			x_prime = perturb_continuous(x, 
										  x_prime, 
										  change_idx,
										  continuous_features,
										  categorical_features,
										  action_meta)

		return x_prime


	def generate_cat_idxs():
		"""
		Get indexes for all categorical features that are one hot encoded
		"""
		
		cat_idxs = list()
		start_idx = len(df[continuous_features].columns)
		for cat in enc.categories_:
			cat_idxs.append([start_idx, start_idx + cat.shape[0]])
			start_idx = start_idx + cat.shape[0]
		return cat_idxs


	def get_actionable_feature_idxs(continuous_features, categorical_features):
		"""
		sample a random actionable feature index
		"""
		
		feature_names = continuous_features.columns.tolist() + categorical_features.columns.tolist()
		actionable_idxs = list() 
		
		for i, f in enumerate(feature_names):
			if action_meta[f]['actionable']:
				actionable_idxs.append( [i, action_meta[f]['can_increase'], action_meta[f]['can_decrease']] )
		
		return actionable_idxs


	def get_rand_actionable_feature_idx(x, actionable_idxs, cat_idxs):
		"""
		sample a random actionable feature index
		"""
		
		instance_specific_actionable_indexes = deepcopy(actionable_idxs)
		
		# Get starting index of categories in actionable index list
		for i in range(len(actionable_idxs)):
			if actionable_idxs[i][0] == cat_idxs[0][0]:
				break    
		starting_index = i
				
		# we don't care about continuous features
		for idx, i in enumerate(list(range(starting_index, len(actionable_idxs)))):
			
			sl = x[ cat_idxs[idx][0] : cat_idxs[idx][1] ]			
			at_top = sl[-1] == 1
			can_only_go_up = actionable_idxs[i][1]
			
			at_bottom = sl[0] == 1
			can_only_go_down = actionable_idxs[i][2]
			
			if can_only_go_up and at_top:
				instance_specific_actionable_indexes.remove(actionable_idxs[i])
				
			if can_only_go_down and at_bottom:
				instance_specific_actionable_indexes.remove(actionable_idxs[i])
			
		rand = np.random.randint(len(instance_specific_actionable_indexes))
		return instance_specific_actionable_indexes[rand]


	def generate_category(x, x_prime, idx, cat_idxs, action_meta, replace=True):
		"""
		Randomly generate a value for a OHE categorical feature using actionability constraints
		replace: this gives the option if the generation should generate the original
		value for the feature that is present in x, or if it should only generate 
		different x_primes with different values for the feature
		
		need to implement "replace" later
		"""
		
		original_rep = x[cat_idxs[idx][0]: cat_idxs[idx][1]]  # To constrain with initial datapoint
		new_rep = x_prime[cat_idxs[idx][0]: cat_idxs[idx][1]]  # to make sure we modify based on new datapoint
		
		cat_name = df[categorical_features].columns[idx]
		
		if replace:  # just for population initialisation

			# If you can generate new feature anywhere
			if action_meta[ cat_name ]['can_increase'] and action_meta[ cat_name ]['can_decrease']:
				new = np.eye( len(original_rep) )[np.random.choice(len(original_rep))]  

			# if you can only increase
			elif action_meta[ cat_name ]['can_increase'] and not action_meta[ cat_name ]['can_decrease']:
				try:
					# To account for when it's the last value in the scale of categories
					new = np.eye( len(original_rep) - (np.argmax(original_rep)) )[np.random.choice( len(original_rep) - (np.argmax(original_rep)) )]
					new = np.append(  np.zeros((np.argmax(original_rep))), new )
				except:
					new = new_rep

			# If you can only decrease
			elif not action_meta[ cat_name ]['can_increase'] and action_meta[ cat_name ]['can_decrease']:
				try:
					# To account for when it's the first value in the scale of categories
					new = np.eye( np.argmax(original_rep) +1 )[np.random.choice(np.argmax(original_rep) +1)]
					new = np.append(new, np.zeros(  ( len(original_rep) - np.argmax(original_rep) ) -1  ) )
				except:
					new = new_rep

			else:
				new = new_rep
			   
		else:  # For MC sampling, and mutation
			
			# If you can generate new feature anywhere
			if action_meta[ cat_name ]['can_increase'] and action_meta[ cat_name ]['can_decrease']:
				new = np.eye( len(original_rep) -1 )[np.random.choice(len(original_rep)-1)]
				new = np.insert(new, np.argmax(new_rep), 0 )

			# if you can only increase
			elif action_meta[ cat_name ]['can_increase'] and not action_meta[ cat_name ]['can_decrease']:
				try:
					# To account for when it's the last value in the scale of categories
					new = np.eye( len(original_rep) - np.argmax(original_rep) -1 )[  np.random.choice(len(original_rep) - np.argmax(original_rep)-1)  ]
					new = np.insert(new, np.argmax(new_rep) - (np.argmax(original_rep)), 0 )
					new = np.concatenate( (np.zeros(  (len(original_rep) -  (len(original_rep) - np.argmax(original_rep))  )  ), new) )
				except:
					new = new_rep

			# If you can only decrease
			elif not action_meta[ cat_name ]['can_increase'] and action_meta[ cat_name ]['can_decrease']:
				
				try:  # To account for when it's the first value in the scale of categories
					new = np.eye( np.argmax(original_rep) )[  np.random.choice(np.argmax(original_rep))  ]
					new = np.insert(new, np.argmax(new_rep), 0 )
					new = np.concatenate( (new, np.zeros(  (len(original_rep) - np.argmax(original_rep) - 1  )  )) )
					
				except:
					new = new_rep

			else:
				new = new_rep  
				
		return new


	def perturb_continuous(x, x_prime, idx, continuous_features, categorical_features, action_meta):
		"""
		slightly perturb continuous feature with actionability constraints
		"""
		
		# Get feature max and min -- and clip it to these
		feature_names = continuous_feature_names + categorical_feature_names
		cat_name = feature_names[idx]
		
		if action_meta[cat_name]['can_increase'] and action_meta[cat_name]['can_decrease']:
			max_value = action_meta[cat_name]['max']
			min_value = action_meta[cat_name]['min']
		
		elif action_meta[cat_name]['can_increase'] and not action_meta[cat_name]['can_decrease']:
			max_value = action_meta[cat_name]['max']
			min_value = x[idx]

		elif not action_meta[cat_name]['can_increase'] and action_meta[cat_name]['can_decrease']:
			max_value = x[idx]
			min_value = action_meta[cat_name]['min']
			
		else:  # not actionable
			max_value = x[idx]
			min_value = x[idx]
			
		perturb = gauss(0, ((max_value - min_value) * CONT_PERTURB_STD)  )
		x_prime[idx] += perturb

		if x_prime[idx] > max_value:
			x_prime[idx] = max_value
		if x_prime[idx] < min_value:
			x_prime[idx] = min_value
		
		return x_prime


	action_meta = actionability_constraints()
	df = get_dataset(seed)
	target = df[TARGET_NAME].values
	del df[TARGET_NAME]
	idx_train = np.load('data/training_idx.npy')
	training = np.zeros(df.shape[0])
	training[idx_train] = 1
	df['training'] = training

	enc = OneHotEncoder().fit( df[categorical_features] )
	categorical_features_enc = enc.transform(df[categorical_features]).toarray()

	#### NB: Continuous features are first
	data = np.concatenate(( df[continuous_features].values, categorical_features_enc), axis=1)

	df_train = df[df.training == 1]
	df_test = df[df.training == 0]
	df_train = df_train.reset_index(inplace=False, drop=True)
	df_test = df_test.reset_index(inplace=False, drop=True)
	del df_train['training']
	del df_test['training']
	X_train = data[(df.training == 1).values]
	X_test = data[(df.training == 0).values]
	scaler = MinMaxScaler().fit(data)
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)

	# Fix DiCE Format
	categorical_features_enc = enc.transform(dice_df[categorical_feature_names]).toarray()
	dice_data = np.concatenate(( dice_df[continuous_feature_names].values, categorical_features_enc), axis=1)
	dice_data = scaler.transform(dice_data)
	temp_df = pd.DataFrame(dice_data, columns=ga_df.columns.tolist()[:-2])
	temp_df['sf_found'] = dice_df['sf_found']
	temp_df['test_idx'] = dice_df['test_idx']
	dice_df = temp_df

	# Filter down ga_df into idxs discovered by DiCE for fair comparison
	test_idxs_ga = np.sort(np.array(ga_df[ga_df.sf_found==1].test_idx.value_counts().index.tolist()))
	test_idxs_dice = np.sort(np.array(dice_df[dice_df.sf_found==1].test_idx.value_counts().index.tolist()))
	test_idxs_neme = np.sort(np.array(neme_df[neme_df.sf_found==1].test_idx.value_counts().index.tolist()))
	test_idxs_piece = np.sort(np.array(piece_df[piece_df.sf_found==1].test_idx.value_counts().index.tolist()))

	common_idxs = sorted(list(set(test_idxs_ga) & set(test_idxs_dice) & set(test_idxs_neme) & set(test_idxs_piece)))

	saved_sf_successes = [ga_df.sf_found.values, dice_df.sf_found.values, neme_df.sf_found.values, piece_df.sf_found.values]
	for xxx in [ga_df, dice_df, piece_df, neme_df]:
		del xxx['test_idx']
		del xxx['sf_found']

	y_train = target[(df.training == 1).values]
	REACH_KNN = KNeighborsClassifier(p=2).fit(X_train, y_train)
	cat_idxs = generate_cat_idxs()
	action_meta = actionability_constraints()
	clf = LogisticRegression(class_weight='balanced', fit_intercept=False).fit(X_train, y_train)
	actionable_idxs = get_actionable_feature_idxs(df[continuous_features], df[categorical_features])

	gain = list()
	diversity = list()
	reachability = list()
	robustness = list()
	failed_sf = list()
	sparsity = list() 
	feature_variety = list()


	def get_sparsity(query, sfs):

		sparsity = list()
		full_sparsity = list()

		for i in range(len(sfs)):
			sf = sfs[i]
			sf = reverse_eng(sf)
			num_changed = sum(np.array(sf) - np.array(reverse_eng(query)) != 0)
			sparsity.append(num_changed)
			full_sparsity.append( ((np.array(sf) - np.array(reverse_eng(query)) != 0) * 1).tolist() )

		full_sparsity = np.array(full_sparsity)
		return sparsity, full_sparsity


	def reverse_eng(inst):
		return scaler.inverse_transform(inst.reshape(1,-1))[0][:len(continuous_feature_names)].tolist() + enc.inverse_transform(inst[len(continuous_feature_names):].reshape(1,-1))[0].tolist()


	def qual_diverse(feature_variety):
		dist1 = list()
		dist2 = list()

		for i in range(feature_variety.shape[0]):  # instances
			for j in range(feature_variety.shape[1]):  # method
				if j == 0:
					dist1.append(feature_variety[i][j].sum(axis=0).tolist())
				if j == 1:
					dist2.append(feature_variety[i][j].sum(axis=0).tolist())
					
		dist1 = np.array(dist1).reshape(-1, feature_variety.shape[-1]).sum(axis=0)
		dist2 = np.array(dist2).reshape(-1, feature_variety.shape[-1]).sum(axis=0)

		return dist1, dist2


	def qual_single(feature_variety):
		dist1 = list()
		dist2 = list()
		dist3 = list()

		for i in range(feature_variety.shape[0]):  # instances
			for j in range(feature_variety.shape[1]):  # method
				if j == 0:
					dist1.append(feature_variety[i][j].sum(axis=0).tolist())
				if j == 1:
					dist2.append(feature_variety[i][j].sum(axis=0).tolist())
				if j == 2:
					dist3.append(feature_variety[i][j].sum(axis=0).tolist())
					
		dist1 = np.array(dist1).reshape(-1, feature_variety.shape[-1]).sum(axis=0)
		dist2 = np.array(dist2).reshape(-1, feature_variety.shape[-1]).sum(axis=0)
		dist3 = np.array(dist3).reshape(-1, feature_variety.shape[-1]).sum(axis=0)

		return dist1, dist2, dist3


	def f_suc(data, succ_idxs):
		"""
		If the instance wasn't a semi-factual, replace the result with a nan
		This is mostly for NeMe
		"""

		if DIVERSITY_SIZE == 1:
			for i in range(len(data)):
				if succ_idxs == 0:
					data[i] = np.nan 
			return data.item()

		else:
			for i in range(len(data)):
				if succ_idxs[i] == 0:
					data[i] = np.nan 
			return data.mean()


	if DIVERSITY_SIZE > 1:

		for idx, t_idx in tqdm(enumerate(test_idxs_ga)):
			query    = X_test[t_idx]
			ga_sfs   = ga_df[ (idx*DIVERSITY_SIZE) : (idx*DIVERSITY_SIZE)+DIVERSITY_SIZE ].values
			dice_sfs = dice_df[ (idx*DIVERSITY_SIZE) : (idx*DIVERSITY_SIZE)+DIVERSITY_SIZE ].values
			neme_sfs = neme_df[ (idx*DIVERSITY_SIZE) : (idx*DIVERSITY_SIZE)+DIVERSITY_SIZE ].values

			# Gain
			ga_gain   = get_gain(query, ga_sfs)
			dice_gain = get_gain(query, dice_sfs)
			neme_gain = get_gain(query, neme_sfs)
			gain.append([f_suc(ga_gain, saved_sf_successes[0][(idx*DIVERSITY_SIZE) : (idx*DIVERSITY_SIZE)+DIVERSITY_SIZE]), f_suc(dice_gain, saved_sf_successes[1][(idx*DIVERSITY_SIZE) : (idx*DIVERSITY_SIZE)+DIVERSITY_SIZE]), f_suc(neme_gain, saved_sf_successes[2][(idx*DIVERSITY_SIZE) : (idx*DIVERSITY_SIZE)+DIVERSITY_SIZE])])

			# Diversity
			ga_div   = get_diversity(ga_sfs)
			dice_div = get_diversity(dice_sfs)
			neme_div = get_diversity(neme_sfs)
			diversity.append([ ga_div if ga_div > 0 else np.nan, dice_div if dice_div > 0 else np.nan, neme_div if neme_div > 0 else np.nan ])
			
			# Reachability
			ga_reach   = get_reachability(ga_sfs)
			dice_reach = get_reachability(dice_sfs)
			neme_reach = get_reachability(neme_sfs)
			reachability.append([f_suc(ga_reach, saved_sf_successes[0][(idx*DIVERSITY_SIZE) : (idx*DIVERSITY_SIZE)+DIVERSITY_SIZE]), f_suc(dice_reach, saved_sf_successes[1][(idx*DIVERSITY_SIZE) : (idx*DIVERSITY_SIZE)+DIVERSITY_SIZE]), f_suc(neme_reach, saved_sf_successes[2][(idx*DIVERSITY_SIZE) : (idx*DIVERSITY_SIZE)+DIVERSITY_SIZE])])

			# Robustness
			ga_rob = get_robustness(query,
					   ga_sfs,
					   clf,
					   cat_idxs,
					   actionable_idxs,
					   action_meta,
					   continuous_features,
					   categorical_features)
			dice_rob = get_robustness(query,
					   dice_sfs,
					   clf,
					   cat_idxs,
					   actionable_idxs,
					   action_meta,
					   continuous_features,
					   categorical_features)
			neme_rob = get_robustness(query,
					   neme_sfs,
					   clf,
					   cat_idxs,
					   actionable_idxs,
					   action_meta,
					   continuous_features,
					   categorical_features)

			robustness.append([f_suc(ga_rob, saved_sf_successes[0][(idx*DIVERSITY_SIZE) : (idx*DIVERSITY_SIZE)+DIVERSITY_SIZE]), f_suc(dice_rob, saved_sf_successes[1][(idx*DIVERSITY_SIZE) : (idx*DIVERSITY_SIZE)+DIVERSITY_SIZE]), f_suc(neme_rob, saved_sf_successes[2][(idx*DIVERSITY_SIZE) : (idx*DIVERSITY_SIZE)+DIVERSITY_SIZE])])

			failed_sf.append([saved_sf_successes[0][(idx*DIVERSITY_SIZE) : (idx*DIVERSITY_SIZE)+DIVERSITY_SIZE].mean(), saved_sf_successes[1][(idx*DIVERSITY_SIZE) : (idx*DIVERSITY_SIZE)+DIVERSITY_SIZE].mean(), saved_sf_successes[2][(idx*DIVERSITY_SIZE) : (idx*DIVERSITY_SIZE)+DIVERSITY_SIZE].mean()])

		gain = np.array(gain)
		diversity = np.array(diversity)
		reachability = np.array(reachability)
		robustness = np.array(robustness)
		failed_sf = abs(np.array(failed_sf) - 1)

		return [np.nanmean(gain.T[0]), np.nanmean(gain.T[1]), np.nanmean(gain.T[2])], [np.nanmean(diversity.T[0]), np.nanmean(diversity.T[1]), np.nanmean(diversity.T[2])], [np.nanmean(reachability.T[0]), np.nanmean(reachability.T[1]), np.nanmean(reachability.T[2])], [np.nanmean(robustness.T[0]), np.nanmean(robustness.T[1]), np.nanmean(robustness.T[2])], [np.nanmean(failed_sf.T[0]), np.nanmean(failed_sf.T[1]), np.nanmean(failed_sf.T[2])]


	elif DIVERSITY_SIZE == 1:

		for idx, t_idx in tqdm(enumerate(test_idxs_ga)):
			query = X_test[t_idx]
			ga_sfs = ga_df[ (idx) : (idx)+1 ].values
			piece_sfs = piece_df[ (idx) : (idx)+1 ].values
			dice_sfs = dice_df[ (idx) : (idx)+1 ].values
			neme_sfs = neme_df[ (idx) : (idx)+1 ].values

			# Gain
			ga_gain = get_gain(query, ga_sfs)
			piece_gain = get_gain(query, piece_sfs)
			dice_gain = get_gain(query, dice_sfs)
			neme_gain = get_gain(query, neme_sfs)
			gain.append([f_suc(ga_gain, saved_sf_successes[0][idx]), f_suc(dice_gain, saved_sf_successes[1][idx]), f_suc(neme_gain, saved_sf_successes[2][idx]), f_suc(piece_gain, saved_sf_successes[3][idx])])

			# Diversity
			ga_div = get_diversity(ga_sfs)
			piece_div = get_diversity(piece_sfs)
			dice_div = get_diversity(dice_sfs)
			neme_div = get_diversity(neme_sfs)
			diversity.append([0., 0., 0., 0.])
			
			# Reachability
			ga_reach = get_reachability(ga_sfs)
			piece_reach = get_reachability(piece_sfs)
			dice_reach = get_reachability(dice_sfs)
			neme_reach = get_reachability(neme_sfs)
			reachability.append([f_suc(ga_reach, saved_sf_successes[0][idx]), f_suc(piece_reach, saved_sf_successes[1][idx]), f_suc(neme_reach, saved_sf_successes[2][idx]), f_suc(piece_reach, saved_sf_successes[3][idx])])
			
			# Robustness
			ga_rob = get_robustness(query,
					   ga_sfs,
					   clf,
					   cat_idxs,
					   actionable_idxs,
					   action_meta,
					   continuous_features,
					   categorical_features)
			piece_rob = get_robustness(query,
					   piece_sfs,
					   clf,
					   cat_idxs,
					   actionable_idxs,
					   action_meta,
					   continuous_features,
					   categorical_features)
			dice_rob = get_robustness(query,
					   dice_sfs,
					   clf,
					   cat_idxs,
					   actionable_idxs,
					   action_meta,
					   continuous_features,
					   categorical_features)
			neme_rob = get_robustness(query,
					   neme_sfs,
					   clf,
					   cat_idxs,
					   actionable_idxs,
					   action_meta,
					   continuous_features,
					   categorical_features)

			robustness.append([f_suc(ga_rob, saved_sf_successes[0][idx]), f_suc(dice_rob, saved_sf_successes[1][idx]), f_suc(neme_rob, saved_sf_successes[2][idx]), f_suc(piece_rob, saved_sf_successes[3][idx])])



			failed_sf.append([ saved_sf_successes[i][idx] for i in range(4) ])


		gain = np.array(gain)
		diversity = np.array(diversity)
		reachability = np.array(reachability)
		robustness = np.array(robustness)
		failed_sf = abs(np.array(failed_sf) - 1)

		return [np.nanmean(gain.T[0]), np.nanmean(gain.T[1]), np.nanmean(gain.T[2]), np.nanmean(gain.T[3])], [np.nanmean(diversity.T[0]), np.nanmean(diversity.T[1]), np.nanmean(diversity.T[2]), np.nanmean(diversity.T[3])], [np.nanmean(reachability.T[0]), np.nanmean(reachability.T[1]), np.nanmean(reachability.T[2]), np.nanmean(reachability.T[3])], [np.nanmean(robustness.T[0]), np.nanmean(robustness.T[1]), np.nanmean(robustness.T[2]), np.nanmean(robustness.T[3])], [np.nanmean(failed_sf.T[0]), np.nanmean(failed_sf.T[1]), np.nanmean(failed_sf.T[2]), np.nanmean(failed_sf.T[3])]







