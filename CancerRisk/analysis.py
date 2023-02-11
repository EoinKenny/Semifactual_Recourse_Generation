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


def analysis(DIVERSITY_SIZE):

	dice_df = pd.read_csv('data/dice_diverse.csv')
	ga_df = pd.read_csv('data/GA_Xps_diverse.csv')
	piece_data = np.load('data/piece_sfs.npy')[:, :-1]

	print(" ")
	print('=====================')
	print(dice_df.shape, ga_df.shape, piece_data.shape)
	print(" ")

	dice_failed = False
	if dice_df.shape[0] == 0:
		dice_failed = True  
		dice_df = deepcopy(ga_df)


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
		return l2s.mean()


	# In[94]:


	def get_gain(x, solution):
		"""
		Return mean distance between query and semifactuals
		"""
		
		l2s = np.sqrt(((x - solution)**2).sum(axis=1))    
		return l2s.mean()


	# In[95]:


	def get_robustness(x, solution, clf, cat_idxs, actionable_idxs, action_meta, continuous_features, categorical_features):
		"""
		Monte Carlo Approximation of e-ball robustness
		"""
		
		perturbations = list()
		
		for x_prime in solution:
			
	#         print(x_prime)
			
			for _ in range(MAX_MC):
				
				x_prime_clone = deepcopy(x_prime)
			
				perturbed_instance = perturb_one_random_feature(x, 
																x_prime_clone,
																df[continuous_features],
																df[categorical_features],
																action_meta,
																cat_idxs,
																actionable_idxs)
				
				
				perturbations.append(perturbed_instance.tolist())

		predictions = clf.predict(perturbations) == POSITIVE_CLASS
			
		return predictions.mean()


	# In[96]:


	def get_diversity(solution):
		"""
		Return L2 distance between all vectors (the mean)
		"""
		
		if DIVERSITY_SIZE == 1:
			return 0
		
		# Take average distance
		score = distance_matrix(solution, solution).sum() / (DIVERSITY_SIZE**2 - DIVERSITY_SIZE)
		return score


	# In[97]:


	def perturb_one_random_feature(x, x_prime, continuous_features, categorical_features, action_meta, cat_idxs, actionable_idxs):
		"""
		perturb one actionable feature for MC robustness optimization
		Really just for Monte Carlo Approximation Robustness
		"""
		
		feature_names = continuous_features.columns.tolist() + categorical_features.columns.tolist()
		change_idx    = get_rand_actionable_feature_idx(x, actionable_idxs, cat_idxs)[0]
		feature_num   = len(feature_names)
		
	#     print("Changing feature:", change_idx)
		
		# if categorical feature
		if feature_names[change_idx] in categorical_features.columns:
			perturbed_feature = generate_category(x,
												  x_prime,
												  change_idx-len(continuous_features.columns),  # index of category for function
												  cat_idxs,
												  action_meta,
												  replace=False)
			
			x_prime[cat_idxs[change_idx-len(continuous_features.columns)][0]: cat_idxs[change_idx-len(continuous_features.columns)][1]] = perturbed_feature

		# if continuous feature
		else:
			x_prime = perturb_continuous(x, 
										  x_prime, 
										  change_idx,
										  continuous_features,
										  categorical_features,
										  action_meta)

		return x_prime


	# In[98]:


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


	# In[100]:


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
			
	#         print(idx, i )
			
			sl = x[ cat_idxs[idx][0] : cat_idxs[idx][1] ]
			
	#         print(sl)
			
			at_top = sl[-1] == 1
			can_only_go_up = actionable_idxs[i][1]
			
			at_bottom = sl[0] == 1
			can_only_go_down = actionable_idxs[i][2]
			
	#         print(at_top, can_only_go_up)
	#         print(at_bottom, can_only_go_down)
			
			if can_only_go_up and at_top:
				instance_specific_actionable_indexes.remove(actionable_idxs[i])
				
			if can_only_go_down and at_bottom:
				instance_specific_actionable_indexes.remove(actionable_idxs[i])
			
		rand = np.random.randint(len(instance_specific_actionable_indexes))
		return instance_specific_actionable_indexes[rand]


	# In[101]:


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


	# In[102]:


	def perturb_continuous(x, x_prime, idx, continuous_features, categorical_features, action_meta):
		"""
		slightly perturb continuous feature with actionability constraints
		"""
		
		# Get feature max and min -- and clip it to these
		feature_names = continuous_features.columns.tolist() + categorical_features.columns.tolist()
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
	df = get_dataset()
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


	if not dice_failed:
		categorical_features_enc = enc.transform(dice_df[categorical_feature_names]).toarray()
		dice_data = np.concatenate(( dice_df[continuous_feature_names].values, categorical_features_enc), axis=1)
		dice_data = scaler.transform(dice_data)


	# Filter down ga_df into idxs discovered by DiCE for fair comparison
	test_idxs1 = np.sort(np.array(ga_df.test_idx.value_counts().index.tolist()))
	test_idxs2 = np.sort(np.array(dice_df.test_idx.value_counts().index.tolist()))
	test_idxs = sorted(list(set(test_idxs1) & set(test_idxs2)))
	ga_df = ga_df[ga_df.test_idx.isin(test_idxs)]



	# print(piece_data.shape)

	piece_idxs_filter = np.zeros(len(test_idxs1))

	for i in range(len(test_idxs2)):
		piece_idxs_filter += test_idxs1 == test_idxs2[i]

	piece_data = piece_data[piece_idxs_filter.astype(bool)]   
	# print(piece_data.shape)






	del ga_df['Unnamed: 0']
	del ga_df['test_idx']
	del ga_df['sf_found']
	ga_data = ga_df.values

	if dice_failed:
		dice_data = deepcopy(ga_data)

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

		# if len(np.array(sfs).shape) == 1:
		# 	sfs = np.array([sfs])
		# query = np.array(query)
		# print(query)
		# print(sfs)

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


	if DIVERSITY_SIZE > 1:

		for idx, t_idx in tqdm(enumerate(test_idxs)):
			query    = X_test[t_idx]
			ga_sfs   = ga_data[ (idx*DIVERSITY_SIZE) : (idx*DIVERSITY_SIZE)+DIVERSITY_SIZE ]
			dice_sfs = dice_data[ (idx*DIVERSITY_SIZE) : (idx*DIVERSITY_SIZE)+DIVERSITY_SIZE ]

			# Gain
			ga_gain   = get_gain(query, ga_sfs)
			dice_gain = get_gain(query, dice_sfs)
			if t_idx not in dice_df.test_idx.values or dice_failed: dice_gain = np.nan
			gain.append([ga_gain, dice_gain])
			
			# Diversity
			ga_div   = get_diversity(ga_sfs)
			dice_div = get_diversity(dice_sfs)
			if t_idx not in dice_df.test_idx.values or dice_failed: dice_div = np.nan
			diversity.append([ga_div, dice_div])
			
			# Reachability
			ga_reach   = get_reachability(ga_sfs)
			dice_reach = get_reachability(dice_sfs)
			if t_idx not in dice_df.test_idx.values or dice_failed: dice_reach = np.nan
			reachability.append([ga_reach, dice_reach])

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

			if t_idx not in dice_df.test_idx.values or dice_failed: dice_rob = np.nan
			robustness.append([ga_rob, dice_rob])

			if dice_failed:
				failed_sf.append([0, 1])
			else:
				failed_sf.append([0, 1 - dice_df.shape[0] / ga_data.shape[0]])

		gain = np.array(gain)
		diversity = np.array(diversity)
		reachability = np.array(reachability)
		robustness = np.array(robustness)
		failed_sf = np.array(failed_sf)

		return [gain.T[0].mean(), gain.T[1].mean()], [diversity.T[0].mean(), diversity.T[1].mean()], [reachability.T[0].mean(), reachability.T[1].mean()], [robustness.T[0].mean(), robustness.T[1].mean()], [failed_sf.T[0].mean(), failed_sf.T[1].mean()] 


	elif DIVERSITY_SIZE == 1:

		for idx, t_idx in tqdm(enumerate(test_idxs)):
			query = X_test[t_idx]
			ga_sfs = ga_data[ (idx) : (idx)+1 ]
			piece_sfs = piece_data[ (idx) : (idx)+1 ]
			dice_sfs = dice_data[ (idx) : (idx)+1 ]
			
			# Gain
			ga_gain = get_gain(query, ga_sfs)
			piece_gain = get_gain(query, piece_sfs)
			dice_gain = get_gain(query, dice_sfs)
			if t_idx not in dice_df.test_idx.values or dice_failed: dice_gain = np.nan
			gain.append([ga_gain, dice_gain, piece_gain])
			
			# Diversity
			ga_div = get_diversity(ga_sfs)
			piece_div = get_diversity(piece_sfs)
			dice_div = get_diversity(dice_sfs)
			if t_idx not in dice_df.test_idx.values or dice_failed: dice_div = np.nan
			diversity.append([ga_div, dice_div, piece_div])
			
			# Reachability
			ga_reach = get_reachability(ga_sfs)
			piece_reach = get_reachability(piece_sfs)
			dice_reach = get_reachability(dice_sfs)
			if t_idx not in dice_df.test_idx.values or dice_failed: dice_reach = np.nan
			reachability.append([ga_reach, dice_reach, piece_reach])
			
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

			if t_idx not in dice_df.test_idx.values or dice_failed: dice_rob = np.nan
			robustness.append([ga_rob, dice_rob, piece_rob])
			failed_sf.append([ 0, 1 - dice_df.shape[0] / piece_data.shape[0], 0 ])


		gain = np.array(gain)
		diversity = np.array(diversity)
		reachability = np.array(reachability)
		robustness = np.array(robustness)
		failed_sf = np.array(failed_sf)


		return [gain.T[0].mean(), gain.T[1].mean(), gain.T[2].mean()], [diversity.T[0].mean(), diversity.T[1].mean(), diversity.T[2].mean()], [reachability.T[0].mean(), reachability.T[1].mean(), reachability.T[2].mean()], [robustness.T[0].mean(), robustness.T[1].mean(), robustness.T[2].mean()], [failed_sf.T[0].mean(), failed_sf.T[1].mean(), failed_sf.T[2].mean()]







