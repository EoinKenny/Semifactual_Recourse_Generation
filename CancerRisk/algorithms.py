import pandas as pd
import numpy as np
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import pickle 
import pdb
import time   
import string
import logging

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from copy import deepcopy
from collections import Counter
from random import gauss
from scipy.spatial import distance_matrix

# DiCE imports
import dice_ml
from dice_ml.utils import helpers  # helper functions

from dataset import *


logging.basicConfig(filename='logger.log', encoding='utf-8')
logging.getLogger('matplotlib.font_manager').disabled = True



def piece_algorithm(seed, DIVERSITY_SIZE, max_num_samples):

	def get_actionable_feature_idxs(continuous_features, categorical_features):
		"""
		sample a random actionable feature index
		"""
		
		feature_names = continuous_feature_names + categorical_feature_names
		actionable_idxs = list() 
		
		for i, f in enumerate(feature_names):
			if action_meta[f]['actionable']:
				actionable_idxs.append( [i, action_meta[f]['can_increase'], action_meta[f]['can_decrease']] )
		
		return actionable_idxs


	action_meta = actionability_constraints()

	df_train = pd.read_csv('data/df_train.csv')
	df_test = pd.read_csv('data/df_test.csv')

	X_train = np.load('data/X_train.npy', )
	X_test = np.load('data/X_test.npy', )
	y_train = np.load('data/y_train.npy', )
	y_test = np.load('data/y_test.npy', )

	# ## Normalization
	scaler = MinMaxScaler().fit(X_train)
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)


	with open('data/enc.pkl', 'rb') as file:
		enc = pickle.load(file)


	# ## Generate Training Column Label
	#### Logistic Regression
	with open('data/clf.pkl', 'rb') as file:
		clf = pickle.load(file)

	test_preds = clf.predict(X_test)
	train_preds = clf.predict(X_train)

	test_probs = clf.predict_proba(X_test)
	train_probs = clf.predict_proba(X_train)

	df_test['preds'] = test_preds
	df_test['probs'] = test_probs.T[1]

	df_train['preds'] = train_preds
	df_train['probs'] = train_probs.T[1]


	# # Make Counterfactual
	cf_df = df_train[(df_train.preds == 0)]
	preds = clf.predict(X_test)

	def find_nearest(array, value):
		array = np.asarray(array)
		idx = (np.abs(array - value)).argmin()
		return array[idx]


	def get_prob_cat(cf_df, x):

		cat_probabilities = list()
		expected_values = list()
		index_current = len(continuous_feature_names)

		for i, cat in enumerate(categorical_feature_names):
			temp0 = df_train[df_train.preds== 0][cat]
			temp1 = df_train[df_train.preds== 1][cat]

			# Expected value
			probs = list()
			for cat2 in enc.categories_[i]:
				probs.append( (temp0 == cat2).sum() / ((temp1 == cat2).sum()+0.0001) )

			probs = np.array(probs) / sum(probs)

			expected_values.append( np.argmax( np.array(probs) ) )

			# Feature prob
			feature_rep = x[index_current: index_current + enc.categories_[i].shape[0]]
			feature_prob = (feature_rep * probs).sum()
			cat_probabilities.append(   feature_prob   )
			actual_feature_value_idx = np.argmax( np.array(feature_rep) )

			index_current += enc.categories_[i].shape[0]   
			
		return cat_probabilities, expected_values


	def get_prob_cont(x):
		"""
		Returns probability of values from normal class, expected value
		"""
		
		cont_probs = list()
		cont_expected = list()
		
		for i, cat in enumerate(continuous_feature_names):
			
			# pick continuous feature (i.e., i), and positive cancer prediction (i.e., 1)
			temp = X_train.T[i][train_preds == 1]
			rv = scipy.stats.gamma
			
			try:
				params = rv.fit(temp)
			except:
				params = (0.5, 0.5, 0.5)
				
			prob = rv.cdf(x[i], *params)
			if prob < 0.5:
				cont_probs.append(prob)
				
				# project mean to nearest recorded value (to allow ordinal variables to work)
				mean = find_nearest(temp, rv.mean(*params))
				cont_expected.append( mean )
			else:
				cont_probs.append(1 - prob)
				
				# project mean to nearest recorded value (to allow ordinal variables to work)
				mean = find_nearest(temp, rv.mean(*params))
				cont_expected.append( mean )
			
		return cont_probs, cont_expected


	def get_feature_probabilities(cf_df, x):
		cont_probs, cont_expected = get_prob_cont(df_test.iloc[test_idx].values)
		cat_probs, expected_cat = get_prob_cat(cf_df, X_test[test_idx])
		return cont_probs, cont_expected, cat_probs, expected_cat


	def flip_category(x, cat_name='menopaus', change_to=1):
		for i, cat in enumerate(categorical_feature_names):
			if cat == cat_name:
				feature_rep = deepcopy(x[cat_idxs[i][0]: cat_idxs[i][1]])
				feature_rep *= 0.
				feature_rep[int(change_to)-1] = 1.
				x[cat_idxs[i][0]: cat_idxs[i][1]] = feature_rep
		return x


	def clip_expected_values(test_idx, expected_values, feature_names):
		
		# iterate each actionable feature
		for idx, f in enumerate(feature_names):
			if action_meta[f]['actionable']:

				if f in continuous_feature_names:
					current_value = X_test[test_idx][idx]
				else:
					current_value = df_test.iloc[test_idx][f]
				
				# current_value = df_test.iloc[test_idx].values[idx]
				e_value = expected_values[idx]

				# if expected value is lower than actionable range and you can't go down
				if e_value < current_value and not action_meta[f]['can_decrease']:
					expected_values[idx] = current_value

				# opposite
				if e_value > current_value and not action_meta[f]['can_increase']:
					expected_values[idx] = current_value

		return expected_values


	def get_counterfactual(test_idx):

		# Totally normalized (0-1)
		x = deepcopy(X_test[test_idx])
		original_query = deepcopy(X_test[test_idx])

		# Get feature probabilities
		cont_probs, expected_conts, cat_probs, expected_cat = get_feature_probabilities(cf_df, test_idx)

		feature_probs = np.array(cont_probs + cat_probs)
		feature_expected = np.array(expected_conts + expected_cat)
		features = continuous_feature_names + categorical_feature_names
		feature_expected = clip_expected_values(test_idx, feature_expected, features)
		feature_order = np.argsort(feature_probs)
		original_prob = clf.predict_proba(X_test[test_idx].reshape(1,-1))[0][1]
		current_prob = clf.predict_proba(X_test[test_idx].reshape(1,-1))[0][1]
		original_pred = clf.predict(X_test[test_idx].reshape(1,-1)).item()

		# Flip the excpetional feature(s) one at a time:
		for i in range(len(feature_order)):

			if action_meta[features[feature_order[i]]]['actionable']:

				temp = deepcopy(x)
				tempx = deepcopy(x)

				if features[feature_order[i]] in continuous_feature_names:
					temp[ feature_order[i] ] = expected_conts[ feature_order[i] ]
				else:
					temp = flip_category(temp, cat_name=features[feature_order[i]],
										 change_to=feature_expected[feature_order[i]])

				new_prob = clf.predict_proba(temp.reshape(1,-1))[0][1]
				new_pred = clf.predict(temp.reshape(1,-1)).item()

				if new_pred != original_pred:
					return temp, original_prob, current_prob

				if new_prob < current_prob:
					x = temp
					current_prob = new_prob

		return temp, original_prob, current_prob


	def generate_cat_idxs():
		"""
		Get indexes for all categorical features that are one hot encoded
		"""
		
		cat_idxs = list()
		start_idx = len(continuous_feature_names)
		for cat in enc.categories_:
			cat_idxs.append([start_idx, start_idx + cat.shape[0]])
			start_idx = start_idx + cat.shape[0]
		return cat_idxs


	cat_idxs = generate_cat_idxs()
	ga_df = pd.read_csv('data/GA_Xps_diverse.csv')
	test_idxs = np.sort(np.array(ga_df.test_idx.value_counts().index.tolist()))
	piece_sfs = list()

	for test_idx in test_idxs:   # range(len(X_test)):
		sf, _, _ = get_counterfactual(test_idx)
		sf = sf.tolist()
		sf.append(test_idx)
		piece_sfs.append( sf )

	piece_sfs = np.array(piece_sfs)
	np.save('data/piece_sfs.npy', piece_sfs)


def dice_algorithm(seed, DIVERSITY_SIZE, max_num_samples):

	action_meta = actionability_constraints()
	df = get_dataset()
	numerical = continuous_feature_names

	train_dataset = pd.read_csv('data/df_train.csv')
	test_dataset = pd.read_csv('data/df_test.csv')

	#### Need to add this text so that DiCE works
	for f in train_dataset.columns:
		if f not in numerical and f != TARGET_NAME:
			train_dataset[f] = train_dataset[f].astype(str) + '-Cat'
	#### Need to add this text so that DiCE works
	for f in test_dataset.columns:
		if f not in numerical and f != TARGET_NAME:
			test_dataset[f] = test_dataset[f].astype(str) + '-Cat'

	y_train = np.load('data/y_train.npy', )
	y_test = np.load('data/y_test.npy', )
	x_train = deepcopy(train_dataset)
	x_test = deepcopy(test_dataset)
	train_dataset[TARGET_NAME] = y_train
	test_dataset[TARGET_NAME] = y_test


	# ## DiCE
	# Step 1: dice_ml.Data
	d = dice_ml.Data(dataframe=train_dataset,
					 continuous_features=continuous_feature_names,
					 outcome_name=TARGET_NAME,
					 method='genetic')

	numerical = continuous_feature_names
	categorical = x_train.columns.difference(numerical)

	categorical_transformer = Pipeline(steps=[
		('onehot', OneHotEncoder(handle_unknown='ignore'))])

	normalizer_transformer = Pipeline(steps=[
		('minmax', MinMaxScaler())])

	transformations = ColumnTransformer(
		transformers=[
			('cat', categorical_transformer, categorical),
			('all', normalizer_transformer, numerical),
		])

	clf = Pipeline(steps=[('preprocessor', transformations),
						  ('classifier', LogisticRegression(class_weight='balanced', fit_intercept=False, max_iter=1000))])
	model = clf.fit(x_train, y_train)


	def get_actionable_range(x):
		
		dice_action = {}

		for feature in action_meta.keys():

			# Only add actionable features for DiCE's constraints
			if action_meta[feature]['actionable']:

				if feature in numerical:
					query_min_value = float(x[feature])
					query_max_value = float(x[feature])
					min_value = min([float(xxx) for xxx in pd.concat([x_train, x_test])[feature].values])
					max_value = max([float(xxx) for xxx in pd.concat([x_train, x_test])[feature].values])
				else:
					query_min_value = int(x[feature][0])
					query_max_value = int(x[feature][0])
					min_value = min([int(xxx[0]) for xxx in pd.concat([x_train, x_test])[feature].values])
					max_value = max([int(xxx[0]) for xxx in pd.concat([x_train, x_test])[feature].values])

				# Is it up or down mutable?
				if action_meta[feature]['can_increase']:
					query_max_value = max_value

				if action_meta[feature]['can_decrease']:
					query_min_value = min_value
					
				# If it is a continuous feature
				if feature in numerical:
					dice_action[feature] = [float(query_min_value), float(query_max_value)]
				else:
					dice_action[feature] = [str(x) + '-Cat' for x in list(range(query_min_value, query_max_value+1))]

		return dice_action


	# ## Explanation Generation Loop
	# Using sklearn backend
	m = dice_ml.Model(model=model, backend="sklearn")
	# Using method=random for generating CFs
	exp = dice_ml.Dice(d, m, method="random")
	original_preds = model.predict(x_test)
	failed = list()
	ga_df = pd.read_csv('data/GA_Xps_diverse.csv')
	test_idxs = np.sort(np.array(ga_df.test_idx.value_counts().index.tolist()))

	for ex_idx in test_idxs:

		try:
			x = x_test.iloc[ex_idx]
			dice_action = get_actionable_range(x)
			e3 = exp.generate_counterfactuals(
											  x_test[ex_idx:ex_idx+1],
											  total_CFs=DIVERSITY_SIZE,
											  desired_class="opposite",
											  permitted_range=dice_action,
											  features_to_vary=list(dice_action.keys())
											 )
			e3.visualize_as_list(ex_idx)
			failed.append(0)

		except:
			failed.append(1)
			li = list()
			for _ in range(DIVERSITY_SIZE):
				li.append(x.values.tolist()+[1])
			with open('DiCE_Xps/test' + str(ex_idx) + '.pkl', 'wb') as fp:
				# print("length of file being dumped into pickel:", ex_idx, len(li))
				pickle.dump(li, fp)

	# ## Convert Test Data Back into original format to save df
	for f in x_test.columns:
		if f not in numerical:
			x_test[f] = [int(xxx[:-4]) for xxx in x_test[f].values]
	x_test.to_csv('DiCE_Xps/test_df.csv')

	#### Convert Explanations
	dice_cfs = list()
	for ex_idx in test_idxs:  
		with open('DiCE_Xps/test'+ str(ex_idx) +'.pkl', 'rb') as fp:
			banana = pickle.load(fp)
		for b in banana:
			for i in range(len(b)):
				if type(b[i]) == str:
					b[i] = int(b[i][:-4])
			dice_cfs.append([int(xxx) for xxx in b])
		# for failed counterfactuals
		if len(banana) < DIVERSITY_SIZE:
			fill_in = DIVERSITY_SIZE - len(banana)
			for _ in range(fill_in):
				dice_cfs.append([int(xxx) for xxx in b])

	dice_cfs = np.array(dice_cfs)
	xp_df = pd.DataFrame(dice_cfs, columns=x_train.columns.tolist() + ['original_pred'])

	# add index to cfs
	new_list = list()
	for item in test_idxs:
		for _ in range(DIVERSITY_SIZE):
			new_list.append(item)

	xp_df['test_idx'] = new_list
	del xp_df['original_pred']
	xp_df.to_csv('DiCE_Xps/xp_df.csv')

	#### Need to add this text so that DiCE works
	for f in x_test.columns:
		if f not in numerical:
			x_test[f] = x_test[f].astype(str) + '-Cat'

	#### Need to add this text so that DiCE works
	for f in xp_df.columns:
		if f not in numerical:
			xp_df[f] = xp_df[f].astype(str) + '-Cat'

	found_sfs = list()
	for ex_idx in test_idxs:
		x = x_test.iloc[ex_idx]
		dice_action = get_actionable_range(x)
		for idx, char in enumerate(list(string.ascii_lowercase)[:DIVERSITY_SIZE]):
			instance = xp_df[x_test.columns][xp_df.test_idx==str(ex_idx)+'-Cat'][idx:idx+1]
			try:
				e3 = exp.generate_counterfactuals(
												  instance,
												  total_CFs=1,
												  desired_class="opposite",
												  permitted_range=dice_action,
												  features_to_vary=list(dice_action.keys())
												 )
				e3.visualize_as_list( str(ex_idx)+char )
				found_sfs.append([ex_idx, char, 1])
			except:
				with open('DiCE_Xps/test' + str(ex_idx) + char + '.pkl', 'wb') as fp:
					pickle.dump([instance.values.tolist()[0] + [1]], fp)
				found_sfs.append([ex_idx, char, 0])


	#### Convert Explanations
	dice_sfs = list()
	for ex_idx in test_idxs:
		for char in list(string.ascii_lowercase)[:DIVERSITY_SIZE]:
			with open('DiCE_Xps/test' + str(ex_idx) + char + '.pkl', 'rb') as fp:
				banana = pickle.load(fp)							
			if len(np.array(banana).shape) == 3:
				banana = banana[0]
			for b in banana:
				for i in range(len(b)):
					if type(b[i]) == str:
						b[i] = int(b[i][:-4])
			dice_sfs.append([int(xxx) for xxx in b])

	dice_sfs = np.array(dice_sfs)

	# add index to cfs
	failed_cf_list = list()
	for item in failed:
		for _ in range(DIVERSITY_SIZE):
			failed_cf_list.append(item)

	sf_df = pd.DataFrame(dice_sfs, columns=x_train.columns.tolist() + ['original_pred'])
	sf_df['found_sf'] = [ int(x) for x in np.array(found_sfs)[:, -1]]
	sf_df['failed_cf'] = failed_cf_list
	sf_df['test_idx'] = new_list

	# remove examples which couldn't make the explanation(s)
	sf_df = sf_df[sf_df.found_sf==1]
	sf_df = sf_df[sf_df.failed_cf==0]

	# Remove indexs with less than m sfs found
	sf_df = sf_df.groupby('test_idx').filter(lambda x : len(x)==DIVERSITY_SIZE)
	sf_df.to_csv('data/dice_diverse.csv')


def genetic_algorithm(seed, DIVERSITY_SIZE, max_num_samples, POPULATION_SIZE):

	df = get_dataset()
	indices = np.arange(df.shape[0])
	target = df[TARGET_NAME].values
	del df[TARGET_NAME]
	_, _, _, _, idx_train, _ = train_test_split(df, 
												target,
												indices, 
												test_size=0.5, 
												random_state=seed,
												stratify=target)

	training = np.zeros(df.shape[0])
	training[idx_train] = 1
	df['training'] = training
	np.save('data/training_idx.npy', idx_train)

	## Name the Continuous & Categorical Features
	continuous_features = df[continuous_feature_names]
	categorical_features = df[categorical_feature_names]
	enc = OneHotEncoder().fit(categorical_features)
	categorical_features_enc = enc.transform(categorical_features).toarray()

	with open('data/enc.pkl', 'wb') as file:
		pickle.dump(enc, file)

	#### NB: Continuous features are first
	data = np.concatenate((continuous_features.values, categorical_features_enc), axis=1)
	df_train = df[df.training == 1]
	df_test = df[df.training == 0]
	df_train = df_train.reset_index(inplace=False, drop=True)
	df_test = df_test.reset_index(inplace=False, drop=True)
	del df_train['training']
	del df_test['training']
	df_train.to_csv('data/df_train.csv')
	df_test.to_csv('data/df_test.csv')
	X_train = data[(df.training == 1).values]
	X_test = data[(df.training == 0).values]

	# ## Convert targets to 0 and 1
	y_train = target[(df.training == 1).values]
	y_test = target[(df.training == 0).values]
	np.save('data/X_train.npy', X_train)
	np.save('data/X_test.npy', X_test)
	np.save('data/y_train.npy', y_train)
	np.save('data/y_test.npy', y_test)

	# ## Normalization
	scaler = MinMaxScaler().fit(X_train)
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)

	## Modelling
	#### Logistic Regression
	clf = LogisticRegression(class_weight='balanced', fit_intercept=False, max_iter=1000).fit(X_train, y_train)
	preds = clf.predict(X_test)
	probs = clf.predict_proba(X_test)
	df_test['preds'] = preds
	df_test['probs'] = probs.T[1]

	with open('data/clf.pkl', 'wb') as file:
		pickle.dump(clf, file)

	## Genetic Algorithm
	def fitness(x, population, cat_idxs, actionable_idxs, clf, action_meta, continuous_features, categorical_features):
		
		fitness_scores = list()
		meta_fitness = list()
		
		for solution in population:
			reachability = get_reachability(solution) 
			gain = get_gain(x, solution)
			robustness_1 = get_robustness(x, solution, clf, cat_idxs,
										actionable_idxs, action_meta,
										continuous_features, categorical_features) * 1
			
			robustness_2 = (clf.predict(solution) == POSITIVE_CLASS) * 1
			diversity = get_diversity(solution)
			
			term1 = np.array(reachability.flatten() * gain)
			robustness_1 = np.array(robustness_1)
			robustness_2 = np.array(robustness_2)
			
					
			robustness_1 *= LAMBDA1
			robustness_2 *= LAMBDA2
			diversity    *= GAMMA
			
			term1 = (term1 + robustness_1 + robustness_2).mean()
			
			correctness = clf.predict(solution).mean()  # hard constraint that the solution MUST contain SF
			fitness_scores.append( (term1 + diversity).item() * correctness )
			meta_fitness.append( [reachability.mean(), gain.mean(), robustness_1.mean(), robustness_2.mean(), diversity] )
		
		return np.array(fitness_scores), np.array(meta_fitness)


	def get_diversity(solution):
		"""
		Return L2 distance between all vectors (the mean)
		"""
		
		if DIVERSITY_SIZE == 1:
			return 0
		
		# Take average distance
		score = distance_matrix(solution, solution).sum() / (DIVERSITY_SIZE**2 - DIVERSITY_SIZE)
		return score


	def get_reachability(solution):
		"""
		OOD Check using NN-dist metric
		"""
		
		l2s, _ = REACH_KNN.kneighbors(X=solution, n_neighbors=1, return_distance=True)    
		l2s = 1 / (l2s**2 + 0.1)
		return l2s


	def get_gain(x, solution):
		"""
		Return mean distance between query and semifactuals
		"""
		
		scores = np.sqrt(((x - solution)**2).sum(axis=1))    
		return scores


	def get_robustness(x, solution, clf, cat_idxs, actionable_idxs, action_meta, continuous_features, categorical_features):
		"""
		Monte Carlo Approximation of e-neighborhood robustness
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
			perturbation_preds.append(predictions.tolist())
		return np.array(perturbation_preds).mean(axis=1)


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


	def perturb_one_random_feature(x, x_prime, continuous_features, categorical_features, action_meta, cat_idxs, actionable_idxs):
		"""
		perturb one actionable feature for MC robustness optimization
		"""
		
		feature_names = continuous_features.columns.tolist() + categorical_features.columns.tolist()
		change_idx    = get_rand_actionable_feature_idx(x, actionable_idxs, cat_idxs)[0]
		feature_num   = len(feature_names)
			
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


	def generate_cat_idxs():
		"""
		Get indexes for all categorical features that are one hot encoded
		"""
		
		cat_idxs = list()
		start_idx = len(continuous_features.columns)
		for cat in enc.categories_:
			cat_idxs.append([start_idx, start_idx + cat.shape[0]])
			start_idx = start_idx + cat.shape[0]
		return cat_idxs


	def generate_category(x, x_prime, idx, cat_idxs, action_meta, replace=True):
		"""
		Randomly generate a value for a OHE categorical feature using actionability constraints
		replace: this gives the option if the generation should generate the original
		value for the feature that is present in x, or if it should only generate 
		different x_primes with different values for the feature
		
		"""
		
		original_rep = x[cat_idxs[idx][0]: cat_idxs[idx][1]]  # To constrain with initial datapoint
		new_rep = x_prime[cat_idxs[idx][0]: cat_idxs[idx][1]]  # to make sure we modify based on new datapoint
		
		cat_name = categorical_features.columns[idx]
		
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


	def init_population(x, X_train, continuous_features, categorical_features, action_meta, replace=True):
		
		num_features = X_train.shape[1]
		population = np.zeros((POPULATION_SIZE, DIVERSITY_SIZE, num_features))

		# iterate continous features
		for i in range(len(continuous_features.columns)):

			cat_name = continuous_features.columns[i]
			value = x[i]

			# If the continuous feature can take any value
			if action_meta[ cat_name ]['can_increase'] and action_meta[ cat_name ]['can_decrease']:
				f_range = action_meta[ cat_name ]['max'] - action_meta[ cat_name ]['min']
				temp = value + np.random.normal(0, CONT_PERTURB_STD, (POPULATION_SIZE, DIVERSITY_SIZE, 1))
				temp *= f_range
				population[:, :, i:i+1] = temp

			# If the continous feature can only go up
			elif action_meta[ cat_name ]['can_increase'] and not action_meta[ cat_name ]['can_decrease']:
				f_range = action_meta[ cat_name ]['max'] - value
				temp = value + abs(np.random.normal(0, CONT_PERTURB_STD, (POPULATION_SIZE, DIVERSITY_SIZE, 1)))
				temp *= f_range
				population[:, :, i:i+1] = temp

			# if the continuous features can only go down
			elif not action_meta[ cat_name ]['can_increase'] and action_meta[ cat_name ]['can_decrease']:
				f_range = value
				temp = value - abs(np.random.normal(0, CONT_PERTURB_STD, (POPULATION_SIZE, DIVERSITY_SIZE, 1)))
				temp *= f_range
				population[:, :, i:i+1] = temp

			# If it's not actionable
			else:
				temp = np.zeros((POPULATION_SIZE, DIVERSITY_SIZE, 1)) + value
				population[:, :, i:i+1] = temp

		# iterate categorical features
		current_idx = len(continuous_features.columns)
		for i in range(len(categorical_features.columns)):
			cat_len = len(x[cat_idxs[i][0]: cat_idxs[i][1]])
			temp = list()

			for j in range(POPULATION_SIZE):
				temp2 = list()
				for k in range(DIVERSITY_SIZE):
					x_prime = deepcopy(x)  # to keep x the same
					temp3 = generate_category(x, x_prime, i, cat_idxs, action_meta, replace=True)
					temp2.append(temp3.tolist())
				temp.append(temp2)

			temp = np.array(temp)
			population[:, :, current_idx:current_idx+cat_len] = temp
			current_idx += cat_len
			
		return population


	def mutation(population, continuous_features, categorical_features, x):
		"""
		Iterate all features and randomly perturb them
		"""
		
		feature_names = continuous_features.columns.tolist() + categorical_features.columns.tolist()
		
		for i in range(len(population)):
			for j in range(DIVERSITY_SIZE):
				x_prime = population[i][j]
				for k in range(len(actionable_idxs)):
					if np.random.rand() < MUTATION_RATE:
						change_idx = actionable_idxs[k][0]
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
		return population


	def natural_selection(population, fitness_scores):
		"""
		Save the top solutions
		"""
		
		tournamet_winner_idxs = list()
		for i in range(POPULATION_SIZE - ELITIST):
			knights = np.random.randint(0, population.shape[0], 2)
			winner_idx = knights[np.argmax(fitness_scores[knights])]
			tournamet_winner_idxs.append(winner_idx)
		return population[tournamet_winner_idxs], population[(-fitness_scores).argsort()[:ELITIST]]


	def crossover(population):
		"""
		mix up the population
		"""
		
		children = list()

		for i in range(0, population.shape[0], 2):

			parent1, parent2 = population[i:i+2]
			child1, child2 = deepcopy(parent1), deepcopy(parent2)

			crossover_idxs = np.random.randint(low=0,
											   high=2,
											   size=DIVERSITY_SIZE*len(actionable_idxs)).reshape(DIVERSITY_SIZE, len(actionable_idxs))

			# Crossover Children
			for j in range(DIVERSITY_SIZE):
				for k in range(len(actionable_idxs)):

					# Child 1
					if crossover_idxs[j][k] == 0:

						# if continuous
						if actionable_idxs[k][0] < len(continuous_features.columns):
							child1[j][actionable_idxs[k][0]] = parent2[j][actionable_idxs[k][0]]

						# if categorical
						else:
							cat_idx = actionable_idxs[k][0] - len(continuous_features.columns)
							child1[j][cat_idxs[cat_idx][0]: cat_idxs[cat_idx][1]] = parent2[j][cat_idxs[cat_idx][0]: cat_idxs[cat_idx][1]]


					# Child 2
					else:
						# if continuous
						if actionable_idxs[k][0] < len(continuous_features.columns):
							child2[j][actionable_idxs[k][0]] = parent1[j][actionable_idxs[k][0]]

						# if categorical
						else:
							cat_idx = actionable_idxs[k][0] - len(continuous_features.columns)
							child2[j][cat_idxs[cat_idx][0]: cat_idxs[cat_idx][1]] = parent1[j][cat_idxs[cat_idx][0]: cat_idxs[cat_idx][1]]

			children.append(child1.tolist())
			children.append(child2.tolist())
		
		return np.array(children)


	def force_sf(result):
		result_preds = clf.predict(result)
		keep = np.where(result_preds==abs(POSITIVE_CLASS))[0]
		sf = result[keep[0]]
		replace_these_idxs = np.where(result_preds==abs(POSITIVE_CLASS-1))[0]
		for idx in replace_these_idxs:
			result[idx] = sf
		return result

	# just for lending club so speed up NN searches
	if X_train.shape[0] > 885300:
		_, X_train, _, y_train = train_test_split(X_train, 
													y_train,
													test_size=0.05, 
													random_state=42)

	action_meta = actionability_constraints()
	cat_idxs = generate_cat_idxs()
	actionable_idxs = get_actionable_feature_idxs(continuous_features, categorical_features)

	# Necessary variables
	REACH_KNN = KNeighborsClassifier(p=2).fit(X_train, y_train)
	MAX_GENERATIONS = 20
	LAMBDA1 = 10  # robustness e-neighborhood
	LAMBDA2 = 10  # robustness instance
	GAMMA = 10  # diversity
	POSITIVE_CLASS = 1  # the semi-factual positive "loan accepted" class number
	CONT_PERTURB_STD = 0.05 # perturb continuous features by 5% STD
	MUTATION_RATE = 0.05
	ELITIST = 4  # how many of the "best" to save
	MAX_MC = 100

	sf_data = list()
	found_sfs = list()
	fails_to_find_sfs = 0
	print("Population Size:", POPULATION_SIZE)

	for test_idx in range(X_test.shape[0]):

		start_time = time.time()
		
		x = X_test[test_idx]
		x_prime = deepcopy(x)

		if clf.predict_proba(x.reshape(1,-1))[0][1] < 0.8:
			continue

		# this while loop exists so that the initial population has at least one semifactual
		avg_preds = 0.0
		counter_xxx = 0
		while avg_preds < 0.3:
			counter_xxx += 1
			population = init_population(x, X_train, continuous_features, categorical_features, action_meta, replace=True)
			avg_preds = clf.predict(population.reshape(-1, x.shape[0])).mean()
			if counter_xxx == 100:
				break
		if counter_xxx == 100:
			continue			
			
		# Start GA
		for generation in range(MAX_GENERATIONS):
			
			# Evaluate fitness (meta = reachability, gain, robustness, diversity)
			fitness_scores, meta_fitness = fitness(x, population, cat_idxs,
												  actionable_idxs, clf, action_meta,
												  continuous_features, categorical_features)

			# Selection
			population, elites = natural_selection(population, fitness_scores)

			# Crossover
			population = crossover(population)

			# Mutate
			population = mutation(population, continuous_features, categorical_features, x)

			# Carry over elite solutions
			population = np.concatenate((population, elites), axis=0)

			# Evaluate fitness (meta = reachability, gain, robustness, diversity)
			fitness_scores, meta_fitness = fitness(x, population, cat_idxs,
												  actionable_idxs, clf, action_meta,
												  continuous_features, categorical_features)

		result = population[np.argmax( fitness_scores )]        
		logging.info( str(time.time() - start_time) )
		if sum(fitness_scores * (meta_fitness.T[-2] == LAMBDA2)) > 0:
			for d in result:
				sf_data.append( d.tolist() )
			found_sfs.append([test_idx, True])
			
		else:
			result = force_sf(result)
			for d in result:
				sf_data.append( d.tolist() )
			found_sfs.append([test_idx, False])

		print("Took Sec:", round(time.time() - start_time, 2))
		if len(sf_data) == max_num_samples*DIVERSITY_SIZE:
			print("Acquired number of test instances specified")
			break

	sf_data = np.array(sf_data)
	np.save('data/GA_sfs.npy', sf_data)
	success_data = list()
	idx_data = list()

	for d in found_sfs:
		for _ in range(DIVERSITY_SIZE):
			success_data.append(int(d[1]))
			idx_data.append(d[0])
			
	success_data = np.array(success_data).reshape(-1, 1)
	idx_data = np.array(idx_data).reshape(-1, 1)
	sf_df = pd.DataFrame(sf_data)
	sf_df['test_idx'] = idx_data
	sf_df['sf_found'] = success_data
	sf_df.to_csv('data/GA_Xps_diverse.csv')



