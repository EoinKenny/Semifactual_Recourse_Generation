import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time 
import pdb
import os  
import shutil
import sys

from algorithms import *
from analysis import *


def main(dataset_name):

	if os.path.exists('DiCE_Xps/'):
		shutil.rmtree('DiCE_Xps/')
	if os.path.exists('figs/'):
		shutil.rmtree('figs/')	

	os.mkdir('DiCE_Xps/')
	os.mkdir('figs/')	

	start_time = time.time()

	df = pd.DataFrame(columns=['gain', 'diversity', 'plausibility', 'robustness', 'failed_sfs', 'method', 'DIVERSITY_SIZE'])

	POP_SIZES = [12,24,48,72,96,120]

	for idx, DIVERSITY_SIZE in enumerate([1, 2, 4, 6, 8, 10]):
		print(" =================================== ")
		print(" ")
		print("Diversity Size:", DIVERSITY_SIZE)
		print(" ")
		print(" =================================== ")

		random_seeds = list(range(3))
		data = list()
		max_num_samples = 30 # max number of test instances to do
		POP_SIZE = POP_SIZES[idx]

		for seed in random_seeds:
			genetic_algorithm(seed, DIVERSITY_SIZE, max_num_samples, POP_SIZE, dataset_name)
			dice_algorithm(seed, DIVERSITY_SIZE, max_num_samples, dataset_name)
			neme_algorithm(seed, DIVERSITY_SIZE, max_num_samples, dataset_name)

			if DIVERSITY_SIZE == 1:
				piece_algorithm(seed, DIVERSITY_SIZE, max_num_samples, dataset_name)

			gain, diversity, plausibility, robustness, failed_sfs = analysis(DIVERSITY_SIZE, seed)
				
			data.append([gain, diversity, plausibility, robustness, failed_sfs])


			metrics = ['gain', 'diversity', 'plausibility', 'robustness', 'failed_sfs']

			if DIVERSITY_SIZE > 1:
				methods = ['S-GEN', 'DiCE*', 'NeMe']
			else:
				methods = ['S-GEN', 'DiCE*', 'NeMe', 'PIECE*']

			for i, method in enumerate(methods):
				temp = list()
				for j, metric in enumerate(metrics):
					result = list()
					for k in range(len(data)):
						result.append( data[k][j][i] )

					result = np.array(result)
					result = result[~np.isnan(result)]

					temp.append(result.mean())
				temp.append(method)
				temp.append(DIVERSITY_SIZE)
				# pdb.set_trace()
				df.loc[len(df)] = temp

			df.to_csv('data_trend_plot.csv')

			df2 = pd.read_csv('data_trend_plot.csv')
			df2 = df2.rename(columns={"gain": "Gain", "diversity": "Diversity", 
							  "plausibility": "Plausibility", "robustness": "Robustness", 
							  "method": "Method", "DIVERSITY_SIZE": "$m$", "failed_sfs": "SF Failed"})

			for f in ['Gain', 'Diversity', 'Plausibility', 'Robustness', 'SF Failed']:
				ax = sns.lineplot(x='$m$', y=f, hue='Method', data=df2, ci=68, marker='o')
				ax.set(xlabel=None)
				ax.set(ylabel=None)
				plt.savefig('figs/' + f + '.pdf')
				plt.close()

	print("Total Time Taken:", time.time() - start_time)


if __name__ == "__main__":
	dataset_name = sys.argv[1]
	main(dataset_name)







