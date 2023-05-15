import utils
import data_utils
import trainers
import recourse
import attacks
import pdb
import numpy as np
import torch
import pandas as pd

from tqdm import tqdm
from scipy.spatial.distance import pdist
from copy import deepcopy


def find_recourse_mlp(model, trainer, method, scmm, X_explain, constraints):

    hyperparams = utils.get_recourse_hyperparams(trainer)

    if method == 'S-GEN':
        explain = recourse.DifferentiableRecourseSGEN(model, hyperparams)
    if method == 'karimi':
        explain = recourse.DifferentiableRecourseKarimi(model, hyperparams)
    if method == 'dominguez':
        explain = recourse.DifferentiableRecourseDominguez(model, hyperparams)

    (gain_recourse, cost_action), robustness, diversity, valids, (interv_mask, actions), counterfacs = recourse.causal_recourse(X_explain, explain,
                                                                                    constraints, scm=scmm)

    return (gain_recourse, cost_action), robustness, diversity, valids, (interv_mask, actions), counterfacs


def eval_recourse(dataset, model_type, method, trainer, random_seed, N_explain, lambd, save_adv=False):
    # Set the random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Load the relevant dataset
    X, Y, constraints = data_utils.process_data(dataset)
    X_train, Y_train, X_test, Y_test = data_utils.train_test_split(X, Y)

    # Load the relevant model
    model_dir = utils.get_model_save_dir(dataset, trainer, model_type, random_seed, lambd) + '.pth'
    model = trainers.LogisticRegression if model_type == 'lin' else trainers.MLP
    model = model(X_train.shape[-1], actionable_features=constraints['actionable'], actionable_mask=trainer=='AF')
    model.load_state_dict(torch.load(model_dir))
    model.set_max_mcc_threshold(X_train, Y_train)

    # Load the SCM
    scmm = utils.get_scm(model_type, dataset)
    
    id_neg = model.predict(X_test) == 1  # choose positive outcome as test data
    X_neg = X_test[id_neg]
    N_Explain = min(N_explain, len(X_neg))

    # Different seed here
    id_explain = np.random.choice(np.arange(X_neg.shape[0]), size=N_Explain, replace=False)
    id_neg_explain = np.argwhere(id_neg)[id_explain]
    X_explain = X_neg[id_explain]

    # Find recourse
    find_recourse = find_recourse_lin if model_type == 'lin' else find_recourse_mlp
    (gain_recourse, cost_action), robustness, diversity, valids, (valid_interv_sets, interv), counterfacs = find_recourse(model, trainer, method, scmm, X_explain, constraints)

    adv_robustness = list()
    valids = np.array(valids)

    for i in range(interv.shape[1]):
        if sum(valids[i]) == 0:
            continue
        attacker = attacks.CW_Adversary(scmm=scmm)
        adv_egs, valid_adv, cost_adv = attacker(model, X_explain[valids[i]], torch.Tensor(interv[:, i, : ][valids[i]]), interv_set=valid_interv_sets[:, i, : ][valids[i]])
        
        final_valids = np.delete(np.where([valids[i]][0]==True)[0], np.where(valid_adv==False)[0])
        cost_padded = np.array([np.nan for _ in range(len(X_explain))])
        cost_padded[final_valids] = cost_adv[valid_adv]
        adv_robustness.append(cost_padded.tolist())

    adv_robustness = np.nanmean(adv_robustness, axis=0)

    print(model.predict(counterfacs))
    print("Robustness:", np.nanmean(adv_robustness))
    print("Gain:", gain_recourse.mean())

    df = pd.DataFrame()
    df['CostAction'] = cost_action
    df['Gain'] = gain_recourse
    df['RobustnessAdv'] = adv_robustness
    df['RobustnessMCMC'] = robustness
    df['Diversity'] = diversity
    df['Dataset'] = dataset
    df['Method'] = method
    df.to_csv('results/' + dataset + method + str(random_seed) + '.csv', index=False)



