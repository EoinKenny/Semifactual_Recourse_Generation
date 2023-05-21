import os
import numpy as np
import torch
import scm
import utils
import data_utils
import train_classifiers
import evaluate_recourse


def run_benchmark(models, datasets, methods, seed, N_explain):

    dirs_2_create = [utils.model_save_dir, utils.metrics_save_dir, utils.scms_save_dir]
    for dir in dirs_2_create:
        if not os.path.exists(dir):
            os.makedirs(dir)

    # ------------------------------------------------------------------------------------------------------------------
    # FIT THE STRUCTURAL CAUSAL MODELS
    # ------------------------------------------------------------------------------------------------------------------
    learned_scms = {'adult': scm.Learned_Adult_SCM, 'compas': scm.Learned_COMPAS_SCM}
    for dataset in datasets:
        for model_type in ['lin', 'mlp']:
            # We only need to learn the SCM for some of the data sets
            if dataset in learned_scms.keys():
                # Check if the structural equations have already been fitted
                print('Fitting SCM for %s...' % (dataset))

                # Learn a single SCM (no need for multiple seeds)
                np.random.seed(0)
                torch.manual_seed(0)

                X, _, _ = data_utils.process_data(dataset)
                myscm = learned_scms[dataset](linear=model_type=='lin')
                myscm.fit_eqs(X.to_numpy(), save=utils.scms_save_dir + dataset)

    # ------------------------------------------------------------------------------------------------------------------
    # TRAIN THE DECISION-MAKING CLASSIFIERS
    # ------------------------------------------------------------------------------------------------------------------
    trainers = ['ERM']
    for model_type in models:
        for trainer in trainers:
            for dataset in datasets:
                lambd = utils.get_lambdas(dataset, model_type, trainer)
                save_dir = utils.get_model_save_dir(dataset, trainer, model_type, seed, lambd)

                # Train the model if it has not been already trained
                if not os.path.isfile(save_dir+'.pth'):
                    print('Training... %s %s %s' % (model_type, trainer, dataset))
                    train_epochs = utils.get_train_epochs(dataset, model_type, trainer)
                    accuracy, mcc = train_classifiers.train(dataset, trainer, model_type, train_epochs, lambd, seed,
                                                            verbose=True, save_dir=save_dir)

                    # Save the performance metrics of the classifier
                    save_name = utils.get_metrics_save_dir(dataset, trainer, lambd, model_type, 0, seed)
                    np.save(save_name + '_accs.npy', np.array([accuracy]))
                    np.save(save_name + '_mccs.npy', np.array([mcc]))
                    print(save_name + '_mccs.npy')


    def run_evaluation(dataset, model_type, method, trainer, seed, N_explain, save_adv):

        lambd = utils.get_lambdas(dataset, model_type, trainer)
        print("\n =========================")
        print("Method")
        print(method)
        print("=========================\n")

        print("\n =========================")
        print("Dataset")
        print(dataset)
        print("=========================\n")

        print("\n =========================")
        print("Seed")
        print(seed)
        print("=========================\n")
        evaluate_recourse.eval_recourse(dataset, model_type, method, trainer, seed, N_explain, lambd, save_adv)


    trainer = 'ERM'
    for model_type in models:
        for dataset in datasets:
            for method in methods:
                run_evaluation(dataset, model_type, method, trainer, seed, N_explain, True)


if __name__ == "__main__":
    models = ['mlp']
    datasets = ['compas', 'adult']
    methods = ['S-GEN', 'karimi', 'dominguez']

    N_explain = 30  # number of points for which recourse is found

    for seed in [0,1,2,3,4]:
        run_benchmark(models, datasets, methods, seed, N_explain)

