import numpy as np
import cvxpy as cp
import torch
import pdb   

from tqdm import tqdm
from scipy.spatial.distance import pdist
from copy import deepcopy


EPSILON = 0.1
NUM_MCMC_SAMPLES = 100
AGE_LIMIT = 5
Compas1year = 0.08542784764702091
Adult1year = 0.07331033338816978


def build_feasibility_sets(X, actionable, constraints):

    bounds = (torch.Tensor([[[-1, 1]]]).repeat(X.shape[0], X.shape[-1], 1) * 1e10).numpy()
    for i in range(X.shape[1]):
        if i in actionable:
            if i in constraints['increasing']:
                bounds[:, i, 0] = 0
            elif i in constraints['decreasing']:
                bounds[:, i, 1] = 0
        else:
            bounds[:, i, 0] = 0
            bounds[:, i, 1] = 0

    # Take into account maximum feature magnitude
    delta_limits = constraints['limits'][None] - X[..., None]  # (N, D, 2)
    bounds[..., 0] = np.maximum(delta_limits[..., 0], bounds[..., 0])
    bounds[..., 1] = np.minimum(delta_limits[..., 1], bounds[..., 1])
    return torch.Tensor(bounds)


def causal_recourse(x, explainer, constraints, scm=None, verbose=True, **kwargs):

    if scm is not None:
        sets = scm.getPowerset(constraints['actionable'])  # every possible intervention set
    else:
        sets = [constraints['actionable']]

    # Arrays to store the best recourse found so far (as one iterates through intervention sets)
    actions = list()
    valids = list()
    counterfacs = list()
    interventions = [None]*x.shape[0]

    for i in range(len(sets)):
        interv_set = list(sets[i])
        bounds = build_feasibility_sets(x, interv_set, constraints)

        # Recourse for that subset
        action, finished, _, cfs = explainer.find_recourse(x, interv_set, bounds, constraints, scm=scm, verbose=verbose, **kwargs)

        # costs.append(cost.tolist())
        counterfacs.append(cfs.tolist())
        actions.append(action.tolist())
        valids.append(finished.tolist())

    # costs = np.array(costs)
    counterfacs = np.rollaxis(np.array(counterfacs), 1,0)
    actions     = np.rollaxis(np.array(actions), 1,0)
    x_repeat    = np.repeat(x, len(sets), axis=0).reshape(counterfacs.shape)

    # Consider positive Gain Directions
    gain_recourse = (counterfacs - x_repeat)
    gain_recourse[:, :, constraints['gain_neg'] ] *= -1.
    gain_recourse = gain_recourse.mean(axis=2).mean(axis=1)

    cost_action = deepcopy(abs(actions)) 
    cost_action = cost_action.mean(axis=2).mean(axis=1)

    robustness = list()
    diversity  = list()

    for i in range(len(counterfacs)):
        rob = robustness_mcmc_check(torch.tensor(counterfacs[i], dtype=torch.float32), constraints, explainer.model)
        robustness.append(rob.mean())

    for i in range(len(counterfacs)):
        div = pdist(counterfacs[i], 'minkowski', p=1.).mean()
        diversity.append(div)

    interv_mask = torch.tensor((actions > 0.) * 1)

    return (gain_recourse, cost_action), robustness, diversity, valids, (interv_mask, actions), counterfacs



class DifferentiableRecourseSGEN:

    def __init__(self, model, hyperparams, inner_max_pgd=False, early_stop=False):

        self.l1_loss = torch.nn.L1Loss(reduction='none')
        self.model = model
        self.lr = hyperparams['lr']
        self.lambd_init = hyperparams['lambd_init']
        self.decay_rate = hyperparams['decay_rate']
        self.inner_iters = hyperparams['inner_iters']
        self.outer_iters = hyperparams['outer_iters']
        self.inner_max_pgd = inner_max_pgd
        self.early_stop = early_stop
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')

    def find_recourse(self, x, interv_set, bounds, constraints, target=1., scm=None, verbose=True):

        D = x.shape[1]
        x_og = torch.Tensor(x)
        x_pertb = torch.autograd.Variable(torch.zeros(x.shape), requires_grad=True)  # to calculate the adversarial
                                                                                     # intervention on the features
        ae_tol = 1e-4  # for early stopping
        actions = torch.zeros(x.shape)  # store here valid recourse found so far

        target_vec = torch.ones(x.shape[0]) * target  # to feed into the BCE loss
        unfinished = torch.ones(x.shape[0])  # instances for which recourse was not found so far

        # Define variable for which to do gradient descent, which can be updated with optimizer
        delta = torch.autograd.Variable(torch.zeros(x.shape), requires_grad=True)
        optimizer = torch.optim.Adam([delta], self.lr)

        # Models the effect of the recourse intervention on the features
        def recourse_model(x, delta):
            if scm is None:
                return x + delta  # IMF
            else:
                return scm.counterfactual(x, delta, interv_set)  # counterfactual

        lambd = self.lambd_init
        prev_batch_loss = np.inf  # for early stopping
        pbar = tqdm(range(self.outer_iters)) if verbose else range(self.outer_iters)

        def sample_spherical(npoints, ndim=3):
            vec = np.random.randn(ndim, npoints)
            vec /= np.linalg.norm(vec, axis=0, ord=1)
            return vec.T * EPSILON

        def robustness_mcmc_check(cfs, constraints, model):
            num_features = constraints['limits'].shape[0]

            samples = sample_spherical(NUM_MCMC_SAMPLES, ndim=num_features)

            robustness = list()
            for cf in cfs:
                # repeat cf 
                temp_cf = cf.repeat(NUM_MCMC_SAMPLES, 1)
                # add samples to cf
                temp_cf += torch.tensor(samples, dtype=torch.float32)
                # predict
                robust_score = (model.predict(temp_cf) == 1).mean()  # 1 is original label
                # add mean to robustness list
                robustness.append(robust_score)
            return np.array(robustness)

        # hardcoded for these datasets ONLY
        if constraints['dataset'] == 'compas':
            targets_inc = constraints['limits'][constraints['increasing']].T[1]
            targets_dec = 0.
        if constraints['dataset'] == 'adult':
            targets_inc = constraints['limits'][constraints['increasing']].T[1]
            targets_dec = 0.

        org_label = self.model.predict_torch(recourse_model(x_og, delta.detach()))[0].item()
        avg_diff = list()

        for outer_iter in pbar:
            for inner_iter in range(self.inner_iters):
                optimizer.zero_grad()
                x_cf = recourse_model(x_og, delta)
                with torch.no_grad():

                    pre_unfinished_1 = self.model.predict_torch(recourse_model(x_og, delta.detach())) == org_label  # cf +1
                    pre_unfinished_2 = robustness_mcmc_check(recourse_model(x_og, delta.detach()), constraints, self.model)
                    pre_unfinished_2 = torch.tensor(pre_unfinished_2 == 1) # is it not 100% robust
                    pre_unfinished   = torch.logical_and(pre_unfinished_1, pre_unfinished_2)

                    # Add new solution to solutions
                    # must be (1) robust currently and (2) always robust so far
                    new_solution = torch.logical_and(unfinished, pre_unfinished)
                    actions[new_solution] = torch.clone(delta[new_solution].detach())
                    unfinished = torch.logical_and(pre_unfinished, unfinished)

                # Compute loss clf
                target_vec_clf = torch.ones(x.shape[0])  # 1 is the semi-factual class
                clf_loss   = self.bce_loss(self.model(x_cf), target_vec_clf)

                # Compute loss L1
                cur_inc = x_cf[:,constraints['increasing']]
                cur_dec = x_cf[:,constraints['decreasing']]
                cur_inc_l = self.l1_loss(cur_inc, torch.zeros(cur_inc.shape)+targets_inc).mean(axis=1)
                cur_dec_l = self.l1_loss(cur_dec, torch.zeros(cur_dec.shape)+targets_dec).mean(axis=1)
                cur_dec_l = torch.nan_to_num(cur_dec_l, nan=0.0)

                l1_loss = (cur_inc_l + cur_dec_l).flatten()

                # Final loss
                loss = lambd * clf_loss + l1_loss

                avg_diff.append(l1_loss.mean().item())

                # Apply mask over the ones where recourse has already been found
                loss_mask = unfinished.to(torch.float) * loss
                loss_mean = torch.mean(loss_mask)

                # Update x_cf
                loss_mean.backward()
                optimizer.step()

                # Satisfy the constraints on the features, by projecting delta
                with torch.no_grad():
                    if constraints['dataset'] == 'adult':
                        if AGE_LIMIT is not None:
                            age_bounds = bounds[..., 1].clone()
                            age_bounds[:, 1] = Adult1year * AGE_LIMIT
                            delta[:] = torch.min(torch.max(delta, bounds[..., 0]), age_bounds)
                        else:
                            delta[:] = torch.min(torch.max(delta, bounds[..., 0]), bounds[..., 1])

                    if constraints['dataset'] == 'compas':
                        if AGE_LIMIT is not None:
                            age_bounds = bounds[..., 1].clone()
                            age_bounds[:, 0] = Compas1year * AGE_LIMIT
                            delta[:] = torch.min(torch.max(delta, bounds[..., 0]), age_bounds)
                        else:
                            delta[:] = torch.min(torch.max(delta, bounds[..., 0]), bounds[..., 1])

                # For early stopping
                if self.early_stop and inner_iter % (inner_iters // 10) == 0:
                    if loss_mean > prev_batch_loss * (1 - ae_tol):
                        break
                    prev_batch_loss = loss_mean

            lambd *= self.decay_rate

            if verbose:
                pbar.set_description("Pct left: %.3f Lambda: %.4f" % (float(unfinished.sum()/x_cf.shape[0]), lambd))

            # Get out of the loop if recourse was found for every individual
            if not torch.any(unfinished):
                break

        valid = (delta.sum(axis=1) != 0).detach().cpu().numpy()  # if it moved at all, it's a semi-factual
        cfs = recourse_model(x_og, actions).detach().cpu().numpy()
        cost = torch.sum(torch.abs(actions), -1).detach().cpu().numpy()

        return actions.detach().cpu().numpy(), valid, cost, cfs


class DifferentiableRecourseKarimi:

    def __init__(self, model, hyperparams, inner_max_pgd=False, early_stop=False):

        self.model = model
        self.lr = hyperparams['lr']
        self.lambd_init = hyperparams['lambd_init']
        self.decay_rate = hyperparams['decay_rate']
        self.inner_iters = hyperparams['inner_iters']
        self.outer_iters = hyperparams['outer_iters']
        self.inner_max_pgd = inner_max_pgd
        self.early_stop = early_stop
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')

    def find_recourse(self, x, interv_set, bounds, constraints, target=1., scm=None, verbose=True):

        D = x.shape[1]
        x_og = torch.Tensor(x)
        x_pertb = torch.autograd.Variable(torch.zeros(x.shape), requires_grad=True)  # to calculate the adversarial
                                                                                     # intervention on the features
        ae_tol = 1e-4  # for early stopping
        actions = torch.zeros(x.shape)  # store here valid recourse found so far

        target_vec = torch.ones(x.shape[0]) * target  # to feed into the BCE loss
        unfinished = torch.ones(x.shape[0])  # instances for which recourse was not found so far

        # Define variable for which to do gradient descent, which can be updated with optimizer
        delta = torch.autograd.Variable(torch.zeros(x.shape), requires_grad=True)
        optimizer = torch.optim.Adam([delta], self.lr)

        # Models the effect of the recourse intervention on the features
        def recourse_model(x, delta):
            if scm is None:
                return x + delta  # IMF
            else:
                return scm.counterfactual(x, delta, interv_set)  # counterfactual

        lambd = self.lambd_init
        prev_batch_loss = np.inf  # for early stopping
        pbar = tqdm(range(self.outer_iters)) if verbose else range(self.outer_iters)
        org_label = self.model.predict_torch(recourse_model(x_og, delta.detach()))[0].item()

        # hardcoded for these datasets ONLY
        if constraints['dataset'] == 'compas':
            targets_inc = constraints['limits'][constraints['increasing']].T[1]
            targets_dec = 0.

        if constraints['dataset'] == 'adult':
            targets_inc = constraints['limits'][constraints['increasing']].T[1]
            targets_dec = 0.

        for outer_iter in pbar:
            for inner_iter in range(self.inner_iters):
                optimizer.zero_grad()
                x_cf = recourse_model(x_og, delta)

                with torch.no_grad():
                    pre_unfinished_1 = self.model.predict_torch(recourse_model(x_og, delta.detach())) == org_label  # cf +1
                    pre_unfinished_2 = torch.tensor([True for _ in range(len(x_og))]) # placeholder for version control
                    pre_unfinished   = torch.logical_and(pre_unfinished_1, pre_unfinished_2)

                    new_solution = torch.logical_and(unfinished, pre_unfinished)
                    actions[new_solution] = torch.clone(delta[new_solution].detach())
                    unfinished = torch.logical_and(pre_unfinished, unfinished)

                # Compute loss
                target_vec = torch.zeros(x.shape[0])
                clf_loss   = self.bce_loss(self.model(x_cf), target_vec)
                l1_loss    = torch.sum(torch.abs(delta), -1)
                loss       = clf_loss + lambd * l1_loss

                # Apply mask over the ones where recourse has already been found
                loss_mask = unfinished.to(torch.float) * loss
                loss_mean = torch.mean(loss_mask)

                # Update x_cf
                loss_mean.backward()
                optimizer.step()

                # Satisfy the constraints on the features, by projecting delta
                with torch.no_grad():
                    if constraints['dataset'] == 'adult':
                        if AGE_LIMIT is not None:
                            age_bounds = bounds[..., 1].clone()
                            age_bounds[:, 1] = Adult1year * AGE_LIMIT
                            delta[:] = torch.min(torch.max(delta, bounds[..., 0]), age_bounds)
                        else:
                            delta[:] = torch.min(torch.max(delta, bounds[..., 0]), bounds[..., 1])

                    if constraints['dataset'] == 'compas':
                        if AGE_LIMIT is not None:
                            age_bounds = bounds[..., 1].clone()
                            age_bounds[:, 0] = Compas1year * AGE_LIMIT
                            delta[:] = torch.min(torch.max(delta, bounds[..., 0]), age_bounds)
                        else:
                            delta[:] = torch.min(torch.max(delta, bounds[..., 0]), bounds[..., 1])

                # For early stopping
                if self.early_stop and inner_iter % (inner_iters // 10) == 0:
                    if loss_mean > prev_batch_loss * (1 - ae_tol):
                        break
                    prev_batch_loss = loss_mean

            lambd *= self.decay_rate

            if verbose:
                pbar.set_description("Pct left: %.3f Lambda: %.4f" % (float(unfinished.sum()/x_cf.shape[0]), lambd))

            # Get out of the loop if recourse was found for every individual
            if not torch.any(unfinished):
                break

        valid = (delta.sum(axis=1) != 0).detach().cpu().numpy()  # if it moved at all, it's a semi-factual
        cfs = recourse_model(x_og, actions).detach().cpu().numpy()
        cost = torch.sum(torch.abs(actions), -1).detach().cpu().numpy()

        return actions.detach().cpu().numpy(), valid, cost, cfs



class DifferentiableRecourseDominguez:

    def __init__(self, model, hyperparams, inner_max_pgd=False, early_stop=False):

        self.model = model
        self.lr = hyperparams['lr']
        self.lambd_init = hyperparams['lambd_init']
        self.decay_rate = hyperparams['decay_rate']
        self.inner_iters = hyperparams['inner_iters']
        self.outer_iters = hyperparams['outer_iters']
        self.inner_max_pgd = True
        self.early_stop = early_stop
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')

    def find_recourse(self, x, interv_set, bounds, constraints, target=1., scm=None, verbose=True):

        D = x.shape[1]
        x_og = torch.Tensor(x)
        x_pertb = torch.autograd.Variable(torch.zeros(x.shape), requires_grad=True)  # to calculate the adversarial
                                                                                     # intervention on the features
        ae_tol = 1e-4  # for early stopping
        actions = torch.zeros(x.shape)  # store here valid recourse found so far

        target_vec = torch.zeros(x.shape[0]) * target  # to feed into the BCE loss
        unfinished = torch.ones(x.shape[0])  # instances for which recourse was not found so far

        # Define variable for which to do gradient descent, which can be updated with optimizer
        delta = torch.autograd.Variable(torch.zeros(x.shape), requires_grad=True)
        optimizer = torch.optim.Adam([delta], self.lr)

        # Models the effect of the recourse intervention on the features
        def recourse_model(x, delta):
            if scm is None:
                return x + delta  # IMF
            else:
                return scm.counterfactual(x, delta, interv_set)  # counterfactual

        # Perturbation model is only used when generating robust recourse, models perturbations on the features
        def perturbation_model(x, pertb, delta):
            if scm is None:
                return recourse_model(x, delta) + pertb
            else:
                x_prime = scm.counterfactual(x, pertb, np.arange(D), [True] * D)
                return recourse_model(x_prime, delta)

        # Solve the first order approximation to the inner maximization problem
        def solve_first_order_approx(x_og, x_pertb, delta, target_vec):
            x_adv = perturbation_model(x_og, x_pertb, delta.detach())  # x_pertb is 0, only to backprop
            loss_x = torch.mean(self.bce_loss(self.model(x_adv), torch.ones(x.shape[0])))
            grad = torch.autograd.grad(loss_x, x_pertb, create_graph=False)[0]
            return grad / torch.linalg.norm(grad, dim=-1, keepdims=True) * EPSILON  # akin to FGSM attack

        lambd = self.lambd_init
        prev_batch_loss = np.inf  # for early stopping
        pbar = tqdm(range(self.outer_iters)) if verbose else range(self.outer_iters)


        # hardcoded for these datasets ONLY
        if constraints['dataset'] == 'compas':
            targets_inc = constraints['limits'][constraints['increasing']].T[1]
            targets_dec = 0.

        if constraints['dataset'] == 'adult':
            targets_inc = constraints['limits'][constraints['increasing']].T[1]
            targets_dec = 0.


        for outer_iter in pbar:
            for inner_iter in range(self.inner_iters):
                optimizer.zero_grad()

                pertb = solve_first_order_approx(x_og, x_pertb, delta, target_vec)
                # Solve inner maximization with projected gradient descent
                pertb = torch.autograd.Variable(pertb, requires_grad=True)
                optimizer2 = torch.optim.SGD([pertb], lr=self.lr)

                for _ in range(10):
                    # print(pertb)
                    optimizer2.zero_grad()
                    loss_pertb = torch.mean(self.bce_loss(self.model(x_og + pertb + delta.detach()), torch.zeros(x.shape[0])))
                    loss_pertb.backward()
                    optimizer2.step()

                    # Project to L2 ball, and with the linearity mask
                    with torch.no_grad():
                        norm = torch.linalg.norm(pertb, dim=-1)
                        too_large = norm > EPSILON
                        pertb[too_large] = pertb[too_large] / norm[too_large, None] * EPSILON
                    x_cf = x_og + pertb.detach() + delta

                with torch.no_grad():
                    # To continue optimazing, both the counterfactual or the adversarial counterfactual must be
                    # positively classified
                    pre_unfinished_1 = self.model.predict_torch(recourse_model(x_og, delta.detach())) == 1  # cf +1
                    pre_unfinished_2 = self.model.predict_torch(x_cf) == 1  # cf adversarial
                    pre_unfinished   = torch.logical_and(pre_unfinished_1, pre_unfinished_2)

                    actions[pre_unfinished] = torch.clone(delta[pre_unfinished].detach())
                    unfinished = pre_unfinished.clone()

                # Compute loss
                clf_loss = self.bce_loss(self.model(x_cf), target_vec)
                l1_loss = torch.sum(torch.abs(delta), -1)
                loss = clf_loss + lambd * l1_loss

                # Apply mask over the ones where recourse has already been found
                loss_mask = unfinished.to(torch.float) * loss
                loss_mean = torch.mean(loss_mask)

                # Update x_cf
                loss_mean.backward()
                optimizer.step()

                # Satisfy the constraints on the features, by projecting delta
                with torch.no_grad():
                    if constraints['dataset'] == 'adult':
                        if AGE_LIMIT is not None:
                            age_bounds = bounds[..., 1].clone()
                            age_bounds[:, 1] = Adult1year * AGE_LIMIT
                            delta[:] = torch.min(torch.max(delta, bounds[..., 0]), age_bounds)
                        else:
                            delta[:] = torch.min(torch.max(delta, bounds[..., 0]), bounds[..., 1])

                    if constraints['dataset'] == 'compas':
                        if AGE_LIMIT is not None:
                            age_bounds = bounds[..., 1].clone()
                            age_bounds[:, 0] = Compas1year * AGE_LIMIT
                            delta[:] = torch.min(torch.max(delta, bounds[..., 0]), age_bounds)

                        else:
                            delta[:] = torch.min(torch.max(delta, bounds[..., 0]), bounds[..., 1])

                if self.early_stop and inner_iter % (inner_iters // 10) == 0:
                    if loss_mean > prev_batch_loss * (1 - ae_tol):
                        break
                    prev_batch_loss = loss_mean

            lambd *= self.decay_rate

            if verbose:
                pbar.set_description("Pct left: %.3f Lambda: %.4f" % (float(unfinished.sum()/x_cf.shape[0]), lambd))

            if sum(unfinished)==0:
                break

        valid = (delta.sum(axis=1) != 0).detach().cpu().numpy()  # if it moved at all, it's a semi-factual
        cfs = recourse_model(x_og, actions).detach().cpu().numpy()
        cost = torch.sum(torch.abs(actions), -1).detach().cpu().numpy()
        return actions.detach().cpu().numpy(), valid, cost, cfs


def robustness_mcmc_check(cfs, constraints, model):
    num_actionable_features = constraints['limits'].shape[0]
    samples = generate_points_inside_sphere(num_actionable_features)
    robustness = list()
    for cf in cfs:
        temp_cf = cf.repeat(NUM_MCMC_SAMPLES, 1)
        temp_cf += torch.tensor(samples, dtype=torch.float32)
        robust_score = robust_score = (model.predict(temp_cf) == 1).mean()  # 1 is original label
        robustness.append(robust_score)
    return np.array(robustness)


def generate_points_inside_sphere(n):
    dim = n
    points = np.empty((NUM_MCMC_SAMPLES, dim))
    volume_sphere = np.pi**(n/2) / np.math.gamma(n/2 + 1) * EPSILON**n
    volume_stratum = volume_sphere / NUM_MCMC_SAMPLES
    radius_stratum = (volume_stratum / np.pi)**(1/n)
    for i in range(NUM_MCMC_SAMPLES):
        while True:
            point = np.random.uniform(low=-radius_stratum, high=radius_stratum, size=n)
            norm = np.linalg.norm(point)
            if norm <= EPSILON:
                points[i] = point
                break
            radius_stratum = ((volume_stratum * (i + 1)) / np.pi)**(1/n)
    return points





