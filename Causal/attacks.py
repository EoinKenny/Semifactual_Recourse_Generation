import numpy as np
import torch
import pdb   

from tqdm import tqdm


class CW_Adversary:
    "Implementation of the Carlini & Wagner attack, adapted from https://github.com/kkew3/pytorch-cw2"
    def __init__(self, confidence=0.0, c_range=(1e-3, 1e10), search_steps=15, max_steps=1000, abort_early=True,
                 optimizer_lr=1e-2, init_rand=False, scmm=None, verbose=False):

        self.confidence = float(confidence)
        self.c_range = (float(c_range[0]), float(c_range[1]))
        self.binary_search_steps = search_steps
        self.max_steps = max_steps
        self.abort_early = abort_early
        self.ae_tol = 1e-4  # tolerance of early abort
        self.optimizer_lr = optimizer_lr
        self.init_rand = init_rand
        self.repeat = (self.binary_search_steps >= 10)  # largest c is attempted at least once, ensure some adversarial
                                                        # is found, even if with poor L2 distance
        self.constrains = None
        self.verbose = verbose
        self.scmm = scmm

    def __call__(self, model, x, interv, interv_set=None):

        self.confidence = model.get_threshold_logits() - 1e-9  # since g(x) >= b rather than g(x) > b
        x = torch.Tensor(x)
        y = torch.ones(x.shape[0])

        batch_size = x.shape[0]
        y_np = y.clone().cpu().numpy()  # for binary search

        # Bounds for binary search
        c = np.ones(batch_size) * self.c_range[0]
        c_lower = np.zeros(batch_size)
        c_upper = np.ones(batch_size) * self.c_range[1]

        # To store the best adversarial examples found so far
        x_adv = x.clone().cpu().numpy()  # adversarial examples
        o_best_l2 = np.ones(batch_size) * np.inf  # L2 distance to the adversary

        # Perturbation variable to optimize. maybe this should be reset at each step (delta, optimizer?)
        init_noise = torch.normal(0, 1e-3, x.shape) if self.init_rand else 0.
        delta = torch.autograd.Variable(torch.zeros(x.shape) + init_noise, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=self.optimizer_lr)

        range_list = tqdm(range(self.binary_search_steps)) if self.verbose else range(self.binary_search_steps)
        # Binary seach steps
        for sstep in range_list:

            # If the last step, then go directly to the maximum c
            if self.repeat and sstep == self.binary_search_steps - 1:
                c = c_upper

            # Best solutions for the inner optimization
            best_l2 = np.ones(batch_size) * np.inf
            prev_batch_loss = np.inf  # for early stopping

            steps_range = tqdm(range(self.max_steps)) if self.verbose else range(self.max_steps)
            for optim_step in steps_range:

                batch_loss, l2, yps, advs = self._optimize(model, optimizer, x, interv, interv_set, delta, y, torch.Tensor(c))

                # Constrains on delta if relevant
                if self.constrains is not None:
                    with torch.no_grad():
                        # Satisfy the constraints on the features
                        delta[:] = torch.min(torch.max(delta, self.constrains[0]), self.constrains[1])

                # Early stopping (every 10 steps check if loss has increased sufficiently)
                if self.abort_early and optim_step % (self.max_steps // 10) == 0:
                    if batch_loss > prev_batch_loss * (1 - self.ae_tol):
                        break
                    prev_batch_loss = batch_loss

                # Update best attack found during optimization
                for i in range(batch_size):
                    if yps[i] == (1 - y_np[i]): # valid adversarial example
                        if l2[i] < best_l2[i]:
                            best_l2[i] = l2[i]
                        if l2[i] < o_best_l2[i]:
                            o_best_l2[i] = l2[i]
                            x_adv[i] = advs[i]

            # Binary search of c
            for i in range(batch_size):
                if best_l2[i] < np.inf:  # found an adversarial example, lower c by halving it
                    if c[i] < c_upper[i]:
                        c_upper[i] = c[i]
                    if c_upper[i] < self.c_range[1] * 0.1: # a solution has been found sufficiently early
                        c[i] = (c_lower[i] + c_upper[i]) / 2
                else:
                    if c[i] > c_lower[i]:
                        c_lower[i] = c[i]
                    if c_upper[i] < self.c_range[1] * 0.1:
                        c[i] = (c_lower[i] + c_upper[i]) / 2
                    else: # one order of magnitude more if no solution has been found yet
                        c[i] *= 10

        valid_adversarial = o_best_l2 < np.inf
        norm = np.sqrt(o_best_l2)
        return x_adv, valid_adversarial, norm

    def _optimize(self, model, optimizer, x, interv, interv_set, delta, y, c):

        # Causal perturbation model
        D = x.shape[1]
        if self.scmm is None:
            x_adv = x + interv + delta
        else:
            x_prime = self.scmm.counterfactual(x, delta, np.arange(D), [True] * D)
            x_adv = self.scmm.counterfactual_batch(x_prime, interv, interv_set)  # counterfactual

        # Compute logits of adversary
        z, yp = model.logits_predict(x_adv)

        # Second term of loss (C&W loss): max(z, -k) for y=0, max(-z, -k) for y=1 (for binary classification)
        mask = 2 * y - 1
        loss_cw = torch.clamp(mask * z - self.confidence, min=0.0)

        # First term of loss: l2 norm of the perturbation
        l2_norm = torch.sum(torch.pow(delta, 2), -1)

        # Overall loss with parameter c
        loss = torch.sum(l2_norm + c * loss_cw)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.detach().item(), l2_norm.detach().cpu().numpy(), yp.detach().cpu().numpy(), x_adv.detach().cpu().numpy()
