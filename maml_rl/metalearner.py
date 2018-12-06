import torch
from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)
from torch.distributions.kl import kl_divergence

from maml_rl.utils.torch_utils import (weighted_mean, detach_distribution,
                                       weighted_normalize)
from maml_rl.utils.optimization import conjugate_gradient
import numpy as np
import torch.optim as optim
from collections import OrderedDict
from maml_rl.episode import BatchEpisodes
import pdb
import logging
import copy

def total_rewards(episodes_rewards, aggregation=torch.mean):
    rewards = torch.mean(torch.stack([aggregation(torch.sum(rewards, dim=0))
        for rewards in episodes_rewards], dim=0))
    return rewards.item()

class MetaLearner(object):
    """Meta-learner

    The meta-learner is responsible for sampling the trajectories/episodes
    (before and after the one-step adaptation), compute the inner loss, compute
    the updated parameters based on the inner-loss, and perform the meta-update.

    [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic
        Meta-Learning for Fast Adaptation of Deep Networks", 2017
        (https://arxiv.org/abs/1703.03400)
    [2] Richard Sutton, Andrew Barto, "Reinforcement learning: An introduction",
        2018 (http://incompleteideas.net/book/the-book-2nd.html)
    [3] John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan,
        Pieter Abbeel, "High-Dimensional Continuous Control Using Generalized
        Advantage Estimation", 2016 (https://arxiv.org/abs/1506.02438)
    [4] John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan,
        Pieter Abbeel, "Trust Region Policy Optimization", 2015
        (https://arxiv.org/abs/1502.05477)
    """

    logger = logging.getLogger("metalearner")

    def __init__(self, sampler, policy, baseline, gamma=0.95,
                 fast_lr=0.5, tau=1.0, device='cpu',cliprange=0.2, noptepochs=4,
                nminibatches=8,ppo_lr=0.01,useSGD=True,ppo_momentum=0):
        self.sampler = sampler
        self.policy = policy
        self.baseline = baseline
        self.gamma = gamma
        self.fast_lr = fast_lr
        self.tau = tau
        self.to(device)
        self.cliprange=cliprange
        self.noptepochs=noptepochs
        self.nminibatches=nminibatches
        self.ppo_lr=ppo_lr
        self.useSGD=useSGD
        self.ppo_momentum=ppo_momentum

    def inner_loss(self, episodes, params=None):
        """Compute the inner loss for the one-step gradient update. The inner
        loss is REINFORCE with baseline [2], computed on advantages estimated
        with Generalized Advantage Estimation (GAE, [3]).
        https://pytorch.org/docs/0.3.1/distributions.html (except using advantag
        instead of rewards.) Implements eq 4.
        """
        values = self.baseline(episodes)
        advantages = episodes.gae(values, tau=self.tau)
        advantages_unnorm = advantages
        sum_adv = torch.sum(advantages_unnorm).numpy()
        logging.info("unnormalized advantages: "+str(sum_adv))
        logging.info("sum of returns:" + str(torch.sum(episodes.returns)))

        advantages = weighted_normalize(advantages, weights=episodes.mask)

        pi = self.policy(episodes.observations, params=params)
        log_probs = pi.log_prob(episodes.actions)
        if log_probs.dim() > 2:
            # sum over all the workers.
            log_probs = torch.sum(log_probs, dim=2)
        loss = -weighted_mean(log_probs * advantages, dim=0,
            weights=episodes.mask)
        logging.info("inner loss: " + str(loss))

        return loss

    def inner_loss_ppo(self, episodes, first_order,params=None, ent_coef=0,
                        vf_coef=0, nenvs = 1):
        """Compute the inner loss for the one-step gradient update. The inner
        loss is PPO with clipped ratio = new_pi/old_pi.
        Can make cliprange adaptable.
        nenvs = number of workers. nsteps defined in env
        """
        #episodes = [num of steps, num of episodes, obs_space]
        #NEED TO CHANGE ADVANTAGE CALCULATION TO CRITIC.

        self.logger.info("cliprange: "+str(self.cliprange)+ "; noptepochs: "+
            str(self.noptepochs) +";nminibaches: "+ str(self.nminibatches) + ";ppo_lr: " + str(self.ppo_lr))
        # Save the old parameters
        old_policy = copy.deepcopy(self.policy)
        old_params = parameters_to_vector(old_policy.parameters())

        #Need to take mini-batch of sampled examples to do gradient update a few times.
        nepisodes = episodes.observations.shape[1]
        nsteps = episodes.observations.shape[0]
        nbatch = nenvs * nsteps * nepisodes
        nbatch_train = nbatch // self.nminibatches
        mblossvals = []

        #Flattern the episode to [steps, observations]
        episodes_flat = BatchEpisodes(batch_size=nbatch)
        i = 0
        for ep in range(nepisodes):
            for step in range(nsteps):
                episodes_flat.append([episodes.observations[step][ep].numpy()],
                                    [episodes.actions[step][ep].numpy()],
                                    [episodes.returns[step][ep].numpy()],(i,))
                i += 1

        inds = np.arange(nbatch)

        for epoch in range(self.noptepochs):

            # Randomize the indexes
            #np.random.shuffle(inds)

            # 0 to batch_size with batch_train_size step
            for start in range(0, nbatch, nbatch_train):

                mb_obs, mb_returns, mb_masks, mb_actions = [],[],[],[]
                mb_episodes = BatchEpisodes(batch_size=nbatch_train)

                end = start + nbatch_train
                mbinds = inds[start:end]

                for i in range(len(mbinds)):
                    mb_obs.append(episodes_flat.observations[0][mbinds[i]].numpy())
                    mb_returns.append(episodes_flat.returns[0][mbinds[i]].numpy())
                    mb_masks.append(episodes_flat.mask[0][mbinds[i]].numpy())
                    mb_actions.append(episodes_flat.actions[0][mbinds[i]].numpy())
                    mb_episodes.append([mb_obs[i]],[mb_actions[i]],[mb_returns[i]],(i,))

                values = self.baseline(mb_episodes)
                advantages = mb_episodes.gae(values, tau=self.tau)
                advantages_unnorm = advantages
                advantages = weighted_normalize(advantages.type(torch.float32), weights=torch.ones(1,advantages.shape[1]))

                mb_returns_sum = np.sum(mb_returns)
                self.logger.info("iter: "+ "epoch:" + str(epoch) + "; mb:" + str(start/nbatch_train))
                self.logger.info("mb returns: "+str(mb_returns_sum))

                # create optimizer
                if(self.useSGD):
                    optimizer = optim.SGD(self.policy.parameters(), lr=self.ppo_lr, momentum=self.ppo_momentum)
                else:
                    optimizer = optim.Adam(self.policy.parameters(), self.ppo_lr)

                optimizer.zero_grad()   # zero the gradient buffers
                pi = self.policy(mb_episodes.observations)
                log_probs = pi.log_prob(mb_episodes.actions)

                #reload old policy.
                vector_to_parameters(old_params, old_policy.parameters())
                pi_old = old_policy(mb_episodes.observations)

                log_probs_old = pi_old.log_prob(mb_episodes.actions)

                if log_probs.dim() > 2:
                    log_probs_old = torch.sum(log_probs_old, dim=2)
                    log_probs = torch.sum(log_probs, dim=2)

                ratio = torch.exp(log_probs - log_probs_old)

                self.logger.info("max pi: ")
                self.logger.info(torch.max(pi.mean))

                for x in ratio[0][:10]:
                    if x > 1E5 or x <1E-5:
                        #pdb.set_trace()
                        self.logger.info("ratio too large or too small.")
                        self.logger.info(ratio[0][:10])

                self.logger.info("policy ratio: ")
                self.logger.info(ratio[0][:10])

                #loss function
                pg_losses = -advantages * ratio
                pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)

                # Final PG loss
                pg_loss = weighted_mean(torch.max(pg_losses, pg_losses2), weights=torch.ones(1,advantages.shape[1]))

                self.logger.debug("policy mu weights: ")
                self.logger.debug(self.policy.mu.weight)

                sum_adv = torch.sum(advantages_unnorm).numpy()
                self.logger.info("unnormalized advantages: "+str(sum_adv))

                # Total loss
                loss = pg_loss

                self.logger.info("max_action: "+str(np.max(mb_actions)))
                self.logger.info("max_action index: "+str(np.argmax(mb_actions)))

                # Save the old parameters
                old_params = parameters_to_vector(self.policy.parameters())

                loss.backward()
                optimizer.step()
                mblossvals.append(loss)

        self.logger.info("inner loss for each mb and epoch: ")
        self.logger.info(mblossvals)

        #format update parameters
        updated_params = OrderedDict()
        old_param = self.policy.named_parameters()
        old_param_unpack = [*old_param]
        for (name, param) in old_param_unpack:
            updated_params[name] = param

        return updated_params

    def adapt(self, episodes, first_order=False):
        """Adapt the parameters of the policy network to a new task, from
        sampled trajectories `episodes`, with a one-step gradient update [1].
        """
        # Fit the baseline to the training episodes
        self.baseline.fit(episodes)

        loss = self.inner_loss(episodes)
        # Get the new parameters after a one-step gradient update
        params = self.policy.update_params(loss, step_size=self.fast_lr,
            first_order=first_order)

        return params

    def adapt_ppo(self, episodes, first_order=False):
        """Adapt the parameters of the policy network to a new task, from
        sampled trajectories `episodes`, with a one-step gradient update [1].
        """
        # Fit the baseline to the training episodes. NEED TO CHANGE TO V(s).
        self.baseline.fit(episodes)
        # Get the loss on the training episodes
        params = self.inner_loss_ppo(episodes,first_order)

        return params

    def sample(self, tasks, first_order=False,use_ppo=True):
        """Sample trajectories (before and after the update of the parameters)
        for all the tasks `tasks`.
        """
        episodes = []
        for task in tasks:
            self.sampler.reset_task(task)
            train_episodes = self.sampler.sample(self.policy,
                gamma=self.gamma, device=self.device)

            if use_ppo:
                params = self.adapt_ppo(train_episodes, first_order=first_order)
            else:
                params = self.adapt(train_episodes, first_order=first_order)

            self.logger.debug('mu parameter after update:')
            self.logger.debug(params['mu.weight'])
            self.logger.debug("\n")

            valid_episodes = self.sampler.sample(self.policy, params=params,
                gamma=self.gamma, device=self.device)
            episodes.append((train_episodes, valid_episodes))

            self.logger.info('total_rewards/before_update'+
                str(total_rewards([ep.rewards for ep, _ in episodes])))
            self.logger.info('total_rewards/after_update'+
                str(total_rewards([ep.rewards for _, ep in episodes])))
        return episodes

    def kl_divergence(self, episodes, old_pis=None):
        kls = []
        if old_pis is None:
            old_pis = [None] * len(episodes)
        i = 0
        for (train_episodes, valid_episodes), old_pi in zip(episodes, old_pis):
            #Adapt for each task!
            i += 1
            #pdb.set_trace()
            self.logger.info("in kl divergence")
            params = self.adapt_ppo(train_episodes)
            pi = self.policy(valid_episodes.observations, params=params)

            if old_pi is None:
                old_pi = detach_distribution(pi)

            mask = valid_episodes.mask
            if valid_episodes.actions.dim() > 2:
                mask = mask.unsqueeze(2)
            kl = weighted_mean(kl_divergence(pi, old_pi), dim=0, weights=mask)
            kls.append(kl)
        self.logger.info("kl:")
        self.logger.info(kls)
        print(i)
        #pdb.set_trace()
        return torch.mean(torch.stack(kls, dim=0))

    def hessian_vector_product(self, episodes, damping=1e-2):
        """Hessian-vector product, based on the Perlmutter method."""
        def _product(vector):
            kl = self.kl_divergence(episodes)
            grads = torch.autograd.grad(kl, self.policy.parameters(),
                create_graph=True)
            flat_grad_kl = parameters_to_vector(grads)

            grad_kl_v = torch.dot(flat_grad_kl, vector)
            grad2s = torch.autograd.grad(grad_kl_v, self.policy.parameters())
            flat_grad2_kl = parameters_to_vector(grad2s)

            return flat_grad2_kl + damping * vector
        return _product

    def surrogate_loss(self, episodes, old_pis=None):
        losses, kls, pis = [], [], []
        if old_pis is None:
            old_pis = [None] * len(episodes)

        for (train_episodes, valid_episodes), old_pi in zip(episodes, old_pis):
            params = self.adapt_ppo(train_episodes)
            self.logger.info("in surrogate_loss")
            with torch.set_grad_enabled(old_pi is None):
                pi = self.policy(valid_episodes.observations, params=params)
                pis.append(detach_distribution(pi))

                if old_pi is None:
                    old_pi = detach_distribution(pi)

                values = self.baseline(valid_episodes)
                advantages = valid_episodes.gae(values, tau=self.tau)
                advantages = weighted_normalize(advantages,
                    weights=valid_episodes.mask)

                log_ratio = (pi.log_prob(valid_episodes.actions)
                    - old_pi.log_prob(valid_episodes.actions))
                if log_ratio.dim() > 2:
                    log_ratio = torch.sum(log_ratio, dim=2)
                ratio = torch.exp(log_ratio)

                loss = -weighted_mean(ratio * advantages, dim=0,
                    weights=valid_episodes.mask)
                losses.append(loss)

                mask = valid_episodes.mask
                if valid_episodes.actions.dim() > 2:
                    mask = mask.unsqueeze(2)
                kl = weighted_mean(kl_divergence(pi, old_pi), dim=0,
                    weights=mask)
                kls.append(kl)

        return (torch.mean(torch.stack(losses, dim=0)),
                torch.mean(torch.stack(kls, dim=0)), pis)

    def step(self, episodes, max_kl=1e-3, cg_iters=10, cg_damping=1e-2,
             ls_max_steps=10, ls_backtrack_ratio=0.5):
        """Meta-optimization step (ie. update of the initial parameters), based
        on Trust Region Policy Optimization (TRPO, [4]).
        """
        old_loss, _, old_pis = self.surrogate_loss(episodes)
        grads = torch.autograd.grad(old_loss, self.policy.parameters())
        grads = parameters_to_vector(grads)

        # Compute the step direction with Conjugate Gradient
        hessian_vector_product = self.hessian_vector_product(episodes,
            damping=cg_damping)
        stepdir = conjugate_gradient(hessian_vector_product, grads,
            cg_iters=cg_iters)

        # Compute the Lagrange multiplier
        shs = 0.5 * torch.dot(stepdir, hessian_vector_product(stepdir))
        lagrange_multiplier = torch.sqrt(shs / max_kl)

        step = stepdir / lagrange_multiplier

        # Save the old parameters
        old_params = parameters_to_vector(self.policy.parameters())

        # Line search
        step_size = 1.0
        for _ in range(ls_max_steps):
            vector_to_parameters(old_params - step_size * step,
                                 self.policy.parameters())
            loss, kl, _ = self.surrogate_loss(episodes, old_pis=old_pis)
            improve = loss - old_loss
            if (improve.item() < 0.0) and (kl.item() < max_kl):
                break
            step_size *= ls_backtrack_ratio
        else:
            vector_to_parameters(old_params, self.policy.parameters())

    def to(self, device, **kwargs):
        self.policy.to(device, **kwargs)
        self.baseline.to(device, **kwargs)
        self.device = device
