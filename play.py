import maml_rl.envs
import gym
import numpy as np
import torch
import json
import os

from maml_rl.metalearner import MetaLearner
from maml_rl.policies import CategoricalMLPPolicy, NormalMLPPolicy
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.sampler import BatchSampler
from maml_rl.episode import BatchEpisodes

from tensorboardX import SummaryWriter

def total_rewards(episodes_rewards, aggregation=torch.mean):
    rewards = torch.mean(torch.stack([aggregation(torch.sum(rewards, dim=0))
        for rewards in episodes_rewards], dim=0))
    return rewards.item()

class args:
    def __init__(self):
        self.env_name = 'PendulumTheta-v0'
        self.num_workers = 1
        self.fast_lr = 0.1
        self.max_kl=0.01
        self.fast_batch_size=1   #number of episodes
        self.meta_batch_size = 40 #number of tasks
        self.num_layers = 2
        self.hidden_size = 100
        self.num_batches=2 #100. Number of iterations
        self.gamma = 0.99 #gae
        self.tau = 1.0 #gae
        self.cg_damping = 1e-5
        self.ls_max_step= 15
        self.output_folder = 'maml-pendulumTheta'
        self.device = 'cpu'
        self.first_order = False
        self.cg_iters = 10
        self.ls_max_steps = 10
        self.ls_backtrack_ratio = 0.5
args = args()

#input
batch = 99 #695 #497 #353
#for half cheetah velocity. gradient seems small. rewards in the [before,after] = [-260,-200]
#seems to be stuck at the same point, not moving..

#ant flips over way too easily.. seems to need to keep kl and fast_lr both small

#try giving it more episodes

continuous_actions = (args.env_name in ['AntVel-v1', 'AntDir-v1',
    'AntPos-v0', 'HalfCheetahVel-v1', 'HalfCheetahDir-v1',
    '2DNavigation-v0','PendulumTheta-v0'])

sampler = BatchSampler(args.env_name, batch_size=args.fast_batch_size,
        num_workers=args.num_workers)

if continuous_actions:
    the_model = NormalMLPPolicy(
        int(np.prod(sampler.envs.observation_space.shape)),
        int(np.prod(sampler.envs.action_space.shape)),
        hidden_sizes=(args.hidden_size,) * args.num_layers)
else:
    the_model = CategoricalMLPPolicy(
        int(np.prod(sampler.envs.observation_space.shape)),
        sampler.envs.action_space.n,
        hidden_sizes=(args.hidden_size,) * args.num_layers)

#loading the model
save_folder = './saves/{0}'.format(args.output_folder)
the_model.load_state_dict(torch.load(os.path.join(save_folder,
        'policy-{0}.pt'.format(batch))))

baseline = LinearFeatureBaseline(
    int(np.prod(sampler.envs.observation_space.shape)))


metalearner = MetaLearner(sampler, the_model, baseline, gamma=args.gamma,
    fast_lr=args.fast_lr, tau=args.tau, device=args.device)

env = gym.make(args.env_name)

# new task!
episodes = []

#randomly sample task
test_task = sampler.sample_tasks(num_tasks=1)

#set specific task.
#test_task = []
#test_task.append({'velocity': 1.9})
sampler.reset_task(test_task[0])

print("new task: ", test_task[0], ", where 1 is forward")

#task = env.unwrapped.sample_tasks(1)
env.unwrapped.reset_task(test_task[0])
observations = env.reset()
print("new task: ", env.step([1])[3]['task'], ", where 1 is forward")
_theta = env.step([1])[3]['task']
degrees = 180*_theta['theta']/np.pi
print("new task in degrees: ",degrees)

train_episodes = metalearner.sampler.sample(the_model,
    gamma=args.gamma, device=args.device)
print("len of train episoid: ",len(train_episodes))
print(train_episodes)
params = metalearner.adapt(train_episodes, first_order=args.first_order)
valid_episodes = metalearner.sampler.sample(the_model, params=params,
    gamma=args.gamma, device=args.device)
episodes.append((train_episodes, valid_episodes))

for param in [None,params]:
    for i in np.arange(1):
        observations = env.reset()
        if param == None:
            print("New episode before gradient update")
        else:
            print("New episode after one gradient update")

        rewards = 0

        for i in np.arange(1000):
            with torch.no_grad():
                observations_tensor = torch.from_numpy(observations).to(device='cpu')
                actions_tensor = the_model(observations_tensor, params=param).sample()
                actions = actions_tensor.cpu().numpy()
            new_observations, reward, dones, info, = env.step(actions)
            observations, info = new_observations, info
            rewards = rewards + reward
            #print("rewards: ", rewards)
            env.render(mode='human')
env.close()

print('total_rewards/before_update',
    total_rewards([ep.rewards for ep, _ in episodes]), batch)
print('total_rewards/after_update',
    total_rewards([ep.rewards for _, ep in episodes]), batch)
