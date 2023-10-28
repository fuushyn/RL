from tqdm import tqdm
from sklearn.metrics import r2_score
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from mujoco_wrappers import Normalize
from torch.distributions.normal import Normal
from collections import defaultdict
import os
from math import gamma
import argparse

# env = gym.make("HalfCheetah-v4", render_mode="rgb_array")
# print("observation space: ", env.observation_space,
#       "\nobservations:", env.reset()[0])
# print("action space: ", env.action_space,
#       "\naction_sample: ", env.action_space.sample())


class Summaries(gym.Wrapper):
    """ Wrapper to write summaries. """
    def __init__(self, env):
        super().__init__(env)
        self.episode_counter = 0
        self.current_step_var = 0

        self.episode_rewards = []
        self.episode_lens = []

        self.current_reward = 0
        self.current_len = 0

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        # rew = min(100, max(rew, -10))
        # rew = info['reward_run']
        self.current_reward += rew
        self.current_len += 1
        self.current_step_var += 1

        if terminated or truncated:
            self.episode_rewards.append((self.current_step_var, self.current_reward))
            self.episode_lens.append((self.current_step_var, self.current_len))

        return obs, rew, terminated, truncated, info

    def reset(self, **kwargs):
        self.episode_counter += 1

        self.current_reward = 0
        self.current_len = 0

        return self.env.reset(**kwargs)

# if torch.backends.mps.is_available():
#     device = torch.device("mps")

# elif torch.cuda.is_available():
#     # Move the entire model to the GPU
#     device = 'cuda'
# else:
#     device = 'cpu'

device ='cpu'
print('device', device)

env = Normalize(Summaries(gym.make("HalfCheetah-v4", render_mode="rgb_array")))
env.reset(seed=0)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PolicyModel(nn. Module):


    def __init__(self):
        super().__init__()
        self.h = 64
        state_shape = env.observation_space.shape[0]
        action_shape = env.action_space.shape[0]

        self.linear1 = nn.Linear(state_shape, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 2*action_shape)

        self.linear4 = nn.Linear(state_shape, 64)
        self.linear5 = nn.Linear(64, 64)
        self.linear6 = nn.Linear(64, 1)


        self.tanh1 = nn.Tanh()
        self.tanh2 = nn.Tanh()
        self.tanh3 = nn.Tanh()
        self.tanh4 = nn.Tanh()

        self.policy_model = nn.Sequential(
            layer_init(self.linear1),
            self.tanh1,
            layer_init(self.linear2),
            self.tanh2,
            layer_init(self.linear3, std=0.01)
        )

        self.value_model = nn.Sequential(
            layer_init(self.linear4),
            self.tanh3,
            layer_init(self.linear5),
            self.tanh4,
            layer_init(self.linear6, std=1.0)
        )

    def get_policy(self, x):
        x= torch.tensor(x, dtype=torch.float32, device =device)
        self.policy_model.to(device)
        means = self.policy_model(x)[:, :6]
        log_std = self.policy_model(x)[:,6:]
        m = nn.Sigmoid()
        means = 2*m(means)-1
        action_std = m(log_std)
        return means, action_std

    def get_value(self, x):
        x= torch.tensor(x, dtype=torch.float32, device =device)
        self.value_model.to(device)
        out = self.value_model(x.float())
        return out

    def forward(self, x):
        x= torch.tensor(x, dtype=torch.float32, device =device)
        self.policy_model.to(device)
        self.value_model.to(device)
        policy = self.get_policy(x)
        value = self.get_value(x)

        return policy, value


class Policy:
    def __init__(self, model):
        self.model = model

    def act(self, inputs, training=False):
        inputs = torch.tensor(inputs)
        if inputs.ndim < 2:
            inputs = inputs.unsqueeze(0)
        # inputs = inputs.cuda()
        inputs = inputs.to(device, dtype=torch.float32)

        batch_size = inputs.shape[0]

        # < insert your code here >
        means, std = self.model.get_policy(inputs)
        probs = Normal(means, std)
        actions = probs.sample()
        log_probs = probs.log_prob(actions).sum(-1)

        values = self.model.get_value(inputs)

        if not training:
            return {'actions': actions.cpu().numpy().tolist()[0],
                    'log_probs': log_probs[0].detach().cpu().numpy(),
                    'values': values[0].detach().cpu().numpy()}
        else:
            return {'distribution': probs, 'values': values}


class AsArray:
    """
    Converts lists of interactions to ndarray.
    """
    def __call__(self, trajectory):
        # Modify trajectory inplace.
        for k, v in filter(lambda kv: kv[0] != "state", trajectory.items()):
            trajectory[k] = np.asarray(v)



class EnvRunner:
    """ Reinforcement learning runner in an environment with given policy """

    def __init__(self, env, policy, nsteps, transforms=None, step_var=None):
        self.env = env
        self.policy = policy
        self.nsteps = nsteps
        self.transforms = transforms or []
        self.step_var = step_var if step_var is not None else 0
        self.state = {"latest_observation": self.env.reset()[0]}

    @property
    def nenvs(self):
        """ Returns number of batched envs or `None` if env is not batched """
        return getattr(self.env.unwrapped, "nenvs", None)

    def reset(self, **kwargs):
        """ Resets env and runner states. """
        self.state["latest_observation"], info = self.env.reset(**kwargs)
        self.policy.reset()

    def get_next(self):
        """ Runs the agent in the environment.  """
        trajectory = defaultdict(list, {"actions": []})
        observations = []
        rewards = []
        resets = []
        self.state["env_steps"] = self.nsteps

        for i in range(self.nsteps):
            observations.append(self.state["latest_observation"])
            act = self.policy.act(self.state["latest_observation"])
            if "actions" not in act:
                raise ValueError("result of policy.act must contain 'actions' "
                                 f"but has keys {list(act.keys())}")
            for key, val in act.items():
                trajectory[key].append(val)

            obs, rew, terminated, truncated, info = self.env.step(trajectory["actions"][-1])
            # rew = info['reward_run']
            done = np.logical_or(terminated, truncated)
            self.state["latest_observation"] = obs
            rewards.append(rew)
            resets.append(done)
            self.step_var += self.nenvs or 1

            # Only reset if the env is not batched. Batched envs should
            # auto-reset.
            if not self.nenvs and np.all(done):
                self.state["env_steps"] = i + 1
                self.state["latest_observation"] = self.env.reset()[0]

        trajectory.update(
            observations=observations,
            rewards=rewards,
            resets=resets)
        trajectory["state"] = self.state

        for transform in self.transforms:
            transform(trajectory)
        return trajectory

class DummyPolicy:
    def act(self, inputs, training=False):
        assert not training
        return {"actions": np.random.randn(6), "values": np.nan}

runner = EnvRunner(env, DummyPolicy(), 3,
                   transforms=[AsArray()])
trajectory = runner.get_next()

{k: v.shape for k, v in trajectory.items() if k != "state"}

class GAE:
    """ Generalized Advantage Estimator. """
    def __init__(self, policy, gamma=0.99, lambda_=0.95):
        self.policy = policy
        self.gamma = gamma
        self.lambda_ = lambda_

    def __call__(self, trajectory):
        gamma = self.gamma
        lambda_ = self.lambda_

        # < insert your code here >
        s_T = trajectory["state"]["latest_observation"]
        T = len(trajectory['values'])
        v_T = self.policy.act([s_T])['values']


        values = trajectory['values']
        values =np.squeeze(values)


        values_ = trajectory['values']
        values_ =np.squeeze(values_)
        values_ = np.append(values_, v_T)
        values_ = values_[1:]

        deltas = self.gamma*values_*(1-trajectory['resets']) + trajectory['rewards'] - values


        A = [deltas[-1]]
        for i in range(1, len(values)):
            if trajectory['resets'][T-i-1]:  # Notice the indexing change here
                A.append(deltas[T-i-1])  # Use the direct index here
            else:
                A.append(A[-1]*self.gamma*self.lambda_ + deltas[T-i-1])
        A.reverse()

        A = np.array(A)
        v_ = A+ values

        trajectory['advantages'] = A
        trajectory['value_targets'] = v_
        return trajectory

def test_gae():

    trajectory = {}
    for key in ['actions', 'log_probs', 'values', 'observations', 'rewards', 'resets']:
        trajectory[key] = np.load(f'{key}.npy', allow_pickle=True)
    trajectory['state'] = {"latest_observation": np.load('state.npy')}

    policy = torch.load(f'policy')
    # print(policy.model)
    gae_to_test = GAE(policy, gamma=0.99, lambda_=0.95)

    gae_to_test(trajectory)

    for key in ['advantages', 'value_targets']:
        diff = np.squeeze(np.load(f'{key}.npy'))- trajectory[key]
        print(np.sum(diff>0.01))
        # print(diff>0.01)
        # print(diff[999])
        # print(diff[1999])
        # print(np.load(f'{key}.npy')[999])
        # print(np.load(f'{key}.npy')[1999])
        # indices = np.where(diff>0.01)[0]
        # print(indices)
        assert np.allclose(np.squeeze(np.load(f'{key}.npy')), trajectory[key], atol=2e-2)

    print("It's all good!")

test_gae()

class TrajectorySampler:
    """ Samples minibatches from trajectory for a number of epochs. """
    def __init__(self, runner, num_epochs, num_minibatches, transforms=None):
        self.runner = runner
        self.num_epochs = num_epochs
        self.num_minibatches = num_minibatches
        self.transforms = transforms or []
        self.minibatch_count = 0
        self.epoch_count = 0
        self.trajectory = None

    def shuffle_trajectory(self):
        """ Shuffles all elements in trajectory.

        Should be called at the beginning of each epoch.
        """
        trajectory_len = self.trajectory["observations"].shape[0]

        permutation = np.random.permutation(trajectory_len)
        for key, value in self.trajectory.items():
            if key != 'state':
                self.trajectory[key] = value[permutation]

    def get_next(self):
        """ Returns next minibatch.  """
        if not self.trajectory:
            self.trajectory = self.runner.get_next()

        if self.minibatch_count == self.num_minibatches:
            self.shuffle_trajectory()
            self.minibatch_count = 0
            self.epoch_count += 1

        if self.epoch_count == self.num_epochs:
            self.trajectory = self.runner.get_next()

            self.shuffle_trajectory()
            self.minibatch_count = 0
            self.epoch_count = 0

        trajectory_len = self.trajectory["observations"].shape[0]

        batch_size = trajectory_len//self.num_minibatches

        minibatch = {}
        for key, value in self.trajectory.items():
            if key != 'state':
                minibatch[key] = value[self.minibatch_count*batch_size: (self.minibatch_count + 1)*batch_size]

        self.minibatch_count += 1

        for transform in self.transforms:
            transform(minibatch)

        return minibatch

class NormalizeAdvantages:
    """ Normalizes advantages to have zero mean and variance 1. """
    def __call__(self, trajectory):
        # < insert your code here >
        adv = trajectory['advantages']
        mean = np.mean(adv)
        std_dev = np.std(adv)

        # Normalize the vector
        norm_adv = (adv - mean) / (std_dev+0.0001)
        trajectory['advantages'] = norm_adv
        return trajectory

def make_ppo_runner(env, policy, num_runner_steps=2048,
                    gamma=0.99, lambda_=0.95,
                    num_epochs=10, num_minibatches=32):
    """ Creates runner for PPO algorithm. """
    runner_transforms = [AsArray(),
                       GAE(policy, gamma=gamma, lambda_=lambda_)]
    runner = EnvRunner(env, policy, num_runner_steps,
                     transforms=runner_transforms)

    sampler_transforms = [NormalizeAdvantages()]
    sampler = TrajectorySampler(runner, num_epochs=num_epochs,
                              num_minibatches=num_minibatches,
                              transforms=sampler_transforms)
    return sampler

class PPO:
    def __init__(self, policy, opt,
               cliprange=0.2,
               value_loss_coef=0.5,
               max_grad_norm=0.5, entropy_coef= 0.01):
        self.policy = policy
        # self.optimizer_p = optimizer_p
        # self.optimizer_v = optimizer_v
        self.opt = opt
        self.cliprange = cliprange
        self.value_loss_coef = value_loss_coef
        # Note that we don't need entropy regularization for this env.
        self.max_grad_norm = max_grad_norm
        self.entropy_coef = entropy_coef

    def policy_loss(self, trajectory, act):
        # print(act)
        """ Computes and returns policy loss on a given trajectory. """
        # < insert your code here >


        actions = torch.tensor(trajectory['actions'], dtype=torch.float32, device=device)
        distr = act['distribution']
        # entropy = distr.entropy()
        new_log_probs = distr.log_prob(actions).sum(-1)

        old_log_probs = torch.tensor(trajectory['log_probs'], dtype=torch.float32, device=device)
        advantages = torch.tensor(trajectory['advantages'], dtype=torch.float32, device=device)

        ratio = torch.exp(new_log_probs - old_log_probs)
        J = ratio*advantages
        J_cl = torch.clamp(ratio, 1-self.cliprange, 1+self.cliprange)* advantages
        loss = -torch.mean(torch.min(J, J_cl))
        return loss


    def value_loss(self, trajectory, act):
        """ Computes and returns value loss on a given trajectory. """
        # < insert your code here >
        rewards = trajectory['rewards']
        resets = trajectory['resets']
        states = trajectory['observations']
        advantages = torch.tensor(trajectory['advantages'], dtype=torch.float32, device=device)
        value_targets = torch.tensor(trajectory['value_targets'], dtype=torch.float32, device=device)

        old_values = torch.tensor(trajectory['values'], dtype=torch.float32, device=device)
        n = len(states)


        new_values = act['values']

        l_simple = (new_values- value_targets)**2
        l_clipped = (old_values + torch.clamp(new_values - old_values, -self.cliprange, self.cliprange)- value_targets)**2
        loss = torch.mean(torch.max(l_simple, l_clipped))
        return loss

    def loss(self, trajectory):
        act = self.policy.act(trajectory["observations"], training=True)
        policy_loss = self.policy_loss(trajectory, act)
        value_loss = self.value_loss(trajectory, act)
        # return policy_loss + self.value_loss_coef * value_loss
        return policy_loss, value_loss

    def step(self, trajectory):
        """ Computes the loss function and performs a single gradient step. """
        self.opt.zero_grad()
        policy_loss, value_loss= self.loss(trajectory)
        loss = policy_loss + self.value_loss_coef*value_loss
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.model.parameters(), self.max_grad_norm)
        self.opt.step()

        return policy_loss.item(), value_loss.item()


parser = argparse.ArgumentParser(description="Script for running experiments.")
parser.add_argument("experiment_name", type=str, help="Name of the experiment")
args = parser.parse_args()

plot_dir = os.path.join(".", args.experiment_name)
os.makedirs(plot_dir, exist_ok=True)  # Create the directory if it doesn't exist

model = PolicyModel()
model = model.to(device, dtype=torch.float32)

policy = Policy(model)

runner = make_ppo_runner(env, policy)

opt = torch.optim.Adam(policy.model.parameters(), lr = 3e-5, eps=1e-5)
epochs = 500000

lr_mult = lambda epoch: (1 - (epoch/epochs))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_mult)
ppo = PPO(policy, opt)

policy_losses = []
value_losses = []
rew_lens = []
for epoch in tqdm(range(epochs)):
    if(epoch%10000==0):
      torch.save(policy.model.state_dict(), f'{os.path.join(plot_dir, f"model_{epoch}")}')

    trajectory = runner.get_next()

    policy_loss, value_loss= ppo.step(trajectory)  # Get the losses
    sched.step()

    if (epoch + 1) % 100 == 0:
        # clear_output(True)
        rewards = np.array(env.env.episode_rewards)
        rew_lens.append(len(rewards))
        policy_losses.append(policy_loss)
        value_losses.append(value_loss)

        if rewards.size > 0:
            plt.figure(figsize=(13, 8))

            # Plot reward
            plt.subplot(2, 2, 1)
            plt.plot(rewards[:, 0], rewards[:, 1], label="episode rewards")
            plt.title("Reward")
            plt.xlabel("Total steps")
            plt.ylabel("Reward")
            plt.grid()
            plt.savefig(os.path.join(plot_dir, "reward_plot.png"))  # Save the reward plot in the experiment folder
            plt.close()  # Close the current figure to avoid displaying it

            # Plot policy loss
            plt.figure(figsize=(13, 8))
            plt.subplot(2, 2, 2)
            plt.plot(policy_losses)
            plt.title("Policy Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.grid()
            plt.savefig(os.path.join(plot_dir, "policy_loss_plot.png"))  # Save the policy loss plot
            plt.close()

            # Plot value loss
            plt.figure(figsize=(13, 8))
            plt.subplot(2, 2, 3)
            plt.plot(value_losses)
            plt.title("Value Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.grid()
            plt.savefig(os.path.join(plot_dir, "value_loss_plot.png"))  # Save the value loss plot
            plt.close()



## record



env2 = gym.wrappers.RecordVideo(gym.make("HalfCheetah-v4", render_mode="rgb_array"), video_folder=os.path.join(plot_dir, "videos"), episode_trigger=lambda episode_number: True)
for i in range(10):
  env2.episode_id += 1
  obs = env2.reset()[0]
  env2.start_video_recorder()
  total_reward = 0
  steps =0
  for i in range(500):
    action = policy.act([obs])
    obs, rew, terminated, truncated, info= env2.step(action['actions'])
    if(truncated or terminated):
      print("termmm")
      break
    total_reward += rew
    steps += 1
  print('total reward', total_reward)
  print('steps', steps)
  env2.close_video_recorder()


