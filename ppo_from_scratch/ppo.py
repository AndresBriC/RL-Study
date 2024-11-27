import torch
import torch.nn as nn

import gymnasium as gym


import numpy as np

from torch.distributions import MultivariateNormal
from torch.optim import Adam
from network import FeedForwardNN

class PPO:
    def __init__(self, env):
        # Extract environment information
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        
        # ALG STEP 1
        # Initialize actor and critic networks
        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)

        self._init_hyperparameters()

        # Create our variable for the matrix
        # 0.5 was chosen abritrarily
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)


        # Create the covariance matrix
        self.cov_mat = torch.diag(self.cov_var)

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
        

    def _init_hyperparameters(self):
        # Defaults values for hyperparameters, will need to change later
        self.timesteps_per_batch = 4800         # timesteps per batch
        self.max_timesteps_per_episode = 1600   # timesteps per episode
        self.gamma = 1                          # discount factor
        self.n_updates_per_iteration = 5        
        self.clip = 0.2                         # As recommended by the paper
        self.lr = 0.005                         # learning rate

    def get_action(self, obs):
        # Query the actor network for a mean action
        # Same thing as calling self.actor.forward(obs)
        mean = self.actor(obs)

        # Create our Multivariate Normal Distribution
        dist = MultivariateNormal(mean, self.cov_mat)

        # Sample an action from the distribution and get its log prob
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Return the sampled action and the log prob of that action
        # Note that I'm calling detach() since the action and log_prob  
        # are tensors with computation graphs, so I want to get rid
        # of the graph and just convert the action to numpy array.
        # log prob as tensor is fine. Our computation graph will
        # start later down the line.
        return  action.detach().numpy(), log_prob.detach()
    
    def rollout(self):
        # Batch data
        batch_obs = []          # batch observations
        batch_acts = []         # batch actions
        batch_log_probs = []    # log probs of each action
        batch_rews = []         # batch rewards
        batch_rtgs = []         # batch rewards-to-go
        batch_lens = []         # episodic lengths in batch

        # Number of timesteps run so far this batch
        t = 0

        while t < self.timesteps_per_batch:

            # Rewards this episode
            ep_rews = []

            obs = self.env.reset()
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                # Increment timesteps ran this batch so far
                t += 1
                
                # Collect observation
                batch_obs.append(obs)

                # Print the type and shape of obs
                print(f"obs type: {type(obs)}")
                if isinstance(obs, np.ndarray):
                    print(f"obs shape: {obs.shape}")
                elif isinstance(obs, list):
                    print(f"obs length: {len(obs)}")
                    if len(obs) > 0 and isinstance(obs[0], np.ndarray):
                        print(f"obs[0] shape: {obs[0].shape}")
                elif isinstance(obs, tuple):
                    print(f"obs length: {len(obs)}")
                    for i, o in enumerate(obs):
                        if isinstance(o, np.ndarray):
                            print(f"obs[{i}] shape: {o.shape}")
                        else:
                            print(f"obs[{i}] type: {type(o)}")
                


                # Handle the case where obs is a tuple with a dictionary
                if isinstance(obs, tuple) and isinstance(obs[1], dict):
                    obs_list = [obs[0]]  # Start with the first element (numpy array)
                    for key, value in obs[1].items():
                        if isinstance(value, np.ndarray):
                            obs_list.append(value)
                        else:
                            obs_list.append(np.array(value))
                    obs_array = np.concatenate(obs_list)
                else:
                    obs_array = np.array(obs)

                obs_tensor = torch.tensor(obs_array, dtype=torch.float).unsqueeze(0)  # Add batch dimension

                action, log_prob = self.get_action(obs_tensor)
                obs, rew, done, _, _ = self.env.step(action)

                # Collect reward, actino and log prob
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break
            
            # Collect episodic length and rewards
            batch_lens.append(ep_t + 1) # plus 1 because timesteps starts at 0
            batch_rews.append(ep_rews)

        # Reshape data as tensors in the shape specified before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)

        # ALG STEP #4
        batch_rtgs - self.compute_rtgs(batch_rews)

        # Return the batch data
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def compute_rtgs(self, batch_rews):
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num steps per episode)
        batch_rtgs = []

        # Iterate through each episode backwards to maintain some order
        # in batch_rtgs
        for ep_rews in reversed(batch_rews):

            discounted_reward = 0 # The discounted reward so far

            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
            
            batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

            return batch_rtgs

    def evaluate(self, batch_obs, batch_acts):
        # Query critic network for a value V for each obs in batch_obs
        v = self.critic(batch_obs).squeeze()

        # Calculate the log provabilites of batch actions using most
        # recent actor network.
        # This segment of code is similat to that in get_action()
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        return v, log_probs

    def learn(self, total_timesteps):
        t_so_far = 0 # Timesteps simulated so far

        while t_so_far < total_timesteps: # ALG STEP 2
            # ALG STEP 3
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)

            V, _ = self.evaluate(batch_obs, batch_acts)

            # ALG STEP 5
            # Calculate advantage
            A_k = batch_rtgs - V.detach()

            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration):
                # Calculate pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                # Calculate ratios
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Calculate surrogate losses
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                # Calculate gradients and perform backward propagation for actor
                # network
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                # Calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()



env = gym.make('Pendulum-v1')
model = PPO(env)
model.learn(10000)
