import gymnasium as gym
import numpy as np
import torch

from base_env import BaseEnv

class nonstat_bandit(BaseEnv):
    def __init__(self, H, dim, init_means = None, distance = 0.1, var = 0.01, k = 0.1, rwd_type = "normal") -> None:
        self.H = H
        self.rwd_type = rwd_type
        self.dim = dim
        self.k = k
        self.var = var
        self.distance = distance

        if init_means is None:
            self.means = np.random.uniform(0, 1, self.dim)
        self.means = init_means
        self.opt_a_index = np.argmax(self.means)
        self.opt_a = np.zeros(self.dim)
        self.opt_a[self.opt_a_index] = 1.0

        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,))
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.dim,))
        self.state = np.array([1])
        self.t = 0
        self.reset()
    
    def reset(self):
        self.t = 0
        self.means = np.random.uniform(0, 1, self.dim)
        self.opt_a_index = np.argmax(self.means)
        self.opt_a = np.zeros(self.dim)
        self.opt_a[self.opt_a_index] = 1.0
        return self.state
    
    def transit(self, x, u):
        if np.random.random() < self.k:
            self.means = np.random.normal(self.means, self.distance, self.dim)
            self.opt_a_index = np.argmax(self.means)
            self.opt_a = np.zeros(self.dim)
            self.opt_a[self.opt_a_index] = 1.0
        a = np.argmax(u)
        if self.rwd_type == "normal":
            r = self.means[a] + np.random.normal(0, self.var)
        elif self.rwd_type == "bernoulli":
            r = np.random.binomial(1, self.means[a])
        else:
            raise NotImplementedError
        self.t += 1
        done = self.t >= self.H
        return self.state, r, done, {}
    
    def step(self, action):
        return self.transit(self.state, action)
    
    def get_arm_value(self, u):
        return np.sum(self.means * u)
    
    def get_optimal_value(self):
        return np.max(self.means)
    
    def deploy_eval(self, ctrl):
    # No variance during evaluation
        tmp = self.var
        self.var = 0.0
        res = self.deploy(ctrl)
        self.var = tmp
        return res
    
class nonstat_bandit_vec(BaseEnv):
    def _init__(self, envs):
        self.envs = envs
        self.num_envs = len(envs)
        self.dim = envs[0].dim
        self.H = envs[0].H

    def reset(self):
        return np.array([env.reset() for env in self.envs])
    
    def step(self, actions):
        return np.array([env.step(a) for env, a in zip(self.envs, actions)])
    
    @property
    def num_envs(self):
        return self.num_envs

    @property
    def envs(self):
        return self.envs

    def deploy_eval(self, ctrl):
        # No variance during evaluation
        tmp = [env.var for env in self.envs]
        for env in self.envs:
            env.var = 0.0
        res = self.deploy(ctrl)
        for env, var in zip(self.envs, tmp):
            env.var = var
        return res

    def deploy(self, ctrl):
        x = self.reset()
        xs = []
        xps = []
        us = []
        rs = []
        done = False

        while not done:
            u = ctrl.act_numpy_vec(x)

            xs.append(x)
            us.append(u)

            x, r, done, _ = self.step(u)
            done = all(done)

            rs.append(r)
            xps.append(x)

        xs = np.concatenate(xs)
        us = np.concatenate(us)
        xps = np.concatenate(xps)
        rs = np.concatenate(rs)
        return xs, us, xps, rs

    def get_arm_value(self, us):
        values = [np.sum(env.means * u) for env, u in zip(self.envs, us)]
        return np.array(values)
    
class UCB:
    def __init__(self, num_arms, c):
        self.num_arms = num_arms
        self.c = c
        self.N = np.zeros(num_arms)
        self.Q = np.zeros(num_arms)
    
    def act(self):
        exploration = np.sqrt(np.log(np.sum(self.N + 1)) / (self.N + 1e-5))
        ucb_values = self.Q + self.c * exploration
        return np.argmax(ucb_values)
    
    def update(self, arm, reward):
        self.N[arm] += 1
        self.Q[arm] += (reward - self.Q[arm]) / self.N[arm]

def test_env(env, num_steps, c):
    agent = UCB(env.dim, c)
    rewards = []
    optimal_values = []
    for _ in range(num_steps):
        action = agent.act()
        optimal_values.append(env.get_optimal_value())
        _, reward, done, _ = env.step(action)
        agent.update(action, reward)
        rewards.append(reward)
        if done:
            break
    return rewards, optimal_values

env = nonstat_bandit(H=100, dim=10)
rewards, optimal_values = test_env(env, num_steps=1000, c=2)
import matplotlib.pyplot as plt

plt.plot(range(len(rewards)), rewards, label='Rewards')
plt.plot(range(len(optimal_values)), optimal_values, label='Optimal Values')

plt.xlabel('Timesteps')
plt.ylabel('Rewards / Optimal Values')
plt.title('Rewards and Optimal Values vs Timesteps')
plt.legend()
plt.show()
