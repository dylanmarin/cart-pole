import gym
from collections import defaultdict
import numpy as np
import math

class CartPoleAgent():
    def __init__(self, num_episodes=1000, lr=0.5, epsilon=0.05, discount=0.95):
        self.num_episodes = num_episodes
        self.lr=lr
        self.epsilon = epsilon
        self.discount = discount

        self.sarsa_table = defaultdict(lambda: np.zeros(2))

    def discretize_state(self, obs):
        if len(obs) == 2:
            obs = obs[0]

        cart_pos, cart_vel, pole_angle, pole_vel = obs.tolist()

        new_obs = (
            _discretize_cart_position(cart_pos),
            _discretize_cart_velocity(cart_vel),
            _discretize_pole_angle(pole_angle),
            _discretize_angular_velocity(pole_vel)
        )

        return new_obs

    def choose_action(self, state):
        if (np.random.random() < self.epsilon):
            return self.env.action_space.sample() 
        else:
            return np.argmax(self.sarsa_table[state])

    def update_sarsa(self, state, action, reward, new_state, new_action):
        self.sarsa_table[state][action] += self.learning_rate * (reward + self.discount * (self.sarsa_table[new_state][new_action]) - self.sarsa_table[state][action])

    def get_epsilon(self, t):
        return self.epsilon

    def get_learning_rate(self, t):
        return self.lr

    def train(self, env):
        self.env = env
        ep_lengths = []
        for e in range(self.num_episodes):
            ep_length = 0
            current_state = self.discretize_state(self.env.reset())

            self.learning_rate = self.get_learning_rate(e)
            self.epsilon = self.get_epsilon(e)
            done = False

            while not done:
                action = self.choose_action(current_state)
                obs, reward, done, _, _= self.env.step(action)
                new_state = self.discretize_state(obs)
                new_action = self.choose_action(new_state)
                self.update_sarsa(current_state, action, reward, new_state, new_action)
                current_state = new_state
                ep_length += 1

                if done:
                    ep_lengths.append(ep_length)
                
        return ep_lengths
            

def _discretize_cart_velocity(velocity):
    return np.round(velocity, 1)

def _discretize_angular_velocity(velocity):
    return np.round(velocity, 1)

def _discretize_cart_position(pos):
    return np.round(pos, 1)

def _discretize_pole_angle(angle):
    return np.round(angle, 1)

if __name__ == "__main__":
    agent = CartPoleAgent()
    agent.train()
    t = agent.run()
    print("Time", t)