import gym
from typing import Callable, Tuple
from collections import defaultdict
from tqdm import trange
import numpy as np
from discretize import discretize


def generate_episode(env: gym.Env, policy: Callable, es: bool = False):
    """A function to generate one episode and collect the sequence of (s, a, r) tuples

    This function will be useful for implementing the MC methods

    Args:
        env (gym.Env): a Gym API compatible environment
        policy (Callable): A function that represents the policy.
        es (bool): Whether to use exploring starts or not
    """
    episode = []
    state = discretize(env.reset())[0]
    while True:
        if es and len(episode) == 0:
            action = env.action_space.sample()
        else:
            action = policy(state)

        next_state, reward, done, _, _ = discretize(env.step(action))
        episode.append((state, action, reward))
        if done:
            break
        state = next_state

    return episode

def generate_episode_custom_reward(env: gym.Env, policy: Callable, es: bool = False):
    """A function to generate one episode and collect the sequence of (s, a, r) tuples

    This function will be useful for implementing the MC methods

    Modifies the reward function so that there is 0 reward at all steps but large negative reward at the end

    Args:
        env (gym.Env): a Gym API compatible environment
        policy (Callable): A function that represents the policy.
        es (bool): Whether to use exploring starts or not
    """
    episode = []
    state = discretize(env.reset())[0]
    while True:
        if es and len(episode) == 0:
            action = env.action_space.sample()
        else:
            action = policy(state)

        next_state, reward, done, _, _ = discretize(env.step(action))

        if done:
            reward = -1000
        else: 
            reward = 0

        episode.append((state, action, reward))
        if done:
            break
        state = next_state

    return episode

def on_policy_mc_control_optimistic_initialization(
    env: gym.Env, num_episodes: int, gamma: float, initial_Q: defaultdict = None, episode_generator_func: Callable = generate_episode
) -> Tuple[defaultdict, Callable]:
    """
    On-policy Monte Carlo control with optimistic initialization for cart-pole

    Args:
        env (gym.Env): a Gym API compatible environment
        num_episodes (int): Number of episodes
        gamma (float): Discount factor of MDP
    """
    # We use defaultdicts here for both Q and N for convenience. The states will be the keys and the values will be numpy arrays with length = num actions

    if initial_Q:
        Q = initial_Q
    else:
        # initialize optimistically
        Q = defaultdict(lambda: 15 * np.ones(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))

    returns = np.zeros(num_episodes)
    episode_lengths = np.zeros(num_episodes)

    # If the state was seen, use the greedy action using Q values.
    # Else, default to the original policy of sticking to 20 or 21.
    policy = create_policy(Q)

    for ep in trange(num_episodes, desc="Episode"):
        episode = episode_generator_func(env, policy)
        episode_lengths[ep] = len(episode)
        G = 0
        for t in range(len(episode) - 1, -1, -1):
            # Update V and N here according to first visit MC
            current_state, current_action, current_reward = episode[t]

            G = gamma * G + current_reward

            if t == 0:
                returns[ep] = G

            first_visit = True
            for i in range(t):
                # if state at time t isn't the first visit
                if episode[i][0] == current_state and episode[i][1] == current_action:
                    # skip for now
                    first_visit = False
                    break

            # if state at time t is the first visit
            if first_visit:
                N[current_state][current_action] += 1
                Q[current_state][current_action] = Q[current_state][current_action] + \
                    (1/N[current_state][current_action]) * \
                    (G - Q[current_state][current_action])


    return Q, policy, returns, episode_lengths

def on_policy_mc_control_es(
    env: gym.Env, num_episodes: int, gamma: float, initial_Q: defaultdict = None
) -> Tuple[defaultdict, Callable]:
    """
    On-policy Monte Carlo control with optimistic initialization for cart-pole

    Args:
        env (gym.Env): a Gym API compatible environment
        num_episodes (int): Number of episodes
        gamma (float): Discount factor of MDP
    """
    # We use defaultdicts here for both Q and N for convenience. The states will be the keys and the values will be numpy arrays with length = num actions

    if initial_Q:
        Q = initial_Q
    else:
        # initialize optimistically
        Q = defaultdict(lambda: 15 * np.ones(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))

    returns = np.zeros(num_episodes)

    # If the state was seen, use the greedy action using Q values.
    # Else, default to the original policy of sticking to 20 or 21.
    policy = create_policy(Q)

    for ep in trange(num_episodes, desc="Episode"):
        episode = generate_episode(env, policy, es=True)

        G = 0
        for t in range(len(episode) - 1, -1, -1):
            # Update V and N here according to first visit MC
            current_state, current_action, current_reward = episode[t]

            G = gamma * G + current_reward

            if t == 0:
                returns[ep] = G

            first_visit = True
            for i in range(t):
                # if state at time t isn't the first visit
                if episode[i][0] == current_state and episode[i][1] == current_action:
                    # skip for now
                    first_visit = False
                    break

            # if state at time t is the first visit
            if first_visit:
                N[current_state][current_action] += 1
                Q[current_state][current_action] = Q[current_state][current_action] + \
                    (1/N[current_state][current_action]) * \
                    (G - Q[current_state][current_action])


    return Q, policy, returns

def create_policy(Q: defaultdict) -> Callable:
    """A function that takes in the state and returns the action according to the policy
    Args:
        Q (defaultdict): A dictionary that maps states to action values

    Returns:
        Callable: A function that takes in the state and returns the action according to the policy
    """
    def policy(state):
        if state not in Q.keys():
            return np.random.choice([0, 1])
        else:
            return np.argmax(Q[state]).item()
    return policy


def run_iterations(env, Q, n=10):
    for i in range(n):
        generate_episode(env, create_policy(Q))