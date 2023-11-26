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


def _calculate_reward_from_state(state):
    '''
    reward is 1 when the pole is vertical and the cart is centered

    reward is 0.5 when the pole is vertical and the cart is not centered or the pole is not vertical and the cart is centered
    
    reward is 0 when the pole is not vertical and the cart is not centered
    '''
    cart_pos, cart_vel, pole_angle, pole_vel = state

    cart_pos_contribution = np.abs(cart_pos) / (2.4)
    pole_angle_contribution = np.abs(pole_angle) / (0.418)

    reward = 1 - (cart_pos_contribution * 0.2 + pole_angle_contribution * 0.8)

    return reward

def generate_episode_reward_central_vertical(env: gym.Env, policy: Callable, es: bool = False):
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

        reward = _calculate_reward_from_state(next_state)

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
    episode_lengths = np.zeros(num_episodes)

    # If the state was seen, use the greedy action using Q values.
    # Else, default to the original policy of sticking to 20 or 21.
    policy = create_policy(Q)

    for ep in trange(num_episodes, desc="Episode"):
        episode = generate_episode(env, policy, es=True)
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


def on_policy_mc_control_epsilon_soft(
    env: gym.Env, num_episodes: int, gamma: float, epsilon: float, episode_generator_func: Callable = generate_episode
):
    """On-policy Monte Carlo policy control for epsilon soft policies.

    Args:
        env (gym.Env): a Gym API compatible environment
        num_episodes (int): Number of episodes
        gamma (float): Discount factor of MDP
        epsilon (float): Parameter for epsilon soft policy (0 <= epsilon <= 1)
    Returns:

    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))

    policy = create_epsilon_policy(Q, epsilon)

    returns = np.zeros(num_episodes)

    episode_lengths = np.zeros(num_episodes)

    for ep in trange(num_episodes, desc="Episode", leave=False):
        # For each episode calculate the return
        # Update Q
        episode = episode_generator_func(env, policy)

        episode_lengths[ep] = len(episode)
        G = 0
        for t in range(len(episode) - 1, -1, -1):

            # Update V and N here according to first visit MC
            current_state, current_action, current_reward = episode[t]

            G = gamma * G + current_reward
            if t == 0:
                returns[ep] = G
                # print(G)
            

            first_visit = True
            for i in range(t):
                # if state at time t isn't the first visit
                if episode[i][0] == current_state and episode[i][1] == current_action:
                    # skip for now
                    first_visit = False
                    break

            # if state at time t is the first visit
            # Note there is no need to update the policy here directly.
            # By updating Q, the policy will automatically be updated.
            if first_visit:
                N[current_state][current_action] += 1
                Q[current_state][current_action] = Q[current_state][current_action] + (1/N[current_state][current_action]) * (G - Q[current_state][current_action])
            
    return Q, policy, returns, episode_lengths


def on_policy_mc_control_epsilon_soft_every_visit(
    env: gym.Env, num_episodes: int, gamma: float, epsilon: float, episode_generator_func: Callable = generate_episode
):
    """On-policy Monte Carlo policy control for epsilon soft policies.

    Args:
        env (gym.Env): a Gym API compatible environment
        num_episodes (int): Number of episodes
        gamma (float): Discount factor of MDP
        epsilon (float): Parameter for epsilon soft policy (0 <= epsilon <= 1)
    Returns:

    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))

    policy = create_epsilon_policy(Q, epsilon)

    returns = np.zeros(num_episodes)

    episode_lengths = np.zeros(num_episodes)

    for ep in trange(num_episodes, desc="Episode", leave=False):
        # For each episode calculate the return
        # Update Q
        episode = episode_generator_func(env, policy)

        episode_lengths[ep] = len(episode)
        G = 0
        for t in range(len(episode) - 1, -1, -1):

            # Update V and N here according to first visit MC
            current_state, current_action, current_reward = episode[t]

            G = gamma * G + current_reward
            if t == 0:
                returns[ep] = G

            # Note there is no need to update the policy here directly.
            # By updating Q, the policy will automatically be updated.
            N[current_state][current_action] += 1
            Q[current_state][current_action] = Q[current_state][current_action] + (1/N[current_state][current_action]) * (G - Q[current_state][current_action])
            
    return Q, policy, returns, episode_lengths


def off_policy_mc_control(
    env: gym.Env, num_episodes: int, gamma: float, epsilon: float, episode_generator_func: Callable = generate_episode
):
    """
    Off-policy Monte Carlo policy control for epsilon soft policies.

    Args:
        env (gym.Env): a Gym API compatible environment
        num_episodes (int): Number of episodes
        gamma (float): Discount factor of MDP
        epsilon (float): Parameter for epsilon soft policy (0 <= epsilon <= 1)
    Returns:

    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    C = defaultdict(lambda: np.zeros(env.action_space.n))

    behavior_policy = create_epsilon_policy(Q, epsilon)
    optimal_policy = create_policy(Q)

    returns = np.zeros(num_episodes)

    episode_lengths = np.zeros(num_episodes)

    for ep in trange(num_episodes, desc="Episode", leave=False):
        # For each episode calculate the return
        # Update Q
        episode = episode_generator_func(env, behavior_policy)

        episode_lengths[ep] = len(episode)
        G = 0
        W = 1

        for t in range(len(episode) - 1, -1, -1):

            # Update V and N here according to first visit MC
            current_state, current_action, current_reward = episode[t]

            G = gamma * G + current_reward
            if t == 0:
                returns[ep] = G

            # Note there is no need to update the policy here directly.
            # By updating Q, the policy will automatically be updated.
            C[current_state][current_action] = C[current_state][current_action] + W

            Q[current_state][current_action] = Q[current_state][current_action] + (W/C[current_state][current_action]) * (G - Q[current_state][current_action])

            if current_action != optimal_policy(current_state):
                break
            
            # 1 / (epsilon/2) is 1/(b(a|s)) because if we aren't taking the argmax
            # then the behavior policy chose this aciton with a 1/2 epsilon chance
            W = W * (1/(epsilon / 2))

            
    return Q, optimal_policy, returns, episode_lengths


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


def create_epsilon_policy(Q: defaultdict, epsilon: float) -> Callable:
    """Creates an epsilon soft policy from Q values.

    A policy is represented as a function here because the policies are simple. More complex policies can be represented using classes.

    Args:
        Q (defaultdict): current Q-values
        epsilon (float): softness parameter
    Returns:
        get_action (Callable): Takes a state as input and outputs an action
    """
    # Get number of actions
    num_actions = 2

    def get_action(state: Tuple) -> int:
        # You can reuse code from ex1
        # Make sure to break ties arbitrarily

        if np.random.random() < epsilon:
            # take random action
            action = np.random.randint(0, num_actions)
        else:
            # get values for each action
            action_values = Q[state]

            best_actions = []
            best_action_value = -np.inf

            for action in range(num_actions):
                if action_values[action] > best_action_value:
                    best_action_value = action_values[action]
                    best_actions = [action]
                elif action_values[action] == best_action_value:
                    best_actions.append(action)


            if len(best_actions) == 0:
                action = np.argmax(Q[state]).item()
            else:
                # choose random action from best actions
                action = np.random.choice(best_actions)

        return action

    return get_action

def run_iterations(env, Q, n=10):
    for i in range(n):
        generate_episode(env, create_policy(Q))








'''
def generate_episode_large_negative_reward(env: gym.Env, policy: Callable, es: bool = False):
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
'''