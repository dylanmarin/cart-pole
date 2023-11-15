import gym
from typing import Optional, Callable, Tuple
from collections import defaultdict
import numpy as np
from tqdm import trange, tqdm
import time
from discretize import discretize

def sarsa(
    env: gym.Env,
    gamma: float,
    epsilon: float,
    step_size: float,
    num_steps: int = 1,
    num_eps: Optional[int] = None, 
    reward_func=None,
):
    """SARSA algorithm.

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = create_epsilon_policy(Q, epsilon)

    ep_num = 1
    episode_lengths = []
    step_count = 0


    if num_eps is not None:
        pbar = tqdm(total=num_eps)
    else:
        pbar = tqdm(total=num_steps)



    # loop until we have taken num_steps steps
    while True:
        if num_eps is None:
            if step_count < num_steps:
                break
        elif num_eps is not None:
            if ep_num > num_eps:
                break


        # start new episode
        state = tuple(discretize(env.reset())[0])

        action = policy(state)
        done = False

        ep_length = 0

        # for each step in episode
        while not done:
            next_state, reward, done, _, _ = discretize(env.step(action))

            if reward_func:
                reward = reward_func(next_state)

            next_state = tuple(next_state)
            next_action = policy(next_state)

            ep_length += 1

            if done:
                episode_lengths.append(ep_length)

                # if done then next_state is terminal so force Q[terminal][-] = 0
                Q[state][action] = Q[state][action] + step_size * (
                    reward + gamma * 0 - Q[state][action])
            else:
                Q[state][action] = Q[state][action] + step_size * (
                    reward + gamma * Q[next_state][next_action] - Q[state][action]
                )
            state = next_state
            action = next_action

            if num_eps is None:
                pbar.update(1)


            step_count += 1
            if step_count == num_steps:
                break

        ep_num += 1
        if num_eps is not None:
            pbar.update(1)

    pbar.close()
    return Q, policy, episode_lengths


def nstep_sarsa(
    env: gym.Env,
    gamma: float,
    epsilon: float,
    step_size: float,
    n: int,
    num_steps: int = 1,
    num_eps: Optional[int] = None, 
    reward_func=None,
):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = create_epsilon_policy(Q, epsilon)

    ep_num = 0
    ep_lengths = []
    step_count = 0

    if num_eps is not None:
        pbar = tqdm(total=num_eps)
    else:
        pbar = tqdm(total=num_steps)

    # loop until we have taken num_steps steps
    while True:
        if num_eps is None:
            if step_count < num_steps:
                break

        elif num_eps is not None:
            if ep_num > num_eps:
                break

        ep_num += 1
            
        # start new episode
        state =  tuple(discretize(env.reset())[0])

        action = policy(state)
        done = False

        experiences = []
        tao = -1

        # for each step in episode
        while True:
            if not done:
                # take a step
                next_state, reward, new_done, _, _ = discretize(env.step(action))

                if reward_func:
                    reward = reward_func(next_state)

                next_state = tuple(next_state)

                # store the experience
                experiences.append((state, action, reward, done))
                done = new_done

            # if we have at least n + 1 experiences
            if len(experiences) >= n + 1:
                # first value will be 0
                # tao is the index of the experience we want to update from
                tao += 1

                # always the last n experiences excluding the most recent experience
                relevant_experiences = experiences[tao:-1]

                # when there are less than n states left, make sure it is however many states are left
                if len(relevant_experiences) < n:
                    relevant_experiences = experiences[tao:]

                G = _calculate_return_from_episode(relevant_experiences, gamma)

                _, _, _, last_relevant_experience_done = relevant_experiences[-1]

                # if the last state isn't terminal
                if not last_relevant_experience_done:
                    # then get the next state after that (should always be the last item in experiences)
                    (
                        last_relevant_state,
                        last_relevant_action,
                        _,
                        last_state_terminal,
                    ) = experiences[-1]

                    # if it isn't terminal
                    if not last_state_terminal:
                        # add to the reward
                        G += gamma**n * Q[last_relevant_state][last_relevant_action]

                state_to_update, action_to_update, _, terminal = experiences[tao]
                if not terminal:
                    Q[state_to_update][action_to_update] = Q[state_to_update][
                        action_to_update
                    ] + step_size * (G - Q[state_to_update][action_to_update])

            # if the episode hasn't terminated, take another step
            if not done:
                next_action = policy(next_state)
                state = next_state
                action = next_action

                step_count += 1
                if num_eps is None:
                    pbar.update(1)

            else:
                # if we are done and if tao is at the second to last experience, break
                # we don't need to update the terminal state
                if tao == len(experiences) - 2:
                    ep_lengths.append(len(experiences))
                    
                    if num_eps is not None:
                    
                        pbar.update(1)

                    break

            if num_eps is None:
                if step_count == num_steps:
                    break

        if num_eps is None:
            if step_count == num_steps:
                break

    pbar.close()
    return Q, policy, ep_lengths


def _calculate_return_from_episode(steps, gamma):
    G = 0
    for i in range(len(steps)):
        _, _, reward, _ = steps[i]
        G += (gamma**i) * reward
    return G


def exp_sarsa(
    env: gym.Env,
    gamma: float,
    epsilon: float,
    step_size: float,
    num_steps: int = 1,
    num_eps: Optional[int] = None, 
    reward_func=None,
):
    """Expected SARSA

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """

    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = create_epsilon_policy(Q, epsilon)

    ep_num = 0
    # episode_nums = []
    ep_lengths = []
    step_count = 0

    if num_eps is not None:
        pbar = tqdm(total=num_eps)
    else:
        pbar = tqdm(total=num_steps)

    # loop until we have taken num_steps steps
    while True:
        if num_eps is None:
            if step_count < num_steps:
                break
        elif num_eps is not None:
            if ep_num > num_eps:
                break
        
        # start new episode
        state = tuple(discretize(env.reset())[0])

        action = policy(state)
        done = False
        ep_length =0

        ep_num += 1
        if num_eps is not None:
            pbar.update(1)

        # for each step in episode
        while not done:
            ep_length += 1
            next_state, reward, done, _, _ = discretize(env.step(action))
            next_state = tuple(next_state)

            if reward_func:
                reward = reward_func(next_state)

            next_action = policy(next_state)
            exp_q_value_of_next_state = 0

            if not done:
                # if not done then calculate expected q value of next state, otherwise it remains 0
                exp_q_value_of_next_state = exp_value_of_state(
                    Q, next_state, epsilon=epsilon
                )
            else:
                ep_lengths.append(ep_length)

            Q[state][action] = Q[state][action] + step_size * (
                reward + gamma * exp_q_value_of_next_state - Q[state][action]
            )

            state = next_state
            action = next_action

            # episode_nums.append(ep_num)
            step_count += 1
            if num_eps is None:
                pbar.update(1)


    return Q, policy, ep_lengths


def q_learning(
    env: gym.Env,
    gamma: float,
    epsilon: float,
    step_size: float,
    num_steps: int = 1,
    num_eps: Optional[int] = None, 
):
    """Q-learning

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """

    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = create_epsilon_policy(Q, epsilon)

    if num_eps is not None:
        pbar = tqdm(total=num_eps)
    else:
        pbar = tqdm(total=num_steps)

    ep_num = 0

    # episode_nums = []
    ep_lengths = []

    step_count = 0
    # loop until we have taken num_steps steps
    while True:
        if num_eps is None:
            if step_count < num_steps:
                break
        elif num_eps is not None:
            if ep_num > num_eps:
                break

        # start new episode
        state = tuple(discretize(env.reset())[0])
        action = policy(state)
        done = False
        ep_length = 0


        # for each step in episode
        while not done:
            ep_length += 1
            next_state, reward, done, _, _ = discretize(env.step(action))
            next_action = policy(next_state)

            if done:
                # if done then next_state is terminal so force Q[terminal][-] = 0
                Q[state][action] = Q[state][action] + step_size * (
                    reward + gamma * 0 - Q[state][action]
                )
                ep_lengths.append(ep_length)
            else:

                Q[state][action] = Q[state][action] + step_size * (
                    reward + gamma * max(Q[next_state]) - Q[state][action]
                )
            state = next_state
            action = next_action

            # episode_nums.append(ep_num)
            step_count += 1
            if num_eps is None:
                pbar.update(1)

            if step_count == num_steps:
                break

        if num_eps is not None:
            pbar.update(1)
        ep_num += 1

    print(step_count)
    pbar.close()
    return Q, policy, ep_lengths


def create_epsilon_policy(Q: defaultdict, epsilon: float):
    """Creates an epsilon soft policy from Q values.

    A policy is represented as a function here because the policies are simple. More complex policies can be represented using classes.

    Args:
        Q (defaultdict): current Q-values
        epsilon (float): softness parameter
    Returns:
        get_action (Callable): Takes a state as input and outputs an action
    """
    # Get number of actions
    num_actions = len(Q[0])

    def get_action(state: Tuple) -> int:
        # You can reuse code from ex1
        # Make sure to break ties arbitrarily
        state = tuple(state)

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

            # choose random action from best actions
            action = np.random.choice(best_actions)

        return action

    return get_action


def exp_value_of_state(Q: defaultdict, state, epsilon: float):
    """
    Given a Q table and a state, get the expected value of the state assuming an epsilon soft policy where ties are broken randomly
    """
    # Get number of actions
    num_actions = len(Q[0])

    # get values for each action
    action_values = Q[state]

    best_actions = []
    action_probs = np.zeros(num_actions)
    best_action_value = -np.inf

    for action in range(num_actions):
        if action_values[action] > best_action_value:
            best_action_value = action_values[action]
            best_actions = [action]
        elif action_values[action] == best_action_value:
            best_actions.append(action)

    # assign all probs equal epsilon probability
    action_probs[:] = epsilon / num_actions

    # assign best actions divided 1-epsilon prob

    for action in best_actions:
        action_probs[action] = (1 - epsilon) / len(best_actions)

    exp_value = 0

    # loop over each action
    for action in range(num_actions):
        exp_value += action_probs[action] * action_values[action]

    return exp_value


def generate_episode(
    env: gym.Env, policy: Callable, es: bool = False, current_count=0, max_count=None
):
    """A function to generate one episode and collect the sequence of (s, a, r) tuples

    This function will be useful for implementing the MC methods

    Args:
        env (gym.Env): a Gym API compatible environment
        policy (Callable): A function that represents the policy.
        es (bool): Whether to use exploring starts or not
    """

    step_count = current_count

    episode = []
    state = discretize(env.reset())[0]
    while True:
        if es and len(episode) == 0:
            action = env.action_space.sample()
        else:
            action = policy(state)

        next_state, reward, done, _ = discretize(env.step(action))
        episode.append((state, action, reward))
        if done:
            break
        state = next_state
        step_count += 1
        if max_count:
            if step_count >= max_count:
                break

    return episode
