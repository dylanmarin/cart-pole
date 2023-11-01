import numpy as np
"""
Cart Pole:
Observation:
0: Cart Position [-4.8, 4.8], with episode ending at [-2.4, 2.4]
1: Cart Velocity [-Inf, Inf]
2: Pole Angle [-0.418, 0.418] (in radians) with episode ending at [-0.2095, 0.2095]
3: Pole Angular Velocity [-Inf, Inf]



Env.Reset():
returns tuple:
0: observation
1: info

Env.Step():
returns tuple:
0: observation
1: reward
2: done
3: truncated
4: info
"""

def discretize(step):
    '''
    Discretizes the observation in the given step using the `_discretize_observation` function.

    Args:
        step (tuple): A tuple containing the observation and additional information, or the observation, reward, done flag, truncated flag, and additional information.

    Returns:
        tuple: A tuple containing the discretized observation and additional information, or the discretized observation, reward, done flag, truncated flag, and additional information.
    '''
    if len(step) == 2:
        observation, info = step   
        return _discretize_observation(observation), info
    else: 
        observation, reward, done, unsure, info = step
        return _discretize_observation(observation), reward, done, unsure, info

def _discretize_observation(observation):
    cart_pos, cart_vel, pole_angle, pole_vel = observation

    new_obs = (
        _discretize_cart_position(cart_pos),
        _discretize_cart_velocity(cart_vel),
        _discretize_pole_angle(pole_angle),
        _discretize_angular_velocity(pole_vel)
    )

    return new_obs

def _discretize_cart_velocity(velocity):
    return np.round(velocity, 1)

def _discretize_angular_velocity(velocity):
    return np.round(velocity, 1)

def _discretize_cart_position(pos):
    return np.round(pos, 1)

def _discretize_pole_angle(angle):
    return np.round(angle, 1)

