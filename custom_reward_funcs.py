import numpy as np

def reward_vertical_central(state):
    '''
    reward is 1 when the pole is vertical and the cart is centered

    reward is 0.5 when the pole is vertical and the cart is not centered or the pole is not vertical and the cart is centered
    
    reward is 0 when the pole is not vertical and the cart is not centered
    '''
    cart_pos, cart_vel, pole_angle, pole_vel = state

    cart_pos_contribution = np.abs(cart_pos) / (2.4)
    pole_angle_contribution = np.abs(pole_angle) / (0.418)

    reward =  1 - (cart_pos_contribution * 0.2 + pole_angle_contribution * 0.8)

    return reward