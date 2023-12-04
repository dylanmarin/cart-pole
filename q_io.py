import numpy as np
from collections import defaultdict

def save_q_values(Q, filename):
    """
    Q is a dictionary of tuple->np.array(float) pairs
    Write to a csv where 

    col # : value
    0: cart position
    1: cart velocity
    2: pole angle
    3: pole angular velocity
    4: action 0 value
    5: action 1 value
    """
    try: 
        # check if file exists
        open(filename, 'r')

        # if it does, increment the filename
        filename, file_extension = filename.split('.')
        filename_num = filename.split('_')[-1]

        if filename_num.isdigit():
            print(filename_num)
            filename = filename.replace("_" + filename_num, "")
            filename_num = int(filename_num) + 1
            filename = filename + f"_{filename_num}." + file_extension
        else:
            filename = filename + "_1." + file_extension

        # recursively call save_q_values with new filename to make sure new file also doesnt exist
        save_q_values(Q, filename)
    except:
        # if file doesn't exist write as normal
        with open(filename, 'w') as f:
            f.write("cart_pos,cart_vel,pole_angle,pole_vel,action_0_value,action_1_value\n")
            for state in Q.keys():
                cart_pos, cart_vel, pole_angle, pole_vel = state
                action_0_value = Q[state][0]
                action_1_value = Q[state][1]
                f.write(f"{cart_pos},{cart_vel},{pole_angle},{pole_vel},{action_0_value},{action_1_value}\n")


def load_q_values(filename):
    """
    Q is a dictionary of tuple->np.array(float) pairs
    return a Q dictionary after loading file at the given filename    
    """    
    Q = defaultdict(lambda: np.zeros(2))

    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            cart_pos, cart_vel, pole_angle, pole_vel, action_0_value, action_1_value = line.split(',')
            state = (float(cart_pos), float(cart_vel), float(pole_angle), float(pole_vel))
            Q[state] = np.array([float(action_0_value), float(action_1_value)])

    return Q

def save_array(returns, filename):
    try: 
        # check if file exists
        open(filename, 'r')

        # if it does, increment the filename
        filename, file_extension = filename.split('.')
        filename_num = filename.split('_')[-1]

        if filename_num.isdigit():
            print(filename_num)
            filename = filename.replace("_" + filename_num, "")
            filename_num = int(filename_num) + 1
            filename = filename + f"_{filename_num}." + file_extension
        else:
            filename = filename + "_1." + file_extension

        # recursively call save_q_values with new filename to make sure new file also doesnt exist
        save_array(returns, filename)
    except:
        # if file doesn't exist write as normal
        with open(filename, 'w') as f:
            for i in range(len(returns)):
                f.write(f"{returns[i]},")

def load_array(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        returns = [float(line) for line in lines[0].split(',')[:-1]]
    return returns