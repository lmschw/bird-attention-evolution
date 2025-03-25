import numpy as np

import simulator.head_movement.weight_options as wo


def move_heads(model, weight_options, num_agents, current_head_angles, distances, angles, perception_strengths_conspecifics):
    """
    Moves the heads of all agents based on the output of the neural network model.
    """
    inputs = np.array([wo.get_input_value_for_weight_option(weight_option=option, 
                                                            current_head_angles=current_head_angles, 
                                                            distances=distances, 
                                                            angles=angles, 
                                                            perception_strengths=perception_strengths_conspecifics) for option in weight_options])
    inputs = np.where(inputs == np.inf, wo.MAX_INPUT, inputs)
    new_head_angles = []
    for i in range(num_agents):
        new_head_angles.append(model.predict([inputs[:,i]])[0][0][0])
    return new_head_angles