import numpy as np

import general.angle_conversion as ac
import general.normalisation as normal
import simulator.head_movement.weight_options as wo


def move_heads(model, weight_options, animal_type, num_agents, current_head_angles, distances, angles, perception_strengths_conspecifics):
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
        predicted = model.predict([inputs[:,i]])[0][0][0]
        """
        angle_2pi = predicted * (2*np.pi)
        angle = ac.wrap_angle_to_pi(angle_2pi)
        if np.absolute(angle) > animal_type.head_range_half:
            angle = animal_type.head_range_half"
        """
        angle = (predicted * (2*animal_type.head_range_half)) - animal_type.head_range_half
        new_head_angles.append(angle)
    return new_head_angles