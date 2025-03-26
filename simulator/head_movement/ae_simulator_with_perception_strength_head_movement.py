import numpy as np

import general.normalisation as normal
import general.angle_conversion as ac
import simulator.head_movement.head_movement as hm
import vision.perception_strength as pstrength
from simulator.ae_simulator import ActiveElasticSimulator

# AE Constants
EPSILON = 12
SIGMA = 0.7
SIGMA_MIN = 0.7
SIGMA_MAX = 0.7
UC = 0.05
UMAX = 0.1
WMAX = np.pi / 2
ALPHA = 2.0
BETA = 0.5
BETA_LEADER = 4
GAMMA = 1.0
G_GAIN = 1
K1 = 0.6
K2 = 0.05
L_0 = 0.5
K_REP = 2.0
DT = 1
DES_DIST = SIGMA * 2**(1/2)
PERCEPTION_STRENGTH_MODIFIER = 5

"""
An implementation of Active Elastic including perception strengths.
"""

class ActiveElasticWithPerceptionStrengthAndHeadMovementSimulator(ActiveElasticSimulator):
    def __init__(self, animal_type, num_agents, domain_size, start_position,
                 model, weight_options, landmarks=[], visualize=True, follow=True, graph_freq=5):
        """
        Params:
            - animal_type (Animal): the type of animal
            - num_agents (int): the number of agents
            - domain_size (tuple of ints): how big the domain is (not bounded by these values though)
            - start_position (tuple of ints): around which points the agents should initially be placed
            - visualize (boolean) [optional, default=True]: whether the results should be visualized directly
            - follow (boolean) [optional, default=True]: whether the visualization should follow the centroid of the swarm or whether it should show the whole domain
            - graph_freq (int) [optional, default=5]: how often the visualization should be updated
        """

        super().__init__(animal_type=animal_type,
                         num_agents=num_agents,
                         domain_size=domain_size,
                         start_position=start_position,
                         landmarks=landmarks,
                         visualize=visualize,
                         follow=follow,
                         graph_freq=graph_freq)
        self.model = model
        self.weight_options = weight_options

    def init_agents(self):
        base = super().init_agents()
        pos_xs = base[:,0]
        pos_ys = base[:,1]
        pos_hs = base[:,2]

        head_angles = np.zeros(self.num_agents)
        self.curr_agents = np.column_stack([pos_xs, pos_ys, pos_hs, head_angles])

        return self.curr_agents

    def get_pi_elements(self, distances_conspecifics, angles_conspecifics, perception_strengths_conspecifics):
        """
        Computes the proximal control vector.
        """
        forces_conspecifics = -EPSILON * perception_strengths_conspecifics * PERCEPTION_STRENGTH_MODIFIER * (2 * (self.sigmas[:, np.newaxis] ** 4 / distances_conspecifics ** 5) - (self.sigmas[:, np.newaxis] ** 2 / distances_conspecifics ** 3))
        forces_conspecifics[distances_conspecifics == np.inf] = 0.0

        p_x_conspecifics = np.sum(np.multiply(forces_conspecifics, np.cos(angles_conspecifics)), axis=1)
        p_y_conspecifics = np.sum(np.multiply(forces_conspecifics, np.sin(angles_conspecifics)), axis=1)

        p_x = p_x_conspecifics
        p_y = p_y_conspecifics

        return p_x, p_y

    def compute_fi(self):
        """
        Computes the force components.
        """
        dists_conspecifics, angles_conspecifics = self.compute_distances_and_angles()
        perception_strengths_conspecifics = pstrength.compute_perception_strengths(distances=dists_conspecifics, 
                                                                                   angles=angles_conspecifics, 
                                                                                   shape=(self.num_agents, self.num_agents), 
                                                                                   animal_type=self.animal_type,
                                                                                   head_orientations=self.curr_agents[:,3])

        p_x, p_y = self.get_pi_elements(distances_conspecifics=dists_conspecifics,
                                        angles_conspecifics=angles_conspecifics,
                                        perception_strengths_conspecifics=perception_strengths_conspecifics)
        h_x, h_y = self.get_hi_elements()

        f_x = ALPHA * p_x + BETA * h_x 
        f_y = ALPHA * p_y + BETA * h_y 
        
        return f_x, f_y, dists_conspecifics, angles_conspecifics, perception_strengths_conspecifics

    def update_agents(self):
        """
        Updates the agents' positions and orientations based on the spring force.
        """
        f_x, f_y, distances, angles, pstrengths = self.compute_fi()
        u, w = self.compute_u_w(f_x, f_y)

        headings = self.curr_agents[:, 2]
        x_vel = np.multiply(u, np.cos(headings))
        y_vel = np.multiply(u, np.sin(headings))

        self.curr_agents[:, 0] = self.curr_agents[:, 0] + x_vel * DT
        self.curr_agents[:, 1] = self.curr_agents[:, 1] + y_vel * DT
        self.curr_agents[:, 2] = ac.wrap_to_pi(self.curr_agents[:, 2] + w * DT)
        self.curr_agents[:, 3] = hm.move_heads(model=self.model,
                                               weight_options=self.weight_options,
                                               animal_type=self.animal_type,
                                               num_agents=self.num_agents,
                                               current_head_angles=self.curr_agents[:,3],
                                               distances=distances,
                                               angles=angles,
                                               perception_strengths_conspecifics=pstrengths)

