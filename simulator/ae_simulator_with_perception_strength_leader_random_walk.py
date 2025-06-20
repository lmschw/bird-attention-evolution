import numpy as np

import vision.perception_strength as pstrength
from simulator.ae_simulator_with_perception_strength import ActiveElasticWithPerceptionStrengthSimulator
import general.angle_conversion as ac

"""
An implementation of Active Elastic including perception strengths.
"""

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

class ActiveElasticWithPerceptionStrengthWithLeaderRandomWalkSimulator(ActiveElasticWithPerceptionStrengthSimulator):
    def __init__(self, animal_type, num_agents, domain_size, start_position,
                 landmarks=[], occlusion_active=True, visualize=True, follow=True, graph_freq=5):
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
        self.occlusion_active = occlusion_active
        self.leaders = np.zeros(num_agents)
    

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
        if self.occlusion_active:
            perception_strengths_conspecifics = pstrength.compute_perception_strengths_with_occlusion_conspecifics(agents=self.curr_agents, distances=dists_conspecifics, angles=angles_conspecifics, shape=(self.num_agents, self.num_agents), animal_type=self.animal_type)
        else:
            perception_strengths_conspecifics = pstrength.compute_perception_strengths(distances=dists_conspecifics, angles=angles_conspecifics, shape=(self.num_agents, self.num_agents), animal_type=self.animal_type)


        p_x, p_y = self.get_pi_elements(distances_conspecifics=dists_conspecifics,
                                        angles_conspecifics=angles_conspecifics,
                                        perception_strengths_conspecifics=perception_strengths_conspecifics)
        h_x, h_y = self.get_hi_elements()

        f_x = ALPHA * p_x + BETA * h_x 
        f_y = ALPHA * p_y + BETA * h_y 
        
        return f_x, f_y, np.sum(perception_strengths_conspecifics, axis=1)
    
    def update_agents(self):
        """
        Updates the agents' positions and orientations based on the spring force.
        """
        f_x, f_y, total_perception_strengths = self.compute_fi()
        u, w = self.compute_u_w(f_x, f_y)

        headings = self.curr_agents[:, 2]
        headings = np.where(((total_perception_strengths == 0) & (self.leaders == 0)), np.random.random() * 2 * np.pi, headings)

        x_vel = np.multiply(u, np.cos(headings))
        y_vel = np.multiply(u, np.sin(headings))

        self.curr_agents[:, 0] = self.curr_agents[:, 0] + x_vel * DT
        self.curr_agents[:, 1] = self.curr_agents[:, 1] + y_vel * DT
        self.curr_agents[:, 2] = ac.wrap_to_pi(self.curr_agents[:, 2] + w * DT)
        self.curr_agents[:,2] = np.where(((total_perception_strengths == 0) & (self.leaders == 1)), headings, self.curr_agents[:,2])




        
