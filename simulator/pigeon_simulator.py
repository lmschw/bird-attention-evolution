import numpy as np
import matplotlib.pyplot as plt

from bird_models.pigeon import Pigeon
import vision.environment_vision as env_vision
import geometry.normalisation as gnormal

"""
I am a pigeon. I am part of a flock. I am flying home.
My head is looking straight ahead to begin with.
I can mainly see what is in my focus areas.
If I recognize a landmark, i.e. it is in my path, I will use this information to try and adjust my heading.
If I can see a conspecific, I will try to get closer to it unless I am too close, in which case I will turn away.
I will then move my head if necessary.
"""

class PigeonSimulator:
    def __init__(self, n, area, start_position, target_position, neural_network):
        self.n = n
        self.area = area
        self.start_position = start_position
        self.target_position = target_position
        self.neural_network = neural_network
        self.bird_type = Pigeon()

    def init_swarm(self):
        rng = np.random
        n_points_x = self.n
        n_points_y = self.n
        spacing = self.bird_type.wingspan
        init_x = 0
        init_y = 0

        x_values = np.linspace(init_x, init_x + (n_points_x - 1) * spacing, n_points_x)
        y_values = np.linspace(init_y, init_y + (n_points_y - 1) * spacing, n_points_y)
        xx, yy = np.meshgrid(x_values, y_values)

        pos_xs = xx.ravel() + (rng.random(n_points_x * n_points_y) * spacing * 0.5) - spacing * 0.25
        pos_ys = yy.ravel() + (rng.random(n_points_x * n_points_y) * spacing * 0.5) - spacing * 0.25
        pos_hs = (rng.random(n_points_x * n_points_x) * np.pi * 2) - np.pi

        num_agents = len(pos_xs)
        self.num_agents = num_agents

        self.paths = np.random.choice(self.area.paths, num_agents)
        hierarchy = np.zeros((num_agents, 1))
        target_heading = np.zeros((num_agents, 1))
        head_heading = np.zeros((num_agents, 1))
        speeds = np.full((num_agents,1), self.bird_type.average_speed)

        return np.column_stack([pos_xs, pos_ys, pos_hs, hierarchy, target_heading, head_heading, speeds])
    
    def compute_local_orders(self, orientations, perception_strengths):
        """
        Computes the local order for every individual.

        Params: 
            - orientations (array of floats): the orientation of every individual at the current timestep
            - neighbours (array of arrays of booleans): the identity of every neighbour of every individual

        Returns:
            An array of floats representing the local order for every individual at the current time step (values between 0 and 1)
        """
        uv_coordinates = self.compute_u_v_coordinates_for_angles(orientations)
        sumOrientation = np.sum(perception_strengths[:,:,np.newaxis]*uv_coordinates[np.newaxis,:,:],axis=1)
        perception_strengths = gnormal.normalise(values=perception_strengths)
        perception_strengths[np.count_nonzero(perception_strengths, axis=1) == 0] = np.inf
        return np.divide(np.sqrt(np.sum(sumOrientation**2,axis=1)), np.count_nonzero(perception_strengths, axis=1))
    
    def compute_comfortable_distance_matches(self, agents, perception_strengths):
        distances = env_vision.compute_distances(agents=agents)
        in_comfortable_distance_min = distances >= self.bird_type.preferred_distance_left_right[0]
        in_comfortable_distance_max = distances <= self.bird_type.preferred_distance_left_right[1]
        in_comfortable_distance = np.where(((in_comfortable_distance_min & in_comfortable_distance_max)), True, False)
        return np.sum(perception_strengths*in_comfortable_distance)

    def get_conspecific_alignment_cohesion(self, agents):
        perception_strengths, min_conspecific = env_vision.compute_perception_strengths_conspecifics(agents=agents, bird_type=self.bird_type)
        local_orders = []
        comfortable_distance_matches = []
        for p_strength in perception_strengths:
            local_orders.append(self.compute_local_orders(agents[:,2], perception_strengths=p_strength))
            comfortable_distance_matches.append(self.compute_comfortable_distance_matches(agents=agents, perception_strengths=perception_strengths))
        return perception_strengths, np.average(local_orders), np.average(comfortable_distance_matches), min_conspecific

    def get_landmark_alignment(self, agents):
        perception_strengths, min_landmark = env_vision.compute_perception_strengths_landmarks(agents=agents, landmarks=self.area.landmarks, bird_type=self.bird_type)
        min_dists_final, min_angles_final, min_dist_idx = min_landmark
        alignment_angles = []
        for i in range(len(agents)):
            if min_dist_idx[i] in self.paths[i].keys():
                if min_angles_final[i] == 0 or (min_angles_final[i] < 0 and self.paths[i][min_dist_idx[i]] == 'l') or (min_angles_final[i] > 0 and self.paths[i][min_dist_idx[i]] == 'r'):
                    alignment_angles.append(0)
                else:
                    alignment_angles.append(min_angles_final[i] * perception_strengths[i][min_dist_idx[i]])
            else:
                alignment_angles.append(0)
        return alignment_angles

    def determine_visual_feedback(self, agents):
        """
        TODO: implement the visual feedback based on the neighbours, landmarks and possibly predators that the
        individual can see. Use the proximity to the focus direction to determine the strength of the input for
        each entity. Convert into a proximity and possibly alignment value for the neighbours, an alignment to
        the landmarks (i.e. the path home) and the proximity to predators
        """

        perception_strengths, conspec_alignment, conspec_cohesion, min_neighbour = self.get_conspecific_alignment_cohesion(agents=agents)
        landmark_alignment = self.get_landmark_alignment(agents=agents)
        return perception_strengths, min_neighbour, [conspec_alignment, conspec_cohesion, landmark_alignment]

    def get_new_head_angles(self, agents, visual_feedback):
        # TODO: implement NN to get new angle for head incl. head turn limits
        return agents[:, 5]

    def get_new_orientations(self, agents, perception_strengths_conspecifics, visual_feeback):
        """
        TODO: implement the new orientation for the pigeon based on the distance to its conspecifics that it can see
        here we can either outright use AE or at least take inspiration from it
        """
        orientations = []
        for p_strength in perception_strengths_conspecifics:
            orient = np.sum(agents[:,np.newaxis, 2] * p_strength, axis=1)
            if np.count_nonzero(orient) != 0:
                orientations.append(orient)
        if len(orientations) == 0:
            return agents[:,2] + visual_feeback[2]
        orientations = np.array(orientations)
        orientation_conspecifics = np.average(orientations.T, axis=1)
        orientation_conspecifics += visual_feeback[2]
        return orientation_conspecifics + np.random.random(orientation_conspecifics.shape)

    def compute_u_v_coordinates_for_angles(self, angles):
        """
        Computes the (u,v)-coordinates based on the angle.

        Params:
            - angle (float): the angle in radians

        Returns:
            An array containing the [u, v]-coordinates corresponding to the angle.
        """
        # compute the uv-coordinates
        U = np.cos(angles)
        V = np.sin(angles)
    
        return np.column_stack((U,V))
    
    def compute_angles_for_orientations(self, orientations):
        """
        Computes the angle in radians based on the (u,v)-coordinates of the current orientation.

        Params:
            - orientation (array of floats): the current orientation in (u,v)-coordinates

        Returns:
            A float representin the angle in radians.
        """
        return np.arctan2(orientations[:, 1], orientations[:, 0])

    def update_positions(self, agents):
        positions = np.column_stack([agents[:,0], agents[:,1]])
        positions += (self.compute_u_v_coordinates_for_angles(agents[:, 2]).T * agents[:, 6]).T
        agents[:, 0] = positions[:, 0]
        agents[:, 1] = positions[:, 1]
        return agents

    def update_agents(self, agents):
        perception_strengths, min_neighbour, visual_feedback = self.determine_visual_feedback(agents)
        agents = self.update_positions(agents=agents)
        new_headings = self.get_new_orientations(agents=agents, perception_strengths_conspecifics=perception_strengths, visual_feeback=visual_feedback)
        new_head_headings = self.get_new_head_angles(agents=agents, visual_feedback=visual_feedback)
        agents[:, 2] = new_headings
        agents[:, 5] = new_head_headings
        return agents

    def simulate(self, tmax):
        """
        TODO: implement the simulation procedure:
        1) determine visual feedback
        2) determine new orientation
        3) determine new head angle
        optionally also determine new speed and update energy levels
        also implement leadership
        """
        # TODO: possibly change tmax to the distance home once energy levels are included
        curr_agents = self.init_swarm()

        fig, ax = plt.subplots()
        ax.set_facecolor((0, 0, 0))  
        centroid_x, centroid_y = np.mean(curr_agents[:, 0]), np.mean(curr_agents[:, 1])
        ax.set_xlim(centroid_x - 5, centroid_x + 5)
        ax.set_ylim(centroid_y - 5, centroid_y + 5)
        for t in range(tmax):
            new_agents = np.copy(curr_agents)
            self.update_agents(new_agents)
            curr_agents = np.copy(new_agents)  

            if (not (t % 5)) and (t > 0):
                # Clear the axes
                ax.clear()

                # Draw target area
                target = plt.Circle(self.target_position, 0.1 * self.area.area_size[0], color='green')
                ax.add_patch(target)

                # Draw landmarks
                landmark_positions = np.array([landmark.position for landmark in self.area.landmarks])
                ax.scatter(landmark_positions[:, 0], landmark_positions[:, 1], color="yellow", s=15)
                for i in range(len(landmark_positions)):
                    ax.annotate(self.area.landmarks[i].name, (landmark_positions[i,0], landmark_positions[i,1]), color="white")

                # Draw curr_agents as a scatter plot
                ax.scatter(curr_agents[:, 0], curr_agents[:, 1], color="white", s=15)

                # Draw agent headings using quiver
                ax.quiver(curr_agents[:, 0], curr_agents[:, 1], np.cos(curr_agents[:, 2]), np.sin(curr_agents[:, 2]),
                        color="white", width=0.005, scale=40)

                # Set plot properties
                ax.set_facecolor((0, 0, 0))  # Set the background color to black
                
                """
                centroid_x, centroid_y = np.mean(curr_agents[:, 0]), np.mean(curr_agents[:, 1])
                ax.set_xlim(centroid_x - 5, centroid_x + 5)
                ax.set_ylim(centroid_y - 5, centroid_y + 5)
                """
                ax.set_xlim(0, self.area.area_size[0])
                ax.set_ylim(0, self.area.area_size[1])
                
                # Show the plot
                plt.pause(0.000001)


