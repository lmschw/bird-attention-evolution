import numpy as np
import matplotlib.pyplot as plt

from bird_models.pigeon import Pigeon

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

        print(pos_xs)
        print(pos_ys)
        print(pos_hs)

        self.agents = np.column_stack([pos_xs, pos_ys, pos_hs, hierarchy, target_heading, head_heading])



    def get_new_head_angles(self, visual_feedback):
        # TODO: implement NN to get new angle for head incl. head turn limits
        pass

    def get_new_orientations(self):
        """
        TODO: implement the new orientation for the pigeon based on the distance to its conspecifics that it can see
        here we can either outright use AE or at least take inspiration from it
        """
        pass

    def determine_visual_feedback(self):
        """
        TODO: implement the visual feedback based on the neighbours, landmarks and possibly predators that the
        individual can see. Use the proximity to the focus direction to determine the strength of the input for
        each entity. Convert into a proximity and possibly alignment value for the neighbours, an alignment to
        the landmarks (i.e. the path home) and the proximity to predators
        """
        pass

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
        self.init_swarm()
        curr_agents = self.agents
        fig, ax = plt.subplots()
        ax.set_facecolor((0, 0, 0))  
        centroid_x, centroid_y = np.mean(curr_agents[:, 0]), np.mean(curr_agents[:, 1])
        ax.set_xlim(centroid_x - 5, centroid_x + 5)
        ax.set_ylim(centroid_y - 5, centroid_y + 5)
        for t in range(tmax):
            new_agents = np.copy(curr_agents)
            #update_agents()
            curr_agents = np.copy(new_agents)  

            if (not (t % 5)) and (t > 0):
                # Clear the axes
                ax.clear()

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


