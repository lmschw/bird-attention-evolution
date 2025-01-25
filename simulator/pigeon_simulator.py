
class PigeonSimulator:
    def __init__(self):
        # TODO: implement init
        pass

    def get_new_head_angles(self, visual_feedback):
        # TODO: implement NN to get new angle for head incl. head turn limits
        pass

    def get_new_orientations(self):
        """
        TODO: implement the new orientation for the pigeon based on the distance to its conspecifics that it can see
        here we can either outright use AE or at least take inspiration from it
        """

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
        pass