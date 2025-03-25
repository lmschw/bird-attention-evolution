
"""
A visual field determined by the foveal projection and the viewing distance.

Attributes:
name                                -   string              -   the name of the focus area, mainly intended for printing and saving
azimuth_angle_position_horizontal   -   float[rad]          -   the horizontal angle between the foveal projection of this focus area and the horizontal direction looking straight ahead (i.e. 0)
angle_field_horizontal              -   float[rad]          -   the horizontal angle between the horizontal foveal projection and the outer edge of the focus area. Represents half of the visual field for the focus area
azimuth_angle_position_vertical     -   float[rad]          -   the vertical angle between the foveal projection of this focus area and the vertical direction looking towards the horizon (i.e. 0)
angle_field_vertical                -   float[rad]          -   the vertical angle between the vertical foveal projection and the outer edge of the focus area. Represents half of the visual field for the focus area
comfortable_distance                -   array of floats [m] -   the comfortable viewing distance for this focus area. Entities between the min and max will be more in focus
"""
class FocusArea:
    def __init__(self, name, azimuth_angle_position_horizontal, angle_fovea_horizontal, angle_field_horizontal,  
                 azimuth_angle_position_vertical, angle_fovea_vertical, angle_field_vertical, comfortable_distance):
        self.name = name
        self.azimuth_angle_position_horizontal = azimuth_angle_position_horizontal
        self.angle_fovea_horizontal = angle_fovea_horizontal
        self.angle_field_horizontal = angle_field_horizontal
        self.azimuth_angle_position_vertical = azimuth_angle_position_vertical
        self.angle_field_vertical = angle_field_vertical
        self.angle_fovea_vertical = angle_fovea_vertical
        self.comfortable_distance = comfortable_distance