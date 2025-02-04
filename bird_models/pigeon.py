import numpy as np

from bird_models.bird import Bird
from bird_models.focus_area import FocusArea

NAME = "Pigeon"
AVERAGE_SPEED = 27 # m/s
AVERAGE_SPEED = 0.27 # m/s
AVERAGE_SPEED = 0.01
MIN_SPEED = AVERAGE_SPEED /2
MAX_SPEED = AVERAGE_SPEED + (AVERAGE_SPEED/2)
SPEEDS = (MIN_SPEED, AVERAGE_SPEED, MAX_SPEED)

WINGSPAN = 0.67 # meters
LENGTH = 0.26 # meters
HEAD_RANGE_HALF = np.deg2rad(120)
PREFERRED_DISTANCE_FRONT_BACK = [LENGTH + 0.1, 2 * LENGTH] # WARNING: THIS IS AN ASSUMPTION
PREFERRED_DISTANCE_LEFT_RIGHT = [WINGSPAN + 2, 2 * WINGSPAN + 1]  # WARNING: THIS IS AN ASSUMPTION
SENSING_RANGE = 1000

FOVEAL_REGION_NAME = "foveal_region_right"
FOVEAL_REGION_AZIMUTH_HORIZONTAL = np.deg2rad(75)
FOVEAL_REGION_FIELD_HORIZONTAL_HALF = np.deg2rad(85)
FOVEAL_REGION_AZIMUTH_VERTICAL = np.deg2rad(1)
FOVEAL_REGION_FIELD_VERTICAL_HALF = np.deg2rad(10)
FOVEAL_REGION_COMFORTABLE_DISTANCE = (0.35, np.inf)
FOVEAL_REGION_RIGHT = FocusArea(name=FOVEAL_REGION_NAME, 
                          azimuth_angle_position_horizontal=FOVEAL_REGION_AZIMUTH_HORIZONTAL,
                          angle_field_horizontal=FOVEAL_REGION_FIELD_HORIZONTAL_HALF,
                          azimuth_angle_position_vertical=FOVEAL_REGION_AZIMUTH_VERTICAL,
                          angle_field_vertical=FOVEAL_REGION_AZIMUTH_VERTICAL,
                          comfortable_distance=FOVEAL_REGION_COMFORTABLE_DISTANCE
                          )

FOVEAL_REGION_NAME = "foveal_region_left"
FOVEAL_REGION_AZIMUTH_HORIZONTAL = np.deg2rad(-75)
FOVEAL_REGION_FIELD_HORIZONTAL_HALF = np.deg2rad(85)
FOVEAL_REGION_AZIMUTH_VERTICAL = np.deg2rad(1)
FOVEAL_REGION_FIELD_VERTICAL_HALF = np.deg2rad(10)
FOVEAL_REGION_COMFORTABLE_DISTANCE = (0.35, np.inf)
FOVEAL_REGION_LEFT = FocusArea(name=FOVEAL_REGION_NAME, 
                          azimuth_angle_position_horizontal=FOVEAL_REGION_AZIMUTH_HORIZONTAL,
                          angle_field_horizontal=FOVEAL_REGION_FIELD_HORIZONTAL_HALF,
                          azimuth_angle_position_vertical=FOVEAL_REGION_AZIMUTH_VERTICAL,
                          angle_field_vertical=FOVEAL_REGION_AZIMUTH_VERTICAL,
                          comfortable_distance=FOVEAL_REGION_COMFORTABLE_DISTANCE
                          )

LOWER_FRONTAL_REGION_NAME = "lower_frontal_region"
LOWER_FRONTAL_REGION_AZIMUTH_HORIZONTAL = np.deg2rad(0)
LOWER_FRONTAL_REGION_FIELD_HORIZONTAL_HALF = np.deg2rad(90)
LOWER_FRONTAL_REGION_AZIMUTH_VERTICAL = np.deg2rad(-55)
LOWER_FRONTAL_REGION_FIELD_VERTICAL_HALF = np.deg2rad(35)
LOWER_FRONTAL_REGION_COMFORTABLE_DISTANCE = (0, 0.35)
LOWER_FRONTAL_REGION = FocusArea(name=LOWER_FRONTAL_REGION_NAME, 
                          azimuth_angle_position_horizontal=LOWER_FRONTAL_REGION_AZIMUTH_HORIZONTAL,
                          angle_field_horizontal=LOWER_FRONTAL_REGION_FIELD_HORIZONTAL_HALF,
                          azimuth_angle_position_vertical=LOWER_FRONTAL_REGION_AZIMUTH_VERTICAL,
                          angle_field_vertical=LOWER_FRONTAL_REGION_AZIMUTH_VERTICAL,
                          comfortable_distance=LOWER_FRONTAL_REGION_COMFORTABLE_DISTANCE
                          )

FOCUS_AREAS = [FOVEAL_REGION_LEFT, FOVEAL_REGION_RIGHT, LOWER_FRONTAL_REGION]

class Pigeon(Bird):
    def __init__(self):
        super().__init__(name=NAME,
                         speeds=SPEEDS,
                         wingspan=WINGSPAN, 
                         length=LENGTH, 
                         head_range_half=HEAD_RANGE_HALF,
                         focus_areas=FOCUS_AREAS, 
                         preferred_distance_front_back=PREFERRED_DISTANCE_FRONT_BACK,
                         preferred_distance_left_right=PREFERRED_DISTANCE_LEFT_RIGHT,
                         sensing_range=SENSING_RANGE)