import numpy as np

from animal_models.bird import Bird
from animal_models.focus_area import FocusArea


"""
Class inheriting from Bird including the relevant data.
"""


NAME = "Zebra finch"
AVERAGE_SPEED = 27 # m/s
AVERAGE_SPEED = 0.27 # m/s
AVERAGE_SPEED = 0.01
MIN_SPEED = AVERAGE_SPEED /2
MAX_SPEED = AVERAGE_SPEED + (AVERAGE_SPEED/2)
SPEEDS = (MIN_SPEED, AVERAGE_SPEED, MAX_SPEED)

WINGSPAN = 0.22 # meters
LENGTH = 0.11 # meters
HEAD_RANGE_HALF = np.deg2rad(120)
MAX_TURN_ANGLE = np.deg2rad(90)
PREFERRED_DISTANCE_FRONT_BACK = [LENGTH + 0.1, 2 * LENGTH] # WARNING: THIS IS AN ASSUMPTION
PREFERRED_DISTANCE_LEFT_RIGHT = [WINGSPAN + 2, 2 * WINGSPAN + 1]  # WARNING: THIS IS AN ASSUMPTION
SENSING_RANGE = 40

FOVEAL_REGION_NAME = "foveal_region_right"
FOVEAL_REGION_AZIMUTH_HORIZONTAL = np.deg2rad(62)
FOVEAL_REGION_FIELD_HORIZONTAL_HALF = np.deg2rad(85)
FOVEAL_REGION_AZIMUTH_VERTICAL = np.deg2rad(16.5)
FOVEAL_REGION_FIELD_VERTICAL_HALF = np.deg2rad(10)
FOVEAL_REGION_COMFORTABLE_DISTANCE = (0, 20)
FOVEAL_REGION_FOVEAL_ANGLE = np.deg2rad(15)
FOVEAL_REGION_RIGHT = FocusArea(name=FOVEAL_REGION_NAME, 
                          azimuth_angle_position_horizontal=FOVEAL_REGION_AZIMUTH_HORIZONTAL,
                          angle_fovea_horizontal=FOVEAL_REGION_FOVEAL_ANGLE,
                          angle_field_horizontal=FOVEAL_REGION_FIELD_HORIZONTAL_HALF,
                          azimuth_angle_position_vertical=FOVEAL_REGION_AZIMUTH_VERTICAL,
                          angle_fovea_vertical=FOVEAL_REGION_FOVEAL_ANGLE,
                          angle_field_vertical=FOVEAL_REGION_AZIMUTH_VERTICAL,
                          comfortable_distance=FOVEAL_REGION_COMFORTABLE_DISTANCE
                          )

FOVEAL_REGION_NAME = "foveal_region_left"
FOVEAL_REGION_AZIMUTH_HORIZONTAL = np.deg2rad(-62)
FOVEAL_REGION_FIELD_HORIZONTAL_HALF = np.deg2rad(85)
FOVEAL_REGION_AZIMUTH_VERTICAL = np.deg2rad(16.5)
FOVEAL_REGION_FIELD_VERTICAL_HALF = np.deg2rad(10)
FOVEAL_REGION_COMFORTABLE_DISTANCE = (0, 20)
FOVEAL_REGION_FOVEAL_ANGLE = np.deg2rad(15)
FOVEAL_REGION_LEFT = FocusArea(name=FOVEAL_REGION_NAME, 
                          azimuth_angle_position_horizontal=FOVEAL_REGION_AZIMUTH_HORIZONTAL,
                          angle_fovea_horizontal=FOVEAL_REGION_FOVEAL_ANGLE,
                          angle_field_horizontal=FOVEAL_REGION_FIELD_HORIZONTAL_HALF,
                          azimuth_angle_position_vertical=FOVEAL_REGION_AZIMUTH_VERTICAL,
                          angle_fovea_vertical=FOVEAL_REGION_FOVEAL_ANGLE,
                          angle_field_vertical=FOVEAL_REGION_AZIMUTH_VERTICAL,
                          comfortable_distance=FOVEAL_REGION_COMFORTABLE_DISTANCE
                          )

FOCUS_AREAS = [FOVEAL_REGION_LEFT, FOVEAL_REGION_RIGHT]
FOVEAL_ACUITY = 1
FOVEA_DISCOUNT_FACTOR = 0.3

class Zebrafinch(Bird):
    def __init__(self):
        super().__init__(name=NAME,
                         speeds=SPEEDS,
                         wingspan=WINGSPAN, 
                         length=LENGTH, 
                         head_range_half=HEAD_RANGE_HALF,
                         max_turn_angle=MAX_TURN_ANGLE,
                         focus_areas=FOCUS_AREAS, 
                         foveal_acuity=FOVEAL_ACUITY,
                         fovea_discount_factor=FOVEA_DISCOUNT_FACTOR,
                         preferred_distance_front_back=PREFERRED_DISTANCE_FRONT_BACK,
                         preferred_distance_left_right=PREFERRED_DISTANCE_LEFT_RIGHT,
                         sensing_range=SENSING_RANGE)