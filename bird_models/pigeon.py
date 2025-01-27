import numpy as np

from bird_models.bird import Bird
from bird_models.focus_area import FocusArea

AVERAGE_SPEED = 27 # m/s
WINGSPAN = 0.67 # meters
LENGTH = 0.26 # meters
HEAD_RANGE_HALF = np.deg2rad(180)
PREFERRED_DISTANCE_FRONT_BACK = LENGTH + 10 # WARNING: THIS IS AN ASSUMPTION
PREFERRED_DISTANCE_LEFT_RIGHT = WINGSPAN + 10  # WARNING: THIS IS AN ASSUMPTION

FOVEAL_REGION_NAME = "foveal_region"
FOVEAL_REGION_AZIMUTH_HORIZONTAL = np.deg2rad(75)
FOVEAL_REGION_FIELD_HORIZONTAL_HALF = np.deg2rad(10)
FOVEAL_REGION_AZIMUTH_VERTICAL = np.deg2rad(0)
FOVEAL_REGION_FIELD_VERTICAL_HALF = np.deg2rad(10)
FOVEAL_REGION_COMFORTABLE_DISTANCE = (35, np.inf)
FOVEAL_REGION = FocusArea(name=FOVEAL_REGION_NAME, 
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
LOWER_FRONTAL_REGION_COMFORTABLE_DISTANCE = (0, 35)
LOWER_FRONTAL_REGION = FocusArea(name=LOWER_FRONTAL_REGION_NAME, 
                          azimuth_angle_position_horizontal=LOWER_FRONTAL_REGION_AZIMUTH_HORIZONTAL,
                          angle_field_horizontal=LOWER_FRONTAL_REGION_FIELD_HORIZONTAL_HALF,
                          azimuth_angle_position_vertical=LOWER_FRONTAL_REGION_AZIMUTH_VERTICAL,
                          angle_field_vertical=LOWER_FRONTAL_REGION_AZIMUTH_VERTICAL,
                          comfortable_distance=LOWER_FRONTAL_REGION_COMFORTABLE_DISTANCE
                          )

FOCUS_AREAS = [FOVEAL_REGION, LOWER_FRONTAL_REGION]

class Pigeon(Bird):
    def __init__(self):
        super().__init__(average_speed=AVERAGE_SPEED,
                         wingspan=WINGSPAN, 
                         length=LENGTH, 
                         head_range_half=HEAD_RANGE_HALF,
                         focus_areas=FOCUS_AREAS, 
                         preferred_distance_front_back=PREFERRED_DISTANCE_FRONT_BACK,
                         preferred_distance_left_right=PREFERRED_DISTANCE_LEFT_RIGHT)