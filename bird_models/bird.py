import numpy as np

class Bird:
    # TODO implement flapping frequency and energy level
    def __init__(self, average_speed, wingspan, length, head_range_half,
                 focus_areas, preferred_distance_front_back, preferred_distance_left_right):
        self.average_speed = average_speed
        self.wingspan = wingspan
        self.length = length
        self.head_range_half = head_range_half
        self.focus_areas = focus_areas
        self.preferred_distance_front_back = preferred_distance_front_back
        self.preferred_distance_left_right = preferred_distance_left_right

    def set_landmarks(self, landmarks):
        self.landmarks = landmarks

    def set_random_subset_of_landmarks(self, landmarks, num_landmarks=None):
        if num_landmarks == None:
            num_landmarks = np.random.randint(1, len(landmarks))
        self.set_landmarks(np.random.choice(landmarks, size=num_landmarks, replace=False))