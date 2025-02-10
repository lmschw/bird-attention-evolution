from animal_models.animal import Animal


class Bird(Animal):
    # TODO implement flapping frequency and energy level
    def __init__(self, name, speeds, wingspan, length, head_range_half, max_turn_angle,
                 focus_areas, preferred_distance_front_back, preferred_distance_left_right,
                 sensing_range):
        super().__init__(name=name,
                         speeds=speeds,
                         width=wingspan,
                         length=length,
                         head_range_half=head_range_half,
                         max_turn_angle=max_turn_angle,
                         focus_areas=focus_areas,
                         preferred_distance_front_back=preferred_distance_front_back,
                         preferred_distance_left_right=preferred_distance_left_right,
                         sensing_range=sensing_range)

    def set_path(self, path):
        self.path = path
