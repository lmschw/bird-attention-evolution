import numpy as np
import general.occlusion as occ
from animal_models.pigeon import Pigeon
from area_models.landmark import Landmark

agents = np.array([[1,1,1*np.pi],
                    [1,2,1*np.pi],
                    [1,3,1*np.pi],
                    [20,20,1*np.pi]])

animal_type = Pigeon()
landmark_1 = Landmark('1', corners=[[0, 1.5], [2, 1.5], [2, 1.3], [0, 1.3]])


landmarks = [landmark_1]
#distances, angles = occ.compute_distances_and_angles(pos_xs_center=agents[:,0], pos_ys_center=agents[:,1], pos_xs_end=agents[:,0], pos_ys_end=agents[:,1], headings=agents[:,2])

print(occ.compute_not_occluded_mask(agents=agents, animal_type=animal_type))
print(occ.compute_not_occluded_mask_landmarks(agents=agents, animal_type=animal_type, landmarks=landmarks))
"""
angle_ranges = np.array([
    [  # Set 1
        [np.radians(10), np.radians(50)],
        [np.radians(30), np.radians(70)],
        [np.radians(100), np.radians(150)],
        [np.radians(340), np.radians(20)]
    ],
    [  # Set 2
        [np.radians(20), np.radians(60)],
        [np.radians(50), np.radians(90)],
        [np.radians(200), np.radians(250)],
        [np.radians(0), np.radians(30)]
    ]
])

overlap_matrix = occ.compute_overlap(angle_ranges)
print(np.round(overlap_matrix, 2))"
"""

"""
positions = np.array([
    [0, 0],
    [2, 0],
    [4, 0],
    [2, 1],
    [2, -1],
])

orientations = np.array([
    0,        # Facing right
    0,        # Facing right
    np.pi,    # Facing left
    -np.pi/2, # Facing down
    np.pi/2   # Facing up
])

visible = occ.get_visible_agents(positions, orientations, animal_type)
for i, vis in enumerate(visible):
    print(f"Agent {i} can see agents: {vis}")"
"""