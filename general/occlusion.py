import numpy as np
from shapely.geometry import LineString

def compute_occluded_mask(agents, animal_type):
    return np.logical_not(compute_not_occluded_mask(agents, animal_type))

def compute_not_occluded_mask(agents, animal_type):
    positions = np.column_stack((agents[:,0], agents[:,1]))
    indices = get_visible_agents(positions, agents[:,2], animal_type)
    mask = np.full((len(agents), len(agents)), False)
    for i in range(len(agents)):
        mask[i][indices[i]] = True
    np.fill_diagonal(mask, True)
    return mask

def compute_occluded_mask_landmarks(agents, animal_type, landmarks):
    return np.logical_not(compute_not_occluded_mask_landmarks(agents, animal_type, landmarks))

def compute_not_occluded_mask_landmarks(agents, animal_type, landmarks):
    positions = np.column_stack((agents[:,0], agents[:,1]))
    indices = get_visible_agents_with_landmarks(positions, agents[:,2], landmarks, animal_type)
    mask = np.full((len(agents), len(agents)), False)
    for i in range(len(agents)):
        mask[i][indices[i]] = True
    np.fill_diagonal(mask, True)
    return mask


def get_visible_agents(positions, orientations, animal_type, fov=2*np.pi):
    """
    Determine which agents are visible (not occluded) from each agent's perspective.
    
    Args:
        positions: (n, 2) array of x, y positions
        orientations: (n,) array of orientations in radians
        animal_type: Animal
        fov: Field of view in radians (default 2*pi)
    
    Returns:
        visibility: list of lists, where visibility[i] contains indices of agents visible to agent i
    """
    n = positions.shape[0]
    visibility = []
    view_distance = animal_type.sensing_range
    agent_radius = animal_type.length

    for i in range(n):
        pos_i = positions[i]
        orient_i = orientations[i]

        # Vector from i to all other agents
        rel_pos = positions - pos_i  # (n, 2)
        distances = np.linalg.norm(rel_pos, axis=1)
        directions = rel_pos / np.clip(distances[:, None], 1e-8, None)

        # Angle between agent i's orientation and the direction to other agents
        forward = np.array([np.cos(orient_i), np.sin(orient_i)])
        cos_angles = directions @ forward  # dot product
        angles = np.arccos(np.clip(cos_angles, -1, 1))

        # Field of view mask
        in_fov = (angles <= fov / 2) & (distances > 0) & (distances <= view_distance)

        # Filter agents in field of view
        candidates = np.where(in_fov)[0]
        candidate_distances = distances[candidates]
        sorted_indices = candidates[np.argsort(candidate_distances)]

        visible = []
        occluded_mask = np.zeros(n, dtype=bool)

        for j in sorted_indices:
            if not occluded_mask[j]:
                visible.append(j)
                # Occlude any agents behind this one, close to the same direction
                rel_vec = positions[j] - pos_i
                rel_dir = rel_vec / np.linalg.norm(rel_vec)
                dot_prods = (positions - pos_i) @ rel_dir
                proj_lens = np.abs(np.cross(positions - pos_i, rel_dir))
                occluded = (dot_prods > distances[j]) & (proj_lens < agent_radius)
                occluded_mask |= occluded

        visibility.append(visible)

    return visibility

def get_visible_agents_with_landmarks(positions, orientations, landmarks, animal_type, fov=2*np.pi):
    """
    Determines visible agents from each other, accounting for polygon landmarks (occlusions).

    Args:
        positions: (n, 2) array of x, y positions
        orientations: (n,) array of orientations in radians
        landmarks: list of shapely.geometry.Polygon objects
        fov: Field of view in radians
        view_distance: max view distance

    Returns:
        visibility: list of lists, where visibility[i] contains indices of agents visible to agent i
    """
    n = positions.shape[0]
    visibility = []
    view_distance = animal_type.sensing_range
    landmarks = [landmark.polygon for landmark in landmarks]

    for i in range(n):
        pos_i = positions[i]
        orient_i = orientations[i]
        forward = np.array([np.cos(orient_i), np.sin(orient_i)])

        visible = []

        for j in range(n):
            if i == j:
                continue

            pos_j = positions[j]
            rel_vec = pos_j - pos_i
            distance = np.linalg.norm(rel_vec)
            if distance > view_distance:
                continue

            # Angle between forward and target
            rel_dir = rel_vec / (distance + 1e-8)
            angle = np.arccos(np.clip(np.dot(forward, rel_dir), -1, 1))

            if angle > fov / 2:
                continue  # Outside FOV

            # Check for line-of-sight occlusion
            line = LineString([pos_i, pos_j])
            occluded = any(poly.intersects(line) for poly in landmarks)
            if not occluded:
                visible.append(j)

        visibility.append(visible)

    return visibility