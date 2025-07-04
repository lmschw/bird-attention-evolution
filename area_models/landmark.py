import matplotlib.patches as mpatches
from shapely import centroid
from shapely.geometry import Polygon
from shapely.plotting import patch_from_polygon

"""
Represents a landmark, e.g. a tree, house etc.. 

Attributes:
id          -   string                                                          -   the ID of the landmark, mostly intended for printing, displaying and saving
corners     -   list of lists representing the corner points of the polygon     -   defines the corners of the polygon representing the shape of the landmark
"""
    
class Landmark:
    def __init__(self, id, corners=[], polygon=None):
        self.id = id
        self.corners = corners
        if polygon:
            self.polygon = polygon
        else:
            self.polygon = Polygon(self.corners)

    def get_geometry(self):
        return self.polygon
    
    def get_annotation_point(self):
        point = centroid(self.polygon)
        return [point.x, point.y]
    
    def get_patch_for_display(self):
        return patch_from_polygon(self.polygon)