"""Module that computes indexes for shapely (2D) and polydata (3D) shapes"""

import math
from shapely.geometry import Point, MultiPoint, Polygon
from momepy.utils_3d import surface_normal
import numpy as np
import pyvista as pv
from .utils_3d import create_polydata

import momepy as mm

def circularity(shape):
    """Returns circularity 2D for a given polygon"""

    return 4 * math.pi * shape.area / math.pow(shape.length, 2)

class Mesh:
    def __init__(self, gdf):
        self.gdf = gdf

        gdf = gdf.copy()

        def get_polydata(feature):
            try:
                return mm.utils_3d.create_polydata(feature['geometry'])
            except:
                return pv.PolyData()

        self.series = gdf.apply(get_polydata, axis=1)

class Hemisphericality:
    def __init__(self, gdf):
        self.gdf = gdf

        gdf = gdf.copy()
        gdf['mesh'] = gdf.apply(lambda f: create_polydata(f['geometry']), axis=1)
