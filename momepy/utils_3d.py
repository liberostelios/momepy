#!/usr/bin/env python

from shapely.geometry import MultiPolygon, Polygon
from shapely import Geometry
import mapbox_earcut as earcut
import numpy as np
import pyvista as pv

def surface_normal(poly):
    n = [0.0, 0.0, 0.0]

    for i, v_curr in enumerate(poly):
        v_next = poly[(i+1) % len(poly)]
        n[0] += (v_curr[1] - v_next[1]) * (v_curr[2] + v_next[2])
        n[1] += (v_curr[2] - v_next[2]) * (v_curr[0] + v_next[0])
        n[2] += (v_curr[0] - v_next[0]) * (v_curr[1] + v_next[1])

    if all([c == 0 for c in n]):
        raise ValueError("No normal. Possible colinear points!")

    normalised = [i/np.linalg.norm(n) for i in n]

    return normalised

def axes_of_normal(normal):
    """Returns an x-axis and y-axis on a plane of the given normal"""
    if normal[2] > 0.001 or normal[2] < -0.001:
        x_axis = [1, 0, -normal[0]/normal[2]]
    elif normal[1] > 0.001 or normal[1] < -0.001:
        x_axis = [1, -normal[0]/normal[1], 0]
    else:
        x_axis = [-normal[1] / normal[0], 1, 0]

    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(normal, x_axis)

    return x_axis, y_axis

def project_2d(points, normal, origin=None):
    if isinstance(points, list):
        points = np.array(points)
    if origin is None:
        origin = points[0]

    x_axis, y_axis = axes_of_normal(normal)

    return [[np.dot(p - origin, x_axis), np.dot(p - origin, y_axis)] for p in points]

def triangulate_polygon(polygon: Polygon):
    """Returns a triangulated MultiPolygon from a Polygon"""

    points = np.array([list(c) for c in polygon.exterior.coords] + [list(c) for interior in polygon.interiors for c in interior])
    normal = surface_normal(points)
    holes = [len(polygon.exterior.coords)]
    for ring in polygon.interiors:
        holes.append(len(ring.coords) + holes[-1])

    points_2d = project_2d(points, normal)

    result = earcut.triangulate_float32(points_2d, holes)

    return [Polygon(points[triangle]) for triangle in result.reshape(-1,3)]

def triangulate_geometry(geometry):
    if geometry.type == "Polygon":
        polygons = triangulate_polygon(geometry)
    else:
        polygons = []
        for poly in geometry.geoms:
            polygons.extend(triangulate_polygon(poly))
    
    return MultiPolygon(polygons)

def create_polydata(geometry: Geometry) -> pv.PolyData:
    if geometry.type != 'MultiPolygon' and geometry.type != 'Polygon':
        raise ValueError("Only (Multi)Polygon geometries allowed")
    
    triangulated_geom = triangulate_geometry(geometry)
    # Create an array with the vertices
    vertices = np.array([geom.exterior.coords[:3] for geom in triangulated_geom.geoms]).reshape(-1, 3)
    # Generate vertices to represent the triangles of the previous vertices
    faces = np.array(range(len(vertices))).reshape(-1, 3)
    # Add 3 (the size of face) at the beginning of every row
    faces = np.insert(faces, 0, 3, axis=1)
    # Flatten
    faces = np.hstack(faces)

    return pv.PolyData(vertices, faces)