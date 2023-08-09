import pytest
import geopandas as gpd
import pyvista as pv
from contextlib import suppress

import momepy as mm

class TestUtils3d:
    def test_create_polydata(self):
        path = mm.datasets.get_path("3dbag")
        buildings = gpd.read_file(path, layer="lod22_3d")

        buildings['mesh'] = mm.shape_3d.Mesh(buildings).series

        assert len(buildings[buildings['mesh'].apply(lambda f: f.volume) > 0]) == 1107
