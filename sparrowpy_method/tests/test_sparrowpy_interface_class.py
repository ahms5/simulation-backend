
from sparrowpy_interface import sparrowpyMethod
import json
import sparrowpy
import numpy.testing as npt
import numpy as np

from sparrowpy_interface.sparrowpy_interface import _import_room_geometry


def test_simple_method(create_temporary_input_file):
    
    sparrowpy_method_object = sparrowpyMethod(create_temporary_input_file)
    sparrowpy_method_object.run_simulation()

    with open(create_temporary_input_file, 'r') as f:
        data = json.load(f)
    
    assert "receiverResults" in data['results'][0]['responses'][0]
    results = data['results'][0]['responses'][0]['receiverResults']
    assert results is not None
    assert len(results) > 0


def test_import_room_geometry(create_temporary_input_file):
    (
        walls_points, walls_normal, walls_up_vector,
        patches_points, n_patches, patch_to_wall_ids,
        material_to_walls, alphas, scattering,
        ) = _import_room_geometry(
            create_temporary_input_file, patch_length=5)
    
    # create radiosity object
    radiosity = sparrowpy.DirectionalRadiosityFast(
        walls_points,
        walls_normal,
        walls_up_vector,
        patches_points,
        n_patches,
        patch_to_wall_ids,
        )

    # check geometry stuff
    radiosity.check()

    # check material stuff 
    npt.assert_equal(np.squeeze(material_to_walls), np.arange(6))
    npt.assert_equal(np.array(alphas).shape, (6, 6))
    npt.assert_equal(np.array(scattering).shape, (6, 6))
    npt.assert_array_less(np.array(alphas), 1)
    npt.assert_array_less(np.array(scattering), 1)
