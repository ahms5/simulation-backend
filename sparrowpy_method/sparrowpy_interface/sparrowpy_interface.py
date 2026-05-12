"""Module implementing a CHORAS interface for sparrowpy.
"""
import json
from pathlib import Path

from .definition import SimulationMethod
import sparrowpy
import gmsh
import pyfar as pf
import numpy as np
import trimesh
import pyrato


class sparrowpyMethod(SimulationMethod):
    """Interface class to run the sparrowpy method.

    The class implements method to run the calculations for the
    sparrowpy simulation method. All required configuration parameters
    are expected to be provided in the input JSON file passed during
    initialization.

    """

    def __init__(self, input_json_path: str | Path | None = None):
        """Initialize the sparrowpy method interface for the given JSON file."""
        super().__init__(input_json_path)

    def run_simulation(self) -> None:
        """Run the simulation.

        Parameters
        ----------
        json_file_path : str | Path | None, optional
            Path to the JSON file. If not provided, uses the path from initialization.
        """
        self._sparrowpy_method(self.input_json_path)

    def _sparrowpy_method(self, json_file_path: str | Path) -> None:
        """
        Run sparrowpy simulation for acoustic wave propagation.

        Args:
            json_file_path: Path to the JSON configuration file
        """
        print('extract simulation settings...')
        # Load the input JSON file
        with open(json_file_path, "r") as json_file:
            result_container = json.load(json_file)

        # extract simulation settings
        frequencies = result_container['results'][0]['frequencies']
        n_bands = len(frequencies)
        simulation_settings = result_container["simulationSettings"]
        etc_time_resolution_s = simulation_settings['etc_time_resolution_s']
        speed_of_sound = simulation_settings['speed_of_sound']
        etc_duration_s = simulation_settings['etc_duration_s']
        max_reflection_order = simulation_settings['max_reflection_order']
        patch_length = simulation_settings['patch_length']
        
        # Read source and receiver positions
        source_coords = pf.Coordinates(
            result_container["results"][0]["sourceX"],
            result_container["results"][0]["sourceY"],
            result_container["results"][0]["sourceZ"],
        )
        n_receivers = len(result_container["results"][0]["responses"])
        receiver_coords = pf.Coordinates(np.zeros((n_receivers)), 0, 0)
        cart = receiver_coords.cartesian
        for i_rec in range(n_receivers):
            rec = result_container["results"][0]["responses"][i_rec]
            cart[i_rec, 0] = rec["x"]
            cart[i_rec, 1] = rec["y"]
            cart[i_rec, 2] = rec["z"]
        receiver_coords.cartesian = cart

        print('extract geometry...')
        set_progress_and_save(5, result_container, json_file_path)
        # read walls and triangular patches
        (
            walls_points, walls_normal, walls_up_vector,
            patches_points, n_patches, patch_to_wall_ids,
            material_to_walls, alphas, scattering,
            ) = _import_room_geometry(json_file_path, patch_length)
    
        radiosity = sparrowpy.DirectionalRadiosityFast(
            walls_points,
            walls_normal,
            walls_up_vector,
            patches_points,
            n_patches,
            patch_to_wall_ids,
            )
        
        print('set materials...')
        set_progress_and_save(10, result_container, json_file_path)
        # apply materials
        incoming = pf.Coordinates(0, 0, 1, weights=1)
        outgoing = pf.Coordinates(0, 0, 1, weights=1)
        for ii, jj in enumerate(material_to_walls):
            brdf = sparrowpy.brdf.create_from_scattering(
                incoming, outgoing,
                pf.FrequencyData(scattering[ii], frequencies),
                pf.FrequencyData(alphas[ii], frequencies),
                )
            radiosity.set_wall_brdf(jj, brdf, incoming, outgoing)

        # run simulation
        print('bake geometry...')
        set_progress_and_save(15, result_container, json_file_path)
        radiosity.bake_geometry()

        print('initialize source...')
        set_progress_and_save(40, result_container, json_file_path)
        radiosity.init_source_energy(source_coords)

        print('compute energy exchange...')
        set_progress_and_save(65, result_container, json_file_path)
        radiosity.calculate_energy_exchange(
            speed_of_sound=speed_of_sound,
            etc_time_resolution=etc_time_resolution_s,
            etc_duration=etc_duration_s,
            max_reflection_order=max_reflection_order)

        print('collect energy at receiver...')
        set_progress_and_save(90, result_container, json_file_path)
        etc_radiosity = radiosity.collect_energy_receiver_mono(
            receivers=receiver_coords, direct_sound=True)

        print('calculating room parameters and writing results...')
        set_progress_and_save(95, result_container, json_file_path)
        # Write results back to JSON
        dynamic_range_db = 100
        result_db = 10*np.log10(etc_radiosity.time/1e-12)
        limit = np.max(result_db) - dynamic_range_db
        result_db[result_db<limit] = limit
        for i_rec in range(n_receivers):
            edc = pyrato.edc.schroeder_integration(etc_radiosity[i_rec, :], is_energy=True)
            edc_db = 10*np.log10(edc.time)
            limit = np.max(edc_db) - dynamic_range_db
            edc_db[edc_db<limit] = limit
            for i_frequency in range(n_bands):
                result_container["results"][0]["responses"][i_rec]["receiverResults"].append(
                    {
                        "data": edc_db[i_frequency].tolist(),
                        "t": etc_radiosity.times.tolist(),
                        "frequency": frequencies[i_frequency],
                        "type": "edc",
                    }
                )
            t20 = pyrato.parameters.reverberation_time_linear_regression(edc, 'T20')
            result_container["results"][0]["responses"][i_rec]["parameters"]['t20'] = t20.tolist()

            t30 = pyrato.parameters.reverberation_time_linear_regression(edc, 'T30')
            result_container["results"][0]["responses"][i_rec]["parameters"]['t30'] = t30.tolist()

            c80 = pyrato.parameters.clarity(edc, 80)
            result_container["results"][0]["responses"][i_rec]["parameters"]['c80'] = c80.tolist()

            d50 = pyrato.parameters.definition(edc, 50)
            result_container["results"][0]["responses"][i_rec]["parameters"]['d50'] = d50.tolist()

            ts = center_time(edc) # TODO replace by pyrato 1.1.0 version
            result_container["results"][0]["responses"][i_rec]["parameters"]['ts'] = ts.tolist()

            spl = 10*np.log10(edc.time[..., 0]/1e-12)
            result_container["results"][0]["responses"][i_rec]["parameters"]['spl_t0_freq'] = spl.tolist()
        
        # Save the updated JSON
        set_progress_and_save(100, result_container, json_file_path)

        print("sparrowpy simulation completed successfully!")

def set_progress_and_save(percentage, result_container, json_file_path):
    result_container["results"][0]["percentage"] = percentage
    # Save the updated JSON
    with open(json_file_path, "w") as json_output:
        json_output.write(json.dumps(result_container, indent=4))


def _import_room_geometry(json_file_path, patch_length):
    """Import room geometry and absorption coefficients.

    The geometry is read from a .geo file specified in the JSON input file.
    The absorption coefficients are directly read from the JSON file.

    Parameters
    ----------
    json_file_path : str
        Path to the JSON file containing room geometry and absorption
        coefficients.


    Raises
    ------
    ValueError
        If absorption coefficients for any surface are not found in the
        input JSON file.
    """

    with open(json_file_path, 'r') as f:
        import json
        input_data = json.load(f)

    frequencies = input_data['results'][0]['frequencies']
    n_bands = len(frequencies)

    # initialize gmsh and load the geometry file
    gmsh.initialize()
    geometry_file = input_data['geo_path']
    gmsh.open(geometry_file)

    # Read the content of the Geo file
    with open(geometry_file, 'r') as file:
        geo_content = file.readlines()

    # If an lc is given in the geo file, we want to compensate for this
    lc_value = 1 # set to 1 by default
    for line in geo_content:
        if "lc =" in line:
            lc_value = float(line.split('=')[1].strip().strip(';'))
            print("Extracted value:", lc_value)
            break
    
    gmsh.option.setNumber('Mesh.MeshSizeFactor', patch_length/lc_value)

    # generate 2d surface mesh
    dim = 2 # 2D surfaces
    gmsh.model.mesh.generate(dim)

    # get all named surfaces in the geometry
    surface_group_tags = gmsh.model.getPhysicalGroups(dim=dim)
    surface_group_names = [
        gmsh.model.getPhysicalName(dim, tag)
        for (dim, tag) in surface_group_tags
    ]

    # get all nodes of the surface mesh
    node_tags_all, coords_all, _ = gmsh.model.mesh.getNodes()
    coords = coords_all.reshape((len(node_tags_all), 3))

    # get the material names from absorption coefficient input
    absorption_names = list(input_data['absorption_coefficients'].keys())

    # check if absorption coefficient data are available for all surfaces
    for material_name in surface_group_names:
        if material_name not in absorption_names:
            raise ValueError(
                "Absorption coefficients for surface "
                f"'{material_name}' not found in input JSON file.")

    # create materials
    alphas = []
    scatterings = []
    material_to_walls = []
    for material_name in absorption_names:
        alphas.append(np.array(input_data['absorption_coefficients'][material_name]))
        scatterings.append(np.ones_like(frequencies))

        # materials
        indies_material  = []
        for ii, s_name in enumerate(surface_group_names):
            if material_name == s_name:
                indies_material.append(ii)
        material_to_walls.append(indies_material)


    # get the element type for surface mesh
    element_type = gmsh.model.mesh.getElementType("Triangle", 1, True)
    
    room_center = np.mean(coords, axis=0)

    alphas = []
    walls_points = []
    walls_normal = []
    walls_up_vector = []
    patches_points = []
    n_patches = 0
    patch_to_wall_ids = []
    for i, surface_name in enumerate(surface_group_names):
        dim_tags = gmsh.model.getEntitiesForPhysicalName(surface_name)
        dim, tag = dim_tags[0]

        face_nodes = gmsh.model.mesh.getElementFaceNodes(
            element_type, 3, tag=tag, )
        faces = np.reshape(face_nodes, (len(face_nodes) // 3, 3))

        # extract wall information
        mesh = trimesh.Trimesh(coords, faces-1)
        wall_points = np.unique(mesh.bounding_box.vertices, axis=0, )
        wall_idx = []
        for p in wall_points:
            wall_idx.append(np.argmin(np.sum(np.abs((mesh.vertices-p)), axis=1)))
        wall_points = np.unique(mesh.vertices[wall_idx], axis=0, ) 

        # flip normals to the center
        wall_normal = np.median(mesh.face_normals, axis=0)
        normal_dimension_mask = np.abs(wall_normal)>1e-3
        surface_center = np.mean(wall_points, axis=0)
        pointing_inwards = np.sign((room_center-surface_center)[normal_dimension_mask]) == np.sign(wall_normal[normal_dimension_mask])
        if not np.all(pointing_inwards):
            wall_normal *= -1
        elif not np.any(pointing_inwards):
            raise ValueError('Flipping normals inwards did not work.')

        # calculate wall up vector
        if np.abs(wall_normal[2]) > 1e-2:
            wall_up_vector = [1, 0, 0] 
        else:
            wall_up_vector = [0, 0, 1] 
        
        walls_points.append(wall_points)
        walls_normal.append(wall_normal)
        walls_up_vector.append(wall_up_vector)

        # write patches
        n_patches_wall = faces.shape[0]
        for jj in range(n_patches_wall):
            patch_to_wall_ids.append(i)
        n_patches += n_patches_wall
        patches_points.append(coords[faces-1, :])

        alpha = np.array(input_data['absorption_coefficients'][surface_name].split(', '), dtype=float)
        alphas.append(alpha)

    # finalizing gmsh
    gmsh.finalize()

    # save wall information
    walls_points = np.array(walls_points)
    walls_normal = np.array(walls_normal)
    walls_up_vector = np.array(walls_up_vector)
    patches_points = np.concatenate(patches_points)

    scattering = np.zeros_like(alphas)

    return (
        walls_points, walls_normal, walls_up_vector,
        patches_points, n_patches, patch_to_wall_ids,
        material_to_walls, alphas, scattering)


# copy pasted from pyrato
def center_time(energy_decay_curve):
    r"""
    Calculate the room-acoustic center time (:math:`T_s`).

    The center time :math:`T_s` is the time of the centroid of the squared
    impulse response. It quantifies the balance between early and late
    sound energy [#isoTs]_.

    The parameter is defined as

    .. math::

        T_s =
        \frac{
            \displaystyle \int_{0}^{\infty} t \cdot p^2(t)\,\mathrm{d}t
        }{
            \displaystyle \int_{0}^{\infty} p^2(t)\,\mathrm{d}t
        }

    where :math:`p(t)` is the room impulse response sound pressure.

    Using the energy decay curve :math:`e(t)`, the parameter can be
    computed efficiently via the EDC identity as

    .. math::

        T_s =
        \frac{
            \displaystyle \int_{0}^{\infty} e(t)\,\mathrm{d}t
        }{
            e(0)
        }.

    Parameters
    ----------
    energy_decay_curve : pyfar.TimeData
        Energy decay curve of the room impulse response. The EDC must
        start at time zero and must have equal time spacing.

    Returns
    -------
    center_time : numpy.ndarray
        Center time (:math:`T_s`) in seconds,
        shaped according to the channel shape of the input EDC.

    References
    ----------
    .. [#isoTs] ISO 3382, Acoustics — Measurement of the reverberation
        time of rooms with reference to other acoustical parameters.
    """

    if not isinstance(energy_decay_curve, pf.TimeData):
        raise TypeError(
            "energy_decay_curve must be a pyfar.TimeData or derived object.")

    if not np.isclose(energy_decay_curve.times[0], 0.0):
        raise ValueError("energy_decay_curve must start at time zero.")

    if np.any(energy_decay_curve.time[..., 0] == 0):
        raise ValueError(
            "Initial energy of energy_decay_curve must not be zero.")

    dt = np.diff(energy_decay_curve.times)
    if not np.allclose(dt, dt[0]):
        raise ValueError(
            "energy_decay_curve must have equal time spacing.")

    sampling_interval = dt[0]
    initial_energy = energy_decay_curve.time[..., 0]
    center_time = (
        np.nansum(energy_decay_curve.time, axis=-1)
        * sampling_interval
        / initial_energy
    )

    return center_time