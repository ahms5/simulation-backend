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

        # Read source and receiver positions
        source_coords = pf.Coordinates(
            result_container["results"][0]["sourceX"],
            result_container["results"][0]["sourceY"],
            result_container["results"][0]["sourceZ"],
        )
        receiver_coords =  pf.Coordinates(
            result_container["results"][0]["responses"][0]["x"],
            result_container["results"][0]["responses"][0]["y"],
            result_container["results"][0]["responses"][0]["z"],
        )

        # read walls and triangular patches
        (
            walls_points, walls_normal, walls_up_vector,
            patches_points, n_patches, patch_to_wall_ids,
            material_to_walls, alphas, scattering,
            ) = _import_room_geometry(json_file_path)
    
        radiosity = sparrowpy.DirectionalRadiosityFast(
            walls_points,
            walls_normal,
            walls_up_vector,
            patches_points,
            n_patches,
            patch_to_wall_ids,
            )
        
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
        radiosity.bake_geometry()

        radiosity.init_source_energy(source_coords)

        radiosity.calculate_energy_exchange(
            speed_of_sound=speed_of_sound,
            etc_time_resolution=etc_time_resolution_s,
            etc_duration=etc_duration_s,
            max_reflection_order=max_reflection_order)
        

        etc_radiosity = radiosity.collect_energy_receiver_mono(
            receivers=receiver_coords)

        # Write results back to JSON
        for i_frequency in range(n_bands):
            result_container["results"][0]["responses"][0]["receiverResults"].append(
                {
                    "data": etc_radiosity.time[i_frequency].tolist(),
                    "t": etc_radiosity.times,
                    "frequency": frequencies[i_frequency],
                    "type": "edc",
                }
            )
        result_container["results"][0]["percentage"] = 100

        # Save the updated JSON
        with open(json_file_path, "w") as json_output:
            json_output.write(json.dumps(result_container, indent=4))

        print("sparrowpy simulation completed successfully!")


def _import_room_geometry(json_file_path):
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
        for s_name in surface_group_names:
            if material_name == s_name:
                indies_material.append(s_name)
        material_to_walls.append(indies_material)


    # get the element type for surface mesh
    element_type = gmsh.model.mesh.getElementType("Triangle", 1, True)


    alphas = []
    walls_points = []
    walls_normal = []
    walls_up_vector = []
    patches_points = []
    n_patches = 0
    patch_to_wall_ids = []
    material_to_walls = []
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
        wall_normal = np.median(mesh.face_normals, axis=0)
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
