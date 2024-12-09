import trimesh
import os
from scipy.spatial.transform import Rotation as R

class MeshGeneration(object):
    def __init__(self, metadata):
        self.metadata = metadata
        self.cwd = './ArucoTag/MeshGeneration/STL_files/'

    def add_quat(self):
        # Convert rpy to quaternion for each entry in metadata
        for _, value in self.metadata.items():
            value['quat'] = R.from_euler('xyz', value['rpy'], degrees=True).as_quat().astype(tuple)  # (x, y, z, w)

    def create_box(self):
        # Create STL files for boxes based on metadata
        for key, value in self.metadata.items():
            if value['visibility'] and value["Name"] == "Box":
                os.makedirs(self.cwd, exist_ok=True)  # Ensure the directory exists
                filename = f"{self.cwd}{value['Name']}_ID{key}.stl"  # Unique file name for each box

                if not os.path.exists(filename) and value["stl_file"] is None:
                    box = trimesh.creation.box(extents=value["whd"])  # Create a box mesh
                    box.export(filename)  # Export to STL
                    value["stl_file"] = filename  # Store STL file path in metadata
                    print(f"Created STL file: {filename}")
                else:
                    print(f"STL file {filename} already exists or is already linked.")
    
    def load_mesh(self, marker_id):
        """Load and return the mesh for the given marker ID."""
        if marker_id in self.metadata:
            stl_file = self.metadata[marker_id].get("stl_file")
            if stl_file:
                return trimesh.load_mesh(stl_file)
        return None
