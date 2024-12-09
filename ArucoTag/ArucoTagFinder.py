import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R
from ArucoTag.MeshGeneration.MeshGeneration import MeshGeneration
from ArucoTag.ArucoCalibrate import ArucoCalibrate

class ArucoTagFinder(object):
    def __init__(self, dictionary=cv2.aruco.DICT_7X7_50, marker_length=0.05, calibration_file="calibration.json", metadata=None):
        """Initialize ArUco tag detection and mesh rendering."""
        # Initialize ArUco dictionary, parameters, and marker length
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary)
        self.parameters = cv2.aruco.DetectorParameters()
        self.marker_length = marker_length

        # Load camera calibration object
        self.calibrator = ArucoCalibrate(calibration_file=calibration_file)

        # Try to load existing calibration
        if not self.calibrator.load_calibration():
            print("Calibration file not found or invalid. Starting camera calibration process.")
            if self.calibrator.calibrate():
                self.calibrator.save_calibration()
            else:
                raise RuntimeError("Camera calibration failed. Please try again.")

        # Load camera matrix and distortion coefficients
        self.camera_matrix = self.calibrator.camera_matrix
        self.dist_coeffs = self.calibrator.dist_coeffs

        # Ensure calibration data is loaded
        if self.camera_matrix is None or self.dist_coeffs is None:
            raise ValueError("Camera calibration data not loaded properly")


        # Initialize mesh generation with metadata for tag mesh handling
        self.mesh_generator = MeshGeneration(metadata)
        if metadata:
            self.mesh_generator.add_quat()
        self.stl_cache = {}

    def detect(self, image):
        """Detect ArUco markers in the image and return their corners and ids."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(image, corners, ids)
        return corners, ids

    def pose(self, corners):
        """Estimate pose of single marker and return rotation and translation vectors"""
        if len(corners) == 0:
            return None, None

        corners = [np.array(corner).reshape((1, 4, 2)) for corner in corners]
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_length, self.camera_matrix, self.dist_coeffs)

        if rvecs is not None and tvecs is not None and rvecs.any() and tvecs.any():
            return rvecs[0], tvecs[0]
        else:
            return None, None
    
    def _getRotationMatrix(self, meta):
        """Get rotation matrix from metadata"""
        if "rpy" in meta:
            return R.from_euler('xyz', meta["rpy"], degrees=True).as_matrix()
        elif "quat" in meta:
            return R.from_quat(meta["quat"]).as_matrix()
        else:
            return np.eye(3)  # Identity matrix if no rotation data
        
    def _translation(self, tvec, rotation_matrix, meta):
        """Adjust the translation vector based on metadata and convert to pixel coordinates."""
        whd = np.array(meta.get("whd", [0, 0, 0]))[:3]  # Ensure 3D "whd" exists
        if whd.shape[0] == 3:
            # Apply rotation and translation in meters
            tvec_meters = tvec + rotation_matrix @ (whd / 2.0)
            
            # Convert tvec from meters to pixels
            tvec_pixels = self.meters_to_pixels(tvec_meters)
            return tvec_pixels
        return self.meters_to_pixels(tvec)  # Fallback if 'whd' doesn't have 3 elements

    def meters_to_pixels(self, point_meters):
        """Convert a point from meters to pixels using the camera matrix."""
        # Assuming point_meters is a 3D point [x, y, z] in meters
        x, y, z = point_meters
        
        # Extract focal length and principal point from camera matrix
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        # Convert to pixels
        x_pixel = (x * fx / z) + cx
        y_pixel = (y * fy / z) + cy
        
        return np.array([x_pixel, y_pixel, z])
    
    def _drawPosesAvg(self, image, poses):
        """Average poses and draw frame axes on the image."""
        for marker_id, pose_data in poses.items():
            if not pose_data["rvecs"] or not pose_data["tvecs"]:
                print(f"No pose data for marker {marker_id}")
                continue

            avg_rvec = np.mean(pose_data["rvecs"], axis=0)
            avg_tvec = np.mean(pose_data["tvecs"], axis=0)

            try:
                # Draw frame axes with averaged pose
                cv2.drawFrameAxes(image, self.camera_matrix, self.dist_coeffs, avg_rvec, avg_tvec, 0.1)  # Increased axis length for visibility
            except Exception as e:
                print(f"Error drawing frame axes for marker {marker_id}: {e}")
    
    def _project_points(self, points, rvec, tvec):
        points_2d, _ = cv2.projectPoints(points, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        return points_2d.reshape(-1, 2)

    def _get_corners_3d(self, meta):
        width, height, depth = meta.get('whd', [0, 0, 0])
        half_width, half_height, half_depth = width/2, height/2, depth/2
        return np.array([
            [-half_width, -half_height, -half_depth],
            [half_width, -half_height, -half_depth],
            [half_width, half_height, -half_depth],
            [-half_width, half_height, -half_depth],
            [-half_width, -half_height, half_depth],
            [half_width, -half_height, half_depth],
            [half_width, half_height, half_depth],
            [-half_width, half_height, half_depth]
        ])

    def _draw_wireframe(self, image, corners_2d, color):
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        for start, end in edges:
            cv2.line(image, tuple(corners_2d[start].astype(int)), tuple(corners_2d[end].astype(int)), color, 2)

    def _render_stl(self, image, meta, rotation_matrix, rvec, tvec):
        stl_file = meta.get("stl_file")
        if not stl_file:
            return

        if stl_file not in self.stl_cache:
            try:
                self.stl_cache[stl_file] = trimesh.load_mesh(stl_file)
            except Exception as e:
                print(f"Error loading STL file {stl_file}: {e}")
                return

        mesh = self.stl_cache[stl_file]
        scale = meta.get("scale", 1.0)
        vertices = np.dot(mesh.vertices * scale, rotation_matrix.T) + tvec
        projected_vertices = self._project_points(vertices, rvec, tvec)

        color = meta.get("color", (0, 255, 0))

        # Create a Viz3d window for 3D visualization
        viz_window = cv2.viz.Viz3d('3D Mesh')
        
        # Create a mesh for Viz
        mesh_viz = cv2.viz.Mesh()
        
        # Add vertices and edges to the Viz mesh
        for edge in mesh.edges:
            pt1 = vertices[edge[0]]
            pt2 = vertices[edge[1]]
            line = cv2.viz.Line(pt1, pt2, color=color)
            mesh_viz.addLine(line)

        # Add the mesh to the Viz window
        viz_window.showWidget('Mesh', mesh_viz)

        # Display the image with OpenCV
        for edge in mesh.edges:
            pt1, pt2 = projected_vertices[edge]
            cv2.line(image, tuple(pt1.astype(int)), tuple(pt2.astype(int)), color, 1)

        # Show the Viz window
        viz_window.spinOnce(1, True)  # Spin once to update the visualization
    
    
    def render_AR(self, image, corners, ids, metadata):
        if ids is None or corners is None:
            return image, {}

        poses = {}
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id not in metadata:
                continue

            meta = metadata[marker_id]
            rvec, tvec = self.pose([corners[i]])
            if rvec is None or tvec is None:
                continue

            rotation_matrix = self._getRotationMatrix(meta)
            adjusted_tvec = self._translation(tvec.flatten()[:3], rotation_matrix, meta)
            adjusted_rvec, _ = cv2.Rodrigues(rotation_matrix)

            poses[marker_id] = {"rvec": adjusted_rvec.flatten(), "tvec": adjusted_tvec}

            # Draw axis and center
            cv2.drawFrameAxes(image, self.camera_matrix, self.dist_coeffs, adjusted_rvec, adjusted_tvec, self.marker_length/2)
            center_pixel = self._project_points(adjusted_tvec.reshape(1, 3), adjusted_rvec, adjusted_tvec)
            cv2.circle(image, tuple(center_pixel[0].astype(int)), 5, (0, 255, 0), -1)

            # Draw rectangular wireframe
            corners_3d = self._get_corners_3d(meta)
            corners_world = np.dot(corners_3d, rotation_matrix.T) + adjusted_tvec
            corners_2d = self._project_points(corners_world, adjusted_rvec, adjusted_tvec)
            self._draw_wireframe(image, corners_2d, meta.get("color", (0, 255, 0)))

            # Render STL if available
            self._render_stl(image, meta, rotation_matrix, adjusted_rvec, adjusted_tvec)

        return image, poses