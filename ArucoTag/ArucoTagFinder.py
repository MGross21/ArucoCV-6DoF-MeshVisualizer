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
        if not corners:
            return None, None
        
        corners = [np.array(corner).reshape((1, 4, 2)) for corner in corners]
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_length, self.camera_matrix, self.dist_coeffs)
        
        return (rvecs[0], tvecs[0]) if rvecs is not None and tvecs is not None else (None, None)
    
    def _getRotationMatrix(self, meta):
        """Get rotation matrix from metadata"""
        if "rpy" in meta:
            return R.from_euler('xyz', meta["rpy"], degrees=True).as_matrix()
        elif "quat" in meta:
            return R.from_quat(meta["quat"]).as_matrix()
        else:
            return np.eye(3)  # Identity matrix if no rotation data
        
    def _translation(self, tvec, rotation_matrix):
        """Convert translation vector from pixels to meters."""
        # Convert pixel coordinates back to meters
        z = tvec[2]  # Use the z value for depth
        x_meters = (tvec[0] - self.camera_matrix[0, 2]) * z / self.camera_matrix[0, 0]
        y_meters = (tvec[1] - self.camera_matrix[1, 2]) * z / self.camera_matrix[1, 1]
        
        return np.array([x_meters, y_meters, z])  # Return in meters

    # def meters_to_pixels(self, point_meters):
    #     """Convert a point from meters to pixels using the camera matrix."""
    #     # Assuming point_meters is a 3D point [x, y, z] in meters
    #     x, y, z = point_meters
        
    #     # Extract focal length and principal point from camera matrix
    #     fx = self.camera_matrix[0, 0]
    #     fy = self.camera_matrix[1, 1]
    #     cx = self.camera_matrix[0, 2]
    #     cy = self.camera_matrix[1, 2]
        
    #     # Convert to pixels
    #     x_pixel = (x * fx / z) + cx
    #     y_pixel = (y * fy / z) + cy
        
    #     return np.array([x_pixel, y_pixel, z])
    
    def _project_points(self, points, rvec, tvec):
        points_2d, _ = cv2.projectPoints(points, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        return points_2d.reshape(-1, 2)

    def _get_corners_3d(self, meta):
        width, height, depth = meta.get('whd', [0, 0, 0])
        half_width, half_height, half_depth = width/2, height/2, depth/2

        # 8 points of rectangle/square
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
        
        for edge in mesh.edges:
            pt1, pt2 = projected_vertices[edge]
            cv2.line(image, tuple(pt1.astype(int)), tuple(pt2.astype(int)), color, 1)
        
    
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
            adjusted_tvec = self._translation(tvec.flatten(), rotation_matrix)
            adjusted_rvec = rvec  # Use the original rvec directly

            poses[marker_id] = {"rvec": adjusted_rvec.flatten(), "tvec": adjusted_tvec}

            # Draw axes and other visual elements...
            cv2.drawFrameAxes(image,
                            self.camera_matrix,
                            self.dist_coeffs,
                            adjusted_rvec,
                            adjusted_tvec,
                            self.marker_length / 2)

            center_pixel = self._project_points(adjusted_tvec.reshape(1, 3),
                                                adjusted_rvec,
                                                adjusted_tvec)
            
            cv2.circle(image,
                    tuple(center_pixel[0].astype(int)),
                    radius=5,
                    color=(0, 255, 0),
                    thickness=-1)
            
            corners_3d = self._get_corners_3d(meta)
            corners_world = np.dot(corners_3d, rotation_matrix.T) + adjusted_tvec
            corners_2d = self._project_points(corners_world, adjusted_rvec, adjusted_tvec)
            
            self._draw_wireframe(image, corners_2d, meta.get("color", (0, 255, 0)))
            self._render_stl(image, meta, rotation_matrix, adjusted_rvec, adjusted_tvec)
        
        return image, poses