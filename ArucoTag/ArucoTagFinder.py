import cv2
import numpy as np
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

    def detect(self, image):
        """Detect ArUco markers in the image and return their corners and ids."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(image, corners, ids)
        return corners, ids

    def pose(self, corners):
        """Estimate the pose of detected ArUco markers."""
        if not corners:
            return None, None

        corners = [np.array(corner).reshape((1, 4, 2)) for corner in corners]
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_length, self.camera_matrix, self.dist_coeffs)

        return (rvecs[0], tvecs[0]) if rvecs is not None and tvecs is not None else (None, None)

    def _getRotationMatrix(self, meta):
        """Get rotation matrix from metadata."""
        if "rpy" in meta:
            return R.from_euler('xyz', meta["rpy"], degrees=True).as_matrix()
        elif "quat" in meta:
            return R.from_quat(meta["quat"]).as_matrix()
        else:
            return np.eye(3)  # Identity matrix if no rotation data

    def render_AR(self, image, corners, ids, metadata):
        """Render augmented reality visuals for detected ArUco markers."""
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
            poses[marker_id] = {"rvec": rvec.flatten(), "tvec": tvec.flatten()}

            # Draw axes
            cv2.drawFrameAxes(image,
                              self.camera_matrix,
                              self.dist_coeffs,
                              rvec,
                              tvec,
                              self.marker_length / 2)

        return image, poses