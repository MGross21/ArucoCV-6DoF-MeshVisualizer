import os
import cv2
import numpy as np
import json

class ArucoCalibrate(object):
    def __init__(self, chessboard_size=(9, 6), square_size=0.025, calibration_file="calibration.json"):
        self.chessboard_size = chessboard_size  # (width, height) in terms of squares
        self.square_size = square_size  # Size of a single square in meters

        script_dir = os.path.dirname(os.path.realpath(__file__))  # Get the directory of this script
        self.calibration_file = os.path.join(script_dir, calibration_file)  # Join with the current directory path


        # Prepare object points (3D points in real world space)
        self.objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        
        # Generate 2D coordinates in x and y, then assign z as 0
        self.objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)


        self.objpoints = []  # 3D points in world space
        self.imgpoints = []  # 2D points in image plane

        self.camera_matrix = None
        self.dist_coeffs = None

    def calibrate(self, video_source=1,max_images=20):
        """
        Perform camera calibration using a chessboard pattern.
        :param video_source: Index of the camera or video file.
        """
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print("Could not open camera.")
            return False


        captured_images = 0
        while captured_images < max_images:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)

            if ret:
                self.objpoints.append(self.objp)
                self.imgpoints.append(corners)

                # Draw and display the corners
                cv2.drawChessboardCorners(frame, self.chessboard_size, corners, ret)
                captured_images += 1  # Increment the counter for captured images

            cv2.putText(frame, f"Calibration: {captured_images}/{max_images}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) # Number of successfully detected chessboard patterns (top left)
            
            cv2.imshow("Chessboard Calibration", frame) # show camera frames

            
            cv2.waitKey(500) # delay to help with calibration
            
            # Exit if the user presses the ESC key
            if cv2.waitKey(1) & 0xFF == 27:
                cap.release()  # Release camera capture
                cv2.destroyAllWindows()  # Close OpenCV windows
                break

        # Perform camera calibration
        if len(self.objpoints) > 0 and len(self.imgpoints) > 0:
            ret, self.camera_matrix, self.dist_coeffs, _, _ = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1], None, None)
            
            if ret:
                print("Camera calibration successful!")
                self.save_calibration()
                return True
            else:
                print("Camera calibration failed.")
                return False
        else:
            print("Not enough points for calibration.")
            return False

    def save_calibration(self):
        """
        Save camera matrix and distortion coefficients to a JSON file.
        """
        calibration_data = {
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist()
        }

        with open(self.calibration_file, 'w') as f:
            json.dump(calibration_data, f)

        print(f"Calibration data saved to {self.calibration_file}")

    def load_calibration(self):
        """Load camera matrix and distortion coefficients from a JSON file"""
        if os.path.exists(self.calibration_file):
            try:
                with open(self.calibration_file, 'r') as f:
                    calibration_data = json.load(f)
                    self.camera_matrix = np.array(calibration_data["camera_matrix"], dtype=np.float32)
                    self.dist_coeffs = np.array(calibration_data["dist_coeffs"], dtype=np.float32)
                print(f"Loaded calibration data from {self.calibration_file}")
                return True
            except Exception as e:
                print(f"Error loading calibration file: {e}")
                return False
        else:
            print(f"No calibration file found at {self.calibration_file}")
            return False
