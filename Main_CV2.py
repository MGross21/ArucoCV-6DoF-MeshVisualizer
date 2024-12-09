import cv2
import pandas as pd
from ArucoTag.ArucoTagFinder import ArucoTagFinder
from ArucoTag.MeshGeneration.MeshGeneration import MeshGeneration
import os
import numpy as np

if __name__ == "__main__":
    # Load metadata and initialize mesh files
    metadata = pd.read_json('./ArucoTag/MeshGeneration/tag_MetaData.json').to_dict()
    mesh_generator = MeshGeneration(metadata)
    mesh_generator.create_box()

    # Initialize ArucoTagFinder with marker length and calibration file
    aruco_processor = ArucoTagFinder(
        marker_length=1.905e-2, # m converted from 0.5 in
        calibration_file="calibration.json", 
        metadata=metadata
    )

    # Setup video capture
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Failed to open video capture")
        exit()

    last_ids = set()
    poses = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Detect ArUco markers
        corners, ids = aruco_processor.detect(frame)

        if corners and ids is not None:
            # Render AR models on detected markers
            frame, new_poses = aruco_processor.render_AR(frame, corners, ids, metadata)
            poses.update(new_poses)

            # Clear terminal screen
            os.system('cls' if os.name == 'nt' else 'clear')

            table_data = ['Tag ID', '\t\tPosition (x, y, z)', '\t\tRotation (Rx, Ry, Rz)']
            print("Detected Markers:")
            print(*table_data)
            # Display poses
            for marker_id, pose_data in poses.items():
                position = pose_data['tvec']
                rotation = pose_data['rvec']
                print(str(marker_id),
                    f"\t\t({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})",
                    f"\t\t({rotation[0]:.3f}, {rotation[1]:.3f}, {rotation[2]:.3f})"
                )

        # Show the frame
        cv2.imshow("AR with ArUco Markers", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Exit on ESC key
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()