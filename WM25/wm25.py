import cv2
import numpy as np
import glob
import os
import open3d as o3d

# --- CONFIGURATION ---

# Checkerboard Calibration Settings
CHECKERBOARD_IMAGES_DIR = 'checkerboard_images/'
CHECKERBOARD_IMAGE_FORMAT = '*.png'
CHECKERBOARD_CORNERS_WIDTH = 9
CHECKERBOARD_CORNERS_HEIGHT = 6
SQUARE_SIZE_MM = 23.3

# Stereo Calibration Settings

# gg2100 - gmach glowny
# mchtr1000 - wydzial mechatroniki
STEREO_CALIBRATION_DIR = 'mchtr1000/'

LEFT_STEREO_IMAGE_PATH = 'stereo_left.png'
RIGHT_STEREO_IMAGE_PATH = 'stereo_right.png'
IMAGE_SCALE_FACTOR = 0.5

# Display script steps:
SHOW_CALIBRATION = True
SHOW_UNDISTORT = True
SHOW_DISPARITY = True
SHOW_POINT_CLOUD = True

# 3D Point Cloud Settings
SAVE_POINT_CLOUD = True
FILTER_POINT_CLOUD = True
POINT_CLOUD_FILENAME = 'point_cloud.ply'

# Image specific settings
if STEREO_CALIBRATION_DIR == 'gg2100/':
    KNOWN_HORIZONTAL_BASELINE_MM = 2100.0
    DISP_COEFF = 2
    MIN_RAW_DISPARITY = 4
    MAX_RAW_DISPARITY = 256
if STEREO_CALIBRATION_DIR == 'mchtr1000/':
    KNOWN_HORIZONTAL_BASELINE_MM = 1000.0
    DISP_COEFF = 8
    MIN_RAW_DISPARITY = 64
    MAX_RAW_DISPARITY = 128

# Camera calibration - https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
def calibrate_camera(images_dir, board_width, board_height, square_size):
    print("\n--- Starting Camera Intrinsic Calibration ---")
    term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((board_height * board_width, 3), np.float32)
    objp[:,:2] = np.mgrid[0:board_width, 0:board_height].T.reshape(-1, 2) * square_size

    obj_points = []  # 3d point in real world space
    img_points = []  # 2d points in image plane

    print("Searching for checkerboard images...")
    images = glob.glob(os.path.join(images_dir, CHECKERBOARD_IMAGE_FORMAT))

    for fname in images:
        img = cv2.imread(fname)

        if IMAGE_SCALE_FACTOR != 1.0:
            img = cv2.resize(img, None, fx=IMAGE_SCALE_FACTOR, fy=IMAGE_SCALE_FACTOR, interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret,corners = cv2.findChessboardCorners(gray, (board_width, board_height), None)
        if ret:
            obj_points.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term_criteria)
            img_points.append(corners2)

            if SHOW_CALIBRATION:
                cv2.drawChessboardCorners(img, (board_width, board_height), corners2, ret)
                cv2.imshow('img', img)
                cv2.waitKey(500)
        else:
            print(f"Checkerboard not found in {fname}")

    print("Calibrating camera...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    if ret:
        print("Camera calibration successful.")
        print("Camera Matrix (K):\n", camera_matrix)
        print("Distortion Coefficients:\n", dist_coeffs)
        return camera_matrix, dist_coeffs
    else:
        print("Camera calibration failed.")
        return None, None


def undistort_image(image, camera_matrix, dist_coeffs):
    print("Undistorting image...")
    h, w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
    x, y, w, h = roi
    undistorted_image = undistorted_image[y:y+h, x:x+w]

    if SHOW_UNDISTORT:
        cv2.imshow('Undistorted Image', undistorted_image)
        cv2.waitKey(500)
    return undistorted_image

def load_image(image_path, median_blur=3, scale_factor=IMAGE_SCALE_FACTOR):
    print(f"Loading Image: {image_path}...")
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    if scale_factor != 1.0:
        img = cv2.resize(img, None, fx=IMAGE_SCALE_FACTOR, fy=IMAGE_SCALE_FACTOR, interpolation=cv2.INTER_AREA)
       
    if median_blur > 0:
        img = cv2.medianBlur(img, median_blur)

    return img



def main():
    mtx, dist = calibrate_camera(CHECKERBOARD_IMAGES_DIR,
                                 CHECKERBOARD_CORNERS_WIDTH,
                                 CHECKERBOARD_CORNERS_HEIGHT,
                                 SQUARE_SIZE_MM)

    print(f"\n--- Loading stereo images from '{STEREO_CALIBRATION_DIR}' ---")
    
    left_image_full_path = os.path.join(STEREO_CALIBRATION_DIR, LEFT_STEREO_IMAGE_PATH)
    imgLeft = load_image(left_image_full_path)
    imgLeft = undistort_image(imgLeft, mtx, dist)

    right_image_full_path = os.path.join(STEREO_CALIBRATION_DIR, RIGHT_STEREO_IMAGE_PATH)
    imgRight = load_image(right_image_full_path)
    imgRight = undistort_image(imgRight, mtx, dist)

    stereo_sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16*DISP_COEFF,
        blockSize=3,
        P1=8 * 3 * 5**2,
        P2=32 * 3 * 5**2,
        disp12MaxDiff=8,
        uniquenessRatio=5,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.StereoSGBM_MODE_SGBM_3WAY
    )
    
    disparity_map_sgbm = stereo_sgbm.compute(imgLeft, imgRight)
    disparity_map_display_normalized = cv2.normalize(disparity_map_sgbm, None, alpha=0, beta=255,
                                                         norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    if SHOW_DISPARITY:
        cv2.imshow('Disparity Map SGBM', disparity_map_display_normalized)
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()

    # --- Stereo Rectification (to get Q)---
    print(f"\n--- Performing Stereo Rectification ---")
    R = np.eye(3)  # Identity rotation matrix
    T = np.float32([-KNOWN_HORIZONTAL_BASELINE_MM, 0, 0]) # Translation vector along the x-axis
    h, w = imgLeft.shape[:2]
    image_size_stereo = (w, h)

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        mtx, dist,
        mtx, dist,  # Same camera intrinsics for both
        image_size_stereo,
        R, T,
        alpha=0.9
    )
        
    # --- 3D Point Cloud Generation ---
    print(f"\n--- Generating 3D Point Cloud ---")
    print("Reprojecting image to 3D...")
    points_3D_cv = cv2.reprojectImageTo3D(disparity_map_sgbm, Q)
    colors_cv = cv2.cvtColor(imgLeft, cv2.COLOR_BGR2RGB)

    mask = (disparity_map_sgbm > MIN_RAW_DISPARITY*16) & (disparity_map_sgbm < MAX_RAW_DISPARITY*16)

    points_for_pcd = points_3D_cv[mask]
    colors_for_pcd = colors_cv[mask]

    print(f"\n--- Rendering point cloud with Open3D ---")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_for_pcd)
    pcd.colors = o3d.utility.Vector3dVector(colors_for_pcd / 255.0)   
    print(f"Initial point cloud created with {len(pcd.points)} points.")
    
    if FILTER_POINT_CLOUD:
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.5)
        pcd_final = pcd.select_by_index(ind)
        print(f"Point cloud filtered by Open3D: {len(pcd_final.points)} points remaining.")
    else:
        pcd_final = pcd

    if SAVE_POINT_CLOUD:
        try:
            o3d.io.write_point_cloud(POINT_CLOUD_FILENAME, pcd_final)
            print(f"Point cloud saved to '{POINT_CLOUD_FILENAME}'")
        except Exception as e:
            print(f"Error saving point cloud: {e}")

    if SHOW_POINT_CLOUD:
        print("Visualizing point cloud with Open3D...")
        o3d.visualization.draw_geometries([pcd_final],
                                          window_name="Reconstructed 3D Point Cloud",
                                          width=1600, height=900)
    print(f"--- Open3D Processing Finished ---")
    print("\n--- Script Completed ---")


if __name__ == "__main__":
    main()
