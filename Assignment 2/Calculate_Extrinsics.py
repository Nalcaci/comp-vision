import cv2 as cv
import numpy as np
import glob
import xml.etree.ElementTree as ET

def load_intrinsics(file_path):
    fs = cv.FileStorage(file_path, cv.FILE_STORAGE_READ)
    camera_matrix = fs.getNode("CameraMatrix").mat()
    dist_coeffs = fs.getNode("DistortionCoeffs").mat()
    fs.release()
    return camera_matrix, dist_coeffs

def load_checkerboard_info(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    width = int(root.find("CheckerBoardWidth").text)
    height = int(root.find("CheckerBoardHeight").text)
    square_size = float(root.find("CheckerBoardSquareSize").text)
    return (width, height), square_size

def select_corners(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        if len(param['corners']) < 4:
            param['corners'].append((x, y))
            print(f"Corner {len(param['corners'])} selected at ({x}, {y})")
            if len(param['corners']) == 4:
                print("All 4 corners selected.")

def find_corners(image_path, chessboard_size):
    img = cv.imread(image_path)
    
    print(f"Select the 4 corners manually for {image_path}")
    manual_corners = []
    cv.namedWindow('img')
    cv.setMouseCallback('img', select_corners, {'corners': manual_corners})
    
    while len(manual_corners) < 4:
        cv.imshow('img', img)
        cv.waitKey(1)
    
    cv.destroyAllWindows()
    manual_corners = np.array(manual_corners, dtype=np.float32)
    
    # Define destination points with Y axis flipped (Y up)
    dst_points = np.array([
        [0, 0],
        [chessboard_size[0] - 1, 0],
        [chessboard_size[0] - 1, -(chessboard_size[1] - 1)],
        [0, -(chessboard_size[1] - 1)]
    ], dtype=np.float32)
    
    H, _ = cv.findHomography(manual_corners, dst_points)
    # Generate grid points for the chessboard with Y axis up
    x_grid, y_grid = np.meshgrid(range(chessboard_size[0]), -np.array(range(chessboard_size[1])))
    grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T.astype(np.float32)
    projected_corners = cv.perspectiveTransform(grid_points.reshape(1, -1, 2), np.linalg.inv(H))
    projected_corners = projected_corners.reshape(-1, 1, 2)
    
    for point in projected_corners:
        cv.circle(img, tuple(point.ravel().astype(int)), 5, (0, 255, 0), -1)
    cv.imshow('Projected Corners', img)
    cv.waitKey(5000)
    cv.destroyAllWindows()
    
    return True, projected_corners

def compute_extrinsics(image_path, chessboard_size, square_size, camera_matrix, dist_coeffs):
    # Create object points with Y axis pointing up (flip y coordinate)
    objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
    grid = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    grid[:, 1] = -grid[:, 1]  # Flip y axis so Y increases upward
    objp[:, :2] = grid * square_size
    
    ret, imgpoints = find_corners(image_path, chessboard_size)
    if not ret:
        print(f"Failed to find chessboard corners in {image_path}")
        return None, None
    
    ret, rvec, tvec = cv.solvePnP(objp, imgpoints, camera_matrix, dist_coeffs)
    return rvec, tvec

def save_extrinsics(file_path, rvec, tvec):
    fs = cv.FileStorage(file_path, cv.FILE_STORAGE_WRITE)
    fs.write("rvec", rvec)
    fs.write("tvec", tvec)
    fs.release()

def visualize_world_coordinates(image_path, camera_matrix, dist_coeffs, rvec, tvec, square_size):
    img = cv.imread(image_path)
    # Define an axis length (e.g., 3 squares long)
    axis_length = 3 * square_size
    # Draw the coordinate axes on the image
    cv.drawFrameAxes(img, camera_matrix, dist_coeffs, rvec, tvec, axis_length)
    # Overlay the translation vector (the world coordinate of the starting corner)
    cv.putText(img, f"T: [{tvec[0][0]:.2f}, {tvec[1][0]:.2f}, {tvec[2][0]:.2f}]", (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv.imshow("World Coordinates", img)
    cv.waitKey(5000)
    cv.destroyAllWindows()

def main():
    base_path = "Assignment 2/data/"
    checkerboard_file = "Assignment 2/data/checkerboard.xml"
    chessboard_size, square_size = load_checkerboard_info(checkerboard_file)
    
    for cam_id in range(1, 5):
        cam_path = f"{base_path}cam{cam_id}/"
        intrinsics_file = f"{cam_path}intrinsics.xml"
        image_file = f"{cam_path}checkerboard.png"
        extrinsics_file = f"{cam_path}extrinsics.xml"
        
        camera_matrix, dist_coeffs = load_intrinsics(intrinsics_file)
        rvec, tvec = compute_extrinsics(image_file, chessboard_size, square_size, camera_matrix, dist_coeffs)
        
        if rvec is not None and tvec is not None:
            save_extrinsics(extrinsics_file, rvec, tvec)
            visualize_world_coordinates(image_file, camera_matrix, dist_coeffs, rvec, tvec, square_size)
            print(f"Extrinsics saved and world coordinates visualized for camera {cam_id}")
        else:
            print(f"Could not compute extrinsics for camera {cam_id}")

if __name__ == "__main__":
    main()
