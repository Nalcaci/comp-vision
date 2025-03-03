import cv2 as cv
import numpy as np
import xml.etree.ElementTree as ET

def load_intrinsics(file_path):
    fs = cv.FileStorage(file_path, cv.FILE_STORAGE_READ)
    camera_matrix = fs.getNode("CameraMatrix").mat()
    dist_coeffs = fs.getNode("DistortionCoeffs").mat()
    fs.release()
    return camera_matrix, dist_coeffs

def load_extrinsics(file_path):
    fs = cv.FileStorage(file_path, cv.FILE_STORAGE_READ)
    rvec = fs.getNode("rvec").mat()
    tvec = fs.getNode("tvec").mat()
    fs.release()
    return rvec, tvec

def load_checkerboard_info(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    square_size = float(root.find("CheckerBoardSquareSize").text)
    return square_size

def draw_axes_on_chessboard(image, mtx, dist, rvec, tvec, square_size):
    origin = np.array([[0, 0, 0]], dtype=np.float32)
    x_axis = np.array([[3 * square_size, 0, 0]], dtype=np.float32)
    y_axis = np.array([[0, 3 * square_size, 0]], dtype=np.float32)
    z_axis = np.array([[0, 0, -3 * square_size]], dtype=np.float32)

    imgpts_origin, _ = cv.projectPoints(origin, rvec, tvec, mtx, dist)
    imgpts_x, _ = cv.projectPoints(x_axis, rvec, tvec, mtx, dist)
    imgpts_y, _ = cv.projectPoints(y_axis, rvec, tvec, mtx, dist)
    imgpts_z, _ = cv.projectPoints(z_axis, rvec, tvec, mtx, dist)

    origin_pt = tuple(imgpts_origin.ravel().astype(int))
    x_pt = tuple(imgpts_x.ravel().astype(int))
    y_pt = tuple(imgpts_y.ravel().astype(int))
    z_pt = tuple(imgpts_z.ravel().astype(int))

    cv.line(image, origin_pt, x_pt, (0, 0, 255), 3)
    cv.line(image, origin_pt, y_pt, (0, 255, 0), 3)
    cv.line(image, origin_pt, z_pt, (255, 0, 0), 3)
    
    cv.putText(image, "X", x_pt, cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv.putText(image, "Y", y_pt, cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv.putText(image, "Z", z_pt, cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    return image

def main():
    base_path = "Assignment 2/data/"
    checkerboard_file = f"{base_path}checkerboard.xml"
    square_size = load_checkerboard_info(checkerboard_file)
    
    for cam_id in range(1, 5):
        cam_path = f"{base_path}cam{cam_id}/"
        intrinsics_file = f"{cam_path}intrinsics.xml"
        extrinsics_file = f"{cam_path}extrinsics.xml"
        image_file = f"{cam_path}checkerboard.png"
        output_file = f"{cam_path}axes_visualization.png"
        
        camera_matrix, dist_coeffs = load_intrinsics(intrinsics_file)
        rvec, tvec = load_extrinsics(extrinsics_file)
        img = cv.imread(image_file)
        
        img_with_axes = draw_axes_on_chessboard(img, camera_matrix, dist_coeffs, rvec, tvec, square_size)
        
        cv.imshow("3D Axes Visualization", img_with_axes)
        cv.waitKey(5000)
        cv.destroyAllWindows()
        
        cv.imwrite(output_file, img_with_axes)
        print(f"3D axes visualization saved for camera {cam_id}")

if __name__ == "__main__":
    main()
