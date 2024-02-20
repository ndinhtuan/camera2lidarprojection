import numpy as np
import cv2
from plyfile import PlyData
import pandas as pd
import copy

ALPHA=3.15815
ZETA=1.5418
KAPPA=1.5265

TRANS_VECT = np.array([0.541, -0.0961, 0.0938])
# TRANS_VECT = np.array([0.874, 0.0009, -0.0039])
LIDAR_TO_PLATFORM_TRANS = np.array([
    [0.000697,-0.999962,0.008683,0.935490],
    [0.999920,0.000806,0.012599,-0.151705],
    [-0.012605,0.008673,0.999883,0.032571],
    [0.000000,0.000000,0.000000,1.000000],
    ])

class LidarCameraCalib(object):

    def __init__(self):
        
        self.lidar2cam_type = "ipi" # or "ipi", t_probe is matrix provided from Dominik

    # Convert lidar point cloud (x, y, z) to (x, y, z) in camera coordination system
    # camera_3d = lidar_3d @ transform_matrix_lidar_to_platform @ invert(tranform_matrix_camera_to_platform)
    def convert_lidarpoint_to_cam_coord(self, lidarpoint, lidar_to_platform_trans, cam_to_platform_trans, filter_behind_points=True):
        
        if self.lidar2cam_type == "ipi":
            tmp = np.matmul(lidar_to_platform_trans, lidarpoint.T).T
            result = np.matmul(np.linalg.inv(cam_to_platform_trans), tmp.T)

        elif self.lidar2cam_type == "t_probe":
            # Matrix from Dominik
            lidar2cam = np.array([
            [-1.0000,   -0.0025,   -0.0039,    0.0035],
            [0.0040,   -0.0047,   -1.0000,   -0.1295],
            [0.0025,   -1.0000,    0.0047,    0.0800],
            [     0,         0,         0,    1.0000],
            ])
            result = np.matmul(lidar2cam, lidarpoint.T)
            
        else:
            print("No specified type")

        result = result.T#*1000

        if filter_behind_points:
            result[result[:,2] < 0] = [0,0,0,1]

        return result[:, :3] # Extract 3D point from homogeneous coordination

    # Convert lidar point (x,y,z) to image plane coordination (x,y)
    def project_3dpoint_to_cam(self, point3d):

        # Left camera
        camera_matrix = np.array([
            [1366.3644624949141, 0., 952.01409341783744], 
            [0., 1366.3644624949141, 595.15378480409288],
            [0., 0., 1. ],
        ])
        # Right camera
        camera_matrix = np.array([
            [1366.448431879, 0., 953.938635794], 
            [0., 1366.448431879, 593.6245488],
            [0., 0., 1. ],
        ])

        distortion_coeff = np.array([-1.5934723176777250e-02, -2.6588261328854116e-02, 0., 0., 0.])
        rvec = np.array([0., 0., 0.])
        tvec = np.array([0., 0., 0.])
        
        point3d = np.array(point3d, dtype=np.float32)
        result = cv2.projectPoints(point3d, rvec, tvec, camera_matrix, distortion_coeff)
        return result

    def draw_point_on_img(self, point2d, h, w, img, depth: np.ndarray):

        depth = (((depth - 2) / 20) * 255).clip(0, 255).astype(np.uint8)
        color = cv2.applyColorMap(depth, cv2.COLORMAP_TURBO)
        color = np.squeeze(color)

        for p, c in zip(point2d, color):
            
            if len(p) == 1:
                if (p[0][0] < 0 or p[0][0] > w) or  (p[0][1] < 0 or p[0][1] > h): continue
                p_img = (int(p[0][0]), int(p[0][1]))
                
            if len(p) == 2:
                if (p[0] < 0 or p[0] > w) or  (p[1] < 0 or p[1] > h): continue
                p_img = (int(p[0]), int(p[1]))

            img = cv2.circle(img, p_img, 2, c.tolist(), -1) 

        return img

# maybe need to swap row 0 and row 2 if projection not right
def create_transformation_mat_from_IPIeuler(alpha, zeta, kappa, translation):
    
    a = alpha #+ np.pi
    z = kappa  + np.pi
    k = zeta #+ np.pi
    cos = np.cos
    sin = np.sin
    
    R = np.zeros((3, 3))
    R[0, 0] = cos(a)*cos(z)*cos(k) - sin(a)*sin(k)
    R[0, 1] = -cos(a)*cos(z)*sin(k) - sin(a)*cos(k)
    R[0, 2] = cos(a)*sin(z)

    R[1, 0] = sin(a)*cos(z)*cos(k)+cos(a)*sin(k)
    R[1, 1] = -sin(a)*cos(z)*sin(k)+cos(a)*cos(k)
    R[1, 2] = sin(a)*sin(z)

    R[2, 0] = -sin(z)*cos(k)
    R[2, 1] = sin(z)*sin(k)
    R[2, 2] = cos(z)
    
    translation = np.expand_dims(translation, 1)
    R = np.hstack((R, translation))
    tmp = np.array([0, 0, 0, 1])
    R = np.vstack((R, tmp))

    return R


def read_ply(ply_path):

    plydata = PlyData.read(ply_path)
    data = plydata.elements[0].data
    data_pd = pd.DataFrame(data)
    data_np = np.zeros(data_pd.shape, dtype=np.float64)
    property_names = data[0].dtype.names

    for i, name in enumerate(property_names):
        data_np[:, i] = data_pd[name]

    return data_np


if __name__=="__main__":
    
    ply_file_path = "./000000.ply"
    img_path = "./image_511_r.png" # right image

    # Read Lidar point
    list_point = read_ply(ply_file_path)
    list_point = list_point[:,:3]
    num_point, _ = list_point.shape
    tmp = np.ones(num_point)
    tmp = np.expand_dims(tmp, 1)
    list_point = np.hstack((list_point, tmp))

    lidar_cam_calib = LidarCameraCalib()

    cam_to_platform_trans = create_transformation_mat_from_IPIeuler(ALPHA, ZETA, KAPPA, TRANS_VECT)

    list_point_cam = lidar_cam_calib.convert_lidarpoint_to_cam_coord(list_point, LIDAR_TO_PLATFORM_TRANS, cam_to_platform_trans)

    list_point_cam = list_point_cam[:,:3]
    depth = np.linalg.norm(list_point_cam, axis=1)

    point2d, _ = lidar_cam_calib.project_3dpoint_to_cam(list_point_cam)

    img = cv2.imread(img_path)
    h, w, _ = img.shape
    img_ = lidar_cam_calib.draw_point_on_img(point2d, h, w, img, depth)

    cv2.imshow("projection", img_)
    cv2.waitKey(0)
    cv2.imwrite("projected_img.png", img_)
