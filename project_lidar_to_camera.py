import numpy as np
import cv2
from plyfile import PlyData
import pandas as pd
import copy

lidar_to_platform_mat = np.array([
   [ 0.000697,-0.999962,0.008683,0.935490],
   [ 0.999920,0.000806,0.012599,-0.151705],
   [ -0.012605,0.008673,0.999883,0.032571],
   [ 0.000000,0.000000,0.000000,1.000000]
   ])

cam_to_platform_mat = None
ALPHA=3.15815
ZETA=1.5418
KAPPA=1.5265

TRANS_VECT = np.array([0.541, 0.0961, 0.0938])
LIDAR_TO_PLATFORM_TRANS = np.array([
    [0.000697,-0.999962,0.008683,0.935490],
    [0.999920,0.000806,0.012599,-0.151705],
    [-0.012605,0.008673,0.999883,0.032571],
    [0.000000,0.000000,0.000000,1.000000],
    ])

# maybe need to swap row 0 and row 2 if projection not right
def create_transformation_mat_from_IPIeuler(alpha, zeta, kappa, translation):
    
    a = alpha
    z = zeta
    k = kappa
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
    
    #tmp = copy.deepcopy(R[0,:])
    #R[0,:] = R[2,:]
    #R[2,:] = tmp
    
    translation = np.expand_dims(translation, 1)
    R = np.hstack((R, translation))
    tmp = np.array([0, 0, 0, 1])
    R = np.vstack((R, tmp))

    return R

# Convert lidar point cloud (x, y, z) to (x, y, z) in camera coordination system
# camera_3d = lidar_3d @ transform_matrix_lidar_to_platform @ invert(tranform_matrix_camera_to_platform)
def convert_lidarpoint_to_cam_coord(lidarpoint, lidar_to_platform_trans, cam_to_platform_trans):
    
    result = np.matmul(lidarpoint, lidar_to_platform_trans.T)
    result = np.matmul(result, np.linalg.inv(cam_to_platform_trans).T)
    return result

# Convert lidar point (x,y,z) to image plane coordination (x,y)
def project_3dpoint_to_cam(point3d):

    camera_matrix = np.array([
        [ -1.3663644624949141, 0., 0.95201409341783744], 
        [0., -1.3663644624949141, 0.59515378480409288],
        [0., 0., 1. ],
    ])

    distortion_coeff = np.array([-1.5934723176777250e-02, -2.6588261328854116e-02, 0., 0., 0.])
    rvec = tvec = np.array([0., 0., 0.])
    
    point3d = np.array(point3d, dtype=np.float32)
    result = cv2.projectPoints(point3d, rvec, tvec, camera_matrix, distortion_coeff)
    return result

def read_ply(ply_path):

    plydata = PlyData.read(ply_path)
    data = plydata.elements[0].data
    data_pd = pd.DataFrame(data)
    data_np = np.zeros(data_pd.shape, dtype=np.float64)
    property_names = data[0].dtype.names

    for i, name in enumerate(property_names):
        data_np[:, i] = data_pd[name]

    return data_np

def draw_point_on_img(point2d, h, w, img):

    for p in point2d:
        if (p[0][0] < 0 or p[0][0] > 1) or  (p[0][1] < 0 or p[0][1] > 1): continue

        p_img = (int(w*p[0][0]), int(h*p[0][1]))
        img = cv2.circle(img, p_img, 2, [0, 0, 255], 0) 

    return img

if __name__=="__main__":
    
    ply_file_path = "./000000.ply"
    img_path = "./image_511.png"

    #list_point = lidar_df["x"]
    list_point = read_ply(ply_file_path)
    list_point = list_point[:,:3]
    num_point, _ = list_point.shape
    tmp = np.ones(num_point)
    tmp = np.expand_dims(tmp, 1)
    list_point = np.hstack((list_point, tmp))

    cam_to_platform_trans = create_transformation_mat_from_IPIeuler(ALPHA, ZETA, KAPPA, TRANS_VECT)
    cam_to_platform_trans = np.array([
        [0.001774, 0.009003, 0.999958, 0.853909],
        [-0.999997, 0.001483, 0.001761,  0.723899],
        [-0.001467, -0.999958,  0.009006, -0.101299],
        [0, 0, 0, 1],
        ])
    print(cam_to_platform_trans)#; exit()

    list_point_cam = convert_lidarpoint_to_cam_coord(list_point, LIDAR_TO_PLATFORM_TRANS, cam_to_platform_trans)
    list_point_cam = list_point_cam[:,:3]
    point2d, _ = project_3dpoint_to_cam(list_point_cam)
    print(point2d.shape)

    img = cv2.imread(img_path)
    h, w, _ = img.shape
    img_ = draw_point_on_img(point2d, h, w, img)

    cv2.imshow("projection", img_)
    cv2.waitKey(0)
