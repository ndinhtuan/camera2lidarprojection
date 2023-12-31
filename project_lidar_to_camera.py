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
    
    result = np.matmul(lidar_to_platform_trans, lidarpoint.T).T
    result = np.matmul(np.linalg.inv(cam_to_platform_trans), result.T)
    lidar2cam = np.array([
    [-1.0000,   -0.0025,   -0.0039,    0.0035],
    [0.0040,   -0.0047,   -1.0000,   -0.1295],
    [0.0025,   -1.0000,    0.0047,    0.0800],
    [     0,         0,         0,    1.0000],
    ])
    result = np.matmul(lidar2cam, lidarpoint.T)
    result = result.T#*1000
    # print(result); exit()
    return result

# Convert lidar point (x,y,z) to image plane coordination (x,y)
def project_3dpoint_to_cam(point3d, cam_transformation_matrix):

    camera_matrix = np.array([
        [1.3663644624949141, 0., 0.95201409341783744], 
        [0., 1.3663644624949141, 0.59515378480409288],
        [0., 0., 1. ],
    ])
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
    # camera_matrix = np.identity(3)

    distortion_coeff = np.array([-1.5934723176777250e-02, -2.6588261328854116e-02, 0., 0., 0.])
    rvec = cam_transformation_matrix[:3, :3]
    rvec, _ = cv2.Rodrigues(rvec)
    rvec = np.squeeze(rvec, 1)
    tvec = cam_transformation_matrix[:-1,-1]
    rvec = np.array([0., 0., 0.])
    tvec = np.array([0., 0., 0.])
    
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

def draw_point_on_img(point2d, h, w, img, depth: np.ndarray):

    # depth = (depth / depth.max() * 255).astype(np.uint8)
    depth = (((depth - 4) / 16) * 255).clip(0, 255).astype(np.uint8)
    color = cv2.applyColorMap(depth, cv2.COLORMAP_TURBO)
    color = np.squeeze(color)
    for p, c in zip(point2d, color):
        
        if len(p) == 1:
            if (p[0][0] < 0 or p[0][0] > w) or  (p[0][1] < 0 or p[0][1] > h): continue
            p_img = (int(p[0][0]), int(p[0][1]))
            
        if len(p) == 2:
            if (p[0] < 0 or p[0] > w) or  (p[1] < 0 or p[1] > h): continue
            p_img = (int(p[0]), int(p[1]))

        # p_img = (int(p[0][1]), int(p[0][0]))
        # color = (0,0,int(d))
        # print("Halo: ", c)
        img = cv2.circle(img, p_img, 2, c.tolist(), -1) 

    return img

if __name__=="__main__":
    
    ply_file_path = "./000000.ply"
    img_path = "./image_511.png"

    #list_point = lidar_df["x"]
    list_point = read_ply(ply_file_path)
    list_point = list_point[:,:3]
    # list_point[:, [0, 2]] = list_point[:, [2, 0]]
    num_point, _ = list_point.shape
    tmp = np.ones(num_point)
    tmp = np.expand_dims(tmp, 1)
    list_point = np.hstack((list_point, tmp))

    # print(list_point[:, 1][:100]); exit()

    cam_to_platform_trans = create_transformation_mat_from_IPIeuler(ALPHA, ZETA, KAPPA, TRANS_VECT)
    cam_to_platform_trans = np.array([
        [0.001774, 0.009003, 0.999958, 0.853909],
        [-0.999997, 0.001483, 0.001761,  0.723899],
        [-0.001467, -0.999958,  0.009006, -0.101299],
        [0, 0, 0, 1],
        ])
    # right camera
    # cam_to_platform_trans = np.array([ 
    #     [0.001784,	-0.003942,	0.999991,	0.854994],
    #     [-0.999961,	-0.008651,	0.00175,	-0.14945],
    #     [0.008644,	-0.999955,	-0.003958,	-0.096643],
    #     [0,	0,	0	,1,],

    #   ])
    # cam_to_platform_trans[:, [0, 2]] = cam_to_platform_trans[:, [2, 0]]
    # LIDAR_TO_PLATFORM_TRANS[:, [1, 2]] = LIDAR_TO_PLATFORM_TRANS[:, [2, 1]]
    # Left camera
    extrinsic_cam = np.array([ 
        [0.99999870737359609, 0.0013758129524782047, -0.00083209966750479893, 0.87407729319618588], 
        [-0.0013751385733047051, 0.99999872605346241, 0.00081048464278761858, 0.00088546374096967848],
        [0.00083321368272364050, -0.00080933934278414944, 0.99999932536216596, -0.0039123668241561977],
        [0.,0.,0.,1.],
      ])
    
    #Right camera
    extrinsic_cam = np.array([
        [0.99998946978274161, 0.0010130161528465337, -0.0044759492742084388, 0.87407729319618588], 
        [-0.0010166536552653224, 0.99999915477098988, -0.00081047680465589538, 0.00088546374096967848],
        [0.0044751244649116400, 0.00081501876034946561, 0.99998965444920662, -0.0039123668241561977],
       [0, 0, 0, 1],
    ])

    # cam_to_platform_trans = LIDAR_TO_PLATFORM_TRANS
    list_point_cam = convert_lidarpoint_to_cam_coord(list_point, LIDAR_TO_PLATFORM_TRANS, cam_to_platform_trans)
    list_point_cam[list_point_cam[:,2] < 0] = [0,0,0,1]

    cam_to_platform_trans = np.linalg.inv(cam_to_platform_trans)
    # print(cam_to_platform_trans); exit()

    list_point_cam = list_point_cam[:,:3]
    depth = np.linalg.norm(list_point_cam, axis=1)
    camera_matrix = np.array([
        [1366.3644624949141, 0., 952.01409341783744], 
        [0., 1366.3644624949141, 595.15378480409288],
        [0., 0., 1. ],
    ])

    print(list_point_cam, list_point_cam.shape)
    # list_point_cam = camera_matrix @ list_point_cam.T
    # list_point_cam = list_point_cam.T
    # list_point_cam[:2,:] /= list_point_cam[2,:]
    print(list_point_cam, list_point_cam.shape)
    # print(list_point_cam); exit()
    point2d, _ = project_3dpoint_to_cam(list_point_cam, extrinsic_cam)
    # point2d = list_point_cam[:, :2]
    print(point2d)

    img = cv2.imread(img_path)
    h, w, _ = img.shape
    img_ = draw_point_on_img(point2d, h, w, img, depth)

    cv2.imshow("projection", img_)
    cv2.waitKey(0)
    cv2.imwrite("img.png", img_)
